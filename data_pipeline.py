import os
import json
import frontmatter
import time
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    DATA_DIR, CACHE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, DYNAMIC_URLS,
    SCRAPE_TIMEOUT, SCRAPE_CACHE_TTL_H, SCRAPE_MAX_WORKERS, SCRAPE_RETRIES,
)

log = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; EWU-RAG/1.0)"}
_NOISE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]


# ──────────────────────────────────────────────────────────────────
# JSON knowledge base
# ──────────────────────────────────────────────────────────────────
def flatten_json(data, parent_key: str = "", file_name: str = ""):
    docs = []
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            docs.extend(flatten_json(value, new_key, file_name))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_key = f"{parent_key}[{i}]"
            docs.extend(flatten_json(item, new_key, file_name))
    else:
        text_value = str(data).strip()
        if text_value:
            docs.append(Document(
                page_content=f"{parent_key}: {text_value}",
                metadata={"source": file_name, "path": parent_key, "type": "json"},
            ))
    return docs


def load_json_docs(data_directory: str = DATA_DIR):
    docs = []
    log.info("Loading JSON files from %s", data_directory)
    if not os.path.isdir(data_directory):
        log.warning("Data directory does not exist: %s", data_directory)
        return docs
    for root, _dirs, files in os.walk(data_directory):
        for file_name in files:
            if not file_name.endswith(".json"):
                continue
            file_path = os.path.join(root, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                docs.extend(flatten_json(data, parent_key="", file_name=file_name))
            except Exception as e:
                log.error("Error loading %s: %s", file_name, e)
    log.info("Loaded %d JSON leaf documents", len(docs))
    return docs


# ──────────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────────
def chunk_documents(docs, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n|", "\n", "। ", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def load_markdown_docs(data_directory: str = DATA_DIR):
    docs = []
    log.info("Loading Markdown files from %s", data_directory)
    if not os.path.isdir(data_directory):
        log.warning("Data directory does not exist: %s", data_directory)
        return docs
    for root, _dirs, files in os.walk(data_directory):
        for file_name in files:
            if not file_name.endswith(".md"):
                continue
            file_path = os.path.join(root, file_name)
            try:
                post = frontmatter.load(file_path)
                inferred_category = (
                    post.metadata.get("category")
                    or os.path.splitext(file_name)[0].replace("_", " ").replace("-", " ").lower()
                )
                docs.append(Document(
                    page_content=post.content,
                    metadata={
                        **post.metadata,
                        "source": post.metadata.get("source", file_path),
                        "category": inferred_category,
                        "type": "markdown",
                    }
                ))
            except Exception as e:
                log.error("Error loading %s: %s", file_name, e)
    log.info("Loaded %d markdown documents", len(docs))
    return docs
# ──────────────────────────────────────────────────────────────────
# Web scraping (cached, parallel, retrying)
# ──────────────────────────────────────────────────────────────────
def _cache_path(url: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, hashlib.md5(url.encode()).hexdigest() + ".txt")


def _cache_fresh(path: str) -> bool:
    if not os.path.exists(path):
        return False
    age_h = (time.time() - os.path.getmtime(path)) / 3600.0
    return age_h < SCRAPE_CACHE_TTL_H


def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n", strip=True)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def _scrape_one(category: str, url: str) -> Document | None:
    cache = _cache_path(url)
    if _cache_fresh(cache):
        with open(cache, "r", encoding="utf-8") as f:
            text = f.read()
        log.info("cache hit  %s", url)
    else:
        text = ""
        for attempt in range(1, SCRAPE_RETRIES + 2):
            try:
                r = requests.get(url, headers=_HEADERS, timeout=SCRAPE_TIMEOUT)
                r.raise_for_status()
                text = _extract_text(r.text)
                if len(text) < 200:
                    log.warning("Very short content (%d chars) at %s — page may require JS rendering", len(text), url)
                with open(cache, "w", encoding="utf-8") as f:
                    f.write(text)
                log.info("scraped    %s (%d chars)", url, len(text))
                break
            except Exception as e:
                log.warning("attempt %d failed for %s: %s", attempt, url, e)
                time.sleep(attempt)  # simple backoff
        if not text and os.path.exists(cache):  # fall back to stale cache
            with open(cache, "r", encoding="utf-8") as f:
                text = f.read()
            log.info("stale cache fallback %s", url)
    if not text:
        return None
    scraped_at = datetime.now(timezone.utc).isoformat()
    return Document(
        page_content=f"LATEST {category.upper()} INFO ({url}):\n{text}",
        metadata={"source": url, "category": category, "type": "web", "scraped_at": scraped_at},
    )


def scrape_dynamic_docs():
    jobs = [(cat, url) for cat, urls in DYNAMIC_URLS.items() for url in urls]
    log.info("Scraping %d EWU pages (max_workers=%d)…", len(jobs), SCRAPE_MAX_WORKERS)
    docs = []
    with ThreadPoolExecutor(max_workers=SCRAPE_MAX_WORKERS) as ex:
        futures = {ex.submit(_scrape_one, cat, url): url for cat, url in jobs}
        for fut in as_completed(futures):
            doc = fut.result()
            if doc:
                docs.append(doc)
    log.info("Scraped %d pages successfully", len(docs))
    return docs
