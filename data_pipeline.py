import os
import json
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DYNAMIC_URLS = {
    "Admission deadlines": [
        "https://ewubd.edu/undergraduate-dates-deadline",
        "https://ewubd.edu/graduate-dates-deadline"
    ],
    "Events": ["https://ewubd.edu/events"],
    "Faculty": [
        "https://fse.ewubd.edu/computer-science-engineering/faculty-members",
        "https://fse.ewubd.edu/electrical-electronic-engineering/faculty-members",
        "https://fse.ewubd.edu/electronics-communications-engineering/faculty-members",
        "https://fse.ewubd.edu/genetic-engineering-biotechnology/faculty-members",
        "https://fse.ewubd.edu/pharmacy-department/faculty-members",
        "https://fse.ewubd.edu/civil-engineering/faculty-members",
        "https://fse.ewubd.edu/mathematical-physical-science/faculty-members",
        "https://fbe.ewubd.edu/business-administration/faculty-members",
        "https://fbe.ewubd.edu/economics-department/faculty-members",
        "https://flass.ewubd.edu/english-department/faculty-members",
        "https://flass.ewubd.edu/law-department/faculty-members",
        "https://flass.ewubd.edu/social-relations-department/faculty-members",
        "https://flass.ewubd.edu/information-studies-library-management/faculty-members",
        "https://flass.ewubd.edu/sociology-department/faculty-members"
    ],
    "Grading": ["https://www.ewubd.edu/grades-rules-and-regulations"],
    "Tuition fees": ["https://ewubd.edu/undergraduate-tuition-fees"]
}

def flatten_json(data, parent_key="", file_name=""):
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
                metadata={"source": file_name, "path": parent_key}
            ))
    return docs

def load_json_docs(data_directory):
    docs = []
    print("Loading JSON files...")
    for file_name in os.listdir(data_directory):
        if not file_name.endswith(".json"): continue
        file_path = os.path.join(data_directory, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            docs.extend(flatten_json(data, file_name, file_name))
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
    return docs

def chunk_documents(docs, chunk_size=512, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

def scrape_dynamic_docs():
    docs = []
    headers = {"User-Agent": "Mozilla/5.0"}
    print("🔍 Scraping live EWU website data...")
    for category, urls in DYNAMIC_URLS.items():
        for url in urls:
            try:
                r = requests.get(url, headers=headers, timeout=15)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                for script in soup(["script", "style"]): script.decompose()
                text = soup.get_text(separator="\n", strip=True)
                if text:
                    docs.append(Document(
                        page_content=f"LATEST {category.upper()} INFO ({url}):\n{text}",
                        metadata={"source": url, "category": category}
                    ))
            except Exception as e:
                print(f" Failed to scrape {url}: {e}")
    return docs