import os
import shutil
shutil.rmtree("./vectorstore", ignore_errors=True)

# ── Secrets / tokens ──────────────────────────────────────────────
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
NGROK_TOKEN = os.environ.get("NGROK_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ── Paths ─────────────────────────────────────────────────────────
DATA_DIR        = os.environ.get("DATA_DIR", "./data")
CACHE_DIR       = os.environ.get("CACHE_DIR", "./cache")
VECTORSTORE_DIR = os.environ.get("VECTORSTORE_DIR", "./vectorstore")

# ── Models ────────────────────────────────────────────────────────
EMBED_MODEL    = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-gemma"
LLM_MODEL      = "llama-3.3-70b-versatile"

# ── GPU assignment ────────────────────────────────────────────────
EMBED_DEVICE   = "cuda:0"
RERANK_DEVICE  = "cuda:1"

# ── Chunking ──────────────────────────────────────────────────────
CHUNK_SIZE    = 256
CHUNK_OVERLAP = 50

# ── Retrieval / fusion ────────────────────────────────────────────
K_DENSE          = 15
K_SPARSE         = 10
RERANK_CANDIDATE = 30
RERANK_TOP_K     = 8
RRF_K            = 60
CATEGORY_BOOST   = 0.02

# ── Generation ────────────────────────────────────────────────────
MAX_NEW_TOKENS = 768
TEMPERATURE    = 0.3
TOP_P          = 0.9

MAX_HISTORY_TURNS = 4

# ── Scraping ──────────────────────────────────────────────────────
SCRAPE_TIMEOUT     = 15
SCRAPE_CACHE_TTL_H = 6
SCRAPE_MAX_WORKERS = 8
SCRAPE_RETRIES     = 2

DYNAMIC_URLS = {
    "Admission deadlines": [
        "https://ewubd.edu/undergraduate-dates-deadline",
        "https://ewubd.edu/graduate-dates-deadline",
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
        "https://flass.ewubd.edu/sociology-department/faculty-members",
    ],
    "Grading": ["https://www.ewubd.edu/grades-rules-and-regulations"],
    "Tuition fees": ["https://ewubd.edu/undergraduate-tuition-fees"],
}