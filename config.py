import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
NGROK_TOKEN = os.environ.get("NGROK_TOKEN", "")

DATA_DIR        = os.environ.get("DATA_DIR", "./data")
CACHE_DIR       = os.environ.get("CACHE_DIR", "./cache")      # scraped-page cache
VECTORSTORE_DIR = os.environ.get("VECTORSTORE_DIR", "./vectorstore")

EMBED_MODEL    = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL      = "mistralai/Ministral-8B-Instruct-2410"

CHUNK_SIZE    = 512
CHUNK_OVERLAP = 100

K_DENSE          = 15   
K_SPARSE         = 10   
RERANK_CANDIDATE = 20   
RERANK_TOP_K     = 6    

MAX_NEW_TOKENS     = 768
TEMPERATURE        = 0.3
TOP_P              = 0.9
REPETITION_PENALTY = 1.05

MAX_HISTORY_TURNS = 4   

SCRAPE_TIMEOUT       = 15   
SCRAPE_CACHE_TTL_H   = 6    
SCRAPE_MAX_WORKERS   = 8    

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
