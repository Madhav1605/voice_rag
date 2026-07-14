import os
from dotenv import load_dotenv
load_dotenv()

HASH_ALGORITHM = "sha256"

# PATHS

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
DATA_DIR = os.path.join(BASE_DIR,"data")
DOCUMENTS_DIR = os.path.join(DATA_DIR,"documents")
IMAGES_DIR = os.path.join(DATA_DIR,"images")
METADATA_DIR = os.path.join(BASE_DIR,"metadata")
DB_ROOT = os.path.join(BASE_DIR,"db")
DB_DIR = os.path.join(DB_ROOT,"chroma_db")
LOGS_DIR = os.path.join(BASE_DIR,"logs")

# CREATE DIRECTORIES

os.makedirs(DATA_DIR,exist_ok=True)
os.makedirs(DOCUMENTS_DIR,exist_ok=True)
os.makedirs(IMAGES_DIR,exist_ok=True)
os.makedirs(METADATA_DIR,exist_ok=True)
os.makedirs(DB_ROOT,exist_ok=True)
os.makedirs(DB_DIR,exist_ok=True)
os.makedirs(LOGS_DIR,exist_ok=True)

# API KEYS

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in .env"
    )

# OCR
# Optional - only needed on Windows if tesseract is not on PATH

TESSERACT_PATH = os.getenv("TESSERACT_PATH")

# MODELS

EMBEDDING_MODEL = ("BAAI/bge-small-en-v1.5")
OLLAMA_MODEL = ("llama3.2:3b")
GROQ_VISION_MODEL = ("qwen/qwen3.6-27b")

GROQ_OCR_MIN_CHARS = 100
GROQ_TIMEOUT = 30

# VECTOR STORE

COLLECTION_NAME = ("rag_collection")

# CHUNKING

CHUNK_SIZE = 4000
NEW_AFTER_N_CHARS = 3000
COMBINE_TEXT_UNDER_N_CHARS = 500
SUBCHUNK_SIZE = 4000
PARENT_CHUNK_ID_PREFIX = "parent"

# QUERY REWRITE

QUERY_REWRITE_WORD_LIMIT = 5

# RETRIEVAL

VECTOR_K = 20
BM25_K = 20
RRF_K = 60
MMR_K = 15
RERANK_TOP_K = 15
FINAL_TOP_K = 5
CONFIDENCE_TOP_SCORE = -7.50
CONFIDENCE_AVG_SCORE = -9.00

# FILE SUPPORT

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tiff",
    ".webp"
}

# MULTI USER

USER_ID_DEFAULT = ("default_user")

# METADATA FILES

DOCUMENT_REGISTRY_FILE = os.path.join(
    METADATA_DIR,
    "documents.json"
)
CHUNKS_METADATA_FILE = os.path.join(
    METADATA_DIR,
    "chunks.json"
)
INGESTION_LOG_FILE = os.path.join(
    METADATA_DIR,
    "ingestion_log.json"
)

# REFERENCE STORE

REFERENCE_STORE_FILE = os.path.join(
    METADATA_DIR,
    "reference_store.json"
)

# LOGGING

QUERY_LOG_FILE = os.path.join(LOGS_DIR,"query.log")

# DOCUMENT IDS

DOC_ID_PREFIX = "doc"
CHUNK_ID_PREFIX = "chunk"
IMG_ID_PREFIX = "img"
TBL_ID_PREFIX = "tbl"

# RERANKER

RERANKER_MODEL = ("cross-encoder/ms-marco-MiniLM-L-6-v2")

# MMR

MMR_LAMBDA = 0.5

# BM25

BM25_INDEX_FILE = os.path.join(DB_ROOT,"bm25.pkl")

# for code 3

# TXT CHUNKING
TXT_CHUNK_SIZE = 1000
TXT_CHUNK_OVERLAP = 200