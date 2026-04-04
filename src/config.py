import os

DATA_DIR = "data/multihop_rag"
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")

# ========== MODELS ==========
LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Cross-encoder for re‑ranking
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_CROSS_ENCODER = True

# Retrieval settings
TOP_K = 5
MMR_LAMBDA = 0.75
SNIPPET_LENGTH = 300
MAX_EVIDENCE_LENGTH = 1500
BATCH_SIZE = 32

# FAISS HNSW index settings
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 40
NPROBE = 10

# Tree settings
MAX_DEPTH = 3
JUDGE_THRESHOLD = 2.5

# LLM settings
TEMPERATURE = 0.2

# Fallback to LLM's own knowledge when retrieval yields low‑confidence answers
FALLBACK_TO_LLM = True

# Web search fallback
ENABLE_WEB_SEARCH = True
WEB_SEARCH_MAX_RESULTS = 3
SEARCH_BACKEND = "duckduckgo"   # Use DuckDuckGo (more reliable with proper user-agent)

# Fact‑checking (optional, disable for speed during evaluation)
ENABLE_FACT_CHECK = False
FACT_CHECK_THRESHOLD = 3.0

# ========== QUERY LIMIT ==========
MAX_QUERIES = None