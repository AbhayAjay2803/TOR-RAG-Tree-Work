import os

DATA_DIR = "data/multihop_rag"
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")

# ========== MODELS ==========
# You can switch between models here
LLM_MODEL = "llama3.2:3b"          # Try "llama3.2:3b" (less censored) or "qwen2.5:3b" (more factual)
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

# LLM settings – higher temperature for creativity
TEMPERATURE = 0.8

# Fallback to LLM's own knowledge
FALLBACK_TO_LLM = True

# Query limit (for testing)
MAX_QUERIES = None