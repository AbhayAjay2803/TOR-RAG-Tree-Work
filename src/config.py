import os

DATA_DIR = "data/multihop_rag"
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")

# ========== MODELS ==========
LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Cross-encoder for re‑ranking (optional, set to None to disable)
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_CROSS_ENCODER = True          # Set to False to disable re‑ranking

# Retrieval settings
TOP_K = 5                         # Number of documents to return after re‑ranking
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
FALLBACK_TO_LLM = True            # If True, when the aggregated answer is "Unknown" or confidence low, ask LLM directly

# ========== QUERY LIMIT (for testing) ==========
MAX_QUERIES = None                # Set to an integer to test only the first N queries