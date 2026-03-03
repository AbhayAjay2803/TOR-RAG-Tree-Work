import os

# Data directories and index path
DATA_DIR = "data/multihop_rag"
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")

# Model settings
LLM_MODEL = "llama3.1:8b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast embedding
EMBEDDING_DIM = 384

# Retrieval settings
TOP_K = 5
MMR_LAMBDA = 0.75
SNIPPET_LENGTH = 300
MAX_EVIDENCE_LENGTH = 1500
BATCH_SIZE = 64

# Tree settings
MAX_DEPTH = 2                     # Reduced from 3 → halves LLM calls
JUDGE_THRESHOLD = 3.0             # Increased from 2.0 → more pruning

# LLM settings
TEMPERATURE = 0.2

# Parallel processing
MAX_WORKERS = 4                    # Number of concurrent queries (adjust based on your CPU/GPU)