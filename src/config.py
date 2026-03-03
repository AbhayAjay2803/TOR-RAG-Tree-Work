import os

# Data directories and index path
DATA_DIR = "data/multihop_rag"
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")

# Model settings
LLM_MODEL = "llama3.1:8b"
EMBEDDING_MODEL = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
EMBEDDING_DIM = 1024

# Retrieval settings
TOP_K = 5
MMR_LAMBDA = 0.75
SNIPPET_LENGTH = 300
MAX_EVIDENCE_LENGTH = 1500
BATCH_SIZE = 8  # Number of documents to embed at once (adjust based on your RAM)

# Tree settings
MAX_DEPTH = 3
JUDGE_THRESHOLD = 2.0

# LLM settings
TEMPERATURE = 0.2