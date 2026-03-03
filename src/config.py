import os

DATA_DIR = "data/multihop_rag"
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")
QUERIES_PATH = os.path.join(DATA_DIR, "queries.jsonl")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")

LLM_MODEL = "llama3.1:8b"
EMBEDDING_MODEL = "Alibaba-NLP/gte-Qwen2-0.5B-instruct"
EMBEDDING_DIM = 1024

TOP_K = 5
MMR_LAMBDA = 0.75
SNIPPET_LENGTH = 300
MAX_EVIDENCE_LENGTH = 1500

MAX_DEPTH = 3
JUDGE_THRESHOLD = 2.0

TEMPERATURE = 0.2