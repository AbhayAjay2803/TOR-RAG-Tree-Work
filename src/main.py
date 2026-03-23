import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from tqdm import tqdm
from datasets import load_dataset
from src.config import *
from src.models import LLMClient, EmbeddingClient
from src.query_decomposer import QueryDecomposer
from src.retriever import Retriever
from src.judge import Judge
from src.tree_processor import TreeProcessor
from src.aggregator import Aggregator

def load_queries():
    print("Loading queries from Hugging Face...")
    ds = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
    queries = []
    # Limit the number of queries if MAX_QUERIES is set
    total = len(ds) if MAX_QUERIES is None else min(MAX_QUERIES, len(ds))
    for idx in range(total):
        item = ds[idx]
        qid = str(idx)
        question = item['query']
        answer = item['answer']
        queries.append({
            "id": qid,
            "question": question,
            "answer": answer
        })
    print(f"Loaded {len(queries)} queries (limit: {MAX_QUERIES if MAX_QUERIES else 'all'})")
    return queries

def main():
    llm = LLMClient(LLM_MODEL, TEMPERATURE)
    embed = EmbeddingClient(EMBEDDING_MODEL)

    retriever = Retriever(
        INDEX_PATH, embed, TOP_K, MMR_LAMBDA,
        batch_size=BATCH_SIZE,
        hnsw_m=HNSW_M,
        hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
        nprobe=NPROBE
    )
    decomposer = QueryDecomposer(llm)
    judge = Judge(llm, JUDGE_THRESHOLD)
    processor = TreeProcessor(decomposer, retriever, judge, llm, MAX_DEPTH)
    aggregator = Aggregator(embed, llm, TOP_K, MMR_LAMBDA, MAX_EVIDENCE_LENGTH)

    queries = load_queries()

    results = []
    for q in tqdm(queries, desc="Processing queries"):
        question = q['question']
        ground_truth = q['answer']

        processor.leaf_evidence = []
        start = time.time()
        root = processor.process(question)
        build_time = time.time() - start

        final_answer = aggregator.aggregate(question, processor.leaf_evidence)

        results.append({
            "id": q['id'],
            "question": question,
            "ground_truth": ground_truth,
            "predicted": final_answer,
            "time": build_time
        })

        print(f"Q: {question}\nA: {final_answer}\nGT: {ground_truth}\n")

    with open("results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()