import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    """Load queries dataset from Hugging Face."""
    print("Loading queries from Hugging Face...")
    ds = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
    queries = []
    for i, item in enumerate(ds):
        queries.append({
            "id": str(i),
            "question": item["query"],
            "answer": item["answer"]
        })
    return queries

def process_single_query(q, llm, embed, retriever, decomposer, judge, processor, aggregator):
    """Process one query and return result."""
    question = q['question']
    ground_truth = q['answer']

    processor.leaf_evidence = []
    start = time.time()
    root = processor.process(question)  # This may still be slow per query, but parallel helps
    build_time = time.time() - start

    final_answer = aggregator.aggregate(question, processor.leaf_evidence)

    return {
        "id": q['id'],
        "question": question,
        "ground_truth": ground_truth,
        "predicted": final_answer,
        "time": build_time
    }

def main():
    llm = LLMClient(LLM_MODEL, TEMPERATURE)
    embed = EmbeddingClient(EMBEDDING_MODEL)

    # Initialize retriever (loads index, should be fast)
    retriever = Retriever(INDEX_PATH, embed, TOP_K, MMR_LAMBDA, batch_size=BATCH_SIZE)
    decomposer = QueryDecomposer(llm)
    judge = Judge(llm, JUDGE_THRESHOLD)
    # Note: processor and aggregator are not thread-safe because they share state.
    # We'll create a new processor per thread to avoid sharing leaf_evidence.
    # We'll define a factory inside the thread function.

    queries = load_queries()
    results = []

    # Use ThreadPoolExecutor to process queries in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all queries
        future_to_q = {}
        for q in queries:
            # Create fresh processor and aggregator for each query (to avoid state leakage)
            processor = TreeProcessor(decomposer, retriever, judge, llm, MAX_DEPTH)
            aggregator = Aggregator(embed, llm, TOP_K, MMR_LAMBDA, MAX_EVIDENCE_LENGTH)
            future = executor.submit(
                process_single_query, q, llm, embed, retriever,
                decomposer, judge, processor, aggregator
            )
            future_to_q[future] = q

        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_q), total=len(queries), desc="Processing queries"):
            try:
                result = future.result()
                results.append(result)
                # Print progress (optional, but can be spammy)
                # print(f"Q: {result['question'][:50]}...\nA: {result['predicted']}\n")
            except Exception as e:
                print(f"Error processing query: {e}")

    # Save results
    with open("results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Print summary
    total_time = sum(r['time'] for r in results)
    avg_time = total_time / len(results) if results else 0
    print(f"\nProcessed {len(results)} queries in {total_time/60:.2f} minutes")
    print(f"Average time per query: {avg_time:.2f} seconds")

if __name__ == "__main__":
    main()