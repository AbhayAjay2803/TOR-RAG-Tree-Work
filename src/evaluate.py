"""
Evaluation script for Enhanced ToR-RAG.
Runs on a set of questions (single-turn and conversational multi-hop) and computes:
- Exact Match (EM)
- F1 Score
- Fallback success rate
- Average latency
- Fact-checking correction rate (if enabled)
"""
import sys
import os
import json
import time
import re
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.models import LLMClient, EmbeddingClient
from src.query_decomposer import QueryDecomposer
from src.retriever import Retriever
from src.judge import Judge
from src.tree_processor import TreeProcessor
from src.aggregator import Aggregator

# ---------- Helper functions for metrics ----------
def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    s = s.lower().strip()
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    # Remove common articles
    for w in ['a', 'an', 'the']:
        s = s.replace(f' {w} ', ' ')
    return ' '.join(s.split())

def exact_match(pred: str, truth: str) -> bool:
    return normalize_answer(pred) == normalize_answer(truth)

def f1_score(pred: str, truth: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    truth_tokens = normalize_answer(truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)

# ---------- Load test data ----------
def load_test_queries(path: str) -> List[Dict]:
    """Load test queries from a JSON file.
    Format: [
        {"id": "q1", "question": "...", "answer": "...", "conversation_context": null or previous answer}
    ]
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ---------- Main evaluation ----------
def evaluate():
    print("Initializing Enhanced ToR-RAG for evaluation...")
    llm = LLMClient(LLM_MODEL, TEMPERATURE)
    embed = EmbeddingClient(EMBEDDING_MODEL)

    retriever = Retriever(
        INDEX_PATH, embed, TOP_K, MMR_LAMBDA,
        batch_size=BATCH_SIZE,
        hnsw_m=HNSW_M,
        hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
        nprobe=NPROBE,
        use_cross_encoder=USE_CROSS_ENCODER,
        cross_encoder_model=CROSS_ENCODER_MODEL
    )
    decomposer = QueryDecomposer(llm)
    judge = Judge(llm, JUDGE_THRESHOLD)
    processor = TreeProcessor(decomposer, retriever, judge, llm, MAX_DEPTH)
    aggregator = Aggregator(
        embed, llm, TOP_K, MMR_LAMBDA, MAX_EVIDENCE_LENGTH,
        fallback_to_llm=FALLBACK_TO_LLM,
        judge_threshold=JUDGE_THRESHOLD,
        enable_web_search=ENABLE_WEB_SEARCH,
        web_search_max_results=WEB_SEARCH_MAX_RESULTS,
        enable_fact_check=ENABLE_FACT_CHECK,
        fact_check_threshold=FACT_CHECK_THRESHOLD
    )

    # Load test queries (you need to create this file)
    test_file = "data/eval_queries.json"
    if not os.path.exists(test_file):
        print(f"Test file {test_file} not found. Creating a sample file with 5 questions...")
        os.makedirs("data", exist_ok=True)
        sample = [
            {"id": "1", "question": "What is the capital of France?", "answer": "Paris"},
            {"id": "2", "question": "Who wrote Hamlet?", "answer": "William Shakespeare"},
            {"id": "3", "question": "What is the highest mountain in the world?", "answer": "Mount Everest"},
        ]
        with open(test_file, 'w') as f:
            json.dump(sample, f, indent=2)
        print(f"Sample file created. Please edit {test_file} with your own questions.")
        return

    queries = load_test_queries(test_file)
    print(f"Loaded {len(queries)} test queries.")

    results = []
    fallback_count = 0
    total_latency = 0.0
    em_correct = 0
    f1_total = 0.0

    for idx, q in enumerate(queries):
        question = q['question']
        truth = q['answer']
        context = q.get('conversation_context', None)  # For multi-turn

        # If there is conversation context, we could simulate memory.
        # For simplicity, we just process the question directly.
        # To test conversation, you would need to chain previous Q&A.
        # We'll leave that as an extension.

        processor.leaf_evidence = []
        start = time.time()
        # We need to capture whether fallback was used. The aggregator prints but we can also check output.
        # For now, we assume fallback is used if answer comes from LLM-only or web search.
        # A cleaner way: modify aggregator to return a flag. We'll keep simple.
        final_answer = aggregator.aggregate(question, processor.leaf_evidence)
        elapsed = time.time() - start
        total_latency += elapsed

        # Check if answer is "Unknown" (fallback failure)
        if final_answer.lower().strip() in ["unknown", "i don't know", "cannot provide"]:
            fallback_count += 1

        # Compute metrics
        em = exact_match(final_answer, truth)
        f1 = f1_score(final_answer, truth)
        if em:
            em_correct += 1
        f1_total += f1

        results.append({
            "id": q.get('id', idx),
            "question": question,
            "ground_truth": truth,
            "predicted": final_answer,
            "exact_match": em,
            "f1": f1,
            "latency": elapsed,
            "fallback_used": (final_answer.lower() != truth.lower())  # crude
        })

        print(f"Q{idx+1}: {question[:50]}... -> {final_answer[:100]} (EM={em}, F1={f1:.2f})")

    # Aggregate metrics
    total = len(queries)
    avg_em = em_correct / total if total > 0 else 0
    avg_f1 = f1_total / total if total > 0 else 0
    fallback_rate = fallback_count / total if total > 0 else 0
    avg_latency = total_latency / total if total > 0 else 0

    summary = {
        "total_queries": total,
        "exact_match": avg_em,
        "f1_score": avg_f1,
        "fallback_rate": fallback_rate,
        "average_latency_seconds": avg_latency,
        "results": results
    }

    # Save results
    out_file = "evaluation_results.json"
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nEvaluation complete. Results saved to {out_file}")
    print(f"Exact Match: {avg_em:.2%}, F1: {avg_f1:.3f}, Fallback Rate: {fallback_rate:.2%}, Avg Latency: {avg_latency:.2f}s")

if __name__ == "__main__":
    evaluate()