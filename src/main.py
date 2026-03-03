import json
import time
from tqdm import tqdm
from src.config import *
from src.models import LLMClient, EmbeddingClient
from src.query_decomposer import QueryDecomposer
from src.retriever import Retriever
from src.judge import Judge
from src.tree_processor import TreeProcessor
from src.aggregator import Aggregator

def load_queries(path: str):
    queries = []
    with open(path, 'r') as f:
        for line in f:
            queries.append(json.loads(line))
    return queries

def main():
    llm = LLMClient(LLM_MODEL, TEMPERATURE)
    embed = EmbeddingClient(EMBEDDING_MODEL)

    retriever = Retriever(CORPUS_PATH, INDEX_PATH, embed, TOP_K, MMR_LAMBDA)
    decomposer = QueryDecomposer(llm)
    judge = Judge(llm, JUDGE_THRESHOLD)
    processor = TreeProcessor(decomposer, retriever, judge, llm, MAX_DEPTH)
    aggregator = Aggregator(embed, llm, TOP_K, MMR_LAMBDA, MAX_EVIDENCE_LENGTH)

    queries = load_queries(QUERIES_PATH)

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