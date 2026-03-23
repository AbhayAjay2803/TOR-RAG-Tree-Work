import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.models import LLMClient, EmbeddingClient
from src.query_decomposer import QueryDecomposer
from src.retriever import Retriever
from src.judge import Judge
from src.tree_processor import TreeProcessor
from src.aggregator import Aggregator

def main():
    print("Initializing ToR-RAG...")
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

    print("Ready! Type your question (or 'exit' to quit):")
    while True:
        question = input("\n🧠 You: ").strip()
        if question.lower() in ["exit", "quit"]:
            break
        if not question:
            continue

        # Reset leaf evidence
        processor.leaf_evidence = []
        # Build tree
        root = processor.process(question)
        # Aggregate final answer
        answer = aggregator.aggregate(question, processor.leaf_evidence)

        print(f"🤖 AI: {answer}")

if __name__ == "__main__":
    main()