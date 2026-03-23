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

class ConversationMemory:
    def __init__(self):
        self.last_question = None
        self.last_answer = None

    def get_context(self, current_question: str) -> str:
        if self.last_question and self.last_answer:
            if len(current_question.split()) < 6:
                return f"Previous question: {self.last_question}\nPrevious answer: {self.last_answer}\nNow: {current_question}"
        return current_question

    def update(self, question: str, answer: str):
        self.last_question = question
        self.last_answer = answer

def main():
    print("Initializing ToR-RAG with enhanced features...")
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
    aggregator = Aggregator(embed, llm, TOP_K, MMR_LAMBDA, MAX_EVIDENCE_LENGTH,
                            fallback_to_llm=FALLBACK_TO_LLM,
                            judge_threshold=JUDGE_THRESHOLD)

    memory = ConversationMemory()

    print("Ready! Type your question (or 'exit' to quit):")
    while True:
        user_input = input("\n🧠 You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue

        context = memory.get_context(user_input)
        if context != user_input:
            print(f"Context added: {context}")

        processor.leaf_evidence = []
        root = processor.process(context if context != user_input else user_input)
        final_answer = aggregator.aggregate(context if context != user_input else user_input,
                                            processor.leaf_evidence)

        print(f"🤖 AI: {final_answer}")

        memory.update(user_input, final_answer)

if __name__ == "__main__":
    main()