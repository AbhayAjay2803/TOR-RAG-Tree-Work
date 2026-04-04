import os
import sys
import warnings

# Suppress huggingface symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Filter out the "position_ids" unexpected key warning from transformers
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.file_download")

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
        self.llm = None   # Will be set later

    def set_llm(self, llm):
        self.llm = llm

    def resolve_followup(self, user_input: str) -> str:
        """Use the LLM to interpret follow‑up questions that refer to previous context."""
        if not self.last_question or not self.llm:
            return user_input

        prompt = f"""You are an assistant that interprets follow‑up questions. The user previously asked:

Previous question: {self.last_question}
Previous answer: {self.last_answer}

Now the user says: {user_input}

What is the actual question they are asking? Output only the clarified question, without any extra text. If it is a new topic, output the user's input unchanged.
Clarified question:"""
        resolved = self.llm.invoke(prompt).strip()
        # Fallback if the LLM returns something strange
        if not resolved or len(resolved) < 3:
            return user_input
        return resolved

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
    aggregator = Aggregator(
        embed, llm, TOP_K, MMR_LAMBDA, MAX_EVIDENCE_LENGTH,
        fallback_to_llm=FALLBACK_TO_LLM,
        judge_threshold=JUDGE_THRESHOLD,
        enable_web_search=ENABLE_WEB_SEARCH,
        web_search_max_results=WEB_SEARCH_MAX_RESULTS,
        enable_fact_check=ENABLE_FACT_CHECK,
        fact_check_threshold=FACT_CHECK_THRESHOLD,
        search_backend=SEARCH_BACKEND
)

    memory = ConversationMemory()
    memory.set_llm(llm)

    print("Ready! Type your question (or 'exit' to quit):")
    while True:
        user_input = input("\n🧠 You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue

        # Resolve follow‑up references using the LLM
        resolved_question = memory.resolve_followup(user_input)
        if resolved_question != user_input:
            print(f"Interpreted as: {resolved_question}")

        # Process the resolved question
        processor.leaf_evidence = []
        root = processor.process(resolved_question)
        final_answer = aggregator.aggregate(resolved_question, processor.leaf_evidence)

        print(f"🤖 AI: {final_answer}")

        # Update memory with the original user input and the answer
        memory.update(user_input, final_answer)

if __name__ == "__main__":
    main()