from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer
import numpy as np

class LLMClient:
    def __init__(self, model_name: str, temperature: float = 0.2):
        self.llm = ChatOllama(model=model_name, temperature=temperature)

    def invoke(self, prompt: str) -> str:
        return self.llm.invoke(prompt).content

class EmbeddingClient:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)