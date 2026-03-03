import faiss
import numpy as np
import os
from typing import List, Tuple
from datasets import load_dataset
from tqdm import tqdm
from .models import EmbeddingClient
from .utils import mmr_selection

class Retriever:
    def __init__(self, index_path: str, embed_client: EmbeddingClient,
                 top_k: int, mmr_lambda: float, batch_size: int = 8):
        self.embed_client = embed_client
        self.top_k = top_k
        self.mmr_lambda = mmr_lambda
        self.index_path = index_path
        self.batch_size = batch_size
        self.doc_ids, self.doc_texts, self.index = self._load_or_create_index()

    def _load_or_create_index(self):
        """Load corpus from Hugging Face dataset and build/load FAISS index."""
        print("Loading corpus from Hugging Face...")
        corpus_ds = load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")
        
        # Corpus has no 'id' column; use the index as document ID
        doc_ids = [str(i) for i in range(len(corpus_ds))]
        doc_texts = corpus_ds["body"]

        # Try to load existing index
        if os.path.exists(self.index_path):
            print(f"Loading FAISS index from {self.index_path}")
            index = faiss.read_index(self.index_path)
            return doc_ids, doc_texts, index

        # Create new index with batched embedding
        print("Creating new FAISS index in batches...")
        dim = self.embed_client.embed([doc_texts[0]]).shape[1]
        index = faiss.IndexFlatIP(dim)

        # Embed in batches to avoid OOM
        for i in tqdm(range(0, len(doc_texts), self.batch_size), desc="Embedding documents"):
            batch_texts = doc_texts[i:i+self.batch_size]
            batch_embeddings = self.embed_client.embed(batch_texts)
            index.add(batch_embeddings)

        # Save index
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(index, self.index_path)
        print(f"Index saved to {self.index_path}")
        return doc_ids, doc_texts, index

    def retrieve(self, query: str) -> List[Tuple[str, str, float]]:
        q_emb = self.embed_client.embed([query])[0]

        # Retrieve more than top_k for MMR diversity
        initial_k = self.top_k * 3
        scores, indices = self.index.search(q_emb.reshape(1, -1), initial_k)
        scores = scores[0]
        indices = indices[0]

        candidates = []
        for i, idx in enumerate(indices):
            if idx == -1:
                continue
            candidates.append({
                'id': self.doc_ids[idx],
                'text': self.doc_texts[idx],
                'relevance': scores[i],
                'embedding': self.index.reconstruct(int(idx))
            })

        selected = mmr_selection(q_emb, candidates, self.mmr_lambda, self.top_k)
        return [(doc['id'], doc['text'], doc['relevance']) for doc in selected]