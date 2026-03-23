import faiss
import numpy as np
import os
from typing import List, Tuple
from datasets import load_dataset
from tqdm import tqdm
from .models import EmbeddingClient
from .utils import mmr_selection

# Try to import cross-encoder if enabled
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

class Retriever:
    def __init__(self, index_path: str, embed_client: EmbeddingClient,
                 top_k: int, mmr_lambda: float, batch_size: int = 8,
                 hnsw_m: int = 16, hnsw_ef_construction: int = 40, nprobe: int = 10,
                 use_cross_encoder: bool = False, cross_encoder_model: str = None):
        self.embed_client = embed_client
        self.top_k = top_k
        self.mmr_lambda = mmr_lambda
        self.index_path = index_path
        self.batch_size = batch_size
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.nprobe = nprobe
        self.use_cross_encoder = use_cross_encoder and CROSS_ENCODER_AVAILABLE
        self.cross_encoder = None
        if self.use_cross_encoder and cross_encoder_model:
            print(f"Loading cross-encoder {cross_encoder_model}...")
            self.cross_encoder = CrossEncoder(cross_encoder_model)
        self._cache = {}   # Simple cache for retrieval results
        self.doc_ids, self.doc_texts, self.index = self._load_or_create_index()

    def _load_or_create_index(self):
        print("Loading corpus from Hugging Face...")
        corpus_ds = load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")
        doc_ids = [str(i) for i in range(len(corpus_ds))]
        doc_texts = corpus_ds["body"]

        # Get expected dimension from first document
        expected_dim = self.embed_client.embed([doc_texts[0]]).shape[1]

        # Check if index exists and has correct dimension
        rebuild = False
        if os.path.exists(self.index_path):
            print(f"Checking existing index at {self.index_path}")
            try:
                index = faiss.read_index(self.index_path)
                if index.d != expected_dim:
                    print(f"Index dimension ({index.d}) does not match expected ({expected_dim}). Rebuilding...")
                    rebuild = True
                    os.remove(self.index_path)
                else:
                    print(f"Loading FAISS index from {self.index_path}")
                    if hasattr(index, 'hnsw'):
                        index.hnsw.nprobe = self.nprobe
                    return doc_ids, doc_texts, index
            except Exception as e:
                print(f"Error reading index: {e}. Rebuilding...")
                rebuild = True

        if rebuild or not os.path.exists(self.index_path):
            print("Creating new FAISS HNSW index...")
            dim = expected_dim
            index = faiss.IndexHNSWFlat(dim, self.hnsw_m)
            index.hnsw.efConstruction = self.hnsw_ef_construction
            index.hnsw.nprobe = self.nprobe

            for i in tqdm(range(0, len(doc_texts), self.batch_size), desc="Embedding documents"):
                batch_texts = doc_texts[i:i+self.batch_size]
                batch_embeddings = self.embed_client.embed(batch_texts)
                index.add(batch_embeddings)

            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(index, self.index_path)
            print(f"Index saved to {self.index_path}")
            return doc_ids, doc_texts, index

    def retrieve_with_cache(self, query: str) -> List[Tuple[str, str, float]]:
        """Retrieve with caching to avoid repeated work for the same query."""
        if query in self._cache:
            return self._cache[query]
        result = self.retrieve(query)
        self._cache[query] = result
        return result

    def retrieve(self, query: str) -> List[Tuple[str, str, float]]:
        q_emb = self.embed_client.embed([query])[0]

        # Retrieve more than top_k for MMR or cross-encoder re-ranking
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

        # If cross-encoder is used, re-rank the candidates and pick top_k
        if self.use_cross_encoder and self.cross_encoder:
            pairs = [(query, doc['text']) for doc in candidates]
            cross_scores = self.cross_encoder.predict(pairs)
            for doc, cs in zip(candidates, cross_scores):
                doc['cross_score'] = cs
            # Sort by cross_score descending
            candidates.sort(key=lambda x: x.get('cross_score', 0), reverse=True)
            selected = candidates[:self.top_k]
        else:
            # Use MMR
            selected = mmr_selection(q_emb, candidates, self.mmr_lambda, self.top_k)

        return [(doc['id'], doc['text'], doc.get('cross_score', doc['relevance'])) for doc in selected]