import faiss
import json
import numpy as np
from typing import List, Tuple
from .models import EmbeddingClient
from .utils import mmr_selection

class Retriever:
    def __init__(self, corpus_path: str, index_path: str, embed_client: EmbeddingClient,
                 top_k: int, mmr_lambda: float):
        self.embed_client = embed_client
        self.top_k = top_k
        self.mmr_lambda = mmr_lambda
        self.doc_ids, self.doc_texts, self.index = self._load_or_create_index(corpus_path, index_path)

    def _load_or_create_index(self, corpus_path: str, index_path: str):
        doc_ids, doc_texts = [], []
        with open(corpus_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                doc_ids.append(item['id'])
                doc_texts.append(item['text'])

        try:
            index = faiss.read_index(index_path)
            print("Loaded existing FAISS index.")
        except:
            print("Creating new FAISS index...")
            embeddings = self.embed_client.embed(doc_texts)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            faiss.write_index(index, index_path)
        return doc_ids, doc_texts, index

    def retrieve(self, query: str) -> List[Tuple[str, str, float]]:
        q_emb = self.embed_client.embed([query])[0]

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