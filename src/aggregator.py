from .models import LLMClient, EmbeddingClient
from .utils import mmr_selection
from typing import List, Dict

FINAL_PROMPT = """Based on the following evidence, answer the original question concisely. If the evidence does not contain enough information, answer "Unknown".

Evidence:
{evidence}

Original question: {question}
Answer:"""

class Aggregator:
    def __init__(self, embed_client: EmbeddingClient, llm: LLMClient,
                 top_k: int, mmr_lambda: float, max_length: int):
        self.embed_client = embed_client
        self.llm = llm
        self.top_k = top_k
        self.mmr_lambda = mmr_lambda
        self.max_length = max_length

    def aggregate(self, question: str, evidence_list: List[Dict]) -> str:
        if not evidence_list:
            return "Unknown"

        q_emb = self.embed_client.embed([question])[0]

        candidates = []
        for ev in evidence_list:
            emb = self.embed_client.embed([ev['text']])[0]
            candidates.append({
                'text': ev['text'],
                'relevance': ev.get('relevance', 0.5),
                'embedding': emb
            })

        selected = mmr_selection(q_emb, candidates, self.mmr_lambda, self.top_k)

        evidence_text = ""
        for doc in selected:
            if len(evidence_text) + len(doc['text']) > self.max_length:
                break
            evidence_text += doc['text'] + "\n\n"

        prompt = FINAL_PROMPT.format(evidence=evidence_text.strip(), question=question)
        return self.llm.invoke(prompt).strip()