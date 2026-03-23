from .models import LLMClient, EmbeddingClient
from .utils import mmr_selection
from typing import List, Dict

FINAL_PROMPT = """Based on the following evidence, answer the original question concisely. If the evidence does not contain enough information, answer "Unknown".

Evidence:
{evidence}

Original question: {question}
Answer:"""

LLM_ONLY_PROMPT = """Answer the following question concisely based on your general knowledge. If you don't know, answer "Unknown". Do not mention evidence or lack thereof.

Question: {question}
Answer:"""

LLM_ONLY_ALT_PROMPT = """I need a concise answer to this question. Use your own knowledge. If you don't know, say "Unknown". Be brief.

Question: {question}
Answer:"""

class Aggregator:
    def __init__(self, embed_client: EmbeddingClient, llm: LLMClient,
                 top_k: int, mmr_lambda: float, max_length: int,
                 fallback_to_llm: bool = True, judge_threshold: float = 2.5):
        self.embed_client = embed_client
        self.llm = llm
        self.top_k = top_k
        self.mmr_lambda = mmr_lambda
        self.max_length = max_length
        self.fallback_to_llm = fallback_to_llm
        self.judge_threshold = judge_threshold

    def _evaluate_answer(self, question: str, answer: str, evidence: str = "") -> float:
        if not evidence:
            eval_prompt = f"Rate the following answer on a scale of 0‑5 (0 = useless, 5 = perfect) based on correctness and completeness:\nQuestion: {question}\nAnswer: {answer}\nScore:"
        else:
            eval_prompt = f"""You are an evaluator. Rate the following answer on a scale of 0‑5 (0 = useless, 5 = perfect) based on how well it answers the question using the evidence.

Evidence: {evidence[:500]}
Question: {question}
Answer: {answer}
Score:"""
        response = self.llm.invoke(eval_prompt).strip()
        try:
            score = float(response)
        except ValueError:
            score = 0.0
        return score

    def aggregate(self, question: str, evidence_list: List[Dict]) -> str:
        # Case 1: No evidence at all
        if not evidence_list:
            if self.fallback_to_llm:
                answer = self.llm.invoke(LLM_ONLY_PROMPT.format(question=question)).strip()
                if answer.lower() == "unknown":
                    answer = self.llm.invoke(LLM_ONLY_ALT_PROMPT.format(question=question)).strip()
                return answer
            else:
                return "Unknown"

        # Build evidence string
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

        # Generate answer from evidence
        prompt = FINAL_PROMPT.format(evidence=evidence_text.strip(), question=question)
        answer = self.llm.invoke(prompt).strip()

        # Self‑critique
        score = self._evaluate_answer(question, answer, evidence_text)

        # Fallback if low score and enabled
        if score < self.judge_threshold and self.fallback_to_llm:
            print(f"[Low confidence ({score:.1f}), falling back to LLM-only]")
            fallback_answer = self.llm.invoke(LLM_ONLY_PROMPT.format(question=question)).strip()
            if fallback_answer.lower() == "unknown":
                fallback_answer = self.llm.invoke(LLM_ONLY_ALT_PROMPT.format(question=question)).strip()
            # Return fallback if it's not "Unknown", otherwise keep original
            if fallback_answer.lower() != "unknown":
                return fallback_answer
        return answer