from .models import LLMClient, EmbeddingClient
from .utils import mmr_selection
from typing import List, Dict

FINAL_PROMPT = """Based on the following evidence, answer the original question concisely. If the evidence does not contain enough information, answer "Unknown".

Evidence:
{evidence}

Original question: {question}
Answer:"""

# More forceful fallback prompts
LLM_ONLY_PROMPT = """Answer the following question directly with a concise factual answer. Do not add commentary, do not say you are ready, do not mention the absence of evidence. If you don't know, just say "Unknown".

Question: {question}
Answer:"""

LLM_ONLY_ALT_PROMPT = """Provide a short, factual answer to this question. Be concise and avoid any extra text. If you don't know, simply say "Unknown".

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

    def _is_meta_response(self, text: str) -> bool:
        """Detect if the answer is a meta statement like 'I'm ready...'"""
        meta_phrases = [
            "i'm ready", "i am ready", "i can provide", "i will answer",
            "please provide", "i'd be happy", "i would be happy", "let me know",
            "what is the question", "what's the question", "i'll answer"
        ]
        lower = text.lower()
        return any(phrase in lower for phrase in meta_phrases)

    def _get_fallback_answer(self, question: str) -> str:
        """Try two fallback prompts; if first gives meta response, try second."""
        answer = self.llm.invoke(LLM_ONLY_PROMPT.format(question=question)).strip()
        if self._is_meta_response(answer):
            answer = self.llm.invoke(LLM_ONLY_ALT_PROMPT.format(question=question)).strip()
        return answer

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
                return self._get_fallback_answer(question)
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
            fallback_answer = self._get_fallback_answer(question)
            # Return fallback if it's not "Unknown" and not a meta response
            if fallback_answer.lower() != "unknown" and not self._is_meta_response(fallback_answer):
                return fallback_answer
        return answer