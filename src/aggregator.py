from .models import LLMClient, EmbeddingClient
from .utils import mmr_selection
from .web_search import web_search
from typing import List, Dict
from src.config import SEARCH_BACKEND

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

WEB_SEARCH_PROMPT = """The following information was retrieved from a web search. Use it to answer the question concisely. If the information is insufficient, answer "Unknown".

Web search results:
{search_results}

Question: {question}
Answer:"""

FACT_CHECK_PROMPT = """You have an answer to a question. You also have web search results. Compare them. If the answer is consistent with the web results, output the original answer. If there is a contradiction or the original answer is incomplete, produce a corrected answer based on the web results. Be concise.

Original answer: {original_answer}

Web search results:
{search_results}

Question: {question}
Corrected answer:"""

class Aggregator:
    def __init__(self, embed_client: EmbeddingClient, llm: LLMClient,
                 top_k: int, mmr_lambda: float, max_length: int,
                 fallback_to_llm: bool = True, judge_threshold: float = 2.5,
                 enable_web_search: bool = False, web_search_max_results: int = 3,
                 enable_fact_check: bool = False, fact_check_threshold: float = 3.0):
        self.embed_client = embed_client
        self.llm = llm
        self.top_k = top_k
        self.mmr_lambda = mmr_lambda
        self.max_length = max_length
        self.fallback_to_llm = fallback_to_llm
        self.judge_threshold = judge_threshold
        self.enable_web_search = enable_web_search
        self.web_search_max_results = web_search_max_results
        self.enable_fact_check = enable_fact_check
        self.fact_check_threshold = fact_check_threshold

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

    def _is_unknown(self, answer: str) -> bool:
        """Check if the answer indicates lack of knowledge."""
        unknown_phrases = ["unknown", "i don't know", "i do not know", "no information", "cannot provide"]
        answer_lower = answer.lower().strip()
        return any(phrase in answer_lower for phrase in unknown_phrases)

    def _web_search_fallback(self, question: str) -> str:
        """Perform web search and return answer."""
        search_results = web_search(question, self.web_search_max_results, backend=SEARCH_BACKEND)
        if search_results:
            return self.llm.invoke(WEB_SEARCH_PROMPT.format(
                search_results=search_results, question=question)).strip()
        return "Unknown"

    def _fact_check(self, question: str, answer: str) -> str:
        """Use web search to verify the answer."""
        search_results = web_search(question, self.web_search_max_results)
        if not search_results:
            return answer
        prompt = FACT_CHECK_PROMPT.format(
            original_answer=answer,
            search_results=search_results,
            question=question
        )
        return self.llm.invoke(prompt).strip()

    def aggregate(self, question: str, evidence_list: List[Dict]) -> str:
        # ----- Case 1: No retrieved evidence -----
        if not evidence_list:
            if self.fallback_to_llm:
                answer = self.llm.invoke(LLM_ONLY_PROMPT.format(question=question)).strip()
                if self._is_unknown(answer):
                    answer = self.llm.invoke(LLM_ONLY_ALT_PROMPT.format(question=question)).strip()
                # If still unknown and web search enabled, try web search
                if self._is_unknown(answer) and self.enable_web_search:
                    print("[No evidence, trying web search]")
                    answer = self._web_search_fallback(question)
                # If answer is not unknown, optionally fact‑check
                if not self._is_unknown(answer) and self.enable_fact_check:
                    print("[Fact‑checking answer with web search]")
                    answer = self._fact_check(question, answer)
                return answer
            else:
                return "Unknown"

        # ----- Case 2: Evidence exists -----
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

        # If confidence is low, try web search first (instead of LLM-only)
        if score < self.judge_threshold and self.enable_web_search:
            print(f"[Low confidence ({score:.1f}), trying web search]")
            web_answer = self._web_search_fallback(question)
            if not self._is_unknown(web_answer):
                answer = web_answer
                # Re‑evaluate confidence (optional)
                score = self._evaluate_answer(question, answer, evidence_text)
            else:
                # Web search failed, fall back to LLM-only
                if self.fallback_to_llm:
                    print("[Web search failed, falling back to LLM-only]")
                    fallback_answer = self.llm.invoke(LLM_ONLY_PROMPT.format(question=question)).strip()
                    if self._is_unknown(fallback_answer):
                        fallback_answer = self.llm.invoke(LLM_ONLY_ALT_PROMPT.format(question=question)).strip()
                    if not self._is_unknown(fallback_answer):
                        answer = fallback_answer
        elif score < self.judge_threshold and self.fallback_to_llm:
            # Fallback to LLM-only if web search is disabled
            print(f"[Low confidence ({score:.1f}), falling back to LLM-only]")
            fallback_answer = self.llm.invoke(LLM_ONLY_PROMPT.format(question=question)).strip()
            if self._is_unknown(fallback_answer):
                fallback_answer = self.llm.invoke(LLM_ONLY_ALT_PROMPT.format(question=question)).strip()
            if not self._is_unknown(fallback_answer):
                answer = fallback_answer

        # Optional fact‑check if answer is not unknown and fact‑check enabled
        if not self._is_unknown(answer) and self.enable_fact_check:
            if score >= self.fact_check_threshold:
                print("[Fact‑checking answer with web search]")
                answer = self._fact_check(question, answer)

        return answer