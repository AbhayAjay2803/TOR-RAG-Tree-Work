from .models import LLMClient

JUDGE_PROMPT = """You are an evaluator of answer quality. Given a question, a context, and a generated answer, rate the answer on the following five criteria (each 0 or 1, total 0-5):

1. Directness: Does the answer directly address the question? (0 or 1)
2. Support: Is the answer fully supported by the context? (0 or 1)
3. Specificity: Is the answer specific (not vague or generic)? (0 or 1)
4. Consistency: Does the answer contradict itself or the context? (0 if contradiction, 1 if consistent)
5. Format: Is the answer in the expected format (short phrase, yes/no, number)? (0 or 1)

Output ONLY a single line with the total score (0-5) and nothing else.

Question: {question}
Context: {context}
Answer: {answer}
Score:"""

class Judge:
    def __init__(self, llm: LLMClient, threshold: float):
        self.llm = llm
        self.threshold = threshold

    def evaluate(self, question: str, context: str, answer: str) -> float:
        prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer)
        response = self.llm.invoke(prompt).strip()
        try:
            score = float(response)
        except ValueError:
            score = 0.0
        return score