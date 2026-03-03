from .models import LLMClient

DECOMPOSITION_PROMPT = """You are an expert at breaking down complex questions into simpler sub-questions.
Given the following question, decompose it into TWO WH- sub-questions (who, what, where, when, why, how, which) that together cover the original question.
The sub-questions should be independent and non-overlapping.
Output only the two sub-questions, one per line, without numbering.

Question: {question}
Sub-questions:"""

class QueryDecomposer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def decompose(self, question: str) -> tuple[str, str]:
        prompt = DECOMPOSITION_PROMPT.format(question=question)
        response = self.llm.invoke(prompt).strip()
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if len(lines) >= 2:
            return lines[0], lines[1]
        return question, question