from typing import List, Dict
from .query_decomposer import QueryDecomposer
from .retriever import Retriever
from .judge import Judge
from .models import LLMClient

PARTIAL_ANSWER_PROMPT = """Answer the following question based on the provided context. Keep the answer short (a phrase, yes/no, or number).

Context: {context}
Question: {question}
Answer:"""

class TreeNode:
    def __init__(self, query: str, depth: int):
        self.query = query
        self.depth = depth
        self.children: List['TreeNode'] = []
        self.answer: str = None
        self.evidence: List[Dict] = []
        self.score: float = None

class TreeProcessor:
    def __init__(self, decomposer: QueryDecomposer, retriever: Retriever,
                 judge: Judge, llm: LLMClient, max_depth: int):
        self.decomposer = decomposer
        self.retriever = retriever
        self.judge = judge
        self.llm = llm
        self.max_depth = max_depth
        self.leaf_evidence = []

    def process(self, query: str, depth: int = 0) -> TreeNode:
        node = TreeNode(query, depth)

        if depth >= self.max_depth:
            evidence = self.retriever.retrieve(query)
            context = "\n\n".join([doc[1] for doc in evidence])
            answer = self.llm.invoke(PARTIAL_ANSWER_PROMPT.format(context=context, question=query))
            node.answer = answer
            node.evidence = [{"id": e[0], "text": e[1], "relevance": e[2]} for e in evidence]
            node.score = self.judge.evaluate(query, context, answer)
            self.leaf_evidence.extend(node.evidence)
            return node

        q1, q2 = self.decomposer.decompose(query)

        for sub_q in [q1, q2]:
            evidence = self.retriever.retrieve(sub_q)
            context = "\n\n".join([doc[1] for doc in evidence])
            answer = self.llm.invoke(PARTIAL_ANSWER_PROMPT.format(context=context, question=sub_q))
            score = self.judge.evaluate(sub_q, context, answer)

            if score >= self.judge.threshold and depth + 1 < self.max_depth:
                child = self.process(sub_q, depth + 1)
                node.children.append(child)
            else:
                child = TreeNode(sub_q, depth + 1)
                child.answer = answer
                child.evidence = [{"id": e[0], "text": e[1], "relevance": e[2]} for e in evidence]
                child.score = score
                node.children.append(child)
                self.leaf_evidence.extend(child.evidence)

        return node