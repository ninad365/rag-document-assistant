from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from app.evaluation.metrics import (
    citation_correctness,
    hit_rate_at_k,
    precision_at_k,
    reciprocal_rank,
    unanswerable_accuracy,
)
from app.models.schemas import EvaluationQuestion
from app.rag.qa import RagService


class JudgeResult(BaseModel):
    answer_correctness: float = Field(ge=0, le=1)
    faithfulness: float = Field(ge=0, le=1)
    answer_relevance: float = Field(ge=0, le=1)


def _source_key(source: str, page: int) -> str:
    return f"{source}#p{page}"


class Evaluator:
    def __init__(self, rag_service: RagService | None = None):
        self.rag_service = rag_service or RagService()

    def _judge(
        self,
        api_key: str,
        question: str,
        expected_answer: str,
        generated_answer: str,
        context: list[str],
    ) -> JudgeResult:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
        structured = llm.with_structured_output(JudgeResult)
        return structured.invoke(
            """
            Score the generated answer from 0 to 1 on:
            - answer_correctness: factual alignment with expected answer
            - faithfulness: supported by provided context only
            - answer_relevance: directness to the question
            Return only numeric scores.
            """
            + f"\nQuestion: {question}"
            + f"\nExpected answer: {expected_answer}"
            + f"\nGenerated answer: {generated_answer}"
            + f"\nContext: {' | '.join(context)}"
        )

    def run(self, api_key: str, dataset: list[EvaluationQuestion], top_k: int) -> dict:
        rows = []
        for item in dataset:
            response = self.rag_service.chat(api_key, item.question, history=[], top_k=top_k)
            retrieved_sources = [_source_key(c["source"], c["page"]) for c in response["retrieved_chunks"]]
            returned_citations = [_source_key(c["source"], c["page"]) for c in response["citations"]]
            predicted_unanswerable = "I could not find this in the uploaded documents." in response["answer"]

            judge = self._judge(
                api_key,
                item.question,
                item.expected_answer,
                response["answer"],
                [c["text"] for c in response["retrieved_chunks"]],
            )

            rows.append(
                {
                    "question": item.question,
                    "answer": response["answer"],
                    "expected_answer": item.expected_answer,
                    "expected_sources": item.expected_sources,
                    "retrieved_sources": retrieved_sources,
                    "answerable": item.answerable,
                    "predicted_unanswerable": predicted_unanswerable,
                    "hit_rate_at_k": hit_rate_at_k(retrieved_sources, item.expected_sources),
                    "precision_at_k": precision_at_k(retrieved_sources, item.expected_sources),
                    "reciprocal_rank": reciprocal_rank(retrieved_sources, item.expected_sources),
                    "answer_correctness": judge.answer_correctness,
                    "faithfulness": judge.faithfulness,
                    "answer_relevance": judge.answer_relevance,
                    "citation_correctness": citation_correctness(returned_citations, item.expected_sources),
                    "unanswerable_accuracy": unanswerable_accuracy(item.answerable, predicted_unanswerable),
                }
            )

        n = max(len(rows), 1)
        summary = {
            "hit_rate_at_k": sum(r["hit_rate_at_k"] for r in rows) / n,
            "precision_at_k": sum(r["precision_at_k"] for r in rows) / n,
            "mrr": sum(r["reciprocal_rank"] for r in rows) / n,
            "answer_correctness": sum(r["answer_correctness"] for r in rows) / n,
            "faithfulness": sum(r["faithfulness"] for r in rows) / n,
            "answer_relevance": sum(r["answer_relevance"] for r in rows) / n,
            "citation_correctness": sum(r["citation_correctness"] for r in rows) / n,
            "unanswerable_accuracy": sum(r["unanswerable_accuracy"] for r in rows) / n,
        }

        return {"summary": summary, "results": rows}
