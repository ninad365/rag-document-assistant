from typing import Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class RetrievedChunk(BaseModel):
    source: str
    page: int
    score: float
    text: str


class Citation(BaseModel):
    source: str
    page: int


class ChatRequest(BaseModel):
    api_key: str = Field(min_length=10)
    question: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    top_k: int = Field(default=4, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    standalone_question: str
    latency_ms: float


class UploadResponse(BaseModel):
    indexed_documents: list[str]
    chunks_added: int


class DocumentRecord(BaseModel):
    document_id: str
    source: str


class EvaluationQuestion(BaseModel):
    question: str
    expected_answer: str
    expected_sources: list[str]
    answerable: bool


class EvaluationRequest(BaseModel):
    api_key: str = Field(min_length=10)
    dataset: list[EvaluationQuestion]
    top_k: int = Field(default=4, ge=1, le=20)


class EvaluationResult(BaseModel):
    question: str
    answer: str
    expected_answer: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    answerable: bool
    predicted_unanswerable: bool
    hit_rate_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    answer_correctness: float
    faithfulness: float
    answer_relevance: float
    citation_correctness: float


class EvaluationSummary(BaseModel):
    hit_rate_at_k: float
    precision_at_k: float
    mrr: float
    answer_correctness: float
    faithfulness: float
    answer_relevance: float
    citation_correctness: float
    unanswerable_accuracy: float


class EvaluationResponse(BaseModel):
    summary: EvaluationSummary
    results: list[EvaluationResult]
