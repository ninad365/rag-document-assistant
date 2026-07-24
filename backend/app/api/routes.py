from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile

from app.core.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from app.evaluation.evaluator import Evaluator
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentRecord,
    EvaluationRequest,
    EvaluationResponse,
    UploadResponse,
)
from app.rag.qa import RagService

router = APIRouter()
rag_service = RagService()
evaluator = Evaluator(rag_service)


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(
    api_key: str | None = Form(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    chunk_size: int = Form(DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(DEFAULT_CHUNK_OVERLAP),
    files: list[UploadFile] = File(...),
):
    resolved_api_key = x_api_key or api_key
    if not resolved_api_key:
        raise HTTPException(status_code=400, detail="API key is required")
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF is required")

    payload = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Unsupported file: {file.filename}")
        payload.append((file.filename, await file.read()))

    return rag_service.index_files(payload, resolved_api_key, chunk_size, chunk_overlap)


@router.get("/documents", response_model=list[DocumentRecord])
def list_documents(api_key: str) -> list[dict[str, str]]:
    return rag_service.list_documents(api_key)


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, api_key: str) -> dict:
    deleted = rag_service.delete_document(api_key, document_id)
    return {"deleted_chunks": deleted}


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> dict:
    return rag_service.chat(
        api_key=request.api_key,
        question=request.question,
        history=request.history,
        top_k=request.top_k,
    )


@router.post("/evaluation/run", response_model=EvaluationResponse)
def run_evaluation(request: EvaluationRequest) -> dict:
    return evaluator.run(request.api_key, request.dataset, request.top_k)
