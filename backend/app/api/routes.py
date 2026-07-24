from fastapi import APIRouter, File, Form, Header, HTTPException, Query, UploadFile
import logging
from openai import AuthenticationError, BadRequestError, RateLimitError

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
logger = logging.getLogger(__name__)


def run_with_api_key_error_handling(route_name: str, action):
    try:
        return action()
    except AuthenticationError as exc:
        logger.warning("Route failed: %s", route_name)
        raise HTTPException(status_code=400, detail="Invalid API key") from exc
    except RateLimitError as exc:
        logger.warning("Route failed: %s", route_name)
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.") from exc
    except BadRequestError as exc:
        message = str(exc).lower()
        if any(
            fragment in message
            for fragment in (
                "maximum context length",
                "context length",
                "token limit",
                "prompt is too long",
            )
        ):
            logger.warning("Route failed: %s", route_name)
            raise HTTPException(status_code=400, detail="Request exceeds the model token limit") from exc
        raise


def resolve_api_key(x_api_key: str | None, api_key: str | None) -> str:
    resolved_api_key = x_api_key or api_key
    if not resolved_api_key:
        raise HTTPException(status_code=400, detail="API key is required")
    return resolved_api_key


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
    resolved_api_key = resolve_api_key(x_api_key, api_key)
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF is required")

    payload = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Unsupported file: {file.filename}")
        payload.append((file.filename, await file.read()))

    return run_with_api_key_error_handling(
        "POST /documents/upload",
        lambda: rag_service.index_files(payload, resolved_api_key, chunk_size, chunk_overlap)
    )


@router.get("/documents", response_model=list[DocumentRecord])
def list_documents(
    api_key: str | None = Query(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> list[dict[str, str]]:
    resolved_api_key = resolve_api_key(x_api_key, api_key)
    return run_with_api_key_error_handling(
        "GET /documents",
        lambda: rag_service.list_documents(resolved_api_key),
    )


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    api_key: str | None = Query(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict:
    resolved_api_key = resolve_api_key(x_api_key, api_key)
    deleted = run_with_api_key_error_handling(
        "DELETE /documents/{document_id}",
        lambda: rag_service.delete_document(resolved_api_key, document_id),
    )
    return {"deleted_chunks": deleted}


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> dict:
    return run_with_api_key_error_handling(
        "POST /chat",
        lambda: rag_service.chat(
            api_key=request.api_key,
            question=request.question,
            history=request.history,
            top_k=request.top_k,
        )
    )


@router.post("/evaluation/run", response_model=EvaluationResponse)
def run_evaluation(request: EvaluationRequest) -> dict:
    return run_with_api_key_error_handling(
        "POST /evaluation/run",
        lambda: evaluator.run(request.api_key, request.dataset, request.top_k),
    )
