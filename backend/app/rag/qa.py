import time
import uuid
from typing import Any

from langchain_openai import ChatOpenAI

from app.core.config import DEFAULT_TOP_K
from app.models.schemas import ChatMessage
from app.rag.chunking import chunk_documents
from app.rag.document_loader import extract_pdf_pages
from app.rag.prompts import build_answer_prompt, build_rewrite_prompt
from app.rag.vector_store import get_vector_store, unique_documents


class RagService:
    def index_files(
        self,
        files: list[tuple[str, bytes]],
        api_key: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> dict[str, Any]:
        vector_store = get_vector_store(api_key)
        all_chunks = []
        indexed = []

        for filename, content in files:
            document_id = str(uuid.uuid4())
            pages = extract_pdf_pages(content, filename, document_id)
            chunks = chunk_documents(pages, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
            indexed.append(filename)

        if all_chunks:
            vector_store.add_documents(all_chunks)

        return {"indexed_documents": indexed, "chunks_added": len(all_chunks)}

    def list_documents(self, api_key: str) -> list[dict[str, str]]:
        vector_store = get_vector_store(api_key)
        payload = vector_store.get(include=["metadatas"])
        return unique_documents(payload.get("metadatas", []))

    def delete_document(self, api_key: str, document_id: str) -> int:
        vector_store = get_vector_store(api_key)
        payload = vector_store.get(where={"document_id": document_id}, include=["metadatas"])
        ids = payload.get("ids", [])
        if ids:
            vector_store.delete(ids=ids)
        return len(ids)

    def _rewrite_question(self, api_key: str, history: list[ChatMessage], question: str) -> str:
        if not history:
            return question
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
        prompt = build_rewrite_prompt([msg.model_dump() for msg in history], question)
        return llm.invoke(prompt).content.strip()

    def chat(
        self,
        api_key: str,
        question: str,
        history: list[ChatMessage],
        top_k: int = DEFAULT_TOP_K,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        standalone_question = self._rewrite_question(api_key, history, question)

        vector_store = get_vector_store(api_key)
        results = vector_store.similarity_search_with_relevance_scores(standalone_question, k=top_k)
        chunks = []
        context_lines = []
        citations = []

        for doc, score in results:
            source = str(doc.metadata.get("source", "unknown"))
            page = int(doc.metadata.get("page", 0))
            text = doc.page_content
            chunks.append({"source": source, "page": page, "score": float(score), "text": text})
            citations.append({"source": source, "page": page})
            context_lines.append(f"[{source} p.{page}] {text}")

        if not context_lines:
            return {
                "answer": "I could not find this in the uploaded documents.",
                "citations": [],
                "retrieved_chunks": [],
                "standalone_question": standalone_question,
                "latency_ms": (time.perf_counter() - t0) * 1000,
            }

        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
        answer = llm.invoke(build_answer_prompt("\n\n".join(context_lines), standalone_question)).content.strip()

        return {
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": chunks,
            "standalone_question": standalone_question,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }
