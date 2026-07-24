from collections.abc import Iterable

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.core.config import CHROMA_DIR

COLLECTION_NAME = "rag_documents"


def get_vector_store(api_key: str) -> Chroma:
    embeddings = OpenAIEmbeddings(api_key=api_key)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def unique_documents(metadatas: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    results: list[dict] = []
    for meta in metadatas:
        doc_id = str(meta.get("document_id", ""))
        source = str(meta.get("source", ""))
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        results.append({"document_id": doc_id, "source": source})
    return results
