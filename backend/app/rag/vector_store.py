from collections.abc import Iterable

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import CHROMA_DIR

COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
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
