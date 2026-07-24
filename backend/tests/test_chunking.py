from langchain_core.documents import Document

from app.rag.chunking import chunk_documents


def test_chunk_documents_splits_content():
    docs = [Document(page_content="A" * 1200, metadata={"source": "x.pdf", "page": 1})]
    chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(c.metadata["source"] == "x.pdf" for c in chunks)
