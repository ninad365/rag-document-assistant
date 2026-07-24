import fitz
from langchain_core.documents import Document


def extract_pdf_pages(file_bytes: bytes, filename: str, document_id: str) -> list[Document]:
    docs: list[Document] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        for idx, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": filename, "page": idx, "document_id": document_id},
                )
            )
    return docs
