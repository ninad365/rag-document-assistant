import fitz

from app.rag.document_loader import extract_pdf_pages


def test_extract_pdf_pages_includes_page_metadata():
    pdf = fitz.open()
    page1 = pdf.new_page()
    page1.insert_text((72, 72), "Hello page one")
    page2 = pdf.new_page()
    page2.insert_text((72, 72), "Hello page two")
    data = pdf.tobytes()
    pdf.close()

    docs = extract_pdf_pages(data, "sample.pdf", "doc-1")

    assert len(docs) == 2
    assert docs[0].metadata["source"] == "sample.pdf"
    assert docs[0].metadata["page"] == 1
    assert docs[1].metadata["page"] == 2
