# rag-document-assistant

A production-style, interview-ready multi-PDF RAG project with a FastAPI backend and Streamlit frontend.

## Overview

This app indexes one or more PDFs into a persistent Chroma vector store and answers questions grounded only in retrieved context with page-level citations.

## Architecture

- **Frontend**: Streamlit UI for API key entry, upload, chat, retrieval viewer, and evaluation.
- **Backend**: FastAPI service for indexing, chat, and evaluation APIs.
- **RAG modules**: PDF extraction, chunking, vector store, retrieval, query rewriting, answer generation.
- **Evaluation module**: retrieval metrics + LLM-as-judge generation metrics.

## RAG pipeline

1. Upload PDFs
2. Extract per-page text + metadata (`source`, `page`, `document_id`)
3. Chunk text using `RecursiveCharacterTextSplitter`
4. Embed chunks with OpenAI embeddings
5. Persist in local Chroma store
6. Rewrite follow-up questions into standalone questions
7. Retrieve Top-K chunks
8. Answer only from retrieved context and include citations

## Technology stack

- Python 3.11+
- FastAPI
- Streamlit
- LangChain
- OpenAI Chat + Embeddings
- ChromaDB
- PyMuPDF
- Pydantic
- Pytest

## Repository structure

```text
backend/
  app/
    api/routes.py
    core/config.py
    evaluation/{evaluator.py,metrics.py}
    models/schemas.py
    rag/{chunking.py,document_loader.py,prompts.py,qa.py,vector_store.py}
    main.py
  tests/
frontend/
  app.py
requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run backend

```bash
uvicorn app.main:app --reload --app-dir backend
```

## Run frontend

```bash
streamlit run frontend/app.py
```

## API endpoints

- `GET /health`
- `POST /documents/upload`
- `GET /documents`
- `DELETE /documents/{document_id}`
- `POST /chat`
- `POST /evaluation/run`

## Example usage

1. Start backend and frontend.
2. Enter OpenAI API key in Streamlit sidebar.
3. Upload PDFs and index.
4. Ask questions and inspect retrieved chunks, scores, and latency.
5. Open evaluation tab and run JSON dataset evaluation.

## Evaluation methodology

Dataset fields per question:

- `question`
- `expected_answer`
- `expected_sources` (e.g. `file.pdf#p2`)
- `answerable` (boolean)

Metrics:

- Hit Rate@K
- Precision@K
- MRR
- Answer Correctness (LLM judge)
- Faithfulness (LLM judge)
- Answer Relevance (LLM judge)
- Citation Correctness
- Unanswerable Question Accuracy

Outputs include summary metrics and per-question records.

## Testing

```bash
pytest backend/tests -q
```

Includes tests for PDF extraction, chunking, metrics, prompt generation, and API validation. OpenAI calls should be mocked in higher-level tests if added.

## Future improvements

- Better duplicate-document handling and source deduplication
- Async batch indexing for large files
- Optional reranking
- Richer evaluation reports and export
