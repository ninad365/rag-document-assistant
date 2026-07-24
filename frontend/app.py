import json

import requests
import streamlit as st

BACKEND_URL = st.secrets.get("backend_url", "http://localhost:8000")

st.set_page_config(page_title="RAG Document Assistant", layout="wide")
st.title("RAG Document Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=150, step=10)
    top_k = st.number_input("Top-K", min_value=1, max_value=20, value=4)

    st.subheader("Upload PDFs")
    uploads = st.file_uploader("Select files", type=["pdf"], accept_multiple_files=True)
    if st.button("Index documents"):
        if not api_key or not uploads:
            st.error("Provide API key and at least one PDF.")
        else:
            files = [("files", (f.name, f.read(), "application/pdf")) for f in uploads]
            data = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
            headers = {"X-API-Key": api_key}
            res = requests.post(
                f"{BACKEND_URL}/documents/upload",
                files=files,
                data=data,
                headers=headers,
                timeout=120,
            )
            if res.ok:
                st.success(f"Indexed {res.json()['chunks_added']} chunks")
            else:
                st.error(res.text)

    if st.button("Refresh documents") and api_key:
        res = requests.get(
            f"{BACKEND_URL}/documents",
            headers={"X-API-Key": api_key},
            timeout=30,
        )
        if res.ok:
            st.session_state.documents = res.json()
        else:
            st.error(res.text)

    for doc in st.session_state.get("documents", []):
        cols = st.columns([4, 1])
        cols[0].write(f"{doc['source']} ({doc['document_id']})")
        if cols[1].button("Delete", key=doc["document_id"]):
            delete_res = requests.delete(
                f"{BACKEND_URL}/documents/{doc['document_id']}",
                headers={"X-API-Key": api_key},
                timeout=30,
            )
            if delete_res.ok:
                st.rerun()
            else:
                st.error(delete_res.text)

tab_chat, tab_eval = st.tabs(["Chat", "Evaluation"])

with tab_chat:
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask a question about uploaded PDFs")
    if user_q:
        st.session_state.history.append({"role": "user", "content": user_q})
        payload = {
            "api_key": api_key,
            "question": user_q,
            "history": st.session_state.history[:-1],
            "top_k": int(top_k),
        }
        res = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=90)
        if res.ok:
            body = res.json()
            answer = body["answer"]
            st.session_state.history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
                st.caption(f"Latency: {body['latency_ms']:.1f}ms")
                st.write("Citations:", body["citations"])
                with st.expander("Retrieved context"):
                    st.json(body["retrieved_chunks"])
        else:
            st.error(res.text)

with tab_eval:
    st.subheader("Run Evaluation")
    sample = st.text_area(
        "Dataset JSON",
        value='[{"question":"...","expected_answer":"...","expected_sources":["file.pdf#p1"],"answerable":true}]',
        height=180,
    )
    if st.button("Run evaluation"):
        try:
            dataset = json.loads(sample)
        except json.JSONDecodeError:
            st.error("Invalid JSON")
        else:
            payload = {"api_key": api_key, "dataset": dataset, "top_k": int(top_k)}
            res = requests.post(f"{BACKEND_URL}/evaluation/run", json=payload, timeout=240)
            if res.ok:
                body = res.json()
                st.write("Summary")
                st.json(body["summary"])
                st.write("Per-question results")
                st.json(body["results"])
            else:
                st.error(res.text)
