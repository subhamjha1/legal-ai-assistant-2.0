from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8004")
API = f"{BACKEND_URL}/api/v1"

st.set_page_config(page_title="Legal & Tax RAG", layout="wide")
st.title("Legal & Tax RAG Assistant")
st.caption("US Acts · Court Judgments · Legal Commentary · IRS Regulations — hybrid retrieval with citations")

tab_ask, tab_upload, tab_docs, tab_eval = st.tabs(["Ask", "Upload", "Documents", "Evaluation"])

with tab_ask:
    col_filters, col_main = st.columns([1, 3])

    with col_filters:
        st.subheader("Filters")
        doc_type = st.selectbox("Document type", ["Any", "act", "judgment", "commentary", "irs_regulation"])
        year = st.text_input("Year (optional)")
        court = st.text_input("Court (optional)")
        use_graph = st.checkbox("Use GraphRAG expansion", value=True)

    with col_main:
        query = st.text_area(
            "Ask a legal or tax question", height=100, placeholder="e.g. What qualifies as a capital asset under IRC?"
        )
        if st.button("Ask", type="primary") and query.strip():
            filters = {}
            if doc_type != "Any":
                filters["document_type"] = doc_type
            if year.strip():
                filters["year"] = int(year)
            if court.strip():
                filters["court"] = court

            with st.spinner("Retrieving and generating answer..."):
                try:
                    resp = requests.post(
                        f"{API}/query",
                        json={"query": query, "filters": filters or None, "use_graph_expansion": use_graph},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                except Exception as e:
                    st.error(f"Request failed: {e}")
                    result = None

            if result:
                if result.get("insufficient_evidence"):
                    st.warning("Insufficient evidence in the provided documents to answer this fully.")

                st.markdown("### Answer")
                st.write(result.get("answer", ""))

                st.markdown("### Summary")
                st.write(result.get("summary", ""))

                confidence = result.get("retrieval_confidence", 0.0)
                st.progress(min(max(confidence, 0.0), 1.0), text=f"Retrieval confidence: {confidence:.0%}")
                st.caption(result.get("confidence_note", ""))

                st.markdown("### Supporting Citations")
                for c in result.get("supporting_citations", []):
                    section = f", §{c.get('section')}" if c.get("section") else ""
                    st.markdown(f"- **{c.get('document_name')}**, p.{c.get('page')}{section} — {c.get('quote_or_paraphrase')}")

                with st.expander("Retrieved chunks (raw)"):
                    for rc in result.get("retrieved_chunks", []):
                        st.markdown(f"**{rc['filename']}** — {rc['citation_label']} (rerank score: {rc['rerank_score']:.3f})")

                st.markdown("---")
                fb_col1, fb_col2 = st.columns(2)
                if fb_col1.button("Helpful"):
                    requests.post(f"{API}/feedback", json={"query": query, "answer": result.get("answer", ""), "rating": 1})
                    st.toast("Thanks for the feedback!")
                if fb_col2.button("Not helpful"):
                    requests.post(f"{API}/feedback", json={"query": query, "answer": result.get("answer", ""), "rating": -1})
                    st.toast("Thanks — we'll use this to improve.")

with tab_upload:
    st.subheader("Upload a legal PDF")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded and st.button("Upload & Index"):
        with st.spinner("Parsing, chunking, and indexing..."):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                resp = requests.post(f"{API}/upload", files=files, timeout=300)
                resp.raise_for_status()
                st.success(f"Indexed: {resp.json()}")
            except Exception as e:
                st.error(f"Upload failed: {e}")

with tab_docs:
    st.subheader("Indexed Documents")
    if st.button("Refresh"):
        st.session_state["docs_refresh"] = True
    try:
        docs = requests.get(f"{API}/documents", timeout=30).json()
        st.dataframe(docs, use_container_width=True)
    except Exception as e:
        st.info(f"Could not load documents (is the backend running?): {e}")

with tab_eval:
    st.subheader("Golden Dataset Evaluation")
    st.caption("Runs retrieval + generation metrics against data/golden/golden_dataset.csv")
    run_gen = st.checkbox("Include RAGAS/DeepEval generation metrics (slower)", value=True)
    if st.button("Run Evaluation"):
        with st.spinner("Running evaluation — this can take a few minutes..."):
            try:
                resp = requests.post(f"{API}/evaluate", params={"run_generation": run_gen}, timeout=1800)
                resp.raise_for_status()
                st.json(resp.json())
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
