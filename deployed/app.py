"""Streamlit UI for the RAG Bug Root Cause Analyzer.

Run: streamlit run deployed/app.py
"""
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402

from src.rag_chain import diagnose, load_vectordb  # noqa: E402
from src.utils import format_source  # noqa: E402

load_dotenv()

st.set_page_config(page_title="Bug Root Cause Analyzer", page_icon="🐛", layout="wide")


@st.cache_resource
def get_vectordb():
    return load_vectordb()


st.title("🐛 RAG Bug Root Cause Analyzer")
st.caption(
    "Diagnoses software bugs using retrieval-augmented generation over "
    "Stack Overflow Q&A data."
)

query = st.text_input(
    "Paste your error message or describe the bug",
    placeholder="e.g. CUDA out of memory even with a small batch size",
)

col1, col2 = st.columns([1, 3])
with col1:
    run = st.button("Diagnose", type="primary", use_container_width=True)

if run and query.strip():
    try:
        vectordb = get_vectordb()
    except Exception as e:
        st.error(
            f"Couldn't load the vector DB ({e}). Have you run "
            "`python setup.py` and `python src/vector_db.py` yet?"
        )
        st.stop()

    with st.spinner("Retrieving similar issues and synthesizing a diagnosis..."):
        start = time.perf_counter()
        result = diagnose(vectordb, query)
        elapsed = time.perf_counter() - start

    st.success(f"Done in {elapsed:.1f}s")

    st.subheader("Diagnosis")
    st.markdown(result["diagnosis"])

    st.subheader("Retrieved sources")
    st.caption("Shown for transparency — this is the context the diagnosis was based on.")
    for i, s in enumerate(result["sources"], start=1):
        with st.expander(f"Source {i} — {format_source(s['metadata'], s['score'])}"):
            st.text(s["content"])
elif run:
    st.warning("Enter a bug description or error message first.")

st.divider()
st.caption(
    "Limitations: diagnoses are only as good as the retrieved Stack Overflow "
    "matches — vague queries or bugs with no close SO precedent will produce "
    "low-confidence or generic answers. See README.md for known failure modes."
)
