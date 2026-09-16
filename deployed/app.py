"""Streamlit UI for the RAG Bug Root Cause Analyzer.

Run: streamlit run deployed/app.py
"""
import re
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

CUSTOM_CSS = """
<style>
.block-container { max-width: 880px; padding-top: 2.5rem; }
h1 { font-weight: 700; letter-spacing: -0.02em; }
.intro-text {
    color: rgba(120, 120, 130, 0.95);
    font-size: 1.05rem;
    line-height: 1.6;
    margin-bottom: 1.4rem;
}
.example-caption {
    font-size: 0.85rem;
    color: rgba(120, 120, 130, 0.85);
    margin-bottom: 0.3rem;
}
div[data-testid="stTextInput"] input {
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    font-size: 1rem;
}
.stButton button {
    border-radius: 8px;
    font-weight: 600;
}
.diagnosis-card {
    background: rgba(130, 130, 140, 0.07);
    border: 1px solid rgba(130, 130, 140, 0.15);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 0.6rem;
}
.confidence-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
}
.confidence-high { background: rgba(46, 160, 67, 0.15); color: #2ea043; }
.confidence-medium { background: rgba(210, 153, 34, 0.18); color: #b8860b; }
.confidence-low { background: rgba(219, 68, 55, 0.15); color: #db4437; }
.footer-note {
    color: rgba(120, 120, 130, 0.85);
    font-size: 0.88rem;
    line-height: 1.5;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_vectordb():
    return load_vectordb()


EXAMPLES = [
    "CUDA out of memory even with a small batch size",
    "TypeError: can't multiply sequence by non-int of type 'float'",
    "React component re-renders infinitely",
]

if "query_text" not in st.session_state:
    st.session_state.query_text = ""

st.title("🐛 Bug Root Cause Analyzer")
st.markdown(
    '<p class="intro-text">Paste an error message or describe what\'s going '
    "wrong, and I'll dig through similar Stack Overflow questions to figure "
    "out what's probably happening and how to fix it.</p>",
    unsafe_allow_html=True,
)

st.markdown('<p class="example-caption">Not sure what to try? Pick one:</p>', unsafe_allow_html=True)
ex_cols = st.columns(len(EXAMPLES))
for col, example in zip(ex_cols, EXAMPLES):
    with col:
        if st.button(example, use_container_width=True, key=f"ex_{example}"):
            st.session_state.query_text = example

query = st.text_input(
    "Your bug, in your own words",
    key="query_text",
    placeholder="e.g. CUDA out of memory even with a small batch size",
)

run = st.button("Diagnose it", type="primary")

if run and query.strip():
    try:
        vectordb = get_vectordb()
    except Exception as e:
        st.error(
            f"I couldn't load the vector database ({e}). Have you run "
            "`python setup.py` and `python src/vector_db.py` yet?"
        )
        st.stop()

    with st.spinner("Reading through similar Stack Overflow threads..."):
        start = time.perf_counter()
        result = diagnose(vectordb, query)
        elapsed = time.perf_counter() - start

    diagnosis_text = result["diagnosis"]
    confidence_match = re.search(r"confidence[:\s\-]*\**\s*(high|medium|low)", diagnosis_text, re.IGNORECASE)
    confidence = confidence_match.group(1).lower() if confidence_match else None

    st.caption(f"Done in {elapsed:.1f}s")

    st.subheader("Here's what I found")
    if confidence:
        st.markdown(
            f'<span class="confidence-badge confidence-{confidence}">'
            f'{confidence.capitalize()} confidence</span>',
            unsafe_allow_html=True,
        )
    st.markdown(f'<div class="diagnosis-card">{diagnosis_text}</div>', unsafe_allow_html=True)

    st.subheader("What I based this on")
    st.caption("The Stack Overflow threads retrieved for this query, shown so you can judge for yourself.")
    for i, s in enumerate(result["sources"], start=1):
        with st.expander(f"Source {i}: {format_source(s['metadata'], s['score'])}"):
            st.text(s["content"])
elif run:
    st.warning("Give me an error message or a short description first.")

st.divider()
st.markdown(
    '<p class="footer-note">A diagnosis is only as good as the Stack Overflow '
    "threads it can find. Vague queries or genuinely novel bugs will get a "
    "low-confidence or generic answer.</p>",
    unsafe_allow_html=True,
)
