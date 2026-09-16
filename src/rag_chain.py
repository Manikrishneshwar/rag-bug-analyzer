"""Retrieval + LLM synthesis: diagnoses a bug from retrieved Stack Overflow context.

Run: python src/rag_chain.py "CUDA out of memory even with a small batch size"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402

from src.vector_db import get_embedding_function  # noqa: E402
from src.utils import format_source, get_logger, timed  # noqa: E402

load_dotenv()
logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a debugging expert. You are given a user's bug \
report and several similar Stack Overflow question/answer pairs retrieved \
for relevance. Using ONLY the retrieved context (plus general debugging \
knowledge if the context is insufficient), produce:

1. Root cause — your best hypothesis for what's causing the bug.
2. Likely solutions — a short ordered list of concrete fixes to try.
3. Confidence — High / Medium / Low, with a one-line reason (e.g. "High: \
retrieved context directly matches the error message").

Be concise. If the retrieved context doesn't clearly match the bug, say so \
explicitly in the confidence line rather than fabricating a diagnosis."""


def load_vectordb():
    # Builds the index from data/so_qa.jsonl if it's missing/empty —
    # needed on ephemeral deployments (e.g. HF Spaces free tier) where
    # data/chroma_db/ doesn't survive a restart.
    from langchain_chroma import Chroma

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "stackoverflow_qa")
    embedding_fn = get_embedding_function()
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_fn,
        persist_directory=persist_dir,
    )

    if vectordb._collection.count() == 0:
        from src.vector_db import build_index, load_records, to_documents

        logger.info(f"'{collection_name}' is empty — building index from data/so_qa.jsonl")
        records = load_records()
        docs = to_documents(records)
        vectordb = build_index(docs, embedding_fn)

    return vectordb


@timed
def retrieve(vectordb, query: str, k: int | None = None):
    k = k or int(os.getenv("APP_TOP_K", "5"))
    return vectordb.similarity_search_with_score(query, k=k)


def build_context_block(results) -> str:
    blocks = []
    for i, (doc, score) in enumerate(results, start=1):
        blocks.append(f"--- Source {i} ({format_source(doc.metadata, score)}) ---\n{doc.page_content}")
    return "\n\n".join(blocks)


def get_llm():
    # LLM_PROVIDER=groq (default, free) or openai (billed — no free tier).
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        logger.info(f"Using OpenAI chat model: {model} (billed — no free tier)")
        return ChatOpenAI(model=model, temperature=0.2)

    from langchain_groq import ChatGroq

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    logger.info(f"Using Groq chat model: {model} (free tier)")
    return ChatGroq(model=model, temperature=0.2)


@timed
def synthesize(query: str, context_block: str) -> str:
    llm = get_llm()

    user_prompt = f"Bug report:\n{query}\n\nRetrieved context:\n{context_block}"
    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
    return response.content


def diagnose(vectordb, query: str) -> dict:
    results = retrieve(vectordb, query)
    context_block = build_context_block(results)
    diagnosis_text = synthesize(query, context_block)
    return {
        "query": query,
        "sources": [
            {"content": doc.page_content, "metadata": doc.metadata, "score": score}
            for doc, score in results
        ],
        "diagnosis": diagnosis_text,
    }


def main():
    query = " ".join(sys.argv[1:]) or "TypeError: can't multiply sequence by non-int of type 'float'"
    vectordb = load_vectordb()
    result = diagnose(vectordb, query)

    print(f"\nQuery: {result['query']}\n")
    print("Retrieved sources:")
    for s in result["sources"]:
        print(" ", format_source(s["metadata"], s["score"]))
    print("\nDiagnosis:\n")
    print(result["diagnosis"])


if __name__ == "__main__":
    main()
