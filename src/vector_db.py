"""Embeds data/so_qa.jsonl and indexes it into a local Chroma collection.

Run: python src/vector_db.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402

from src.utils import DATA_DIR, get_logger, timed  # noqa: E402

load_dotenv()
logger = get_logger(__name__)

SO_DATA_PATH = DATA_DIR / "so_qa.jsonl"


def load_records() -> list[dict]:
    if not SO_DATA_PATH.exists():
        raise FileNotFoundError(f"{SO_DATA_PATH} not found. Run `python setup.py` first.")
    with open(SO_DATA_PATH) as f:
        return [json.loads(line) for line in f]


def to_documents(records: list[dict]):
    # One document per Q&A pair. Chroma metadata can't hold lists, so
    # `tags` gets flattened to a comma-joined string.
    from langchain_core.documents import Document

    docs = []
    for r in records:
        text = f"Q: {r['question']}\n\nA: {r.get('answer', '')}"
        tags = r.get("tags", [])
        tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags or "")
        metadata = {
            "tags": tags_str,
            "score": r.get("score", 0),
            "url": r.get("url", ""),
            "date": r.get("date", ""),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def get_embedding_function():
    mode = os.getenv("EMBEDDING_MODE", "local:all-MiniLM-L6-v2")
    kind, _, model_name = mode.partition(":")

    if kind == "openai":
        from langchain_openai import OpenAIEmbeddings

        logger.info(f"Using OpenAI embeddings: {model_name or 'text-embedding-3-small'}")
        return OpenAIEmbeddings(model=model_name or "text-embedding-3-small")

    # local sentence-transformers, free, no key
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info(f"Using local sentence-transformers embeddings: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name or "all-MiniLM-L6-v2")


@timed
def build_index(docs, embedding_fn):
    import chromadb
    from langchain_chroma import Chroma

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "stackoverflow_qa")

    # Chroma.from_documents appends to an existing collection rather than
    # replacing it, so rerunning this script on refreshed data would just
    # accumulate duplicates. Clear any prior collection of the same name first.
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection(name=collection_name)
        logger.info(f"Cleared existing collection '{collection_name}' before rebuilding")
    except Exception:
        pass  # collection didn't exist yet, nothing to clear

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding_fn,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    logger.info(f"Indexed {len(docs)} documents into '{collection_name}' at {persist_dir}")
    return vectordb


def test_retrieval(vectordb, query: str = "how to fix memory leak Python", k: int = 5):
    results = vectordb.similarity_search_with_score(query, k=k)
    logger.info(f"Test query: {query!r} -> {len(results)} results")
    for doc, score in results:
        print(f"  score={score:.3f} | {doc.page_content[:100]!r}")
    return results


def main():
    records = load_records()
    docs = to_documents(records)
    embedding_fn = get_embedding_function()
    vectordb = build_index(docs, embedding_fn)
    test_retrieval(vectordb)
    print("\nDone. Next: python src/rag_chain.py")


if __name__ == "__main__":
    main()
