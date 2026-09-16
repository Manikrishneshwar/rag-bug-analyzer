# Quickstart

Get from a clean checkout to a working demo in about 5 minutes.

```bash
cd rag-bug-analyzer

# 1. Create and activate a virtualenv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Fill in GROQ_API_KEY — free, no card, get one at console.groq.com/keys

# 4. Download + prep data
python setup.py

# 5. Build the vector index
python src/vector_db.py

# 6. Try a CLI diagnosis
python src/rag_chain.py "CUDA out of memory even with a small batch size"

# 7. Launch the UI
streamlit run deployed/app.py
```

## Local-only path (no API key at all)

Steps 4-5 above run fully locally, no key needed (`EMBEDDING_MODE=local:...`
by default). Steps 6-7 need `GROQ_API_KEY` for the synthesis step — until
then, retrieval alone is testable via `src/vector_db.py`'s built-in
`test_retrieval()` call.

## Troubleshooting

- **`FileNotFoundError: data/so_qa.jsonl`** — run `python setup.py` first.
- **Hugging Face download fails / no network** — `setup.py` falls back to
  a tiny bundled sample so you can still exercise the pipeline end-to-end.
- **Groq 404 "model_not_found"** — see `.env.example` for how to list
  currently available models with your key.
- **`chromadb` import errors on Apple Silicon / older Python** — pin to
  the versions in `requirements.txt`; newer/older combos sometimes clash.
