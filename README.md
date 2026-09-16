# RAG Bug Root Cause Analyzer

A retrieval-augmented generation (RAG) system that diagnoses software bugs
by retrieving similar Stack Overflow Q&A pairs and synthesizing a
root-cause + fix recommendation with an LLM.

Built as a fast, resume-ready project: working retrieval + synthesis +
demo UI, shipped end-to-end rather than polished to perfection.

## Architecture

```
              ┌────────────────────┐
 user query → │  Chroma retriever  │ → top-k similar SO Q&A pairs
              └────────────────────┘
                        │
                        ▼
              ┌────────────────────┐
              │  LLM synthesis     │ → root cause + solutions + confidence
              │  (Groq, free tier) │
              └────────────────────┘
                        │
                        ▼
              Streamlit UI (query box, sources, diagnosis)
```

- **Data**: Stack Overflow question + accepted-answer pairs (`setup.py`),
  chunked as one document per Q&A pair so context stays coherent.
- **Embeddings**: pluggable, local `sentence-transformers` (free, no key,
  default) or OpenAI `text-embedding-3-small` (fast, tiny cost, not free).
  Set via `EMBEDDING_MODE` in `.env`.
- **Vector store**: Chroma, persisted locally to `data/chroma_db/`.
- **Synthesis**: pluggable via `LLM_PROVIDER` in `.env`. **Groq**
  (`openai/gpt-oss-20b` by default, free tier, no cost, this is the
  default) or OpenAI (`gpt-4o-mini` by default, billed, OpenAI's API has
  no free tier at all). Given the retrieved context, prompted to return
  root cause / solutions / confidence.
- **UI**: Streamlit (`deployed/app.py`), query box, diagnosis, and an
  expandable list of retrieved sources for transparency.

### Cost

With the defaults (local embeddings + Groq), this project costs **$0** to
run. No OpenAI key required at all unless you deliberately switch
`LLM_PROVIDER=openai` or `EMBEDDING_MODE=openai:...` in `.env`.

## Project layout

```
rag-bug-analyzer/
├── setup.py                       # env + data download
├── src/
│   ├── vector_db.py                # embeddings + Chroma indexing
│   ├── rag_chain.py                # retrieval + LLM synthesis chain
│   └── utils.py                   # logging, timing, formatting helpers
├── deployed/
│   └── app.py                     # Streamlit UI
├── tests/
│   └── test_smoke.py
├── data/                          # so_qa.jsonl + chroma_db/ (gitignored)
├── requirements.txt
├── .env.example
└── QUICKSTART.md
```

## Running it

See [QUICKSTART.md](./QUICKSTART.md) for the full copy-paste setup. Short
version:

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY (free, see .env.example)
python setup.py
python src/vector_db.py
streamlit run deployed/app.py
```

## Deploying to Streamlit Community Cloud (free)

Hugging Face Spaces dropped native Streamlit support in April 2025.
Streamlit apps there now need the Docker SDK, which requires a paid
plan. Streamlit's own free hosting doesn't have that restriction and
needs zero code changes here (its secrets panel exposes keys as real
env vars, so `os.getenv(...)` in this code works unmodified).

The app auto-builds its vector index from the committed `data/so_qa.jsonl`
on first launch (see `load_vectordb()` in `src/rag_chain.py`), so it
works on a fresh deployment with no extra setup step.

1. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub (free, no card).
2. **Create app** → **Yup, I have an app** → pick this repo, branch
   `main`, main file path `deployed/app.py`.
3. **Advanced settings** → set **Python version to 3.11** (newer default
   versions lack prebuilt wheels for `torch`/`chromadb` and will fail to
   build) → **Secrets** → paste:
   ```toml
   GROQ_API_KEY = "your-actual-key"
   GROQ_MODEL = "openai/gpt-oss-20b"
   LLM_PROVIDER = "groq"
   ```
4. **Deploy**. First build takes a few minutes (installing PyTorch etc.).
   You get a public URL like `https://<something>.streamlit.app` for
   your resume.

Every push to `main` auto-redeploys to that same URL, so the link stays
current with no extra steps.

## Example queries to try

- `TypeError: can't multiply sequence by non-int of type 'float'`
- `CUDA out of memory even with a small batch size`
- `React component re-renders infinitely`

## Roadmap / where this stands

This is the **boilerplate scaffold**. Data loading (real Stack Overflow
pull, not just the fallback sample), indexing, retrieval, and the chain
have been run end-to-end against the live Hugging Face dataset (300
question/accepted-answer pairs, local embeddings, retrieval test passing),
see "Verified so far" below. What's still open, in the order the
original project plan calls for:

- [ ] Scale the real Stack Overflow pull from a few hundred pairs
      (verified working) up to 50K-100K, raise `SO_DATA_LIMIT` and
      `HF_SCAN_LIMIT` in `.env`; expect the download+join step to take
      proportionally longer
- [ ] Benchmark local vs. OpenAI embeddings on latency/quality for the
      full dataset
- [ ] Prompt-tune `SYSTEM_PROMPT` in `rag_chain.py` against ~10 realistic
      bug queries
- [ ] Run and document a quality spot-check (10 queries, manually
      graded), capture failure modes for the "what would you improve"
      interview talking point
- [ ] Deploy, finalize README metrics section, update resume bullet with
      real numbers

## Verified so far

- `setup.py` against the real `mikex86/stackoverflow-posts` dataset: it's
  a flat table of individual posts (`PostTypeId` 1=question, 2=answer),
  not pre-paired Q&A, so `setup.py` streams both types and joins each
  question to its accepted answer by id. Confirmed working: 300 matched
  pairs from a 2000-post scan window.
- `src/vector_db.py` with local embeddings (`all-MiniLM-L6-v2`): indexed
  those 300 docs into Chroma and ran a test retrieval ("how to fix
  memory leak Python"), top results were all genuinely
  memory/corruption-related. Retrieval quality looks sound even at this
  small scale.
- `src/rag_chain.py` with a real Groq key (`LLM_PROVIDER=groq`, model
  `openai/gpt-oss-20b`): full chain end-to-end, both on a query outside
  the sample (correctly flagged **Low confidence** on a CUDA question
  instead of fabricating a match) and on one inside it (correctly **High
  confidence**, exact right source, on a DateTime/age question).
  Note: Groq retires models periodically. If `GROQ_MODEL` 404s, see
  `.env.example` for how to list what's currently live.
- The Streamlit UI (`deployed/app.py`) and the auto-build-on-cold-start
  path in `load_vectordb()` (for ephemeral deployments) are both verified
  working too.
- Deployed live on Streamlit Community Cloud (Python 3.11 pinned in
  Advanced settings to avoid build failures on newer default versions).

## Known limitations (update as you test)

- Diagnosis quality is bounded by retrieval quality. Vague queries or
  bugs with no close Stack Overflow precedent will get generic or
  low-confidence answers.
- The fallback dataset is 3 hand-written examples; real usefulness
  requires running `setup.py` against the full Hugging Face pull.
- No caching/rate-limiting on the OpenAI calls yet, fine for a demo,
  not for production traffic.

## Resume bullet (draft, fill in real numbers once deployed)

> **Bug Root Cause Analyzer (RAG System)**: Built a retrieval-augmented
> generation system diagnosing software bugs from 50K+ Stack Overflow
> Q&A pairs; implemented a Chroma vector retrieval pipeline with
> pluggable local/OpenAI embeddings and an LLM synthesis chain producing
> root-cause + confidence-scored fixes; deployed an interactive
> Streamlit demo. [GitHub] | [Live demo]
