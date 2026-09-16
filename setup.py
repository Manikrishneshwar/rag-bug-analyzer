"""Downloads a Stack Overflow Q&A sample (or falls back to a bundled one)
and writes it to data/so_qa.jsonl.

Run: python setup.py
"""
import json
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_PATH = DATA_DIR / "so_qa.jsonl"


def load_env():
    """Load .env if present; warn (don't fail) if missing."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv

        load_dotenv(env_path)
        print(f"[setup] Loaded environment from {env_path}")
    else:
        print(
            "[setup] No .env found. Copy .env.example to .env if you plan to "
            "use OpenAI for embeddings or synthesis. Local embedding mode "
            "works without any keys."
        )


def download_from_huggingface(limit: int) -> list[dict]:
    # The dataset is a flat table of posts, not pre-paired Q&A: each row
    # has PostTypeId (1=question, 2=answer), and a question's
    # AcceptedAnswerId points at an answer's Id. We stream both types and
    # join them ourselves. Only a fraction of questions get their answer
    # within any given window, so we scan more rows (HF_SCAN_LIMIT) than
    # the number of pairs we actually want (limit).
    from datasets import load_dataset

    dataset_name = os.getenv("HF_DATASET_NAME", "mikex86/stackoverflow-posts")
    # Scan more rows than we need pairs, since only a fraction of questions
    # will have their accepted answer land inside the same window.
    scan_limit = int(os.getenv("HF_SCAN_LIMIT", str(max(20000, limit * 8))))
    print(
        f"[setup] Downloading '{dataset_name}' from Hugging Face "
        f"(target {limit} Q&A pairs, scanning up to {scan_limit} posts)..."
    )

    ds = load_dataset(dataset_name, split="train", streaming=True)

    questions: dict[int, dict] = {}
    answers: dict[int, dict] = {}
    scanned = 0

    def try_build_records() -> list[dict]:
        out = []
        for qid, q in questions.items():
            aid = q.get("AcceptedAnswerId")
            if aid is None or aid not in answers:
                continue
            a = answers[aid]
            title = q.get("Title") or ""
            body = q.get("Body") or ""
            out.append(
                {
                    "question": f"{title}\n\n{body}".strip(),
                    "answer": a.get("Body") or "",
                    "tags": q.get("Tags") or [],
                    "score": q.get("Score") or 0,
                    "url": f"https://stackoverflow.com/questions/{qid}",
                    "date": q.get("CreationDate") or "",
                }
            )
        return out

    for row in ds:
        scanned += 1
        post_type = row.get("PostTypeId")
        if post_type == 1:
            questions[row["Id"]] = row
        elif post_type == 2:
            answers[row["Id"]] = row

        if scanned % 2000 == 0:
            matched = try_build_records()
            print(f"[setup]   scanned {scanned} posts -> {len(matched)} matched pairs so far")
            if len(matched) >= limit:
                return matched[:limit]

        if scanned >= scan_limit:
            break

    matched = try_build_records()
    if len(matched) < limit:
        print(
            f"[setup] Only found {len(matched)}/{limit} matched pairs within "
            f"{scan_limit} scanned posts. Raise HF_SCAN_LIMIT in .env to get more."
        )
    return matched[:limit]


def load_fallback_sample() -> list[dict]:
    """Small hand-written sample so setup never hard-fails offline."""
    sample_path = DATA_DIR / "sample_fallback.jsonl"
    if sample_path.exists():
        with open(sample_path) as f:
            return [json.loads(line) for line in f]

    print("[setup] No fallback sample found — writing a tiny bootstrap sample.")
    bootstrap = [
        {
            "question": "TypeError: can't multiply sequence by non-int of type 'float'",
            "answer": "This happens when you multiply a list/string by a float instead "
            "of an int, e.g. `[1,2,3] * 2.5`. Cast the multiplier with `int()` "
            "or restructure to use numpy arrays if you need float scaling.",
            "tags": ["python", "typeerror"],
            "score": 42,
            "url": "https://stackoverflow.com/questions/example-1",
            "date": "2019-05-01",
        },
        {
            "question": "CUDA out of memory even with a small batch size",
            "answer": "Check for tensors not being freed between iterations (missing "
            "`.detach()` or `torch.no_grad()`), gradient accumulation left on, "
            "or a memory leak from storing loss tensors with their graph "
            "attached. Also try `torch.cuda.empty_cache()` and reduce "
            "`num_workers` in your DataLoader.",
            "tags": ["python", "pytorch", "cuda", "out-of-memory"],
            "score": 88,
            "url": "https://stackoverflow.com/questions/example-2",
            "date": "2021-02-14",
        },
        {
            "question": "React component re-renders infinitely",
            "answer": "Usually caused by calling setState directly in the render body, "
            "or an object/array literal in a useEffect dependency array that "
            "gets a new reference every render. Memoize the dependency with "
            "useMemo/useCallback or move the state update into an event "
            "handler / effect with a stable dependency list.",
            "tags": ["javascript", "reactjs", "react-hooks"],
            "score": 156,
            "url": "https://stackoverflow.com/questions/example-3",
            "date": "2020-08-22",
        },
    ]
    with open(sample_path, "w") as f:
        for row in bootstrap:
            f.write(json.dumps(row) + "\n")
    return bootstrap


def inspect(records: list[dict]) -> None:
    print(f"\n[setup] Loaded {len(records)} records.")
    if not records:
        return
    sample = records[0]
    print("[setup] Sample record fields:", list(sample.keys()))
    print("[setup] Sample question (truncated):", str(sample["question"])[:120])
    tag_counts: dict[str, int] = {}
    for r in records:
        tags = r.get("tags") or []
        if isinstance(tags, str):
            tags = tags.split(",")
        for t in tags:
            t = str(t).strip()
            if t:
                tag_counts[t] = tag_counts.get(t, 0) + 1
    top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:10]
    print("[setup] Top tags:", top_tags)


def main():
    load_env()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    limit = int(os.getenv("SO_DATA_LIMIT", "2000"))

    records: list[dict] = []
    try:
        records = download_from_huggingface(limit)
    except Exception as e:
        print(f"[setup] Hugging Face download failed ({e!r}). Falling back to sample data.")

    if not records:
        records = load_fallback_sample()

    with open(OUTPUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[setup] Wrote {len(records)} records to {OUTPUT_PATH}")
    inspect(records)
    print("\n[setup] Done. Next: python src/vector_db.py")


if __name__ == "__main__":
    sys.exit(main())
