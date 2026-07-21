#!/usr/bin/env python3
"""Build a local instruction dataset from curated knowledge, project docs, and feedback examples."""

import argparse
import json
import os
from pathlib import Path

from quality_utils import is_high_signal_text, normalize_text


PROJECT_DOCS = [
    "README.md",
    "QUICKSTART.md",
    "run_menu.py",
    "jarvis_gui.py",
    "jarvis_ai.py",
    "local_growth_cycle.py",
    "staged_training_cycle.py",
    "train_large_corpus.py",
    "train_from_chatgpt_json.py",
    "build_synthetic_curriculum.py",
    "eval_harness.py",
    "dialogue_regression.py",
    "quality_utils.py",
]


def build_local_instruction_dataset(curated_path="data/curated_knowledge.jsonl", feedback_examples="feedback_examples.jsonl", project_root=None, limit_chars=250000):
    project_root = Path(project_root or Path(__file__).resolve().parent)
    records = []
    records.extend(_load_jsonl_text(curated_path, limit_chars=limit_chars // 2))
    records.extend(_load_feedback_examples(feedback_examples, limit_chars=limit_chars // 3))
    records.extend(_load_project_docs(project_root, limit_chars=limit_chars // 3))

    seen = set()
    deduped = []
    for item in records:
        key = (item.get("prompt", "")[:160] + "||" + item.get("response", "")[:200]).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _load_jsonl_text(path, limit_chars=120000):
    if not os.path.exists(path):
        return []
    out = []
    used = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                text = normalize_text(item.get("text", ""))
                if len(text) < 40 or not is_high_signal_text(text, min_score=0.4):
                    continue
                source = item.get("source", "curated")
                topic = normalize_text(item.get("topic", ""))
                out.append({
                    "prompt": f"Explain {topic or source} clearly.",
                    "response": text[:800],
                    "source": source,
                    "mode": "curated_knowledge",
                })
                used += len(text)
                if used >= limit_chars:
                    break
    except Exception:
        return []
    return out


def _load_feedback_examples(path, limit_chars=90000):
    if not os.path.exists(path):
        return []
    out = []
    used = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                user = normalize_text(item.get("user", ""))
                assistant = normalize_text(item.get("assistant", ""))
                sentiment = item.get("sentiment", "")
                note = normalize_text(item.get("note", ""))
                if len(user) < 12 or len(assistant) < 20:
                    continue
                prompt = user
                if sentiment == "-":
                    response = "A better response should be safer, clearer, and more grounded."
                else:
                    response = assistant
                out.append({
                    "prompt": prompt[:600],
                    "response": response[:900],
                    "source": "feedback_example",
                    "mode": f"feedback_{sentiment or 'unknown'}",
                    "note": note,
                })
                used += len(user) + len(assistant)
                if used >= limit_chars:
                    break
    except Exception:
        return []
    return out


def _load_project_docs(project_root, limit_chars=90000):
    base = Path(project_root)
    out = []
    used = 0
    for rel in PROJECT_DOCS:
        path = base / rel
        if not path.exists() or not path.is_file():
            continue
        try:
            lines = []
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    low = stripped.lower()
                    if low.startswith(("def ", "class ", "async def ", "# ", "##", "###")) or any(
                        term in low for term in ["usage", "checkpoint", "eval", "reward", "rollback", "retrieval", "memory", "safety"]
                    ):
                        lines.append(stripped.lstrip("# ").strip())
                    if sum(len(x) for x in lines) >= 1600:
                        break
            snippet = " ".join(lines).strip()
            if len(snippet) < 80 or not is_high_signal_text(snippet, min_score=0.35):
                continue
            out.append({
                "prompt": f"Summarize the purpose of {rel}.",
                "response": snippet[:1000],
                "source": "project_doc",
                "mode": "repo_context",
                "path": rel,
            })
            used += len(snippet)
            if used >= limit_chars:
                break
        except Exception:
            continue
    return out


def main():
    parser = argparse.ArgumentParser(description="Build a local instruction dataset for Jarvis")
    parser.add_argument("--curated-path", default="data/curated_knowledge.jsonl")
    parser.add_argument("--feedback-examples", default="feedback_examples.jsonl")
    parser.add_argument("--out", default="data/local_instruction_dataset.jsonl")
    parser.add_argument("--limit-chars", type=int, default=250000)
    args = parser.parse_args()

    deduped = build_local_instruction_dataset(
        curated_path=args.curated_path,
        feedback_examples=args.feedback_examples,
        project_root=Path(__file__).resolve().parent,
        limit_chars=args.limit_chars,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for item in deduped:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")

    print(f"Built {len(deduped)} instruction records -> {args.out}")


if __name__ == "__main__":
    main()
