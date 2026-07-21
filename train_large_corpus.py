"""Large-scale local corpus trainer with byte budget and shard-based quality-gated updates.

Designed for high-volume training workflows (e.g., up to 250GB) while keeping
memory bounded and checkpoint updates quality-gated.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import fine_tune_utils
from inference import LLMChat
from model_paths import resolve_active_checkpoint
from quality_utils import content_hash, is_high_signal_text, normalize_text, text_quality_score


SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".jsonl", ".csv", ".py", ".log"}


def clean_text(text):
    text = normalize_text(text)
    if len(text) < 40:
        return ""
    alpha = sum(1 for ch in text if ch.isalpha())
    if alpha < 14:
        return ""
    symbol_count = sum(1 for ch in text if not (ch.isalnum() or ch.isspace() or ch in ".,!?;:'\"()-"))
    if symbol_count / max(len(text), 1) > 0.25:
        return ""
    return text


def collect_strings_from_json(obj, parent_key=""):
    noisy_keys = {
        "id",
        "conversation_id",
        "user_id",
        "create_time",
        "update_time",
        "model_slug",
        "safe_urls",
        "asset_pointer",
        "moderation_results",
        "metadata",
    }

    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in noisy_keys:
                continue
            out.extend(collect_strings_from_json(v, parent_key=k))
        return out

    if isinstance(obj, list):
        for item in obj:
            out.extend(collect_strings_from_json(item, parent_key=parent_key))
        return out

    if isinstance(obj, str):
        if parent_key in {"parts", "text", "title", "content", "message", "prompt", "response"} or len(obj) >= 80:
            t = clean_text(obj)
            if t:
                out.append(t)
    return out


def iter_files(root, allowed_exts):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in allowed_exts:
                yield p


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def train_shard(chat, shard_text, checkpoint_path, steps, learning_rate, min_improvement):
    eval_text = shard_text[-220000:]
    gate = fine_tune_utils.fine_tune_with_quality_gate(
        chat.model,
        shard_text,
        chat.tokenizer,
        chat.device,
        learning_rate=learning_rate,
        num_steps=steps,
        checkpoint_path=checkpoint_path,
        eval_text=eval_text,
        min_improvement=min_improvement,
        verbose=True,
    )
    return gate


def main():
    parser = argparse.ArgumentParser(description="Large-scale corpus training with max byte budget")
    parser.add_argument("--data-root", required=True, help="Root folder containing training files")
    parser.add_argument("--checkpoint", default=resolve_active_checkpoint(), help="Checkpoint path")
    parser.add_argument("--max-gb", type=float, default=250.0, help="Maximum data budget in GB")
    parser.add_argument("--shard-max-chars", type=int, default=1200000, help="Chars per training shard")
    parser.add_argument("--steps", type=int, default=10, help="Fine-tuning steps per shard")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate per shard")
    parser.add_argument("--min-improvement", type=float, default=0.003, help="Quality-gate improvement threshold")
    parser.add_argument("--max-dedup", type=int, default=500000, help="Max dedup hashes retained in memory")
    parser.add_argument("--min-quality-score", type=float, default=0.48, help="Minimum snippet quality score")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists() or not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    max_bytes = int(args.max_gb * (1024 ** 3))
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    ledger_path = Path("large_training_ledger.jsonl")

    print("=" * 70)
    print("Large Corpus Training")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print(f"Data root: {data_root}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Byte budget: {max_bytes} bytes (~{args.max_gb} GB)")
    print(f"Shard max chars: {args.shard_max_chars}")
    print(f"Steps per shard: {args.steps}")

    chat = LLMChat(checkpoint_path=args.checkpoint)

    dedup = set()
    shard_parts = []
    shard_chars = 0
    shard_index = 0

    bytes_seen = 0
    files_seen = 0
    files_used = 0
    snippets_used = 0
    snippets_rejected = 0
    shards_saved = 0

    start = time.time()

    def maybe_train_current_shard(force=False):
        nonlocal shard_parts, shard_chars, shard_index, shards_saved
        if (not force) and shard_chars < args.shard_max_chars:
            return
        if shard_chars < 4000:
            return

        shard_index += 1
        text = "\n".join(shard_parts)
        print(f"\n=== Training shard {shard_index} | chars={len(text)} ===")
        gate = train_shard(
            chat=chat,
            shard_text=text,
            checkpoint_path=args.checkpoint,
            steps=args.steps,
            learning_rate=args.learning_rate,
            min_improvement=args.min_improvement,
        )
        if gate.get("saved", False):
            shards_saved += 1

        append_jsonl(
            ledger_path,
            {
                "run_id": run_id,
                "type": "shard",
                "shard_index": shard_index,
                "chars": len(text),
                "gate": gate,
                "timestamp": datetime.now().isoformat(),
            },
        )

        shard_parts = []
        shard_chars = 0

    for fp in iter_files(data_root, SUPPORTED_EXTENSIONS):
        if bytes_seen >= max_bytes:
            break

        files_seen += 1
        fsize = fp.stat().st_size if fp.exists() else 0
        if fsize <= 0:
            continue

        # Respect byte budget based on raw file sizes.
        if bytes_seen + fsize > max_bytes:
            continue

        snippets_before = snippets_used

        try:
            if fp.suffix.lower() in {".json", ".jsonl"}:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
                texts = collect_strings_from_json(data)
            else:
                texts = []
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        t = clean_text(line)
                        if t:
                            texts.append(t)
        except Exception:
            continue

        for t in texts:
            if not is_high_signal_text(t, min_score=args.min_quality_score):
                snippets_rejected += 1
                continue

            h = content_hash(t, prefix_chars=260)
            if h in dedup:
                continue
            dedup.add(h)
            if len(dedup) > args.max_dedup:
                # Drop oldest-ish by resetting set to keep memory bounded.
                dedup = set(list(dedup)[-args.max_dedup:])

            cleaned = normalize_text(t)
            shard_parts.append(cleaned)
            shard_chars += len(cleaned)
            snippets_used += 1

            if shard_chars >= args.shard_max_chars:
                maybe_train_current_shard(force=True)

        if snippets_used > snippets_before:
            files_used += 1
            bytes_seen += fsize

    maybe_train_current_shard(force=True)

    elapsed = round(time.time() - start, 2)
    summary = {
        "run_id": run_id,
        "type": "run_summary",
        "timestamp": datetime.now().isoformat(),
        "data_root": str(data_root),
        "checkpoint": args.checkpoint,
        "max_bytes": max_bytes,
        "bytes_seen": bytes_seen,
        "files_seen": files_seen,
        "files_used": files_used,
        "snippets_used": snippets_used,
        "snippets_rejected": snippets_rejected,
        "shards_trained": shard_index,
        "shards_saved": shards_saved,
        "min_quality_score": args.min_quality_score,
        "elapsed_seconds": elapsed,
    }
    append_jsonl(ledger_path, summary)

    print("\n" + "=" * 70)
    print("Training complete")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print(f"Ledger: {ledger_path}")


if __name__ == "__main__":
    main()
