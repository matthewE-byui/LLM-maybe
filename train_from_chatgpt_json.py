"""Train JARVIS checkpoint from local ChatGPT JSON export files."""

import argparse
import json
import os
import re
from pathlib import Path

import fine_tune_utils
from inference import LLMChat
from model_paths import resolve_active_checkpoint


NOISY_KEYS = {
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


def clean_text(text):
    text = " ".join((text or "").split())
    if len(text) < 30:
        return ""
    if text.startswith("http://") or text.startswith("https://"):
        return ""
    alpha = len(re.findall(r"[A-Za-z]", text))
    if alpha < 12:
        return ""
    symbol_ratio = len(re.findall(r"[^A-Za-z0-9\s.,!?;:'\"()\-]", text)) / max(len(text), 1)
    if symbol_ratio > 0.25:
        return ""
    return text


def collect_strings(obj, parent_key=""):
    collected = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in NOISY_KEYS:
                continue
            collected.extend(collect_strings(value, parent_key=key))
        return collected

    if isinstance(obj, list):
        for item in obj:
            collected.extend(collect_strings(item, parent_key=parent_key))
        return collected

    if isinstance(obj, str):
        # Prioritize message-like keys but still allow general long text.
        if parent_key in {"parts", "text", "title", "content", "message", "prompt", "response"} or len(obj) > 60:
            t = clean_text(obj)
            if t:
                collected.append(t)
        return collected

    return collected


def build_corpus_from_folder(folder_path, max_chars=1_200_000):
    root = Path(folder_path)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {root}")

    json_files = sorted(root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under: {root}")

    unique = set()
    ordered = []

    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
        except Exception:
            continue

        snippets = collect_strings(data)
        for s in snippets:
            key = s[:180].lower()
            if key in unique:
                continue
            unique.add(key)
            ordered.append(s)

    corpus = "\n".join(ordered)
    corpus = corpus[:max_chars]
    return corpus, len(json_files), len(ordered)


def main():
    parser = argparse.ArgumentParser(description="Train checkpoint from ChatGPT JSON export")
    parser.add_argument("--data-folder", required=True, help="Path to ChatGPT data folder")
    parser.add_argument("--checkpoint", default=resolve_active_checkpoint(), help="Checkpoint path")
    parser.add_argument("--steps", type=int, default=14, help="Fine-tuning steps")
    parser.add_argument("--max-chars", type=int, default=1200000, help="Max corpus chars")
    args = parser.parse_args()

    print("=" * 60)
    print("Training from local ChatGPT JSON export")
    print("=" * 60)
    print(f"Data folder: {args.data_folder}")
    print(f"Checkpoint: {args.checkpoint}")

    corpus, file_count, snippet_count = build_corpus_from_folder(args.data_folder, max_chars=args.max_chars)
    print(f"JSON files scanned: {file_count}")
    print(f"Text snippets extracted: {snippet_count}")
    print(f"Corpus length: {len(corpus)} chars")

    if len(corpus) < 2000:
        raise RuntimeError("Corpus too small after extraction. Nothing to train.")

    chat = LLMChat(checkpoint_path=args.checkpoint)
    result = fine_tune_utils.fine_tune_with_quality_gate(
        chat.model,
        corpus,
        chat.tokenizer,
        chat.device,
        learning_rate=2e-5,
        num_steps=max(4, args.steps),
        checkpoint_path=args.checkpoint,
        eval_text=corpus[-200000:],
        min_improvement=0.003,
        verbose=True,
    )

    print("\nTraining result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
