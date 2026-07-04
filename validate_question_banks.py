#!/usr/bin/env python3
"""Validate question bank manifest and bank JSONL files."""

import json
import os
from collections import defaultdict

MANIFEST_PATH = os.path.join("data", "question_bank_manifest.json")


def load_manifest(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "banks" not in data:
        raise ValueError("Manifest must be a JSON object with a 'banks' list")
    if not isinstance(data["banks"], list):
        raise ValueError("Manifest 'banks' must be a list")
    return data


def validate_bank_file(path):
    stats = {
        "file": path,
        "entries": 0,
        "errors": 0,
        "duplicates": 0,
        "topics": defaultdict(int),
    }
    seen_q = set()

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                stats["errors"] += 1
                print(f"ERROR {path}:{idx} invalid JSON")
                continue

            question = " ".join(str(item.get("question", "")).split())
            answer = " ".join(str(item.get("answer", "")).split())
            topic = " ".join(str(item.get("topic", "")).split())

            if len(question) < 12:
                stats["errors"] += 1
                print(f"ERROR {path}:{idx} question too short")
                continue
            if len(answer) < 40:
                stats["errors"] += 1
                print(f"ERROR {path}:{idx} answer too short")
                continue
            if not topic:
                stats["errors"] += 1
                print(f"ERROR {path}:{idx} missing topic")
                continue

            q_key = question.lower()
            if q_key in seen_q:
                stats["duplicates"] += 1
            else:
                seen_q.add(q_key)

            stats["entries"] += 1
            stats["topics"][topic.lower()] += 1

    return stats


def main():
    manifest = load_manifest(MANIFEST_PATH)
    banks = manifest.get("banks", [])

    total_entries = 0
    total_errors = 0
    total_duplicates = 0

    print("=" * 68)
    print("Question Bank Validation")
    print("=" * 68)

    for bank in banks:
        if not isinstance(bank, dict):
            print("ERROR manifest entry is not an object")
            total_errors += 1
            continue
        if bank.get("enabled", True) is False:
            continue

        rel = bank.get("path", "")
        name = bank.get("name", "unnamed")
        if not rel:
            print(f"ERROR manifest bank '{name}' missing path")
            total_errors += 1
            continue

        if not os.path.exists(rel):
            print(f"ERROR bank file missing: {rel}")
            total_errors += 1
            continue

        stats = validate_bank_file(rel)
        total_entries += stats["entries"]
        total_errors += stats["errors"]
        total_duplicates += stats["duplicates"]
        print(
            f"OK {name}: entries={stats['entries']} errors={stats['errors']} duplicates={stats['duplicates']}"
        )

    print("-" * 68)
    print(
        f"TOTAL entries={total_entries} errors={total_errors} duplicates={total_duplicates}"
    )
    print("=" * 68)

    if total_errors > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
