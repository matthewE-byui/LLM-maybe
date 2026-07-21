#!/usr/bin/env python3
"""Promote a checkpoint into the local model registry and optionally activate it."""

import argparse
import json
import os
import shutil
from datetime import datetime

from model_paths import MODEL_BANK_DIR, register_checkpoint, resolve_active_checkpoint


def _load_eval_snapshot(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Promote a checkpoint into the local model registry")
    parser.add_argument("--source", default=resolve_active_checkpoint(), help="Source checkpoint to promote")
    parser.add_argument("--name", required=True, help="Model name to register")
    parser.add_argument("--notes", default="", help="Short notes for this checkpoint")
    parser.add_argument("--activate", action="store_true", help="Make this the active checkpoint")
    parser.add_argument("--eval-report", default="eval_reports/latest_eval.json", help="Eval report to attach")
    parser.add_argument("--dialogue-report", default="eval_reports/dialogue_regression_latest.json", help="Dialogue report to attach")
    args = parser.parse_args()

    if not os.path.exists(args.source):
        raise FileNotFoundError(f"Source checkpoint not found: {args.source}")

    MODEL_BANK_DIR.mkdir(parents=True, exist_ok=True)
    dest = str(MODEL_BANK_DIR / f"{args.name}.pt")
    shutil.copy2(args.source, dest)

    metrics = {
        "eval": _load_eval_snapshot(args.eval_report),
        "dialogue": _load_eval_snapshot(args.dialogue_report),
        "promoted_at": datetime.now().isoformat(),
        "source": args.source,
    }

    if args.activate:
        current = MODEL_BANK_DIR / "current.pt"
        shutil.copy2(dest, current)
        dest = str(current)

    registry = register_checkpoint(args.name, dest, notes=args.notes, metrics=metrics, activate=args.activate)

    print("=" * 60)
    print("Model upgrade complete")
    print("=" * 60)
    print(f"Registered: {args.name}")
    print(f"Checkpoint: {dest}")
    print(f"Active: {'yes' if args.activate else 'no'}")
    print(f"Registry: checkpoints/model_registry.json")
    print(f"Current active checkpoint: {registry.get('active_checkpoint')}")


if __name__ == "__main__":
    main()