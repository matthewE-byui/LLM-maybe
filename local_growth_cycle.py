#!/usr/bin/env python3
"""Run a local-only staged growth cycle with eval gates and rollback.

This script turns the current training recipe into one command:
- build a local instruction mix from curated knowledge, exports, and curriculum
- fine-tune in small stages with quality gates
- score the result with eval harness + dialogue regression
- roll back automatically if the cycle regresses
"""

import argparse
import json
import os
import shutil
from datetime import datetime

from dialogue_regression import DialogueRegression
from eval_harness import EvalHarness
from inference import LLMChat
from jarvis_ai import JARVIS
from build_local_instruction_dataset import build_local_instruction_dataset
from model_paths import resolve_active_checkpoint
from staged_training_cycle import build_curriculum_text, read_curated_text, train_stage
from train_from_chatgpt_json import build_corpus_from_folder


def read_feedback_examples(path, max_chars=140000):
    if not os.path.exists(path):
        return ""
    parts = []
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
                user = str(item.get("user", "")).strip()
                assistant = str(item.get("assistant", "")).strip()
                sentiment = str(item.get("sentiment", "")).strip()
                note = str(item.get("note", "")).strip()
                if len(user) < 12 or len(assistant) < 20:
                    continue
                parts.append(f"User: {user}\nAssistant: {assistant}\nFeedback: {sentiment}\nNote: {note or 'none'}\n")
                if sum(len(p) for p in parts) >= max_chars:
                    break
    except Exception:
        return ""
    return "\n".join(parts)[:max_chars]


def _load_checkpoint_state(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        return False
    try:
        import torch

        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return True
    except Exception:
        return False


def _snapshot_metrics(checkpoint_path):
    jarvis = JARVIS(local_only=True, checkpoint_path=checkpoint_path)
    try:
        jarvis.stop_autolearn()
        jarvis.stop_proactive_mode()
        eval_result = EvalHarness(jarvis).run()
    finally:
        jarvis.stop_autolearn()
        jarvis.stop_proactive_mode()

    dialogue_result = DialogueRegression(local_only=True).run()
    return {
        "eval_score": float(eval_result.get("score", 0.0)),
        "dialogue_score": float(dialogue_result.get("score", 0.0)),
        "eval_report": eval_result,
        "dialogue_report": dialogue_result,
    }


def _combined_score(metrics):
    return round((0.65 * metrics["eval_score"]) + (0.35 * metrics["dialogue_score"]), 2)


def _backup_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None
    backup_path = checkpoint_path + ".backup"
    shutil.copy2(checkpoint_path, backup_path)
    return backup_path


def _restore_checkpoint(backup_path, checkpoint_path, chat):
    if not backup_path or not os.path.exists(backup_path):
        return False
    shutil.copy2(backup_path, checkpoint_path)
    return _load_checkpoint_state(chat.model, checkpoint_path, chat.device)


def _write_instruction_mix(path, sections):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for title, text in sections:
            if not text:
                continue
            f.write(f"### {title}\n")
            f.write(text.strip())
            f.write("\n\n")


def main():
    parser = argparse.ArgumentParser(description="Run a local-only staged growth cycle for JARVIS")
    parser.add_argument("--checkpoint", default=resolve_active_checkpoint())
    parser.add_argument("--curated-path", default="data/curated_knowledge.jsonl")
    parser.add_argument("--chatgpt-data-folder", default="..\\ChatGPT data 2024-26")
    parser.add_argument("--export-max-chars", type=int, default=180000)
    parser.add_argument("--curriculum-per-level", type=int, default=12)
    parser.add_argument("--feedback-examples", default="feedback_examples.jsonl")
    parser.add_argument("--report", default="eval_reports/local_growth_cycle_report.json")
    parser.add_argument("--instruction-mix-out", default="data/local_instruction_mix.txt")
    parser.add_argument("--instruction-dataset-out", default="data/local_instruction_dataset.jsonl")
    parser.add_argument("--rollback-margin", type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs("eval_reports", exist_ok=True)
    backup_path = _backup_checkpoint(args.checkpoint)

    print("=" * 70)
    print("Local Growth Cycle")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Backup: {backup_path or 'none'}")

    baseline = _snapshot_metrics(args.checkpoint)
    baseline_combined = _combined_score(baseline)

    chat = LLMChat(checkpoint_path=args.checkpoint)
    stages = []

    curated_text = read_curated_text(args.curated_path)
    export_text, file_count, snippet_count = build_corpus_from_folder(args.chatgpt_data_folder, max_chars=args.export_max_chars)
    curriculum_text = build_curriculum_text(per_level=args.curriculum_per_level)
    feedback_text = read_feedback_examples(args.feedback_examples)
    instruction_records = build_local_instruction_dataset(
        curated_path=args.curated_path,
        feedback_examples=args.feedback_examples,
        project_root=os.path.dirname(__file__),
        limit_chars=max(120000, args.export_max_chars // 2),
    )
    instruction_dataset_text = "\n\n".join(
        f"Prompt: {item.get('prompt', '')}\nResponse: {item.get('response', '')}" for item in instruction_records
    )
    mix_text = "\n".join(part for part in [curated_text, export_text, curriculum_text, feedback_text] if part)

    _write_instruction_mix(
        args.instruction_mix_out,
        [
            ("curated_knowledge", curated_text),
            ("chatgpt_export", export_text),
            ("synthetic_curriculum", curriculum_text),
            ("feedback_examples", feedback_text),
            ("instruction_dataset", instruction_dataset_text),
        ],
    )

    os.makedirs(os.path.dirname(args.instruction_dataset_out) or ".", exist_ok=True)
    with open(args.instruction_dataset_out, "w", encoding="utf-8") as f:
        for item in instruction_records:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")

    if curated_text:
        stages.append(train_stage(chat.model, chat.tokenizer, chat.device, args.checkpoint, curated_text, "curated_knowledge", 8, 2e-5))

    if export_text:
        export_stage = train_stage(chat.model, chat.tokenizer, chat.device, args.checkpoint, export_text, "chatgpt_export", 8, 2e-5)
        export_stage["files"] = file_count
        export_stage["snippets"] = snippet_count
        stages.append(export_stage)

    if curriculum_text:
        stages.append(train_stage(chat.model, chat.tokenizer, chat.device, args.checkpoint, curriculum_text, "synthetic_curriculum", 6, 1.5e-5))

    if instruction_dataset_text:
        stages.append(train_stage(chat.model, chat.tokenizer, chat.device, args.checkpoint, instruction_dataset_text[:240000], "instruction_dataset", 6, 1.5e-5))

    if mix_text:
        stages.append(train_stage(chat.model, chat.tokenizer, chat.device, args.checkpoint, mix_text[:240000], "instruction_mix", 6, 1.5e-5))

    final = _snapshot_metrics(args.checkpoint)
    final_combined = _combined_score(final)
    improved = final_combined >= (baseline_combined + args.rollback_margin)
    rolled_back = False

    if not improved and backup_path:
        rolled_back = _restore_checkpoint(backup_path, args.checkpoint, chat)
        final = _snapshot_metrics(args.checkpoint)
        final_combined = _combined_score(final)

    report = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": args.checkpoint,
        "backup": backup_path,
        "instruction_mix_out": args.instruction_mix_out,
        "instruction_dataset_out": args.instruction_dataset_out,
        "baseline": baseline,
        "final": final,
        "baseline_combined": baseline_combined,
        "final_combined": final_combined,
        "improved": improved,
        "rolled_back": rolled_back,
        "stages": stages,
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 70)
    print("Local Growth Cycle Complete")
    print("=" * 70)
    print(f"Baseline combined: {baseline_combined}")
    print(f"Final combined: {final_combined}")
    print(f"Rolled back: {'yes' if rolled_back else 'no'}")
    print(f"Instruction mix: {args.instruction_mix_out}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()