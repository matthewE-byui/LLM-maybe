"""Run a staged local training cycle and compare before/after transcripts."""

import argparse
import json
import os
import re
from datetime import datetime

import fine_tune_utils
from build_synthetic_curriculum import build_task
from dialogue_regression import DialogueRegression
from inference import LLMChat
from model_paths import resolve_active_checkpoint
from transformers import AutoTokenizer


TRANSCRIPT_PROMPTS = [
    "what is overfitting in machine learning",
    "why is that bad",
    "explain gradient descent simply",
    "what is a transformer in ai",
    "i want to learn python where should i start",
]


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _uniq_ratio(text):
    toks = _tokenize(text)
    if not toks:
        return 0.0
    return len(set(toks)) / max(len(toks), 1)


def build_transcript_deltas(before_transcripts, after_transcripts):
    deltas = []
    for before, after in zip(before_transcripts, after_transcripts):
        b = before.get("response", "")
        a = after.get("response", "")
        b_conf = int("confidence:" in b.lower())
        a_conf = int("confidence:" in a.lower())
        b_urls = len(re.findall(r"https?://|\bwww\.", b.lower()))
        a_urls = len(re.findall(r"https?://|\bwww\.", a.lower()))
        b_len = len(_tokenize(b))
        a_len = len(_tokenize(a))

        deltas.append(
            {
                "user": before.get("user", ""),
                "before_preview": " ".join(b.split())[:220],
                "after_preview": " ".join(a.split())[:220],
                "metrics": {
                    "confidence_before": b_conf,
                    "confidence_after": a_conf,
                    "confidence_delta": a_conf - b_conf,
                    "url_hits_before": b_urls,
                    "url_hits_after": a_urls,
                    "url_hits_delta": a_urls - b_urls,
                    "token_len_before": b_len,
                    "token_len_after": a_len,
                    "token_len_delta": a_len - b_len,
                    "lexical_diversity_before": round(_uniq_ratio(b), 4),
                    "lexical_diversity_after": round(_uniq_ratio(a), 4),
                    "lexical_diversity_delta": round(_uniq_ratio(a) - _uniq_ratio(b), 4),
                },
            }
        )

    summary = {
        "count": len(deltas),
        "confidence_gain": sum(d["metrics"]["confidence_delta"] for d in deltas),
        "url_noise_reduction": -sum(d["metrics"]["url_hits_delta"] for d in deltas),
        "avg_lexical_diversity_delta": round(
            sum(d["metrics"]["lexical_diversity_delta"] for d in deltas) / max(len(deltas), 1),
            4,
        ),
    }
    return {"summary": summary, "cases": deltas}


def read_curated_text(path):
    parts = []
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            text = item.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(parts)


def build_curriculum_text(per_level=12):
    tasks = []
    for level in range(1, 6):
        for idx in range(1, per_level + 1):
            task = build_task("llm retrieval grounding", level, idx)
            tasks.append(task["prompt"])
    return "\n".join(tasks)


def capture_transcripts(checkpoint_path):
    from jarvis_ai import JARVIS

    jarvis = JARVIS()
    jarvis.stop_autolearn()
    jarvis.stop_proactive_mode()
    transcripts = []
    try:
        for prompt in TRANSCRIPT_PROMPTS:
            response = jarvis.generate_response(prompt)
            transcripts.append({"user": prompt, "response": response})
    finally:
        jarvis.stop_autolearn()
        jarvis.stop_proactive_mode()
    return transcripts


def train_stage(model, tokenizer, device, checkpoint_path, text, label, num_steps, learning_rate):
    result = fine_tune_utils.fine_tune_with_quality_gate(
        model,
        text,
        tokenizer,
        device,
        learning_rate=learning_rate,
        num_steps=num_steps,
        checkpoint_path=checkpoint_path,
        eval_text=text[-120000:],
        min_improvement=0.003,
        verbose=True,
    )
    result["label"] = label
    return result


def main():
    parser = argparse.ArgumentParser(description="Run staged training and compare transcripts")
    parser.add_argument("--checkpoint", default=resolve_active_checkpoint())
    parser.add_argument("--chatgpt-data-folder", default="..\\ChatGPT data 2024-26")
    parser.add_argument("--curated-path", default="data/curated_knowledge.jsonl")
    parser.add_argument("--report", default="eval_reports/staged_training_report.json")
    parser.add_argument("--export-max-chars", type=int, default=180000)
    parser.add_argument("--curriculum-per-level", type=int, default=12)
    args = parser.parse_args()

    from train_from_chatgpt_json import build_corpus_from_folder

    before_dialogue = DialogueRegression().run()
    before_transcripts = capture_transcripts(args.checkpoint)

    chat = LLMChat(checkpoint_path=args.checkpoint)
    tokenizer = chat.tokenizer
    device = chat.device

    stages = []

    curated_text = read_curated_text(args.curated_path)
    if curated_text:
        stages.append(train_stage(chat.model, tokenizer, device, args.checkpoint, curated_text, "curated_knowledge", 8, 2e-5))

    export_text, file_count, snippet_count = build_corpus_from_folder(args.chatgpt_data_folder, max_chars=args.export_max_chars)
    if export_text:
        stage = train_stage(chat.model, tokenizer, device, args.checkpoint, export_text, "chatgpt_export", 8, 2e-5)
        stage["files"] = file_count
        stage["snippets"] = snippet_count
        stages.append(stage)

    curriculum_text = build_curriculum_text(per_level=args.curriculum_per_level)
    if curriculum_text:
        stages.append(train_stage(chat.model, tokenizer, device, args.checkpoint, curriculum_text, "synthetic_curriculum", 6, 1.5e-5))

    after_dialogue = DialogueRegression().run()
    after_transcripts = capture_transcripts(args.checkpoint)

    report = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": args.checkpoint,
        "stages": stages,
        "dialogue_regression_before": before_dialogue.get("score"),
        "dialogue_regression_after": after_dialogue.get("score"),
        "transcripts_before": before_transcripts,
        "transcripts_after": after_transcripts,
        "transcript_deltas": build_transcript_deltas(before_transcripts, after_transcripts),
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("Staged Training Cycle Complete")
    print("=" * 60)
    print(f"Dialogue regression: {before_dialogue.get('score')} -> {after_dialogue.get('score')}")
    print(f"Transcript delta summary: {report['transcript_deltas']['summary']}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()