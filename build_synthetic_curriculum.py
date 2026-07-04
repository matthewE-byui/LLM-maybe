"""Generate synthetic training curriculum tasks with increasing difficulty.

This script creates JSONL tasks suitable for local training or eval-style rehearsal.
"""

import argparse
import json
import random
from datetime import datetime


BASE_TOPICS = [
    "python debugging",
    "llm retrieval grounding",
    "vector memory design",
    "quality-gated training",
    "safe task execution",
    "confidence calibration",
    "data cleaning",
    "checkpoint evaluation",
]


def build_task(topic, level, idx):
    if level == 1:
        prompt = f"Explain {topic} in two concise sentences with one example."
    elif level == 2:
        prompt = f"Give a 5-step implementation plan for improving {topic}."
    elif level == 3:
        prompt = f"Analyze tradeoffs in {topic} and recommend one approach with rationale."
    elif level == 4:
        prompt = f"Given noisy evidence about {topic}, produce a grounded answer with confidence and citations."
    else:
        prompt = f"Design an end-to-end workflow for {topic} including safety checks, eval metrics, and rollback criteria."

    return {
        "id": f"curr-{level}-{idx:04d}",
        "difficulty": level,
        "topic": topic,
        "prompt": prompt,
        "expected": {
            "must_include": ["concise", "grounded" if level >= 4 else "clear"],
            "max_lines": 8 if level >= 3 else 5,
        },
        "created_at": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Build synthetic curriculum tasks")
    parser.add_argument("--out", default="data/synthetic_curriculum.jsonl", help="Output JSONL path")
    parser.add_argument("--per-level", type=int, default=60, help="Tasks per difficulty level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    tasks = []
    for level in range(1, 6):
        for idx in range(1, args.per_level + 1):
            topic = random.choice(BASE_TOPICS)
            tasks.append(build_task(topic, level, idx))

    with open(args.out, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=True) + "\n")

    print(f"Generated {len(tasks)} tasks -> {args.out}")


if __name__ == "__main__":
    main()
