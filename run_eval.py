"""Run evaluation harness against JARVIS and print score."""

import argparse
import os

from eval_harness import EvalHarness
from jarvis_ai import JARVIS
from model_paths import resolve_active_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Run evaluation harness against JARVIS")
    parser.add_argument("--checkpoint", default=resolve_active_checkpoint(), help="Local checkpoint to load")
    parser.add_argument("--no-local-only", action="store_true", help="Allow non-local runtime behavior")
    args = parser.parse_args()

    jarvis = JARVIS(local_only=not args.no_local_only, checkpoint_path=args.checkpoint)
    result = EvalHarness(jarvis).run()
    print("=" * 60)
    print("JARVIS Evaluation Complete")
    print("=" * 60)
    print(f"Score: {result['score']}/100")
    print("Report: eval_reports/latest_eval.json")


if __name__ == "__main__":
    main()
