"""Run evaluation harness against JARVIS and print score."""

from eval_harness import EvalHarness
from jarvis_ai import JARVIS


def main():
    jarvis = JARVIS()
    result = EvalHarness(jarvis).run()
    print("=" * 60)
    print("JARVIS Evaluation Complete")
    print("=" * 60)
    print(f"Score: {result['score']}/100")
    print("Report: eval_reports/latest_eval.json")


if __name__ == "__main__":
    main()
