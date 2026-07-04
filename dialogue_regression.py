"""Run multi-turn dialogue regression cases against JARVIS."""

import json
import os
import re
from datetime import datetime

from jarvis_ai import JARVIS


class DialogueRegression:
    def __init__(self, cases_path="data/dialogue_regression_cases.json", out_dir="eval_reports"):
        self.cases_path = cases_path
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def load_cases(self):
        with open(self.cases_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _score_contains(self, text, required_terms):
        lower = (text or "").lower()
        if not required_terms:
            return 1.0
        hits = sum(1 for term in required_terms if term.lower() in lower)
        return hits / max(len(required_terms), 1)

    def _score_forbidden(self, text, forbidden_terms):
        lower = (text or "").lower()
        if not forbidden_terms:
            return 1.0
        hits = sum(1 for term in forbidden_terms if term.lower() in lower)
        if hits == 0:
            return 1.0
        return max(0.0, 1.0 - (hits / max(len(forbidden_terms), 1)))

    def _score_confidence(self, text):
        lower = (text or "").lower()
        return 1.0 if "confidence:" in lower else 0.0

    def run(self):
        cases = self.load_cases()
        jarvis = JARVIS()
        jarvis.stop_autolearn()
        jarvis.stop_proactive_mode()

        case_results = []
        total = 0.0
        count = 0

        try:
            for case in cases:
                turns_out = []
                case_score_sum = 0.0
                case_turns = case.get("turns", [])
                for turn in case_turns:
                    response = jarvis.generate_response(turn.get("user", ""))
                    required_score = self._score_contains(response, turn.get("required_terms", []))
                    forbidden_score = self._score_forbidden(response, turn.get("forbidden_terms", []))
                    confidence_score = self._score_confidence(response)
                    turn_score = 0.5 * required_score + 0.25 * forbidden_score + 0.25 * confidence_score
                    case_score_sum += turn_score
                    count += 1
                    total += turn_score
                    turns_out.append(
                        {
                            "user": turn.get("user", ""),
                            "response_preview": re.sub(r"\s+", " ", response)[:320],
                            "score": round(turn_score, 4),
                        }
                    )

                case_score = case_score_sum / max(len(case_turns), 1)
                case_results.append(
                    {
                        "id": case.get("id", "unknown"),
                        "score": round(case_score, 4),
                        "turns": turns_out,
                    }
                )
        finally:
            jarvis.stop_autolearn()
            jarvis.stop_proactive_mode()

        final_score = 100.0 * (total / max(count, 1))
        result = {
            "timestamp": datetime.now().isoformat(),
            "score": round(final_score, 2),
            "cases": case_results,
        }

        out_path = os.path.join(self.out_dir, "dialogue_regression_latest.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result


def main():
    result = DialogueRegression().run()
    print("=" * 60)
    print("Dialogue Regression Complete")
    print("=" * 60)
    print(f"Score: {result['score']}/100")
    print("Report: eval_reports/dialogue_regression_latest.json")


if __name__ == "__main__":
    main()