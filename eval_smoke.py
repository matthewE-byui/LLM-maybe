#!/usr/bin/env python3

import traceback
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path

from eval_harness import EvalHarness
from jarvis_ai import JARVIS
from model_paths import resolve_active_checkpoint


def main():
    out_path = Path(__file__).with_name("eval_smoke_result.txt")
    try:
        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            jarvis = JARVIS(local_only=True, checkpoint_path=resolve_active_checkpoint())
            jarvis.stop_autolearn()
            jarvis.stop_proactive_mode()
            result = EvalHarness(jarvis).run()
        out_path.write_text(f"OK\n{result['score']}\n", encoding="utf-8")
        print(result["score"])
    except Exception:
        out_path.write_text("ERR\n" + traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()