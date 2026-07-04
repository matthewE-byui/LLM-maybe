#!/usr/bin/env python3
"""Run JARVIS in unattended learning mode with periodic maintenance cycles."""

import argparse
import json
import os
import time
from datetime import datetime

from jarvis_ai import JARVIS


def _write_jsonl(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _run_cycle(jarvis, force_maintenance=False):
    jarvis._autolearn_wake_event.set()
    jarvis._adaptive_autolearn_update(force=True)

    maintenance = jarvis._daily_learning_maintenance(force=force_maintenance)
    intelligence = jarvis.intelligence_snapshot()
    ready, readiness = jarvis.can_use_pc_actions()

    return {
        "timestamp": datetime.now().isoformat(),
        "maintenance": maintenance,
        "intelligence": intelligence,
        "pc_ready": bool(ready),
        "readiness": readiness,
        "learning_status": jarvis.learning_status(),
        "autolearn_status": jarvis.autolearn_status(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Unattended JARVIS learning daemon")
    parser.add_argument("--cycle-minutes", type=float, default=20.0, help="Minutes between maintenance cycles")
    parser.add_argument("--max-cycles", type=int, default=0, help="Stop after N cycles (0 means run forever)")
    parser.add_argument("--force-maintenance-every", type=int, default=18, help="Force maintenance every N cycles")
    parser.add_argument("--log-path", default=os.path.join("logs", "autopilot_daemon.jsonl"), help="JSONL output log path")
    parser.add_argument("--once", action="store_true", help="Run one forced cycle and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 64)
    print("JARVIS Autopilot Daemon")
    print("=" * 64)
    print(f"Cycle minutes: {args.cycle_minutes}")
    print(f"Max cycles: {'infinite' if args.max_cycles <= 0 else args.max_cycles}")
    print(f"Force maintenance every: {args.force_maintenance_every} cycle(s)")
    print(f"Log path: {args.log_path}")

    jarvis = JARVIS()
    cycle = 0
    sleep_seconds = max(30.0, args.cycle_minutes * 60.0)

    try:
        if args.once:
            result = _run_cycle(jarvis, force_maintenance=True)
            _write_jsonl(args.log_path, result)
            print(json.dumps(result, indent=2))
            return

        while True:
            cycle += 1
            force = args.force_maintenance_every > 0 and (cycle % args.force_maintenance_every == 0)
            result = _run_cycle(jarvis, force_maintenance=force)
            result["cycle"] = cycle
            _write_jsonl(args.log_path, result)

            band = (result.get("intelligence") or {}).get("band", "unknown")
            overall = (result.get("intelligence") or {}).get("overall", 0.0)
            print(
                f"[{result['timestamp']}] cycle={cycle} intelligence={overall} ({band}) "
                f"pc_ready={result['pc_ready']}"
            )

            if args.max_cycles > 0 and cycle >= args.max_cycles:
                print("Reached max cycles. Stopping autopilot daemon.")
                break

            time.sleep(sleep_seconds)

    except KeyboardInterrupt:
        print("\nAutopilot daemon interrupted by user.")
    finally:
        jarvis.stop_autolearn()
        jarvis.stop_proactive_mode()
        jarvis.stop_watch_mode()


if __name__ == "__main__":
    main()
