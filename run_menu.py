#!/usr/bin/env python3
"""Compact launcher focused on JARVIS conversation, with optional tools."""

import os
import platform
import subprocess
import sys
from pathlib import Path


TOOLS = [
    ("1", "run_eval.py", "Run eval harness"),
    ("2", "dialogue_regression.py", "Run dialogue regression"),
    ("3", "staged_training_cycle.py", "Run staged training cycle"),
    ("4", "train_from_chatgpt_json.py", "Train from ChatGPT export"),
    ("5", "build_synthetic_curriculum.py", "Build synthetic curriculum"),
    ("6", "build_local_instruction_dataset.py", "Build local instruction dataset"),
    ("7", "verify_setup.py", "Verify setup"),
    ("8", "validate_question_banks.py", "Validate question bank files"),
    ("9", "upgrade_model.py", "Promote checkpoint / activate model"),
    ("10", "local_growth_cycle.py", "Run local growth cycle"),
]


def get_venv_python():
    """Resolve the preferred python executable, prioritizing project virtual envs."""
    project_dir = Path(__file__).parent

    if platform.system() == "Windows":
        candidates = [
            project_dir / "venv" / "Scripts" / "python.exe",
            project_dir.parent / ".venv" / "Scripts" / "python.exe",
            project_dir.parent / "venv" / "Scripts" / "python.exe",
        ]
    else:
        candidates = [
            project_dir / "venv" / "bin" / "python",
            project_dir.parent / ".venv" / "bin" / "python",
            project_dir.parent / "venv" / "bin" / "python",
        ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    active_python = sys.executable
    if active_python and Path(active_python).exists():
        return active_python

    return sys.executable


def run_script(script_name, extra_args=None):
    extra_args = extra_args or []
    venv_python = get_venv_python()
    cmd = [venv_python, script_name] + extra_args
    print(f"\nRunning: {script_name}\n")
    subprocess.run(cmd)


def run_tools_menu():
    """Show compact advanced tools while keeping JARVIS chat as the primary flow."""
    while True:
        print("\n" + "=" * 50)
        print("Advanced Training/Eval Tools")
        print("=" * 50)
        for key, script, description in TOOLS:
            print(f"  {key}. {script:<28} - {description}")
        print("  0. Back")

        choice = input("Select tool: ").strip()
        if choice == "0":
            return

        match = next((item for item in TOOLS if item[0] == choice), None)
        if not match:
            print("Invalid choice.")
            continue

        _, script_name, _ = match
        args = []
        if script_name == "train_from_chatgpt_json.py":
            args = ["--data-folder", "..\\ChatGPT data 2024-26"]
        elif script_name == "upgrade_model.py":
            model_name = input("Model name to register: ").strip()
            if not model_name:
                print("Model name is required.")
                continue
            notes = input("Short notes (optional): ").strip()
            activate = input("Activate this checkpoint now? (y/n): ").strip().lower() in {"y", "yes"}
            args = ["--name", model_name]
            if notes:
                args.extend(["--notes", notes])
            if activate:
                args.append("--activate")
        run_script(script_name, args)


def is_interactive_session():
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def main():
    project_dir = Path(__file__).parent
    os.chdir(project_dir)

    if not is_interactive_session():
        print("Non-interactive launch detected. Opening the JARVIS window instead of waiting for menu input.")
        run_script("jarvis_gui.py")
        return

    while True:
        print("\n" + "=" * 58)
        print("JARVIS Launcher")
        print("=" * 58)
        print("  1. Open JARVIS window (local-only, memory + lookup)")
        print("  2. Start local-only JARVIS in terminal")
        print("  3. Start self-learning chat")
        print("  4. Start unattended learning daemon")
        print("  5. Run local growth cycle")
        print("  6. Build local instruction dataset")
        print("  7. Training/eval tools")
        print("  8. Rebuild local memory from the instruction dataset")
        print("  0. Exit")

        try:
            choice = input("Choose an option: ").strip().lower()
        except EOFError:
            print("\nNo terminal input detected. Opening the JARVIS window instead.\n")
            run_script("jarvis_gui.py")
            return
        if choice in {"", "1", "jarvis", "chat"}:
            run_script("jarvis_gui.py")
        elif choice in {"2", "local", "local-only"}:
            run_script("jarvis_ai.py", ["--local-only"])
        elif choice in {"3", "self", "self-learning"}:
            run_script("chat_self_learning.py")
        elif choice in {"4", "autopilot", "daemon", "auto"}:
            run_script("autopilot_daemon.py")
        elif choice in {"5", "growth", "grow"}:
            run_script("local_growth_cycle.py")
        elif choice in {"6", "instruction", "dataset"}:
            run_script("build_local_instruction_dataset.py")
        elif choice in {"7", "tools", "train", "eval"}:
            run_tools_menu()
        elif choice in {"8", "memory", "refresh", "rebuild"}:
            run_script("build_local_instruction_dataset.py")
        elif choice in {"0", "exit", "quit"}:
            print("Goodbye.")
            return
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
