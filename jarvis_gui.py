#!/usr/bin/env python3
"""Tkinter window for JARVIS with live transcript, logs, and runtime status."""

import argparse
import queue
import re
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext

from jarvis_ai import JARVIS


class QueueTee:
    """Forward text to the original stream and a UI queue."""

    def __init__(self, original_stream, message_queue):
        self.original_stream = original_stream
        self.message_queue = message_queue

    def write(self, text):
        if not text:
            return
        try:
            if self.original_stream is not None:
                self.original_stream.write(text)
                self.original_stream.flush()
        except Exception:
            pass
        try:
            self.message_queue.put(("log", text))
        except Exception:
            pass

    def flush(self):
        try:
            if self.original_stream is not None:
                self.original_stream.flush()
        except Exception:
            pass


class JARVISWindowApp:
    """Interactive GUI that keeps the assistant visible while it works."""

    def __init__(self, root, local_only=True):
        self.root = root
        self.local_only = local_only
        self.queue = queue.Queue()
        self.assistant = None
        self.assistant_ready = False
        self.busy = False
        self.shutdown_requested = False
        self.last_response_text = ""
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.stream_proxy = QueueTee(self.original_stdout, self.queue)
        sys.stdout = self.stream_proxy
        sys.stderr = self.stream_proxy

        self.root.title("JARVIS Window")
        self.root.geometry("1200x780")
        self.root.minsize(980, 640)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_ui()
        self._append_system_line("Starting JARVIS in window mode...")
        self._append_system_line("Local-only mode is {}.".format("ON" if self.local_only else "OFF"))
        self._append_system_line("Use 'lookup <topic>' to search local memory, question banks, and instruction data.")
        self._start_assistant_thread()
        self._poll_queue()
        self._refresh_status_loop()

    def _build_ui(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        self.root.configure(bg="#111317")
        style.configure("App.TFrame", background="#111317")
        style.configure("Panel.TFrame", background="#171a21")
        style.configure("Title.TLabel", background="#111317", foreground="#f5f7fb", font=("Segoe UI", 16, "bold"))
        style.configure("Sub.TLabel", background="#111317", foreground="#aab2c0", font=("Segoe UI", 10))
        style.configure("PanelTitle.TLabel", background="#171a21", foreground="#f5f7fb", font=("Segoe UI", 11, "bold"))
        style.configure("Info.TLabel", background="#171a21", foreground="#d7dce7", font=("Segoe UI", 9))
        style.configure("CardTitle.TLabel", background="#1f2430", foreground="#9fb0cc", font=("Segoe UI", 9, "bold"))
        style.configure("CardValue.TLabel", background="#1f2430", foreground="#f5f7fb", font=("Segoe UI", 10))
        style.configure("Chip.TButton", font=("Segoe UI", 9), padding=(10, 4))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 8))

        outer = ttk.Frame(self.root, style="App.TFrame", padding=14)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer, style="App.TFrame")
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="JARVIS Window", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Live transcript, runtime logs, and status updates in one view.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        body = ttk.Frame(outer, style="App.TFrame")
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body, style="Panel.TFrame", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = ttk.Frame(body, style="Panel.TFrame", padding=10)
        right.grid(row=0, column=1, sticky="nsew")

        ttk.Label(left, text="Conversation", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 8))

        quick_row = ttk.Frame(left, style="Panel.TFrame")
        quick_row.pack(fill="x", pady=(0, 8))
        ttk.Label(quick_row, text="Quick prompts:", style="Info.TLabel").pack(side="left")
        ttk.Button(
            quick_row,
            text="Explain this simply",
            style="Chip.TButton",
            command=lambda: self._insert_quick_prompt("Explain this simply: "),
        ).pack(side="left", padx=(8, 4))
        ttk.Button(
            quick_row,
            text="Plan next steps",
            style="Chip.TButton",
            command=lambda: self._insert_quick_prompt("Create a step-by-step plan for: "),
        ).pack(side="left", padx=4)
        ttk.Button(
            quick_row,
            text="Improve this answer",
            style="Chip.TButton",
            command=lambda: self._insert_quick_prompt("Improve this answer: "),
        ).pack(side="left", padx=4)
        ttk.Button(
            quick_row,
            text="Lookup locally",
            style="Chip.TButton",
            command=lambda: self._insert_quick_prompt("lookup "),
        ).pack(side="left", padx=4)

        self.transcript = scrolledtext.ScrolledText(
            left,
            wrap=tk.WORD,
            height=20,
            bg="#0f1318",
            fg="#e8edf5",
            insertbackground="#ffffff",
            relief="flat",
            font=("Segoe UI", 10),
        )
        self.transcript.pack(fill="both", expand=True)
        self.transcript.tag_configure("user", foreground="#7dd3fc")
        self.transcript.tag_configure("assistant", foreground="#f5f7fb")
        self.transcript.tag_configure("system", foreground="#a7f3d0")
        self.transcript.tag_configure("log", foreground="#b9c3d4")

        entry_row = ttk.Frame(left, style="Panel.TFrame")
        entry_row.pack(fill="x", pady=(10, 0))
        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(entry_row, textvariable=self.input_var)
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.input_entry.bind("<Return>", self._on_send)
        send_btn = ttk.Button(entry_row, text="Send", style="Accent.TButton", command=self._on_send)
        send_btn.pack(side="right")

        ttk.Label(right, text="Runtime", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 8))

        cards = ttk.Frame(right, style="Panel.TFrame")
        cards.pack(fill="x", pady=(0, 8))
        for i in range(3):
            cards.columnconfigure(i, weight=1)

        ready_card = ttk.Frame(cards, style="Panel.TFrame")
        ready_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ready_inner = tk.Frame(ready_card, bg="#1f2430", highlightthickness=0)
        ready_inner.pack(fill="both", expand=True)
        ttk.Label(ready_inner, text="READINESS", style="CardTitle.TLabel").pack(anchor="w", padx=8, pady=(6, 2))
        self.card_ready_var = tk.StringVar(value="loading")
        ttk.Label(ready_inner, textvariable=self.card_ready_var, style="CardValue.TLabel").pack(anchor="w", padx=8, pady=(0, 8))

        eval_card = ttk.Frame(cards, style="Panel.TFrame")
        eval_card.grid(row=0, column=1, sticky="nsew", padx=3)
        eval_inner = tk.Frame(eval_card, bg="#1f2430", highlightthickness=0)
        eval_inner.pack(fill="both", expand=True)
        ttk.Label(eval_inner, text="EVAL SCORE", style="CardTitle.TLabel").pack(anchor="w", padx=8, pady=(6, 2))
        self.card_eval_var = tk.StringVar(value="n/a")
        ttk.Label(eval_inner, textvariable=self.card_eval_var, style="CardValue.TLabel").pack(anchor="w", padx=8, pady=(0, 8))

        dialog_card = ttk.Frame(cards, style="Panel.TFrame")
        dialog_card.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        dialog_inner = tk.Frame(dialog_card, bg="#1f2430", highlightthickness=0)
        dialog_inner.pack(fill="both", expand=True)
        ttk.Label(dialog_inner, text="DIALOGUE", style="CardTitle.TLabel").pack(anchor="w", padx=8, pady=(6, 2))
        self.card_dialogue_var = tk.StringVar(value="n/a")
        ttk.Label(dialog_inner, textvariable=self.card_dialogue_var, style="CardValue.TLabel").pack(anchor="w", padx=8, pady=(0, 8))

        self.status_text = tk.StringVar(value="Assistant loading...")
        ttk.Label(right, textvariable=self.status_text, style="Info.TLabel", justify="left", wraplength=360).pack(anchor="w", fill="x")

        self.identity_text = tk.StringVar(value="Identity: loading")
        ttk.Label(right, textvariable=self.identity_text, style="Info.TLabel", justify="left", wraplength=360).pack(anchor="w", fill="x", pady=(8, 0))

        self.answer_meta_var = tk.StringVar(value="Last answer: confidence n/a | citations n/a")
        ttk.Label(right, textvariable=self.answer_meta_var, style="Info.TLabel", justify="left", wraplength=360).pack(anchor="w", fill="x", pady=(8, 0))

        ttk.Label(right, text="Live Log", style="PanelTitle.TLabel").pack(anchor="w", pady=(14, 8))
        self.log_text = scrolledtext.ScrolledText(
            right,
            wrap=tk.WORD,
            height=14,
            bg="#0f1318",
            fg="#d7dce7",
            insertbackground="#ffffff",
            relief="flat",
            font=("Consolas", 9),
        )
        self.log_text.pack(fill="both", expand=True)
        self.log_text.tag_configure("log", foreground="#c9d1e0")
        self.log_text.tag_configure("error", foreground="#fca5a5")

        button_row = ttk.Frame(right, style="Panel.TFrame")
        button_row.pack(fill="x", pady=(10, 0))
        ttk.Button(button_row, text="Wake", command=self._wake_assistant).pack(side="left")
        ttk.Button(button_row, text="Status", command=self._push_status_snapshot).pack(side="left", padx=8)
        ttk.Button(button_row, text="Clear Log", command=self._clear_log).pack(side="left")
        ttk.Button(button_row, text="Copy Last Reply", command=self._copy_last_response).pack(side="left", padx=8)

        reward_row = ttk.Frame(right, style="Panel.TFrame")
        reward_row.pack(fill="x", pady=(10, 0))
        ttk.Button(reward_row, text="Thumbs Up", command=lambda: self._rate_last_response("+")).pack(side="left")
        ttk.Button(reward_row, text="Thumbs Down", command=lambda: self._rate_last_response("-")).pack(side="left", padx=8)
        ttk.Label(reward_row, text="Reward the last assistant reply", style="Info.TLabel").pack(side="left", padx=6)

        footer = ttk.Frame(outer, style="App.TFrame")
        footer.pack(fill="x", pady=(10, 0))
        self.footer_var = tk.StringVar(value="Loading assistant...")
        ttk.Label(footer, textvariable=self.footer_var, style="Sub.TLabel").pack(anchor="w")

    def _append_transcript(self, who, text, tag):
        timestamp = time.strftime("%H:%M:%S")
        self.transcript.insert(tk.END, f"[{timestamp}] {who}: {text}\n\n", tag)
        self.transcript.see(tk.END)

    def _append_system_line(self, text):
        self._append_transcript("System", text, "system")

    def _append_user_line(self, text):
        self._append_transcript("You", text, "user")

    def _append_assistant_line(self, text):
        self.last_response_text = text or ""
        self._append_transcript("JARVIS", text, "assistant")
        self._update_answer_meta(text)

    def _append_log_line(self, text, tag="log"):
        self.log_text.insert(tk.END, text, tag)
        self.log_text.see(tk.END)

    def _clear_log(self):
        self.log_text.delete("1.0", tk.END)

    def _insert_quick_prompt(self, text):
        self.input_var.set(text)
        self.input_entry.icursor(tk.END)
        self.input_entry.focus_set()

    def _copy_last_response(self):
        if not self.last_response_text:
            self._append_system_line("No assistant response available to copy yet.")
            return
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.last_response_text)
            self._append_system_line("Copied last assistant response to clipboard.")
        except Exception as exc:
            self._append_system_line(f"Copy failed: {exc}")

    def _update_answer_meta(self, response_text):
        text = response_text or ""
        conf = "n/a"
        cites = "n/a"
        m = re.search(r"Confidence:\s*([^\n|]+)", text, flags=re.IGNORECASE)
        if m:
            conf = m.group(1).strip()
        c = re.search(r"Citations:\s*([^\n]+)", text, flags=re.IGNORECASE)
        if c:
            cites = c.group(1).strip()[:80]
        self.answer_meta_var.set(f"Last answer: confidence {conf} | citations {cites}")

    def _rate_last_response(self, sentiment):
        if not self.assistant_ready or self.assistant is None:
            self._append_system_line("Assistant is still loading.")
            return
        try:
            msg = self.assistant.apply_feedback(sentiment)
            self._append_system_line(msg)
            self._push_status_snapshot()
        except Exception as exc:
            self._append_system_line(f"Reward failed: {exc}")

    def _start_assistant_thread(self):
        def worker():
            try:
                self.queue.put(("log", "Initializing assistant model...\n"))
                assistant = JARVIS(local_only=self.local_only)
                self.assistant = assistant
                self.assistant_ready = True
                self.queue.put(("system", "JARVIS is ready."))
                self.queue.put(("status", None))
            except Exception as exc:
                self.queue.put(("error", f"Assistant startup failed: {exc}\n"))
                self.assistant_ready = False

        threading.Thread(target=worker, daemon=True).start()

    def _wake_assistant(self):
        if not self.assistant_ready or self.assistant is None:
            self._append_system_line("Assistant is still loading.")
            return
        try:
            response = self.assistant.wake_up()
            self._append_system_line(response)
            self._push_status_snapshot()
        except Exception as exc:
            self._append_system_line(f"Wake failed: {exc}")

    def _on_send(self, event=None):
        text = self.input_var.get().strip()
        if not text:
            return "break"
        if not self.assistant_ready or self.assistant is None:
            self._append_system_line("Assistant is still loading.")
            return "break"
        if self.busy:
            self._append_system_line("JARVIS is busy. Wait for the current response to finish.")
            return "break"

        self.input_var.set("")
        self._append_user_line(text)
        self.busy = True
        self.footer_var.set("JARVIS is thinking...")

        def worker():
            try:
                response = self.assistant.generate_response(text)
            except Exception as exc:
                response = f"Error generating response: {exc}"
                self.queue.put(("error", f"{response}\n"))
            self.queue.put(("assistant", response))
            self.queue.put(("status", None))
            self.queue.put(("busy", False))

        threading.Thread(target=worker, daemon=True).start()
        return "break"

    def _push_status_snapshot(self):
        if not self.assistant_ready or self.assistant is None:
            self.status_text.set("Assistant loading...")
            self.identity_text.set("Identity: loading")
            self.footer_var.set("Loading assistant...")
            return
        try:
            local_status = self.assistant.local_status()
        except Exception as exc:
            local_status = f"Local status unavailable: {exc}"
        try:
            learning_status = self.assistant.learning_status()
        except Exception as exc:
            learning_status = f"Learning status unavailable: {exc}"
        try:
            readiness = self.assistant.readiness_snapshot()
            readiness_text = (
                f"Readiness: {'READY' if readiness.get('ready') else 'LOCKED'} | "
                f"eval={readiness.get('eval_score')} | dialogue={readiness.get('dialogue_score')}"
            )
        except Exception as exc:
            readiness_text = f"Readiness unavailable: {exc}"
        try:
            summary = self.assistant.self_summary_text()
        except Exception as exc:
            summary = f"Summary unavailable: {exc}"

        self.status_text.set(f"{local_status}\n{learning_status}\n{readiness_text}")
        self.identity_text.set(f"Identity: {summary}")
        ready_label = "READY" if readiness_text.startswith("Readiness: READY") else "LOCKED"
        self.card_ready_var.set(ready_label)
        if isinstance(readiness, dict):
            eval_score = readiness.get("eval_score")
            dialogue_score = readiness.get("dialogue_score")
            self.card_eval_var.set("n/a" if eval_score is None else str(eval_score))
            self.card_dialogue_var.set("n/a" if dialogue_score is None else str(dialogue_score))
        if self.busy:
            self.footer_var.set("JARVIS is thinking...")
        else:
            self.footer_var.set("Ready for input.")

    def _refresh_status_loop(self):
        self._push_status_snapshot()
        self.root.after(1500, self._refresh_status_loop)

    def _handle_queue_item(self, item_type, payload):
        if item_type == "log":
            self._append_log_line(payload)
        elif item_type == "assistant":
            self._append_assistant_line(payload)
        elif item_type == "system":
            self._append_system_line(payload)
        elif item_type == "error":
            self._append_log_line(payload, tag="error")
        elif item_type == "status":
            self._push_status_snapshot()
        elif item_type == "busy":
            self.busy = bool(payload)
            if self.busy:
                self.footer_var.set("JARVIS is thinking...")
            else:
                self.footer_var.set("Ready for input.")

    def _poll_queue(self):
        try:
            while True:
                item_type, payload = self.queue.get_nowait()
                self._handle_queue_item(item_type, payload)
        except queue.Empty:
            pass
        self.root.after(60, self._poll_queue)

    def on_close(self):
        self.shutdown_requested = True
        try:
            if self.assistant is not None:
                self.assistant.stop_autolearn()
                self.assistant.stop_proactive_mode()
                self.assistant.stop_watch_mode()
        except Exception:
            pass
        try:
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
        except Exception:
            pass
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Open the JARVIS GUI window")
    parser.add_argument("--no-local-only", action="store_true", help="Allow web access features instead of forcing local-only mode")
    args = parser.parse_args()

    root = tk.Tk()
    app = JARVISWindowApp(root, local_only=not args.no_local_only)
    root.mainloop()


if __name__ == "__main__":
    main()
