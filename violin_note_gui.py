#!/usr/bin/env python3
"""Minimal GUI for the realtime violin note detector."""

from __future__ import annotations

import argparse
import queue
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import sounddevice as sd

from violin_note_detector import ViolinPitchDetector


class NoteDetectorApp:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.detector = ViolinPitchDetector(
            sample_rate=args.sample_rate,
            fft_size=args.fft_size,
            min_freq=args.min_freq,
            max_freq=args.max_freq,
            threshold_db=args.threshold_db,
            prominence_db=args.prominence_db,
            min_harmonic_matches=args.min_harmonic_matches,
            harmonics=args.harmonics,
            a4=args.a4,
            smoothing=args.smoothing,
        )

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=16)
        self.status_queue: queue.Queue[str] = queue.Queue(maxsize=4)
        self.ring = np.zeros(args.fft_size, dtype=np.float32)
        self.stream: sd.InputStream | None = None
        self.running = False
        self.silent_ticks = 0

        self.root.title("Violin Note")
        self.root.minsize(520, 360)
        self.root.configure(bg="#ffffff")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self._build_styles()
        self._build_ui()
        self.start()
        self.root.after(50, self.poll_audio)

    def _build_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Root.TFrame", background="#ffffff")
        style.configure("Status.TLabel", background="#ffffff", foreground="#5f6267", font=("Segoe UI", 11))
        style.configure("Note.TLabel", background="#ffffff", foreground="#15171a", font=("Segoe UI", 96, "bold"))
        style.configure("Hz.TLabel", background="#ffffff", foreground="#272a2f", font=("Segoe UI", 30))
        style.configure("Hint.TLabel", background="#ffffff", foreground="#777b82", font=("Segoe UI", 10))
        style.configure(
            "Primary.TButton",
            font=("Segoe UI", 11),
            padding=(18, 10),
            background="#15171a",
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=2,
            focuscolor="#b56b00",
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#2b2f35"), ("disabled", "#b9bcc2")],
            foreground=[("disabled", "#ffffff")],
        )
        style.configure(
            "Secondary.TButton",
            font=("Segoe UI", 11),
            padding=(18, 10),
            background="#eef0f3",
            foreground="#15171a",
            borderwidth=0,
            focusthickness=2,
            focuscolor="#b56b00",
        )
        style.map("Secondary.TButton", background=[("active", "#e0e3e8"), ("disabled", "#f4f5f7")])

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, style="Root.TFrame", padding=(28, 22, 28, 22))
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        top = ttk.Frame(frame, style="Root.TFrame")
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Listening")
        self.status_label = ttk.Label(top, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.grid(row=0, column=0, sticky="w")

        self.activity = tk.Canvas(top, width=14, height=14, highlightthickness=0, bg="#ffffff")
        self.activity.grid(row=0, column=1, sticky="e")
        self.activity_dot = self.activity.create_oval(3, 3, 11, 11, fill="#b56b00", outline="")

        center = ttk.Frame(frame, style="Root.TFrame")
        center.grid(row=1, column=0, sticky="nsew")
        center.columnconfigure(0, weight=1)
        center.rowconfigure(0, weight=1)
        center.rowconfigure(1, weight=0)

        self.note_var = tk.StringVar(value="--")
        self.hz_var = tk.StringVar(value="-- Hz")
        ttk.Label(center, textvariable=self.note_var, anchor="center", style="Note.TLabel").grid(
            row=0, column=0, sticky="s"
        )
        ttk.Label(center, textvariable=self.hz_var, anchor="center", style="Hz.TLabel").grid(
            row=1, column=0, sticky="n", pady=(0, 20)
        )

        bottom = ttk.Frame(frame, style="Root.TFrame")
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)

        self.detail_var = tk.StringVar(value="Default input")
        ttk.Label(bottom, textvariable=self.detail_var, style="Hint.TLabel").grid(row=0, column=0, sticky="w")

        self.start_button = ttk.Button(bottom, text="Start", command=self.start, style="Primary.TButton")
        self.start_button.grid(row=0, column=1, padx=(12, 8), sticky="e")
        self.stop_button = ttk.Button(bottom, text="Stop", command=self.stop, style="Secondary.TButton")
        self.stop_button.grid(row=0, column=2, sticky="e")

    def audio_callback(self, indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags) -> None:
        if status:
            try:
                self.status_queue.put_nowait(str(status))
            except queue.Full:
                pass
        try:
            self.audio_queue.put_nowait(indata.copy())
        except queue.Full:
            pass

    def start(self) -> None:
        if self.running:
            return
        try:
            self.stream = sd.InputStream(
                device=self.args.device,
                channels=1,
                samplerate=self.args.sample_rate,
                blocksize=self.args.block_size,
                callback=self.audio_callback,
            )
            self.stream.start()
        except Exception as exc:
            self.status_var.set("Input error")
            messagebox.showerror("Audio input error", str(exc))
            return

        self.running = True
        self.status_var.set("Listening")
        self.detail_var.set("Default input" if self.args.device is None else f"Input: {self.args.device}")
        self.start_button.state(["disabled"])
        self.stop_button.state(["!disabled"])
        self.activity.itemconfigure(self.activity_dot, fill="#b56b00")

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.running = False
        self.status_var.set("Stopped")
        self.note_var.set("--")
        self.hz_var.set("-- Hz")
        self.start_button.state(["!disabled"])
        self.stop_button.state(["disabled"])
        self.activity.itemconfigure(self.activity_dot, fill="#c8ccd2")

    def poll_audio(self) -> None:
        processed = False
        try:
            input_status = self.status_queue.get_nowait()
        except queue.Empty:
            input_status = ""

        while True:
            try:
                block = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            mono = block[:, 0].astype(np.float32, copy=False)
            if mono.size >= self.ring.size:
                self.ring[:] = mono[-self.ring.size :]
            else:
                self.ring[:-mono.size] = self.ring[mono.size :]
                self.ring[-mono.size :] = mono
            processed = True

        if self.running and processed:
            result = self.detector.detect(self.ring)
            if result is None:
                self.silent_ticks += 1
                if self.silent_ticks >= 4:
                    self.note_var.set("--")
                    self.hz_var.set("-- Hz")
                    self.status_var.set(input_status or "No stable note")
                    self.activity.itemconfigure(self.activity_dot, fill="#c8ccd2")
            else:
                self.silent_ticks = 0
                self.note_var.set(f"{result.note}{result.octave}")
                self.hz_var.set(f"{result.frequency:.2f} Hz")
                self.status_var.set(input_status or "Listening")
                self.activity.itemconfigure(self.activity_dot, fill="#b56b00")

        self.root.after(50, self.poll_audio)

    def close(self) -> None:
        self.stop()
        self.root.destroy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal GUI for realtime violin note detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default=None, help="input device id or name")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--fft-size", type=int, default=16_384)
    parser.add_argument("--min-freq", type=float, default=150.0)
    parser.add_argument("--max-freq", type=float, default=3_200.0)
    parser.add_argument("--threshold-db", type=float, default=18.0)
    parser.add_argument("--prominence-db", type=float, default=7.0)
    parser.add_argument("--min-harmonic-matches", type=int, default=2)
    parser.add_argument("--harmonics", type=int, default=8)
    parser.add_argument("--a4", type=float, default=440.0)
    parser.add_argument("--smoothing", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = tk.Tk()
    NoteDetectorApp(root, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
