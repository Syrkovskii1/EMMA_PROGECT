#!/usr/bin/env python3
"""Scenario editor and simple player for the robot violin script format."""

from __future__ import annotations

import argparse
import re
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import sounddevice as sd


OPEN_STRINGS = {4: 55, 3: 62, 2: 69, 1: 76}  # G3, D4, A4, E5
STRING_NAMES = {1: "Mi", 2: "La", 3: "Re", 4: "Sol"}
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
WHITE_NOTES = {0, 2, 4, 5, 7, 9, 11}
DURATIONS = (250, 500, 750, 1000, 1250, 1500, 1750, 2000)
BOW_SPEEDS = {250: 200, 500: 100, 750: 90, 1000: 80, 1250: 75, 1500: 70, 1750: 65, 2000: 60}
BOW_STOP_MS = 50
MIDI_MIN = OPEN_STRINGS[4]
MIDI_MAX = OPEN_STRINGS[1] + 7


@dataclass(frozen=True)
class PlayEvent:
    line: int
    string: int
    position: int
    direction: int
    duration_ms: int
    speed: int
    pause_before_ms: int = 0

    @property
    def midi(self) -> int:
        return OPEN_STRINGS[self.string] + self.position

    @property
    def label(self) -> str:
        return f"{midi_name(self.midi)} / {STRING_NAMES[self.string]} {self.position}"


def midi_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def midi_to_freq(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def speed_for_duration(duration_ms: int) -> int:
    closest = min(BOW_SPEEDS, key=lambda item: abs(item - duration_ms))
    return BOW_SPEEDS[closest]


def playable_positions(midi: int) -> list[tuple[int, int]]:
    choices = []
    for string, open_midi in OPEN_STRINGS.items():
        position = midi - open_midi
        if 0 <= position <= 7:
            choices.append((string, position))
    return choices


def choose_position(midi: int, previous_string: int | None) -> tuple[int, int]:
    choices = playable_positions(midi)
    if not choices:
        raise ValueError(f"{midi_name(midi)} is outside G3-B5")
    if previous_string is None:
        return min(choices, key=lambda item: (item[1], item[0]))
    return min(choices, key=lambda item: (0 if item[0] == previous_string else 10, item[1], item[0]))


def parse_last_state(text: str) -> tuple[int | None, int]:
    last_string: int | None = None
    last_direction: int | None = None
    for line in text.splitlines():
        parts = line.strip().lower().split(";")
        if len(parts) >= 3 and parts[0] == "/sn":
            try:
                last_string = int(parts[1])
            except ValueError:
                pass
        elif len(parts) >= 4 and parts[0] == "/bow":
            try:
                last_direction = int(parts[1])
            except ValueError:
                pass
    return last_string, (1 if last_direction is None else 1 - last_direction)


def parse_events(text: str) -> list[PlayEvent]:
    events: list[PlayEvent] = []
    current_note: tuple[int, int] | None = None
    pending_pause_ms = 0
    previous_direction: int | None = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip().lower()
        parts = line.split(";")
        if len(parts) >= 3 and parts[0] == "/sn":
            try:
                string, position = int(parts[1]), int(parts[2])
            except ValueError:
                current_note = None
                continue
            current_note = (string, position) if string in OPEN_STRINGS and 0 <= position <= 7 else None
        elif len(parts) >= 2 and parts[0] == "/p":
            try:
                pending_pause_ms = max(0, int(parts[1]))
            except ValueError:
                pass
        elif len(parts) >= 4 and parts[0] == "/bow" and current_note:
            try:
                direction, duration_ms, speed = int(parts[1]), int(parts[2]), int(parts[3])
            except ValueError:
                continue
            if direction in (0, 1) and duration_ms > 0 and speed > 0:
                pause_ms = pending_pause_ms or (BOW_STOP_MS if direction == previous_direction else 0)
                events.append(PlayEvent(line_number, current_note[0], current_note[1], direction, duration_ms, speed, pause_ms))
                pending_pause_ms = 0
                previous_direction = direction
    return events


def ensure_bow_stops(text: str) -> str:
    """Adds an independent 50 ms stop between neighbouring same-direction bows."""
    lines = text.splitlines()
    bows: list[tuple[int, int]] = []
    for index, raw in enumerate(lines):
        parts = raw.strip().lower().split(";")
        if len(parts) >= 4 and parts[0] == "/bow":
            try:
                direction = int(parts[1])
            except ValueError:
                continue
            if direction in (0, 1):
                bows.append((index, direction))
    insert_after = {
        previous_index
        for (previous_index, previous_direction), (current_index, direction) in zip(bows, bows[1:])
        if direction == previous_direction
        and any(line.strip().lower().startswith("/sn;") for line in lines[previous_index + 1:current_index])
        and not any(line.strip().lower().startswith("/p;") for line in lines[previous_index + 1:current_index])
    }
    if not insert_after:
        return text
    result = []
    for index, line in enumerate(lines):
        result.append(line)
        if index in insert_after:
            result.append(f"/p;{BOW_STOP_MS}")
    return "\n".join(result) + ("\n" if text.endswith("\n") else "")


def make_tone(midi: int, duration_ms: int, sample_rate: int = 44_100) -> np.ndarray:
    count = max(1, int(sample_rate * duration_ms / 1000.0))
    t = np.arange(count, dtype=np.float32) / sample_rate
    freq = midi_to_freq(midi)
    wave = (
        0.75 * np.sin(2.0 * np.pi * freq * t)
        + 0.25 * np.sin(2.0 * np.pi * freq * 2.0 * t)
        + 0.10 * np.sin(2.0 * np.pi * freq * 3.0 * t)
    )
    attack = max(1, int(sample_rate * 0.015))
    release = max(1, int(sample_rate * min(0.045, duration_ms / 4000.0)))
    envelope = np.ones(count, dtype=np.float32)
    envelope[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    envelope[-release:] = np.linspace(1.0, 0.0, release, dtype=np.float32)
    return (wave * envelope * 0.18).astype(np.float32)


def safe_filename_from_title(title: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", title).strip(" .")
    name = re.sub(r"\s+", " ", name)
    return (name[:80].strip() or "scenario") + ".txt"


class ScenarioEditorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.path: Path | None = None
        self.play_thread: threading.Thread | None = None
        self.stop_flag = threading.Event()
        self.key_items: dict[int, list[int]] = {}
        self.key_rects: dict[int, int] = {}
        self.key_base_fill: dict[int, str] = {}
        self.duration_buttons: dict[int, ttk.Button] = {}

        self.root.title("Violin Scenario")
        self.root.minsize(980, 640)
        self.root.configure(bg="#f6f7f9")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.duration_var = tk.IntVar(value=500)
        self.line_var = tk.StringVar(value="Line 1")
        self.status_var = tk.StringVar(value="Ready")

        self._build_styles()
        self._build_ui()
        self._bind_shortcuts()
        self._insert_starter()

    def _build_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Root.TFrame", background="#f6f7f9")
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("Toolbar.TFrame", background="#f6f7f9")
        style.configure("Title.TLabel", background="#f6f7f9", foreground="#121417", font=("Segoe UI", 18, "bold"))
        style.configure("Label.TLabel", background="#ffffff", foreground="#3b4148", font=("Segoe UI", 10))
        style.configure("Hint.TLabel", background="#ffffff", foreground="#66707a", font=("Segoe UI", 9))
        style.configure("Status.TLabel", background="#f6f7f9", foreground="#4d5660", font=("Segoe UI", 10))
        style.configure("Primary.TButton", font=("Segoe UI", 10), padding=(14, 8), background="#1c4f63", foreground="#ffffff")
        style.configure("Secondary.TButton", font=("Segoe UI", 10), padding=(12, 8), background="#e8edf1", foreground="#121417")
        style.configure("Duration.TButton", font=("Segoe UI", 9), padding=(3, 7), background="#eef2f5", foreground="#26313a")
        style.configure("Selected.Duration.TButton", font=("Segoe UI", 9, "bold"), padding=(3, 7), background="#1c4f63", foreground="#ffffff")
        style.map("Primary.TButton", background=[("active", "#173f50"), ("disabled", "#aab7be")])
        style.map("Secondary.TButton", background=[("active", "#d9e1e7"), ("disabled", "#eef1f4")])
        style.map("Duration.TButton", background=[("active", "#d9e4ea")])
        style.map("Selected.Duration.TButton", background=[("active", "#173f50")], foreground=[("active", "#ffffff")])
        style.configure("TNotebook", background="#ffffff", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 10))

    def _build_ui(self) -> None:
        root_frame = ttk.Frame(self.root, style="Root.TFrame", padding=(18, 16, 18, 12))
        root_frame.pack(fill="both", expand=True)
        root_frame.columnconfigure(0, weight=1)
        root_frame.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(root_frame, style="Toolbar.TFrame")
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        toolbar.columnconfigure(1, weight=1)
        ttk.Label(toolbar, text="Violin Scenario", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        buttons = ttk.Frame(toolbar, style="Toolbar.TFrame")
        buttons.grid(row=0, column=2, sticky="e")
        ttk.Button(buttons, text="New", command=self.new_file, style="Secondary.TButton").pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Open", command=self.open_file, style="Secondary.TButton").pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Save", command=self.save_file, style="Secondary.TButton").pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Mark line", command=self.toggle_line_mark, style="Secondary.TButton").pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(buttons, text="Play all", command=self.play_all, style="Primary.TButton").pack(side="left", padx=(8, 8))
        ttk.Button(buttons, text="Play from cursor", command=self.play_from_cursor, style="Secondary.TButton").pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(buttons, text="Stop", command=self.stop, style="Secondary.TButton").pack(side="left")

        main = ttk.PanedWindow(root_frame, orient="horizontal")
        main.grid(row=1, column=0, sticky="nsew")

        editor_panel = ttk.Frame(main, style="Panel.TFrame", padding=(12, 12, 12, 12))
        editor_panel.rowconfigure(1, weight=1)
        editor_panel.columnconfigure(0, weight=1)
        ttk.Label(editor_panel, text="Editor", style="Label.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        text_frame = ttk.Frame(editor_panel, style="Panel.TFrame")
        text_frame.grid(row=1, column=0, sticky="nsew")
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        self.editor = tk.Text(
            text_frame,
            wrap="none",
            undo=True,
            bg="#fbfcfd",
            fg="#121417",
            insertbackground="#1c4f63",
            selectbackground="#c9dce5",
            font=("Cascadia Mono", 11),
            relief="flat",
            padx=12,
            pady=12,
        )
        self.editor.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.editor.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.editor.configure(yscrollcommand=yscroll.set)
        self.editor.tag_configure("playing", background="#d7edf4", foreground="#0b2c38")
        self.editor.tag_configure("marked", background="#fff0a8")
        self.editor.tag_configure("error", background="#f6d7d2")

        composer = ttk.Frame(main, style="Panel.TFrame", padding=(14, 12, 14, 12))
        composer.columnconfigure(0, weight=1)
        composer.rowconfigure(1, weight=1)
        ttk.Label(composer, text="Auto write", style="Label.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        tabs = ttk.Notebook(composer)
        tabs.grid(row=1, column=0, sticky="nsew")
        keyboard_tab = ttk.Frame(tabs, style="Panel.TFrame", padding=(0, 10, 0, 0))
        keyboard_tab.columnconfigure(0, weight=1)
        keyboard_tab.rowconfigure(1, weight=1)
        tabs.add(keyboard_tab, text="Keyboard")

        controls = ttk.Frame(keyboard_tab, style="Panel.TFrame")
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(controls, text="Length", style="Label.TLabel").pack(side="left", padx=(0, 8))
        for duration_ms in DURATIONS:
            button = ttk.Button(
                controls,
                text=str(duration_ms),
                command=lambda item=duration_ms: self.set_duration(item),
                style="Duration.TButton",
                width=4,
            )
            button.pack(side="left", padx=(0, 3))
            self.duration_buttons[duration_ms] = button
        ttk.Label(controls, text="ms", style="Hint.TLabel").pack(side="left", padx=(2, 0))
        self.set_duration(self.duration_var.get())

        self.keyboard = tk.Canvas(keyboard_tab, height=180, bg="#ffffff", highlightthickness=0)
        self.keyboard.grid(row=1, column=0, sticky="nsew")
        self.keyboard.bind("<Configure>", lambda _event: self.draw_keyboard())

        legend = ttk.Frame(keyboard_tab, style="Panel.TFrame")
        legend.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.choice_var = tk.StringVar(value="Click a key to insert /sn and /bow")
        ttk.Label(legend, textvariable=self.choice_var, style="Hint.TLabel").pack(anchor="w")

        helper = ttk.Frame(composer, style="Panel.TFrame")
        helper.grid(row=2, column=0, sticky="ew", pady=(16, 0))
        ttk.Button(helper, text="Insert header", command=self.insert_header, style="Secondary.TButton").pack(side="left")
        ttk.Button(helper, text="Append finish", command=self.ensure_footer, style="Secondary.TButton").pack(
            side="left", padx=(8, 0)
        )

        main.add(editor_panel, weight=3)
        main.add(composer, weight=2)

        footer = ttk.Frame(root_frame, style="Toolbar.TFrame")
        footer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(footer, textvariable=self.line_var, style="Status.TLabel").pack(side="left")
        ttk.Label(footer, textvariable=self.status_var, style="Status.TLabel").pack(side="left", padx=(18, 0))

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-s>", lambda _event: self.save_file())
        self.root.bind("<Control-o>", lambda _event: self.open_file())
        self.editor.bind("<ButtonRelease-1>", self.update_cursor_line, add="+")
        self.editor.bind("<KeyRelease>", self.update_cursor_line, add="+")
        self.editor.bind("<KeyPress>", self.handle_global_key, add="+")
        self.root.bind_all("<KeyPress>", self.handle_global_key, add="+")
        for index, duration_ms in enumerate(DURATIONS, start=1):
            self.root.bind_all(f"<Key-{index}>", lambda event, item=duration_ms: self.duration_shortcut(event, item))
            self.root.bind_all(f"<KP_{index}>", lambda event, item=duration_ms: self.duration_shortcut(event, item))
        self.root.bind_all("<Button-1>", self.clear_editor_focus, add="+")
        self.root.bind("<F5>", lambda _event: self.play_all())
        self.root.bind("<Escape>", lambda _event: self.stop())

    def update_cursor_line(self, _event: tk.Event | None = None) -> None:
        self.line_var.set(f"Line {self.editor.index('insert').split('.')[0]}")

    def _insert_starter(self) -> None:
        self.editor.insert("1.0", "New scenario\n/load_test\n/end\n/play\n")
        self.editor.mark_set("insert", "3.0")
        self.editor.edit_reset()
        self.editor.edit_modified(False)

    def draw_keyboard(self) -> None:
        self.keyboard.delete("all")
        self.key_items.clear()
        self.key_rects.clear()
        self.key_base_fill.clear()
        width = max(1, self.keyboard.winfo_width())
        height = max(1, self.keyboard.winfo_height())
        white_midis = [midi for midi in range(MIDI_MIN, MIDI_MAX + 1) if midi % 12 in WHITE_NOTES]
        white_w = width / len(white_midis)
        white_h = height - 4
        white_x: dict[int, float] = {}

        for idx, midi in enumerate(white_midis):
            x0 = idx * white_w
            x1 = x0 + white_w
            white_x[midi] = x0
            self._make_key(midi, x0, 0, x1, white_h, "#fbfcfd", "#ccd4dc", text=midi_name(midi))

        for midi in range(MIDI_MIN, MIDI_MAX + 1):
            if midi % 12 in WHITE_NOTES:
                continue
            previous_white = midi - 1
            while previous_white >= MIDI_MIN and previous_white % 12 not in WHITE_NOTES:
                previous_white -= 1
            if previous_white not in white_x:
                continue
            black_w = white_w * 0.62
            x0 = white_x[previous_white] + white_w - black_w / 2
            self._make_key(midi, x0, 0, x0 + black_w, white_h * 0.62, "#121417", "#121417", text="")

    def _make_key(self, midi: int, x0: float, y0: float, x1: float, y1: float, fill: str, outline: str, text: str) -> None:
        tag = f"key_{midi}"
        rect = self.keyboard.create_rectangle(x0, y0, x1, y1, fill=fill, outline=outline, tags=(tag,))
        self.key_items.setdefault(midi, []).append(rect)
        self.key_rects[midi] = rect
        self.key_base_fill[midi] = fill
        if text:
            label = self.keyboard.create_text(
                (x0 + x1) / 2,
                y1 - 18,
                text=text,
                fill="#4d5660",
                font=("Segoe UI", 8),
                tags=(tag,),
            )
            self.key_items[midi].append(label)
        self.keyboard.tag_bind(tag, "<Button-1>", lambda _event, item=midi: self.add_note(item))

    def set_duration(self, duration_ms: int) -> None:
        self.duration_var.set(duration_ms)
        for value, button in self.duration_buttons.items():
            button.configure(style="Selected.Duration.TButton" if value == duration_ms else "Duration.TButton")

    def clear_editor_focus(self, event: tk.Event) -> None:
        if event.widget != self.editor:
            self.root.focus_set()

    def duration_shortcut(self, event: tk.Event, duration_ms: int) -> str | None:
        if event.widget == self.editor:
            return None
        self.set_duration(duration_ms)
        self.status_var.set(f"Length {duration_ms} ms")
        return "break"

    def handle_global_key(self, event: tk.Event) -> str | None:
        control_pressed = bool(event.state & 0x4)
        z_pressed = event.keycode == 90 or event.keysym.lower() == "z" or event.keysym in {
            "Cyrillic_ya",
            "Cyrillic_YA",
        }
        if control_pressed and z_pressed:
            return self.undo(event)
        return None

    def animate_key(self, midi: int) -> None:
        rect = self.key_rects.get(midi)
        if rect is None:
            return
        base = self.key_base_fill.get(midi, "#fbfcfd")
        flash = "#d7edf4" if base != "#121417" else "#2c7892"
        try:
            self.keyboard.itemconfigure(rect, fill=flash)
            self.keyboard.scale(f"key_{midi}", *self._key_center(rect), 0.97, 0.97)
        except tk.TclError:
            return
        self.root.after(90, lambda: self._restore_key(midi, base))

    def _key_center(self, rect: int) -> tuple[float, float]:
        x0, y0, x1, y1 = self.keyboard.coords(rect)
        return (x0 + x1) / 2, (y0 + y1) / 2

    def _restore_key(self, midi: int, fill: str) -> None:
        rect = self.key_rects.get(midi)
        if rect is None:
            return
        try:
            self.keyboard.scale(f"key_{midi}", *self._key_center(rect), 1 / 0.97, 1 / 0.97)
            self.keyboard.itemconfigure(rect, fill=fill)
        except tk.TclError:
            pass

    def add_note(self, midi: int) -> None:
        self.animate_key(midi)
        duration = int(self.duration_var.get())
        if self.editor.compare("insert", ">=", self.footer_start_index()):
            self.editor.mark_set("insert", self.footer_start_index())
        previous_string, next_direction = parse_last_state(self.editor.get("1.0", "insert"))
        string, position = choose_position(midi, previous_string)
        speed = speed_for_duration(duration)
        block = f"/sn;{string};{position}\n/bow;{next_direction};{duration};{speed}\n"
        if self.editor.compare("insert", "!=", "1.0"):
            previous = self.editor.get("insert -1c", "insert")
            if previous and previous != "\n":
                block = "\n" + block
        self.editor.edit_separator()
        self.editor.insert("insert", block)
        self.editor.edit_separator()
        self.editor.see("insert")
        self.choice_var.set(
            f"{midi_name(midi)} -> /sn;{string};{position}, /bow;{next_direction};{duration};{speed}"
        )
        self.status_var.set("Inserted note")

    def undo(self, _event: tk.Event | None = None) -> str:
        try:
            self.editor.edit_undo()
            self.status_var.set("Undo")
        except tk.TclError:
            self.status_var.set("Nothing to undo")
        return "break"

    def insert_header(self) -> None:
        self.editor.insert("insert", "Scenario\n/load_test\n")

    def footer_start_index(self) -> str:
        lines = self.editor.get("1.0", "end-1c").splitlines()
        stripped = [line.strip().lower() for line in lines]
        if len(stripped) >= 2 and stripped[-2:] == ["/end", "/play"]:
            return f"{len(lines) - 1}.0"
        return "end-1c"

    def ensure_footer(self) -> None:
        text = self.editor.get("1.0", "end-1c")
        lines = text.splitlines()
        stripped = [line.strip().lower() for line in lines]
        if len(stripped) >= 2 and stripped[-2:] == ["/end", "/play"]:
            self.status_var.set("Finish already exists")
            return
        while lines and lines[-1].strip().lower() in {"/end", "/play"}:
            lines.pop()
        body = "\n".join(lines).rstrip()
        final_text = f"{body}\n/end\n/play\n" if body else "/end\n/play\n"
        self.editor.edit_separator()
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", final_text)
        self.editor.edit_separator()
        self.status_var.set("Finish added")

    def ensure_bow_stops(self) -> None:
        current = self.editor.get("1.0", "end-1c")
        updated = ensure_bow_stops(current)
        if updated == current:
            return
        self.editor.edit_separator()
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", updated)
        self.editor.edit_separator()
        self.status_var.set(f"Added {BOW_STOP_MS} ms bow stop")

    def toggle_line_mark(self) -> None:
        line = int(self.editor.index("insert").split(".")[0])
        start = f"{line}.0"
        end = f"{line}.end"
        if "marked" in self.editor.tag_names(start):
            self.editor.tag_remove("marked", start, end)
            self.status_var.set(f"Line {line} unmarked")
        else:
            self.editor.tag_add("marked", start, end)
            self.status_var.set(f"Line {line} marked")

    def new_file(self) -> None:
        if not self._confirm_discard():
            return
        self.path = None
        self.editor.delete("1.0", "end")
        self._insert_starter()
        self.status_var.set("New scenario")

    def open_file(self) -> None:
        if not self._confirm_discard():
            return
        path = filedialog.askopenfilename(
            title="Open scenario",
            filetypes=(("Scenario files", "*.txt *.scenario *.scr"), ("All files", "*.*")),
        )
        if not path:
            return
        self.path = Path(path)
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", self.path.read_text(encoding="utf-8"))
        self.editor.edit_reset()
        self.editor.edit_modified(False)
        self.status_var.set(f"Opened {self.path.name}")

    def save_file(self) -> None:
        self.ensure_footer()
        self.ensure_bow_stops()
        if self.path is None:
            default_name = safe_filename_from_title(self.editor.get("1.0", "1.end"))
            path = filedialog.asksaveasfilename(
                title="Save scenario",
                defaultextension=".txt",
                initialfile=default_name,
                filetypes=(("Scenario files", "*.txt"), ("All files", "*.*")),
            )
            if not path:
                return
            self.path = Path(path)
        self.path.write_text(self.editor.get("1.0", "end-1c"), encoding="utf-8")
        self.editor.edit_modified(False)
        self.status_var.set(f"Saved {self.path.name}")

    def play_all(self) -> None:
        self.play_from_line(1)

    def play_from_cursor(self) -> None:
        self.play_from_line(int(self.editor.index("insert").split(".")[0]))

    def play_from_line(self, start_line: int) -> None:
        if self.play_thread and self.play_thread.is_alive():
            return
        self.ensure_bow_stops()
        events = [event for event in parse_events(self.editor.get("1.0", "end-1c")) if event.line >= start_line]
        if not events:
            messagebox.showinfo("Nothing to play", "No valid /sn + /bow pairs found.")
            return
        self.stop_flag.clear()
        self.play_thread = threading.Thread(target=self._play_events, args=(events,), daemon=True)
        self.play_thread.start()

    def _play_events(self, events: list[PlayEvent]) -> None:
        self.root.after(0, lambda: self.status_var.set(f"Playing {len(events)} notes"))
        try:
            for event in events:
                if self.stop_flag.is_set():
                    break
                pause_end = time.monotonic() + event.pause_before_ms / 1000.0
                while time.monotonic() < pause_end:
                    if self.stop_flag.is_set():
                        break
                    time.sleep(0.005)
                if self.stop_flag.is_set():
                    break
                self.root.after(0, lambda item=event: self._highlight_event(item))
                sd.play(make_tone(event.midi, event.duration_ms), 44_100, blocking=False)
                end_at = time.monotonic() + event.duration_ms / 1000.0
                while time.monotonic() < end_at:
                    if self.stop_flag.is_set():
                        sd.stop()
                        break
                    time.sleep(0.015)
            sd.stop()
        except Exception as exc:
            self.root.after(0, lambda error=exc: messagebox.showerror("Playback error", str(error)))
        finally:
            self.root.after(0, self._clear_playback)

    def _highlight_event(self, event: PlayEvent) -> None:
        self.editor.tag_remove("playing", "1.0", "end")
        start = f"{event.line}.0"
        end = f"{event.line}.end"
        self.editor.tag_add("playing", start, end)
        self.editor.see(start)
        self.status_var.set(f"Playing {event.label} for {event.duration_ms} ms")

    def stop(self) -> None:
        self.stop_flag.set()
        sd.stop()
        self._clear_playback()

    def _clear_playback(self) -> None:
        self.editor.tag_remove("playing", "1.0", "end")
        if self.stop_flag.is_set():
            self.status_var.set("Stopped")
        else:
            self.status_var.set("Ready")

    def _confirm_discard(self) -> bool:
        if not self.editor.edit_modified():
            return True
        return messagebox.askyesno("Discard changes?", "Current scenario has unsaved changes.")

    def close(self) -> None:
        if not self._confirm_discard():
            return
        self.stop()
        self.root.destroy()


def run_self_test() -> int:
    assert choose_position(76, None) == (1, 0)
    assert choose_position(76, 2) == (2, 7)
    assert speed_for_duration(500) == 100
    text = "/sn;2;0\n/bow;1;500;100\n/sn;2;4\n/bow;0;250;200\n"
    events = parse_events(text)
    assert len(events) == 2
    assert events[0].midi == 69 and events[1].direction == 0
    assert parse_last_state(text) == (2, 1)
    stopped = ensure_bow_stops("/sn;2;0\n/bow;1;750;67\n/sn;2;4\n/bow;1;250;200\n")
    assert "/p;50\n/sn;2;4" in stopped
    assert parse_events(stopped)[1].pause_before_ms == 50
    print("OK")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robot violin scenario editor.")
    parser.add_argument("--self-test", action="store_true", help="run lightweight parser tests and exit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        return run_self_test()
    root = tk.Tk()
    ScenarioEditorApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
