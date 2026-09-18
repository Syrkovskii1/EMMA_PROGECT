#!/usr/bin/env python3
"""Unified Scenario Recorder and Editor for the robot violin (with Voice AI, Standalone Recognition and TCP Receiver)."""

from __future__ import annotations

import os
# Токен Hugging Face для быстрой загрузки моделей
os.environ["HF_TOKEN"] = "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"

import argparse
import re
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from dataclasses import dataclass
from pathlib import Path
import queue
import math
import sys
import serial
import socket
import config

import numpy as np
import sounddevice as sd
from scipy.signal import find_peaks

# --- ИМПОРТЫ ДЛЯ ГОЛОСОВОГО ИИ ---
from RealtimeSTT import AudioToTextRecorder
from sentence_transformers import SentenceTransformer, util
import torch

# --- CONSTANTS & HELPERS FROM ORIGINAL SCENARIO EDITOR ---
OPEN_STRINGS = {4: 55, 3: 62, 2: 69, 1: 76}  # G3, D4, A4, E5
STRING_NAMES = {1: "Mi", 2: "La", 3: "Re", 4: "Sol"}
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
WHITE_NOTES = {0, 2, 4, 5, 7, 9, 11}
DURATIONS = (250, 500, 750, 1000, 1250, 1500, 1750, 2000)
BOW_SPEEDS = {250: 200, 500: 100, 750: 90, 1000: 80, 1250: 75, 1500: 70, 1750: 65, 2000: 60}
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
        # Если нота выше или ниже диапазона, пытаемся транспонировать
        for alt_midi in (midi - 12, midi + 12):
            choices = playable_positions(alt_midi)
            if choices:
                break
    if not choices:
        raise ValueError(f"{midi_name(midi)} вне диапазона скрипки")
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
        elif len(parts) >= 4 and parts[0] == "/bow" and current_note:
            try:
                direction, duration_ms, speed = int(parts[1]), int(parts[2]), int(parts[3])
            except ValueError:
                continue
            if direction in (0, 1) and duration_ms > 0 and speed > 0:
                events.append(PlayEvent(line_number, current_note[0], current_note[1], direction, duration_ms, speed))
    return events

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

def optimize_bow_directions(notes: list[tuple[int, int]], X_start: float, initial_dir: int) -> list[int]:
    """Планирует оптимальное распределение смычка (0 или 1) с помощью динамического программирования."""
    n = len(notes)
    if n == 0:
        return []
        
    disps = []
    for midi, duration in notes:
        speed = speed_for_duration(duration)
        disps.append(duration * speed)
        
    memo_strict = {}
    def search_strict(idx, X, stroke_dir, stroke_disp):
        if idx == n: return 0, []
        state = (idx, int(round(X)), stroke_dir, int(round(stroke_disp)))
        if state in memo_strict: return memo_strict[state]
            
        d = disps[idx]
        duration = notes[idx][1]
        best_cost = float('inf')
        best_path = []
        
        for dir_choice in (0, 1):
            is_same = (dir_choice == stroke_dir)
            if dir_choice == 1: X_new = X + d
            else: X_new = X - d
                
            if X_new < -0.1 or X_new > 100000.1: continue
                
            if is_same: new_disp = stroke_disp + d
            else: new_disp = d
                
            if is_same and new_disp > 100000.1: continue
            if is_same and duration >= 1000: continue
                
            cost = 0
            if not is_same:
                if stroke_disp < 75000 and duration < 1000: cost += 10
            else:
                if duration >= 750: cost += 5
                    
            sub_cost, sub_path = search_strict(idx + 1, X_new, dir_choice, new_disp)
            total_cost = cost + sub_cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = [dir_choice] + sub_path
                
        memo_strict[state] = (best_cost, best_path)
        return best_cost, best_path

    cost, path = search_strict(0, X_start, 1 - initial_dir, 100000)
    if cost != float('inf'):
        return path
        
    memo_soft = {}
    def search_soft(idx, X, stroke_dir, stroke_disp):
        if idx == n: return 0, []
        X_q = int(round(X / 5000)) * 5000
        disp_q = int(round(stroke_disp / 5000)) * 5000
        state = (idx, X_q, stroke_dir, disp_q)
        
        if state in memo_soft: return memo_soft[state]
            
        d = disps[idx]
        duration = notes[idx][1]
        best_cost = float('inf')
        best_path = []
        
        for dir_choice in (0, 1):
            is_same = (dir_choice == stroke_dir)
            if dir_choice == 1: X_new = X + d
            else: X_new = X - d
                
            if is_same: new_disp = stroke_disp + d
            else: new_disp = d
                
            cost = 0
            if not is_same:
                if stroke_disp < 75000 and duration < 1000: cost += 10000
            else:
                if duration >= 750: cost += 5000
                if duration >= 1000: cost += 50000
                    
            overrun = max(0.0, -X_new) + max(0.0, X_new - 100000)
            cost += overrun * 2.0
            
            stroke_overrun = max(0.0, new_disp - 100000)
            cost += stroke_overrun * 2.0
            
            sub_cost, sub_path = search_soft(idx + 1, X_new, dir_choice, new_disp)
            total_cost = cost + sub_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = [dir_choice] + sub_path
                
        memo_soft[state] = (best_cost, best_path)
        return best_cost, best_path

    cost, path = search_soft(0, X_start, 1 - initial_dir, 100000)
    return path

from mpm_pitch_detector import (
    PitchResult,
    MpmPitchDetector,
    process_detections_to_notes,
    estimate_human_tempo,
    quantize_duration,
    scale_and_quantize,
    fit_to_violin_range,
)

# --- ROBOT CONTROLLER & AUTOCALIBRATION LOGIC ---
ROBOT_COM_PORT = "COM11"
ROBOT_BAUD = 115200

TARGET_FREQS = {
    s: {pos: 440.0 * (2.0 ** ((OPEN_STRINGS[s] + pos - 69) / 12.0)) for pos in range(1, 8)}
    for s in OPEN_STRINGS
}

class RobotController:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None

    def connect(self):
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            return True, f"Успешное подключение на {self.port}"
        except Exception as e:
            self.serial_conn = None
            return False, f"Ошибка подключения: {e}"

    def send_cmd(self, cmd):
        if self.serial_conn and self.serial_conn.is_open:
            full_cmd = f"{cmd}\n"
            self.serial_conn.write(full_cmd.encode('utf-8'))
            return f"[PC -> EMMA] {cmd}"
        return "[PC] Робот не подключен!"

class ViolinRecorderEditorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.path: Path | None = None
        self.play_thread: threading.Thread | None = None
        self.stop_flag = threading.Event()
        
        # Audio stream state
        self.detector: MpmPitchDetector | None = None
        self.stream: sd.InputStream | None = None
        self.running = False
        self.silent_ticks = 0
        self.ring = np.zeros(16384, dtype=np.float32)
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=16)
        self.status_queue: queue.Queue[str] = queue.Queue(maxsize=4)
        
        # Recording state (Top Panel standard recording)
        self.recording = False
        self.raw_detections: list[tuple[float, int | None]] = []
        self.recorded_notes: list[tuple[int, float]] = []
        self.recording_start_time = 0.0
        self.min_confidence = 0.45
        
        # Editor keyboard variables
        self.key_items: dict[int, list[int]] = {}
        self.key_rects: dict[int, int] = {}
        self.key_base_fill: dict[int, str] = {}
        self.duration_buttons: dict[int, ttk.Button] = {}
        
        # Network Receiver state
        self.net_server_sock = None
        self.net_server_running = False
        
        self.root.title("Violin Scenario Recorder & Editor")
        self.root.minsize(1100, 800)
        self.root.configure(bg="#f6f7f9")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        
        # Tkinter variables
        self.duration_var = tk.IntVar(value=500)
        self.line_var = tk.StringVar(value="Line 1")
        self.status_var = tk.StringVar(value="Ready")
        
        # Robot & Calib variables
        self.robot = RobotController(ROBOT_COM_PORT, ROBOT_BAUD)
        self.auto_calib_active = False
        self.ac_state = "IDLE"
        self.ac_string = 2
        self.ac_note = 1
        self.ac_wait_time = 0
        self.ac_listen_timeout = 0
        self.ac_bow_dir = 1
        self.ac_freq_buffer = []
        self.current_mm = [[0.0]*7 for _ in range(4)]
        self.robot_port_var = tk.StringVar(value=ROBOT_COM_PORT)

        # Tuner parameters variables
        self.device_var = tk.StringVar()
        self.buffer_size_var = tk.IntVar(value=8192)
        self.block_size_var = tk.IntVar(value=512)
        self.clarity_var = tk.DoubleVar(value=0.45)
        self.cutoff_k_var = tk.DoubleVar(value=0.85)
        self.a4_var = tk.DoubleVar(value=443.0)
        self.min_freq_var = tk.DoubleVar(value=150.0)
        self.max_freq_var = tk.DoubleVar(value=3200.0)

        # Tuner stability variable (used in poll_audio)
        self.confirm_frames_var = tk.IntVar(value=3)

        # Tempo variables
        self.robot_tempo_var = tk.IntVar(value=500)
        self.human_tempo_var = tk.IntVar(value=600)
        
        # Live display variables
        self.live_note_var = tk.StringVar(value="--")
        self.live_hz_var = tk.StringVar(value="-- Hz")
        self.live_conf_var = tk.StringVar(value="Conf: 0.00")
        
        # Build interface
        self._build_styles()
        self._build_ui()
        self._bind_shortcuts()
        self._insert_starter()
        
        # Initialize default audio settings and start listening
        try:
            self.apply_audio_settings()
            self.start_detector()
        except Exception as startup_err:
            print(f"Startup audio warning: {startup_err}")
            self.status_var.set("Аудио не запущено. Выберите устройство вручную.")
        
        # Start GUI poll loop
        self.root.after(50, self.poll_audio)
        self.root.after(100, self.auto_calib_tick)
        self.serial_thread = threading.Thread(target=self.serial_reader_thread, daemon=True)
        self.serial_thread.start()
        
        # Connect to EMMA robot automatically on startup
        self.root.after(500, self.connect_robot)

    def _build_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Root.TFrame", background="#f6f7f9")
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("Toolbar.TFrame", background="#f6f7f9")
        style.configure("Title.TLabel", background="#f6f7f9", foreground="#121417", font=("Segoe UI", 16, "bold"))
        style.configure("Label.TLabel", background="#ffffff", foreground="#3b4148", font=("Segoe UI", 10))
        style.configure("Hint.TLabel", background="#ffffff", foreground="#66707a", font=("Segoe UI", 9))
        style.configure("Status.TLabel", background="#f6f7f9", foreground="#4d5660", font=("Segoe UI", 10))
        style.configure("Primary.TButton", font=("Segoe UI", 10), padding=(14, 8), background="#1c4f63", foreground="#ffffff")
        style.configure("Secondary.TButton", font=("Segoe UI", 10), padding=(12, 8), background="#e8edf1", foreground="#121417")
        style.configure("Record.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 8), background="#d32f2f", foreground="#ffffff")
        style.configure("Convert.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 6), background="#2e7d32", foreground="#ffffff")
        style.configure("Duration.TButton", font=("Segoe UI", 9), padding=(3, 7), background="#eef2f5", foreground="#26313a")
        style.configure("Selected.Duration.TButton", font=("Segoe UI", 9, "bold"), padding=(3, 7), background="#1c4f63", foreground="#ffffff")
        
        style.map("Primary.TButton", background=[("active", "#173f50"), ("disabled", "#aab7be")])
        style.map("Secondary.TButton", background=[("active", "#d9e1e7"), ("disabled", "#eef1f4")])
        style.map("Record.TButton", background=[("active", "#b71c1c"), ("disabled", "#e0e0e0")], foreground=[("disabled", "#9e9e9e")])
        style.map("Convert.TButton", background=[("active", "#1b5e20")])
        style.map("Duration.TButton", background=[("active", "#d9e4ea")])
        style.map("Selected.Duration.TButton", background=[("active", "#173f50")], foreground=[("active", "#ffffff")])
        
        style.configure("TNotebook", background="#ffffff", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 10))

    def _build_ui(self) -> None:
        root_frame = ttk.Frame(self.root, style="Root.TFrame", padding=(18, 12, 18, 12))
        root_frame.pack(fill="both", expand=True)
        root_frame.columnconfigure(0, weight=1)
        root_frame.rowconfigure(2, weight=1)
        
        # --- TOP PANEL (TUNER, SETTINGS, RECORDER) ---
        top_frame = ttk.Frame(root_frame, style="Toolbar.TFrame")
        top_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        top_frame.columnconfigure(0, weight=2)
        top_frame.columnconfigure(1, weight=3)
        top_frame.columnconfigure(2, weight=2)
        
        # 1. Live Tuner Display
        tuner_frame = ttk.LabelFrame(top_frame, text=" Тюнер в реальном времени ", padding=10)
        tuner_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        tuner_frame.columnconfigure(0, weight=1)
        
        self.live_note_label = tk.Label(tuner_frame, textvariable=self.live_note_var, font=("Segoe UI", 36, "bold"), fg="#1c4f63", bg="#ffffff")
        self.live_note_label.grid(row=0, column=0, sticky="ew")
        self.live_hz_label = tk.Label(tuner_frame, textvariable=self.live_hz_var, font=("Segoe UI", 14), fg="#3b4148", bg="#ffffff")
        self.live_hz_label.grid(row=1, column=0, sticky="ew")
        
        live_sub_frame = ttk.Frame(tuner_frame)
        live_sub_frame.grid(row=2, column=0, sticky="ew")
        live_sub_frame.columnconfigure(0, weight=1)
        self.live_conf_label = ttk.Label(live_sub_frame, textvariable=self.live_conf_var, style="Hint.TLabel")
        self.live_conf_label.grid(row=0, column=0, sticky="w")
        self.activity_canvas = tk.Canvas(live_sub_frame, width=14, height=14, highlightthickness=0, bg="#ffffff")
        self.activity_canvas.grid(row=0, column=1, sticky="e")
        self.activity_dot = self.activity_canvas.create_oval(2, 2, 12, 12, fill="#c8ccd2", outline="")
        
        # 2. Audio & Tuner Settings
        settings_frame = ttk.LabelFrame(top_frame, text=" Настройки аудио и тюнера ", padding=10)
        settings_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 5))
        settings_frame.columnconfigure(1, weight=1)
        settings_frame.columnconfigure(3, weight=1)
        
        ttk.Label(settings_frame, text="Вход:", style="Label.TLabel").grid(row=0, column=0, sticky="w", pady=2)
        self.device_combo = ttk.Combobox(settings_frame, textvariable=self.device_var, state="readonly", font=("Segoe UI", 9))
        self.device_combo.grid(row=0, column=1, columnspan=3, sticky="ew", pady=2)
        
        ttk.Label(settings_frame, text="Буфер:", style="Label.TLabel").grid(row=1, column=0, sticky="w", pady=2)
        self.fft_combo = ttk.Combobox(settings_frame, textvariable=self.buffer_size_var, values=[4096, 8192, 16384, 32768], state="readonly", width=8)
        self.fft_combo.grid(row=1, column=1, sticky="w", pady=2)
        
        ttk.Label(settings_frame, text="Блок:", style="Label.TLabel").grid(row=1, column=2, sticky="w", pady=2, padx=(10, 0))
        self.block_combo = ttk.Combobox(settings_frame, textvariable=self.block_size_var, values=[256, 512, 1024, 2048], state="readonly", width=8)
        self.block_combo.grid(row=1, column=3, sticky="w", pady=2)
        
        ttk.Label(settings_frame, text="Clarity:", style="Label.TLabel").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Spinbox(settings_frame, from_=0.1, to=0.9, increment=0.05, textvariable=self.clarity_var, width=8).grid(row=2, column=1, sticky="w", pady=2)
        ttk.Label(settings_frame, text="Cutoff K:", style="Label.TLabel").grid(row=2, column=2, sticky="w", pady=2, padx=(10, 0))
        ttk.Spinbox(settings_frame, from_=0.5, to=0.98, increment=0.02, textvariable=self.cutoff_k_var, width=8).grid(row=2, column=3, sticky="w", pady=2)
        
        ttk.Label(settings_frame, text="A4 (Гц):", style="Label.TLabel").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Spinbox(settings_frame, from_=430.0, to=450.0, increment=0.5, textvariable=self.a4_var, width=8).grid(row=3, column=1, sticky="w", pady=2)
        
        freq_frame = ttk.Frame(settings_frame)
        freq_frame.grid(row=4, column=0, columnspan=3, sticky="w", pady=5)
        ttk.Label(freq_frame, text="Мин:", style="Label.TLabel").pack(side="left")
        ttk.Entry(freq_frame, textvariable=self.min_freq_var, width=5).pack(side="left", padx=2)
        ttk.Label(freq_frame, text="Гц  Макс:", style="Label.TLabel").pack(side="left", padx=(5, 0))
        ttk.Entry(freq_frame, textvariable=self.max_freq_var, width=5).pack(side="left", padx=2)
        ttk.Label(freq_frame, text="Гц", style="Label.TLabel").pack(side="left")
        
        ttk.Button(settings_frame, text="Применить", command=self.apply_audio_settings, style="Secondary.TButton").grid(row=4, column=3, sticky="e", pady=5)
        
        self.populate_audio_devices()
        
        # 3. Performance Recorder controls
        rec_frame = ttk.LabelFrame(top_frame, text=" Запись исполнения ", padding=10)
        rec_frame.grid(row=0, column=2, sticky="nsew", padx=(5, 0))
        rec_frame.columnconfigure(0, weight=1)
        
        rec_buttons = ttk.Frame(rec_frame)
        rec_buttons.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        self.btn_start_rec = ttk.Button(rec_buttons, text="● Запись", command=self.start_recording, style="Record.TButton")
        self.btn_start_rec.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.btn_stop_rec = ttk.Button(rec_buttons, text="■ Стоп", command=self.stop_recording, style="Secondary.TButton")
        self.btn_stop_rec.pack(side="right", fill="x", expand=True, padx=(4, 0))
        self.btn_stop_rec.state(["disabled"])
        
        self.rec_status_label = tk.Label(rec_frame, text="Запись не ведется", font=("Segoe UI", 10), fg="#66707a", bg="#ffffff", height=2)
        self.rec_status_label.grid(row=1, column=0, sticky="ew")
        
        # --- MAIN WORKSPACE NOTEBOOK ---
        self.main_notebook = ttk.Notebook(root_frame)
        self.main_notebook.grid(row=1, column=0, rowspan=2, sticky="nsew", pady=(0, 10))
        root_frame.rowconfigure(1, weight=1)
        root_frame.rowconfigure(2, weight=0)

        # TAB 1: SCENARIO EDITOR
        tab_editor = ttk.Frame(self.main_notebook, style="Root.TFrame")
        self.main_notebook.add(tab_editor, text="Редактор сценария")
        tab_editor.columnconfigure(0, weight=1)
        tab_editor.rowconfigure(1, weight=1)
        
        # TAB 2: AUTO CALIBRATION
        tab_calib = ttk.Frame(self.main_notebook, style="Root.TFrame", padding=10)
        self.main_notebook.add(tab_calib, text="Автокалибровка EMMA")
        tab_calib.columnconfigure(1, weight=1)
        tab_calib.rowconfigure(1, weight=1)
        self._build_calib_tab(tab_calib)

        # TAB 3: QUICK LAUNCH
        tab_quick = ttk.Frame(self.main_notebook, style="Root.TFrame", padding=10)
        self.main_notebook.add(tab_quick, text="Быстрый запуск мелодии")
        self._build_quick_launch_tab(tab_quick)
        
        # TAB 4: VOICE CONTROL 
        tab_voice = ttk.Frame(self.main_notebook, style="Root.TFrame", padding=15)
        self.main_notebook.add(tab_voice, text="🎙️ Голосовое управление")
        self._build_voice_tab(tab_voice)

        # TAB 5: NETWORK RECEIVER 
        tab_network = ttk.Frame(self.main_notebook, style="Root.TFrame", padding=15)
        self.main_notebook.add(tab_network, text="🌐 Прием от другого ПК")
        self._build_network_tab(tab_network)

        # --- MIDDLE PANEL (TEMPO & QUANTIZATION) IN TAB 1 ---
        tempo_frame = ttk.LabelFrame(tab_editor, text=" Обработка темпа и ритма ", padding=10)
        tempo_frame.grid(row=0, column=0, sticky="ew", pady=(10, 10))
        tempo_frame.columnconfigure(5, weight=1)
        
        ttk.Label(tempo_frame, text="Четверть робота:", style="Label.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 5))
        ttk.Spinbox(tempo_frame, from_=200, to=2000, increment=50, textvariable=self.robot_tempo_var, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(tempo_frame, text="мс", style="Hint.TLabel").grid(row=0, column=2, sticky="w", padx=(2, 15))
        
        ttk.Label(tempo_frame, text="Четверть человека:", style="Label.TLabel").grid(row=0, column=3, sticky="w", padx=(0, 5))
        self.human_tempo_entry = ttk.Spinbox(tempo_frame, from_=200, to=2000, increment=10, textvariable=self.human_tempo_var, width=8)
        self.human_tempo_entry.grid(row=0, column=4, sticky="w")
        ttk.Label(tempo_frame, text="мс", style="Hint.TLabel").grid(row=0, column=5, sticky="w", padx=(2, 0))
        
        self.btn_auto_tempo = ttk.Button(tempo_frame, text="Авто-подбор темпа", command=self.auto_detect_tempo, style="Secondary.TButton")
        self.btn_auto_tempo.grid(row=0, column=6, sticky="e", padx=(0, 8))
        self.btn_auto_tempo.state(["disabled"])
        
        self.btn_insert_scenario = ttk.Button(tempo_frame, text="Конвертировать и вставить в редактор", command=self.apply_and_insert_scenario, style="Primary.TButton")
        self.btn_insert_scenario.grid(row=0, column=7, sticky="e")
        self.btn_insert_scenario.state(["disabled"])
        
        # --- BOTTOM PANEL (SCENARIO EDITOR & KEYBOARD) ---
        editor_title_frame = ttk.Frame(tab_editor, style="Toolbar.TFrame")
        editor_title_frame.grid(row=2, column=0, sticky="nsew")
        editor_title_frame.columnconfigure(0, weight=1)
        editor_title_frame.rowconfigure(1, weight=1)
        
        toolbar = ttk.Frame(editor_title_frame, style="Toolbar.TFrame")
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        toolbar.columnconfigure(1, weight=1)
        ttk.Label(toolbar, text="Редактор сценария", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        
        buttons = ttk.Frame(toolbar, style="Toolbar.TFrame")
        buttons.grid(row=0, column=2, sticky="e")
        ttk.Button(buttons, text="Новый", command=self.new_file, style="Secondary.TButton").pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Открыть", command=self.open_file, style="Secondary.TButton").pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Сохранить", command=self.save_file, style="Secondary.TButton").pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Отметить строку", command=self.toggle_line_mark, style="Secondary.TButton").pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Играть всё (F5)", command=self.play_all, style="Primary.TButton").pack(side="left", padx=(6, 6))
        ttk.Button(buttons, text="С курсора", command=self.play_from_cursor, style="Secondary.TButton").pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Стоп (Esc)", command=self.stop, style="Secondary.TButton").pack(side="left")
        
        main_workspace = ttk.PanedWindow(editor_title_frame, orient="horizontal")
        main_workspace.grid(row=1, column=0, sticky="nsew")
        
        editor_panel = ttk.Frame(main_workspace, style="Panel.TFrame", padding=(8, 8, 8, 8))
        editor_panel.rowconfigure(1, weight=1)
        editor_panel.columnconfigure(0, weight=1)
        ttk.Label(editor_panel, text="Текст сценария", style="Label.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        
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
            font=("Cascadia Mono", 10),
            relief="flat",
            padx=10,
            pady=10,
        )
        self.editor.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.editor.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.editor.configure(yscrollcommand=yscroll.set)
        
        self.editor.tag_configure("playing", background="#d7edf4", foreground="#0b2c38")
        self.editor.tag_configure("marked", background="#fff0a8")
        self.editor.tag_configure("error", background="#f6d7d2")
        
        composer = ttk.Frame(main_workspace, style="Panel.TFrame", padding=(10, 8, 10, 8))
        composer.columnconfigure(0, weight=1)
        composer.rowconfigure(1, weight=1)
        ttk.Label(composer, text="Быстрый ввод нот", style="Label.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        
        tabs = ttk.Notebook(composer)
        tabs.grid(row=1, column=0, sticky="nsew")
        keyboard_tab = ttk.Frame(tabs, style="Panel.TFrame", padding=(0, 8, 0, 0))
        keyboard_tab.columnconfigure(0, weight=1)
        keyboard_tab.rowconfigure(1, weight=1)
        tabs.add(keyboard_tab, text="Пианино")
        
        controls = ttk.Frame(keyboard_tab, style="Panel.TFrame")
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(controls, text="Длина:", style="Label.TLabel").pack(side="left", padx=(0, 6))
        for duration_ms in DURATIONS:
            button = ttk.Button(
                controls,
                text=str(duration_ms),
                command=lambda item=duration_ms: self.set_duration(item),
                style="Duration.TButton",
                width=4,
            )
            button.pack(side="left", padx=(0, 2))
            self.duration_buttons[duration_ms] = button
        ttk.Label(controls, text="мс", style="Hint.TLabel").pack(side="left", padx=(2, 0))
        self.set_duration(self.duration_var.get())
        
        self.keyboard = tk.Canvas(keyboard_tab, height=140, bg="#ffffff", highlightthickness=0)
        self.keyboard.grid(row=1, column=0, sticky="nsew")
        self.keyboard.bind("<Configure>", lambda _event: self.draw_keyboard())
        
        legend = ttk.Frame(keyboard_tab, style="Panel.TFrame")
        legend.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.choice_var = tk.StringVar(value="Кликните на клавишу для ввода /sn и /bow")
        ttk.Label(legend, textvariable=self.choice_var, style="Hint.TLabel").pack(anchor="w")
        
        helper = ttk.Frame(composer, style="Panel.TFrame")
        helper.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(helper, text="Вставить заголовок", command=self.insert_header, style="Secondary.TButton").pack(side="left")
        ttk.Button(helper, text="Добавить концовку", command=self.ensure_footer, style="Secondary.TButton").pack(side="left", padx=(6, 0))
        
        main_workspace.add(editor_panel, weight=3)
        main_workspace.add(composer, weight=2)
        
        footer = ttk.Frame(root_frame, style="Toolbar.TFrame")
        footer.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(footer, textvariable=self.line_var, style="Status.TLabel").pack(side="left")
        ttk.Label(footer, textvariable=self.status_var, style="Status.TLabel").pack(side="left", padx=(18, 0))

    # ==========================================
    # --- БЛОК СЕТЕВОГО РЕСИВЕРА (5 ВКЛАДКА) ---
    # ==========================================
    def _build_network_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        # 1. Панель управления сервером
        ctrl_frame = ttk.LabelFrame(parent, text=" Управление сервером (TCP) ", padding=15)
        ctrl_frame.grid(row=0, column=0, sticky="ew", pady=(0, 15))

        ttk.Label(ctrl_frame, text="Порт для прослушивания:", font=("Segoe UI", 10)).pack(side="left", padx=5)
        self.net_port_var = tk.IntVar(value=9000)
        ttk.Entry(ctrl_frame, textvariable=self.net_port_var, width=10, font=("Segoe UI", 10)).pack(side="left", padx=5)

        self.btn_start_net = ttk.Button(ctrl_frame, text="▶ Запустить сервер", command=self.start_network_server, style="Primary.TButton")
        self.btn_start_net.pack(side="left", padx=(15, 5))

        self.btn_stop_net = ttk.Button(ctrl_frame, text="■ Остановить", command=self.stop_network_server, style="Secondary.TButton")
        self.btn_stop_net.pack(side="left", padx=5)
        self.btn_stop_net.state(["disabled"])

        self.net_status_var = tk.StringVar(value="Сервер остановлен")
        status_label = ttk.Label(ctrl_frame, textvariable=self.net_status_var, font=("Segoe UI", 11, "bold"), foreground="#66707a")
        status_label.pack(side="left", padx=20)

        # 2. Логи приема данных
        log_frame = ttk.LabelFrame(parent, text=" Логи приема команд от другого ПК ", padding=10)
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.net_log_text = tk.Text(log_frame, bg="#fbfcfd", fg="#121417", font=("Cascadia Mono", 10), relief="flat")
        self.net_log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.net_log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.net_log_text.configure(yscrollcommand=scroll.set)
        
        self.net_log_text.tag_configure("info", foreground="#1c4f63", font=("Cascadia Mono", 10, "bold"))
        self.net_log_text.tag_configure("recv", foreground="#2e7d32")
        self.net_log_text.tag_configure("error", foreground="#d32f2f", font=("Cascadia Mono", 10, "bold"))

    def log_net(self, msg: str, tag: str = "info") -> None:
        def update_gui():
            self.net_log_text.insert(tk.END, msg + "\n", tag)
            self.net_log_text.see(tk.END)
        self.root.after(0, update_gui)

    def start_network_server(self):
        if self.net_server_running: 
            return
        self.net_server_running = True
        self.btn_start_net.state(["disabled"])
        self.btn_stop_net.state(["!disabled"])
        self.net_status_var.set("🟢 Сервер запущен (Ожидание...)")
        
        threading.Thread(target=self._network_server_worker, daemon=True).start()

    def stop_network_server(self):
        self.net_server_running = False
        if self.net_server_sock:
            try:
                self.net_server_sock.close()
            except Exception:
                pass
            self.net_server_sock = None
            
        self.btn_start_net.state(["!disabled"])
        self.btn_stop_net.state(["disabled"])
        self.net_status_var.set("🔴 Сервер остановлен")
        self.log_net("Сервер остановлен пользователем.", "info")

    def _network_server_worker(self):
        port = self.net_port_var.get()
        try:
            self.net_server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Разрешаем повторное использование адреса
            self.net_server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.net_server_sock.bind(("0.0.0.0", port))
            self.net_server_sock.listen(5)
            self.net_server_sock.settimeout(1.0) # Таймаут для корректного выхода из цикла
            
            self.log_net(f"[*] Ожидание подключений на порту {port}...", "info")
            
            while self.net_server_running:
                try:
                    conn, addr = self.net_server_sock.accept()
                    data = conn.recv(4096)
                    if data:
                        message = data.decode('utf-8', errors='replace')
                        self.log_net(f"\n--- Получено от {addr[0]}:{addr[1]} ---", "info")
                        
                        # Пересылаем роботу полученные команды построчно
                        commands = message.splitlines()
                        for cmd in commands:
                            cmd = cmd.strip()
                            if cmd and not cmd.startswith("#") and not cmd.startswith("//"):
                                self.log_net(f">> {cmd}", "recv")
                                self.robot.send_cmd(cmd)
                                time.sleep(0.04) # Пауза для стабильности отправки
                                
                        self.log_net("---------------------------", "info")
                    conn.close()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.net_server_running:
                        self.log_net(f"Ошибка соединения: {e}", "error")
                        
        except Exception as e:
            if self.net_server_running:
                self.log_net(f"Критическая ошибка сервера: {e}", "error")
        finally:
            if self.net_server_sock:
                try: 
                    self.net_server_sock.close() 
                except Exception: 
                    pass
            self.root.after(0, self.stop_network_server)

    # ==========================================
    # --- БЛОК ГОЛОСОВОГО УПРАВЛЕНИЯ (VOICE) ---
    # ==========================================
    def _build_voice_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        # 1. Верхняя панель: Статус состояния
        status_frame = ttk.LabelFrame(parent, text=" Текущее состояние бота ", padding=15)
        status_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        self.voice_state_var = tk.StringVar(value="IDLE")
        state_label = tk.Label(status_frame, textvariable=self.voice_state_var, font=("Segoe UI", 24, "bold"), fg="#1c4f63", bg="#ffffff")
        state_label.pack(pady=10)

        # 2. Панель управления (Запуск)
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.btn_start_voice = ttk.Button(control_frame, text="▶ Запустить голосовой движок (AI)", command=self.start_voice_engine, style="Primary.TButton")
        self.btn_start_voice.pack(side="left")

        # === ИНДИКАТОР МИКРОФОНА И STT РЕАЛЬНОГО ВРЕМЕНИ ===
        self.mic_status_var = tk.StringVar(value="Микрофон выключен")
        self.mic_status_label = ttk.Label(control_frame, textvariable=self.mic_status_var, font=("Segoe UI", 11, "bold"), foreground="#66707a")
        self.mic_status_label.pack(side="left", padx=15)

        # --- СЛОВАРИ ДЛЯ ГОЛОСОВОГО ИИ ---
        self.voice_commands = {
            "KEY": ["emma hello", "hello emma", "hi emma", "hey emma"],

            "START": ["hello emma are you ready", "emma are you ready", "are you ready emma", "emma raise your hands"],

            
            
            # Melodies (Emma Solo)
            "PLAY_MELODY_1": ["emma play star wars", "play the imperial march", "play star wars"],
            "PLAY_MELODY_2": ["emma play bach", "play classical music", "play bach"],
            "PLAY_ODE_TO_JOY": ["emma play ode to joy", "play ode to joy emma"],
            "PLAY_GODFATHER": ["emma play the godfather", "play the godfather emma", "emma play godfather"],
            "PLAY_EDITOR": ["emma play something", "emma play a game", "play something emma", "emma play from editor"],
            
            # Melodies (Together with Musician)
            "PLAY_GHOST": ["emma let's play the phantom of the opera together", "emma play the phantom of the opera", "emma play the ghost of the opera"],
            "PLAY_MOSCOW_NIGHTS": ["emma let's play moscow nights together", "let's play moscow nights", "emma play moscow nights together"],
            "PLAY_CANON_IN_D": ["emma let's play canon in d together", "let's play canon in d", "emma play canon in d together"],
            
            # Base Movements & Service Commands
            "CHILL": ["emma chill", "chill emma", "rest emma", "emma rest", "emma break", "emma put your hands down"],
            "ARM_OFF": ["emma let go of the hand", "emma let go of your hand", "emma release your arm", "let go of the hand emma", "emma relax your arm"],
            "TURN_AROUND": ["emma turn around", "turn around emma", "emma look around"],
            
            # Head and Emotions
            "THANK_YOU": ["emma thank you", "thank you emma", "thanks emma", "good job emma", "well done emma"],
            "EMOTIONS_ON": ["emma turn on emotions", "turn on emotions emma", "emma be emotional", "emma smile", "emma show emotions"],
            "EMOTIONS_OFF": ["emma turn off emotions", "turn off emotions emma", "emma stop emotions", "emma reset face"],
            "BLINK_OFF": ["emma do not blink", "emma don't blink", "stop blinking emma"],
            "BLINK_ON": ["emma blink", "start blinking emma", "emma you can blink"],
            "LOOK_AT_ME": ["emma look at me", "look at me emma", "emma look straight"],

            # --- ОДИНОЧНЫЕ КОМАНДЫ ---
            "STOP_ROBOT": ["emma stop", "stop emma", "stop everything", "freeze emma"],
            "OPEN_STRING": ["emma open string", "play open string"]
        }
        
        # Связь: Статус -> Текстовый файл (сценарий)
        self.voice_file_map = {
            "START": "start.txt",            
            "PLAY_MELODY_1": "star_wars.txt",    
            "PLAY_MELODY_2": "bach.txt",
            "CHILL": "chill.txt",
            "PLAY_GHOST": "призрак оперы 2 last.txt",
            "PLAY_ODE_TO_JOY": "Ода к радосте 2 last.txt",
            "PLAY_GODFATHER": "Крестный батя 2 last.txt",
            "PLAY_MOSCOW_NIGHTS": "подмосковные вечера.txt",
            "PLAY_CANON_IN_D": "Canon in D 3.0 last.txt"
        }

        # Связь: Статус -> Одиночная команда в Serial (без создания файла)
        self.voice_single_cmd_map = {
            #"KEY": "/mega;/head_rot;center\n/mega;/blink;on",

            "KEY": "/load_test\n/mega;/tilt;mid1\n/p;1500\n/mega;/tilt;forward\n/end\n/play",


            "STOP_ROBOT": "/stop\n/bow_stop\n/mega;/head_stop",
            "OPEN_STRING": "/sn;2;0",
            "ARM_OFF": "/arm_off\n/mg_off",
            "TURN_AROUND": "/violin;90\n/mega;/head_rot;left", # Поворот скрипки и головы
            "THANK_YOU": "/load_test\n/mega;/tilt;down\n/p;1500\n/mega;/tilt;forward\n/mega;/eyes;close\n/p;500\n/mega;/eyes;center\n/end\n/play",

            "EMOTIONS_ON": "/mega;/head_smile\n/mega;/eyes;random\n/mega;/head_rot;random",
            "EMOTIONS_OFF": "/mega;/head_stop",
            "BLINK_OFF": "/mega;/blink;off",
            "BLINK_ON": "/mega;/blink;on",
            "LOOK_AT_ME": "/mega;/head_rot;center\n/mega;/eyes;center\n/mega;/tilt;forward"
        }
        
        self.all_phrases = []
        self.phrase_to_state = {}
        for state, phrases in self.voice_commands.items():
            for phrase in phrases:
                self.all_phrases.append(phrase)
                self.phrase_to_state[phrase] = state

        # --- РАЗДЕЛЯЕМ НИЖНЮЮ ЧАСТЬ НА ЛОГИ И ПОДСКАЗКИ ---
        bottom_container = ttk.Frame(parent)
        bottom_container.grid(row=2, column=0, sticky="nsew")
        bottom_container.rowconfigure(0, weight=1)
        bottom_container.columnconfigure(0, weight=3) # Логи (шире)
        bottom_container.columnconfigure(1, weight=1) # Подсказки (поуже)

        # 3. Логи распознавания голоса (Слева)
        log_frame = ttk.LabelFrame(bottom_container, text=" Логи распознавания и команд ", padding=10)
        log_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.voice_log_text = tk.Text(log_frame, bg="#fbfcfd", fg="#121417", font=("Cascadia Mono", 10), relief="flat")
        self.voice_log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.voice_log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.voice_log_text.configure(yscrollcommand=scroll.set)
        
        self.voice_log_text.tag_configure("heard", foreground="#66707a")
        self.voice_log_text.tag_configure("match", foreground="#2e7d32", font=("Cascadia Mono", 10, "bold"))
        self.voice_log_text.tag_configure("unknown", foreground="#d32f2f")

        # 4. Шпаргалка с командами (Справа)
        hint_frame = ttk.LabelFrame(bottom_container, text=" 📖 Шпаргалка фраз ", padding=10)
        hint_frame.grid(row=0, column=1, sticky="nsew")
        hint_frame.rowconfigure(0, weight=1)
        hint_frame.columnconfigure(0, weight=1)
        
        hint_text = tk.Text(hint_frame, bg="#f6f7f9", fg="#3b4148", font=("Segoe UI", 9), relief="flat", wrap="word", width=35)
        hint_text.grid(row=0, column=0, sticky="nsew")
        
        for state, phrases in self.voice_commands.items():
            hint_text.insert(tk.END, f"■ {state}\n", "header")
            hint_text.insert(tk.END, f"  \"{phrases[0]}\"\n\n", "body")
            
        hint_text.tag_configure("header", font=("Segoe UI", 9, "bold"), foreground="#1c4f63")
        hint_text.tag_configure("body", foreground="#66707a")
        hint_text.configure(state="disabled")

        self.nlp_model = None
        self.command_embeddings = None
        self.voice_recorder = None

    def log_voice(self, msg: str, tag: str = "heard") -> None:
        def update_gui():
            self.voice_log_text.insert(tk.END, msg + "\n", tag)
            self.voice_log_text.see(tk.END)
        self.root.after(0, update_gui)

    # === КОЛБЕКИ ДЛЯ ИНТЕРФЕЙСА (ОТОБРАЖЕНИЕ СОСТОЯНИЯ) ===
    def _on_stt_start(self):
        self.root.after(0, lambda: [
            self.mic_status_var.set("🔴 Идет запись речи..."),
            self.mic_status_label.configure(foreground="#d32f2f")
        ])

    def _on_stt_stop(self):
        self.root.after(0, lambda: [
            self.mic_status_var.set("⚙️ Нейросеть распознает текст..."),
            self.mic_status_label.configure(foreground="#ff9800")
        ])
        
    def _on_stt_realtime(self, text: str):
        if text.strip():
            self.root.after(0, lambda: self.mic_status_var.set(f"🔴 Слышу: {text}"))
            
    def _set_stt_waiting(self):
        self.root.after(0, lambda: [
            self.mic_status_var.set("🟢 Ожидание (слушаю фон)..."),
            self.mic_status_label.configure(foreground="#2e7d32")
        ])
    # ======================================================

    def start_voice_engine(self) -> None:
        self.btn_start_voice.config(text="Загрузка моделей (Ждите...)", state="disabled")
        self.log_voice("Инициализация моделей ИИ на видеокарте... Это займет время.", "heard")
        threading.Thread(target=self._voice_engine_worker, daemon=True).start()

    def _voice_engine_worker(self) -> None:
        try:
            cfg = config.load_config()
            
            nlp_device = cfg.get("nlp", {}).get("device", "cpu")
            nlp_model_name = cfg.get("nlp", {}).get("model", "all-MiniLM-L6-v2")
            
            self.nlp_model = SentenceTransformer(nlp_model_name, device=nlp_device)
            self.command_embeddings = self.nlp_model.encode(self.all_phrases, convert_to_tensor=True)
            self.log_voice(f"NLP модель ({nlp_model_name} на {nlp_device}) загружена.", "match")

            stt_model = cfg.get("stt", {}).get("model", "base")
            stt_device = cfg.get("stt", {}).get("device", "cpu")
            stt_compute = cfg.get("stt", {}).get("compute_type", "float32")

            self.root.after(0, lambda: [
                self.mic_status_var.set(f"⏳ Загрузка STT ({stt_model})..."),
                self.mic_status_label.configure(foreground="#ff9800")
            ])

            self.voice_recorder = AudioToTextRecorder(
                model=stt_model,
                language="en",
                device=stt_device,
                compute_type=stt_compute,
                initial_prompt="Emma.",
                post_speech_silence_duration=1,
                min_length_of_recording=0.3,
                on_recording_start=self._on_stt_start,
                on_recording_stop=self._on_stt_stop,
                on_realtime_transcription_update=self._on_stt_realtime
            )
            
            self.log_voice(f"STT модель ({stt_model} на {stt_device}) готова. Микрофон активен. Можно говорить!", "match")
            self.root.after(0, lambda: self.btn_start_voice.config(text="Аудио движок АКТИВЕН", state="disabled"))
            self._set_stt_waiting()
            
            while True:
                self.voice_recorder.text(self._process_voice_text)
                
        except Exception as e:
            self.log_voice(f"Ошибка аудио движка: {e}", "unknown")
            self.root.after(0, lambda: [
                self.mic_status_var.set("❌ Ошибка микрофона/ИИ"),
                self.mic_status_label.configure(foreground="#d32f2f")
            ])

    def _process_voice_text(self, text: str) -> None:
        if not text.strip():
            return
        
        self.log_voice(f"🗣️ {text}", "heard")
        
        # NLP matching
        user_emb = self.nlp_model.encode(text, convert_to_tensor=True)
        scores = util.cos_sim(user_emb, self.command_embeddings)[0]
        best_idx = torch.argmax(scores).item()
        best_score = scores[best_idx].item()
        
        if best_score > 0.60:
            matched_phrase = self.all_phrases[best_idx]
            state = self.phrase_to_state[matched_phrase]
            self.log_voice(f"⚡ Распознано: [{state}] (уверенность: {best_score:.2f})", "match")
            self.execute_voice_action(state)
        else:
            self.log_voice(f"❓ Неизвестная команда (макс. совпадение: {best_score:.2f})", "unknown")
        
        self._set_stt_waiting()


    def execute_voice_action(self, state: str) -> None:
        if state in self.voice_file_map:
            file_name = self.voice_file_map[state]
            
            # Берем директорию, которая сейчас выбрана во вкладке "Быстрый запуск"
            # Если ничего не выбрано, используем текущую рабочую папку (Path.cwd())
            base_dir = getattr(self, 'quick_launch_dir', Path.cwd()) 
            
            file_path = base_dir / file_name
            
            if file_path.is_file():
                self.log_gui(f"[Voice AI] Запуск файла сценария: {file_name}")
                self.send_melody_file(str(file_path), file_name)
            else:
                # В лог ошибки выводим полный путь, чтобы сразу видеть, где ИИ искал файл
                self.log_voice(f"[ОШИБКА] Файл {file_name} не найден в папке:\n{base_dir}", "unknown")
            return

        if state in self.voice_single_cmd_map:
            cmd_string = self.voice_single_cmd_map[state]
            self.log_gui(f"[Voice AI] Отправка команды: {state}")
            for cmd in cmd_string.split('\n'):
                if cmd.strip():
                    self.robot.send_cmd(cmd.strip())
            return

        if state == "PLAY_EDITOR":
            self.log_gui("[Voice AI] Запуск воспроизведения из текстового редактора!")
            self.root.after(0, self.play_all)

    def send_melody_file(self, file_path: str, title: str = "") -> None:
        path = Path(file_path)
        if not path.is_file():
            messagebox.showerror("Ошибка файла", f"Файл не найден:\n{path.resolve()}")
            self.status_var.set(f"Ошибка: файл не найден ({file_path})")
            return
        try:
            content = path.read_text(encoding="utf-8")
            lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]
            if not lines:
                messagebox.showwarning("Пустой файл", f"Файл {path.name} не содержит команд.")
                return
            self.log_gui(f"=== ОТПРАВКА МЕЛОДИИ: {title or path.name} ({len(lines)} команд) ===")
            for cmd in lines:
                res = self.robot.send_cmd(cmd)
                self.log_gui(res)
                time.sleep(0.02)
            self.status_var.set(f"Успешно отправлена мелодия: {title or path.name}")
        except Exception as e:
            messagebox.showerror("Ошибка отправки", f"Не удалось прочитать или отправить файл:\n{e}")

    # ==========================================

    def _build_calib_tab(self, parent: ttk.Frame) -> None:
        # Connection
        conn_frame = ttk.LabelFrame(parent, text=" Подключение к EMMA ", padding=10)
        conn_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Label(conn_frame, text="COM порт:").pack(side="left", padx=5)
        ttk.Entry(conn_frame, textvariable=self.robot_port_var, width=20).pack(side="left", padx=5)
        ttk.Button(conn_frame, text="Подключиться", command=self.connect_robot, style="Primary.TButton").pack(side="left", padx=5)
        ttk.Button(conn_frame, text="Проверка связи (/cal)", command=self.check_connection, style="Secondary.TButton").pack(side="left", padx=5)
        
        # Controls
        ctrl_frame = ttk.LabelFrame(parent, text=" Управление ", padding=10)
        ctrl_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        
        self.calib_dev_label = ttk.Label(ctrl_frame, text="Отклонение: 0 цент. 00 коп.", font=("Segoe UI", 10, "bold"))
        self.calib_dev_label.pack(pady=(0, 5))
        self.calib_dev_bar = ttk.Progressbar(ctrl_frame, orient="horizontal", length=200, mode="determinate")
        self.calib_dev_bar.pack(pady=(0, 15))
        self.calib_dev_bar['value'] = 50

        self.global_cent_offset = 12
        self.custom_freq_vars = {1: {}, 2: {}, 3: {}, 4: {}}
        
        for s in [1, 2, 3, 4]:
            str_frame = ttk.Frame(ctrl_frame)
            str_frame.pack(fill="x", pady=2)
            ttk.Button(str_frame, text=f"ГАММА\nСтр. {s}", width=8, command=lambda string=s: self.start_auto_calib(string), style="Primary.TButton").grid(row=0, column=0, rowspan=2, padx=2, sticky="nsew")
            for n in range(1, 8):
                btn = ttk.Button(str_frame, text=str(n), width=4, 
                           command=lambda note=n, string=s: self.start_single_note_test(note, string))
                btn.grid(row=0, column=n, padx=1, pady=1)
                base_freq = TARGET_FREQS[s][n]
                shifted_freq = base_freq * (2.0 ** (self.global_cent_offset / 1200.0))
                var = tk.DoubleVar(value=round(shifted_freq, 2))
                self.custom_freq_vars[s][n] = var
                entry = ttk.Entry(str_frame, textvariable=var, width=6, font=("Segoe UI", 7))
                entry.grid(row=1, column=n, padx=1, pady=1)

        shift_frame = ttk.Frame(ctrl_frame)
        shift_frame.pack(fill="x", pady=5)
        ttk.Button(shift_frame, text="-1 Цент", width=8, command=lambda: self.shift_cents(-1)).pack(side="left", padx=2, expand=True)
        
        self.offset_label_var = tk.StringVar(value=f"Сдвиг: {self.global_cent_offset:+} центов")
        ttk.Label(shift_frame, textvariable=self.offset_label_var, font=("Segoe UI", 9, "bold")).pack(side="left", padx=2)
        
        ttk.Button(shift_frame, text="+1 Цент", width=8, command=lambda: self.shift_cents(1)).pack(side="left", padx=2, expand=True)

        ttk.Button(ctrl_frame, text="ОТПУСТИТЬ ПАЛЕЦ (Открытая стр)", command=lambda: [self.robot.send_cmd(f"/sn;{getattr(self, 'ac_string', 2)};0"), self.log_gui("[Manual] Открытая струна")], style="Secondary.TButton").pack(fill="x", pady=5)
        ttk.Button(ctrl_frame, text="СТОП (/stop)", command=self.emergency_stop, style="Record.TButton").pack(fill="x", pady=5)
        
        # Logs
        log_frame = ttk.LabelFrame(parent, text=" Логи робота ", padding=10)
        log_frame.grid(row=1, column=1, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        
        self.log_text = tk.Text(log_frame, bg="#fbfcfd", fg="#121417", font=("Cascadia Mono", 9), height=15, relief="flat")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)
        
        self.log_text.tag_configure("recv", foreground="#2e7d32")
        self.log_text.tag_configure("send", foreground="#0b2c38", font=("Cascadia Mono", 9, "bold"))
        self.log_text.tag_configure("error", foreground="#d32f2f", font=("Cascadia Mono", 9, "bold"))
        self.log_text.tag_configure("info", foreground="#1c4f63")

        # Ручной ввод команд (без дерева)
        parent.rowconfigure(1, weight=1)
        parent.rowconfigure(2, weight=0)
        
        manual_frame = ttk.LabelFrame(parent, text=" Ручной ввод команд ", padding=10)
        manual_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        
        cmd_entry_frame = ttk.Frame(manual_frame)
        cmd_entry_frame.pack(fill="x")
        
        ttk.Label(cmd_entry_frame, text="Команда:").pack(side="left", padx=(0, 5))
        self.manual_cmd_var = tk.StringVar()
        cmd_entry = ttk.Entry(cmd_entry_frame, textvariable=self.manual_cmd_var, font=("Cascadia Mono", 10))
        cmd_entry.pack(side="left", fill="x", expand=True, padx=5)
        
        self.cmd_history = []
        self.cmd_history_idx = -1

        def on_up(event):
            if self.cmd_history:
                if self.cmd_history_idx < len(self.cmd_history) - 1:
                    self.cmd_history_idx += 1
                self.manual_cmd_var.set(self.cmd_history[self.cmd_history_idx])
                cmd_entry.icursor(tk.END)
            return "break"
            
        def on_down(event):
            if self.cmd_history:
                if self.cmd_history_idx > 0:
                    self.cmd_history_idx -= 1
                    self.manual_cmd_var.set(self.cmd_history[self.cmd_history_idx])
                elif self.cmd_history_idx == 0:
                    self.cmd_history_idx = -1
                    self.manual_cmd_var.set("")
                cmd_entry.icursor(tk.END)
            return "break"
            
        cmd_entry.bind("<Up>", on_up)
        cmd_entry.bind("<Down>", on_down)
        
        def send_manual_cmd(*args):
            cmd = self.manual_cmd_var.get().strip()
            if cmd:
                self.robot.send_cmd(cmd)
                self.log_gui(f"[Manual] Отправлено: {cmd}")
                
                if not self.cmd_history or self.cmd_history[0] != cmd:
                    self.cmd_history.insert(0, cmd)
                self.cmd_history_idx = -1
                
                self.manual_cmd_var.set("")
                
        cmd_entry.bind("<Return>", send_manual_cmd)
        ttk.Button(cmd_entry_frame, text="Отправить", command=send_manual_cmd, style="Primary.TButton").pack(side="left", padx=5)

    def _build_quick_launch_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        self.quick_launch_stop_flag = False
        self.quick_launch_dir = Path.cwd()

        # Left side - Scenarios List
        left_frame = ttk.Frame(parent)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=1)
        
        # Top bar with folder selector
        top_frame_left = ttk.Frame(left_frame)
        top_frame_left.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        self.ql_dir_var = tk.StringVar(value=f"Папка: {self.quick_launch_dir.name}")
        ttk.Label(top_frame_left, textvariable=self.ql_dir_var, style="Label.TLabel").pack(side="left", padx=(0, 10))
        ttk.Button(top_frame_left, text="Выбрать папку", command=self.choose_quick_launch_dir, style="Secondary.TButton").pack(side="left", padx=5)
        
        # Listbox for files
        list_frame = ttk.Frame(left_frame, style="Panel.TFrame")
        list_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self.scenarios_listbox = tk.Listbox(list_frame, font=("Cascadia Mono", 15), bg="#fbfcfd", fg="#121417", selectbackground="#c9dce5", selectforeground="#0b2c38", relief="flat")
        self.scenarios_listbox.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.scenarios_listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns", pady=10)
        self.scenarios_listbox.configure(yscrollcommand=scroll.set)
        
        self.scenarios_listbox.bind("<Double-Button-1>", self.load_selected_quick_scenario)
        self.scenarios_listbox.bind("<Button-3>", self._show_ql_context_menu)
        
        # Bottom controls left
        bot_frame_left = ttk.Frame(left_frame)
        bot_frame_left.grid(row=2, column=0, sticky="ew")
        ttk.Button(bot_frame_left, text="Обновить", command=self.refresh_scenarios, style="Secondary.TButton").pack(side="left")
        ttk.Button(bot_frame_left, text="В редактор", command=self.load_selected_quick_scenario, style="Secondary.TButton").pack(side="left", padx=5)
        ttk.Button(bot_frame_left, text="Остановить", command=self.stop_quick_launch, style="Record.TButton").pack(side="right", padx=(5, 0))
        ttk.Button(bot_frame_left, text="Отправить на робота", command=self.send_scenario_to_robot, style="Primary.TButton").pack(side="right", padx=5)

        # Right side - File creation
        right_frame = ttk.LabelFrame(parent, text=" Просмотр / Редактирование / Создание ", padding=10)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)

        name_frame = ttk.Frame(right_frame)
        name_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        ttk.Button(name_frame, text="Новый файл", command=self.clear_quick_scenario_editor, style="Secondary.TButton").pack(side="left", padx=(0, 5))
        ttk.Label(name_frame, text="Имя файла:").pack(side="left")
        self.new_scenario_name_var = tk.StringVar(value="melody.txt")
        ttk.Entry(name_frame, textvariable=self.new_scenario_name_var).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(name_frame, text="Сохранить файл", command=self.save_new_quick_scenario, style="Primary.TButton").pack(side="left")

        text_frame = ttk.Frame(right_frame, style="Panel.TFrame")
        text_frame.grid(row=1, column=0, sticky="nsew")
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        self.new_scenario_text = tk.Text(text_frame, font=("Cascadia Mono", 10), bg="#fbfcfd", fg="#121417", wrap="none")
        self.new_scenario_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        text_scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.new_scenario_text.yview)
        text_scroll.grid(row=0, column=1, sticky="ns", pady=5)
        self.new_scenario_text.configure(yscrollcommand=text_scroll.set)

        self.refresh_scenarios()
        
    def clear_quick_scenario_editor(self):
        self.new_scenario_name_var.set("new_melody.txt")
        self.new_scenario_text.delete("1.0", tk.END)
        self.status_var.set("Редактор очищен для нового файла.")

    def choose_quick_launch_dir(self):
        d = filedialog.askdirectory(title="Выберите папку со сценариями")
        if d:
            self.quick_launch_dir = Path(d)
            self.ql_dir_var.set(f"Папка: {self.quick_launch_dir.name}")
            self.refresh_scenarios()
            
    def load_selected_quick_scenario(self, event=None):
        selection = self.scenarios_listbox.curselection()
        if not selection:
            if event is None:
                messagebox.showwarning("Внимание", "Выберите файл для загрузки!")
            return
            
        file_path = self.scenario_files[selection[0]]
        
        try:
            content = file_path.read_text(encoding="utf-8")
            self.new_scenario_name_var.set(file_path.name)
            self.new_scenario_text.delete("1.0", tk.END)
            self.new_scenario_text.insert("1.0", content)
            self.status_var.set(f"Файл {file_path.name} загружен в редактор.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать файл:\n{e}")

    def refresh_scenarios(self):
        self.scenarios_listbox.delete(0, tk.END)
        self.scenario_files = []
        if hasattr(self, 'quick_launch_dir') and self.quick_launch_dir.exists():
            for p in sorted(self.quick_launch_dir.glob('*.txt')):
                self.scenario_files.append(p)
                self.scenarios_listbox.insert(tk.END, f"📜 {p.name}")

    def _show_ql_context_menu(self, event):
        index = self.scenarios_listbox.nearest(event.y)
        if index >= 0:
            self.scenarios_listbox.selection_clear(0, tk.END)
            self.scenarios_listbox.selection_set(index)
            self.scenarios_listbox.activate(index)
            
            menu = tk.Menu(self.root, tearoff=0)
            menu.add_command(label="Отправить на робота", command=self.send_scenario_to_robot)
            menu.add_command(label="В редактор", command=self.load_selected_quick_scenario)
            menu.add_separator()
            menu.add_command(label="Переименовать", command=self.ql_rename_file)
            menu.add_command(label="Скопировать весь текст", command=self.ql_copy_text)
            menu.add_separator()
            menu.add_command(label="Удалить", command=self.ql_delete_file)
            
            menu.tk_popup(event.x_root, event.y_root)

    def ql_rename_file(self):
        selection = self.scenarios_listbox.curselection()
        if not selection: return
        file_path = self.scenario_files[selection[0]]
        
        from tkinter import simpledialog
        new_name = simpledialog.askstring("Переименовать", "Новое имя файла:", initialvalue=file_path.name, parent=self.root)
        if new_name and new_name != file_path.name:
            if not new_name.endswith(".txt"):
                new_name += ".txt"
            new_path = file_path.with_name(new_name)
            if new_path.exists():
                messagebox.showerror("Ошибка", "Файл с таким именем уже существует!")
                return
            try:
                file_path.rename(new_path)
                self.status_var.set(f"Переименовано в {new_name}")
                self.refresh_scenarios()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось переименовать:\n{e}")

    def ql_copy_text(self):
        selection = self.scenarios_listbox.curselection()
        if not selection: return
        file_path = self.scenario_files[selection[0]]
        try:
            content = file_path.read_text(encoding="utf-8")
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self.status_var.set("Текст сценария скопирован в буфер обмена.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать файл:\n{e}")

    def ql_delete_file(self):
        selection = self.scenarios_listbox.curselection()
        if not selection: return
        file_path = self.scenario_files[selection[0]]
        
        if messagebox.askyesno("Удалить файл?", f"Вы уверены, что хотите удалить '{file_path.name}'?"):
            try:
                file_path.unlink()
                self.status_var.set(f"Файл {file_path.name} удален.")
                self.refresh_scenarios()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось удалить файл:\n{e}")

    def save_new_quick_scenario(self):
        name = self.new_scenario_name_var.get().strip()
        if not name.endswith(".txt"):
            name += ".txt"
        
        if not hasattr(self, 'quick_launch_dir') or not self.quick_launch_dir.exists():
            messagebox.showerror("Ошибка", "Сначала выберите папку назначения!")
            return
            
        path = self.quick_launch_dir / name
        content = self.new_scenario_text.get("1.0", "end-1c")
        
        try:
            path.write_text(content, encoding="utf-8")
            self.status_var.set(f"Файл {name} сохранен!")
            self.refresh_scenarios()
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))
            
    def send_scenario_to_robot(self):
        selection = self.scenarios_listbox.curselection()
        if not selection:
            messagebox.showwarning("Внимание", "Выберите файл для запуска!")
            return
            
        file_path = self.scenario_files[selection[0]]
        
        if not self.robot.serial_conn or not self.robot.serial_conn.is_open:
            messagebox.showerror("Ошибка", "Робот не подключен!")
            return
            
        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.splitlines()
            self.quick_launch_stop_flag = False
            
            def send_task():
                self.root.after(0, lambda: self.status_var.set(f"Отправка {file_path.name}..."))
                self.root.after(0, lambda: self.log_gui(f"=== ОТПРАВКА СЦЕНАРИЯ {file_path.name} ==="))
                for line in lines:
                    if self.quick_launch_stop_flag:
                        self.root.after(0, lambda: self.log_gui("=== ОТПРАВКА ПРЕРВАНА ==="))
                        return
                    line = line.strip()
                    if line and not line.startswith("//") and not line.startswith("#"):
                        self.robot.send_cmd(line)
                        self.root.after(0, lambda l=line: self.log_gui(f"[Auto] {l}"))
                        time.sleep(0.04)
                self.root.after(0, lambda: self.status_var.set(f"Сценарий {file_path.name} успешно отправлен!"))
                self.root.after(0, lambda: self.log_gui(f"=== СЦЕНАРИЙ ОТПРАВЛЕН ==="))
                
            threading.Thread(target=send_task, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать или отправить файл:\n{e}")

    def stop_quick_launch(self):
        self.quick_launch_stop_flag = True
        if self.robot.serial_conn and self.robot.serial_conn.is_open:
            self.log_gui("\n[!] ОСТАНОВКА БЫСТРОГО ЗАПУСКА")
            self.robot.send_cmd("/stop")
            self.robot.send_cmd("/bow_stop")
            self.robot.send_cmd("/finger;-1")
            self.robot.send_cmd("/sn;2;0")
            self.robot.send_cmd("/bow;0;3000;100")
            self.status_var.set("Мелодия остановлена, смычок отведен.")
        else:
            self.status_var.set("Робот не подключен.")

    def log_gui(self, msg: str) -> None:
        tag = "info"
        if "[EMMA -> PC]" in msg or "<< " in msg:
            tag = "recv"
        elif "[Manual]" in msg or ">> " in msg or "Отправлено" in msg or "Отправлен" in msg:
            tag = "send"
        if "Ошибка" in msg or "ERROR" in msg or "ОСТАНОВКА" in msg or "[!]" in msg:
            tag = "error"
            
        self.log_text.insert(tk.END, msg + "\n", tag)
        self.log_text.see(tk.END)

    def connect_robot(self):
        self.robot.port = self.robot_port_var.get()
        ok, msg = self.robot.connect()
        self.log_gui(msg)

    def check_connection(self):
        self.log_gui(self.robot.send_cmd("/cal"))
        self.log_gui("Проверка связи: отправлен /cal")

    def shift_cents(self, delta: int):
        self.global_cent_offset += delta
        self.offset_label_var.set(f"Сдвиг: {self.global_cent_offset:+} центов")
        for s in [1, 2, 3, 4]:
            for n in range(1, 8):
                base_freq = TARGET_FREQS[s][n]
                shifted_freq = base_freq * (2.0 ** (self.global_cent_offset / 1200.0))
                self.custom_freq_vars[s][n].set(round(shifted_freq, 2))
        self.log_gui(f"[Manual] Все частоты сдвинуты на {self.global_cent_offset:+} центов от номинала.")

    def emergency_stop(self):
        self.auto_calib_active = False
        self.ac_state = "IDLE"
        self.robot.send_cmd("/stop")
        self.robot.send_cmd("/bow_stop")
        self.log_gui("\n[!] ЭКСТРЕННАЯ ОСТАНОВКА. Процесс прерван.")

    def start_auto_calib(self, string: int = 2):
        self.log_text.delete(1.0, tk.END)
        self.log_gui(f"=== СТАРТ ТЕСТОВОГО ПРОГОНА (ГАММА, СТРУНА {string}) ===")
        self.auto_calib_active = True
        self.ac_mode = "GAMMA"
        self.ac_string = string
        self.ac_note = 1
        self.ac_bow_dir = 1
        self.ac_needs_save = False
        self.ac_ideal_count = 0
        self.ac_state = "INIT"

    def start_single_note_test(self, note: int = 1, string: int = 2):
        self.log_text.delete(1.0, tk.END)
        self.log_gui(f"=== СТАРТ ТЕСТА ОДНОЙ НОТЫ (ЦИКЛ, Струна {string}, Нота {note}) ===")
        self.auto_calib_active = True
        self.ac_mode = "SINGLE"
        self.ac_string = string
        self.ac_note = note
        self.ac_bow_dir = 1
        self.ac_needs_save = False
        self.ac_ideal_count = 0
        self.ac_state = "INIT"

    def auto_calib_tick(self):
        self.root.after(100, self.auto_calib_tick)
        
        if not self.auto_calib_active:
            return
        if time.time() < self.ac_wait_time:
            return

        if self.ac_state == "INIT":
            self.log_gui(self.robot.send_cmd("/save"))
            self.log_gui("1. Отправлен /save (бекап).")
            self.log_gui(self.robot.send_cmd("/cal"))
            self.log_gui("2. Ждем калибровку концевика (2 сек)...")
            self.ac_wait_time = time.time() + 2.0
            self.ac_state = "GET_MM"
            
        elif self.ac_state == "GET_MM":
            self.log_gui(self.robot.send_cmd("/get_mm"))
            self.log_gui("3. Запрашиваем миллиметры...")
            self.ac_wait_time = time.time() + 1.5
            self.ac_state = "PLAY"
            
        elif self.ac_state == "PLAY":
            if self.ac_note > 7:
                self.log_gui("\n=== ГАММА УСПЕШНО ЗАВЕРШЕНА! ===")
                self.log_gui("Отжимаем палец и уводим смычок в другую сторону...")
                self.robot.send_cmd(f"/sn;{self.ac_string};0")
                self.robot.send_cmd(f"/bow;{self.ac_bow_dir};6000;70")
                self.log_gui("Режим ЧТЕНИЯ. Изменения не вносились.")
                self.auto_calib_active = False
                return
                
            self.log_gui(f"\n[Струна {self.ac_string} | Нота {self.ac_note}] Играем...")
            self.robot.send_cmd(f"/sn;{self.ac_string};{self.ac_note};1")
            self.ac_wait_time = time.time() + 1.5
            self.ac_state = "BOW"
            
        elif self.ac_state == "BOW":
            dir_text = "Вперед" if self.ac_bow_dir == 1 else "Назад"
            self.log_gui(f"Смычок: {dir_text} (макс. 6 сек)")
            self.robot.send_cmd(f"/bow;{self.ac_bow_dir};6000;70")
            self.ac_bow_dir = 0 if self.ac_bow_dir == 1 else 1
            self.ac_freq_buffer = []
            self.ac_wait_time = time.time() + 0.5
            self.ac_listen_timeout = time.time() + 6.0
            self.ac_state = "LISTEN"
            
        elif self.ac_state == "LISTEN":
            freq = 0.0
            try:
                hz_str = self.live_hz_var.get()
                if "Hz" in hz_str and "--" not in hz_str:
                    freq = float(hz_str.replace(" Hz", ""))
            except ValueError:
                pass

            if freq > 0:
                self.ac_freq_buffer.append(freq)
                
            if len(self.ac_freq_buffer) >= 12:
                self.log_gui("-> Звук пойман! Останавливаем смычок досрочно.")
                self.robot.send_cmd("/bow_stop")
                self.ac_state = "ANALYZE"
            elif time.time() >= self.ac_listen_timeout:
                self.ac_state = "ANALYZE"
                
        elif self.ac_state == "ANALYZE":
            is_ideal = False
            if len(self.ac_freq_buffer) >= 5:
                import numpy as np
                med_freq = float(np.median(self.ac_freq_buffer))
                if hasattr(self, 'custom_freq_vars') and self.ac_string in self.custom_freq_vars and self.ac_note in self.custom_freq_vars[self.ac_string]:
                    try:
                        ideal_freq = float(self.custom_freq_vars[self.ac_string][self.ac_note].get())
                    except ValueError:
                        ideal_freq = TARGET_FREQS[self.ac_string][self.ac_note]
                else:
                    ideal_freq = TARGET_FREQS[self.ac_string][self.ac_note]
                med_cents = 1200 * math.log2(med_freq / ideal_freq)
                
                self.log_gui(f"-> Идеал: {ideal_freq:.1f} Hz | Услышано: {med_freq:.1f} Hz")
                self.log_gui(f"-> Отклонение: {med_cents:+.1f} центов")
                
                if abs(med_cents) <= 7.0:
                    self.log_gui("-> ИДЕАЛЬНО!")
                    self.ac_ideal_count += 1
                    
                    if getattr(self, 'ac_needs_save', False):
                        self.log_gui("-> Позиция подобрана. Сохраняем во флэш память... /save")
                        self.robot.send_cmd("/save")
                        self.ac_needs_save = False
                    is_ideal = True
                else:
                    self.ac_ideal_count = 0
                    if med_cents > 0:
                        self.log_gui("-> ВЫСИТ. Корректируем позицию (влево/меньше мм)...")
                    else:
                        self.log_gui("-> НИЗИТ. Корректируем позицию (вправо/больше мм)...")
                    
                    delta_mm = med_cents * 0.15
                    delta_mm = np.clip(delta_mm, -5.0, 5.0)
                    
                    try:
                        current_val = self.current_mm[self.ac_string - 1][self.ac_note - 1]
                        if current_val == 0.0:
                            self.log_gui("[!] mm = 0 (не считано). Корректировка невозможна. Пропускаем.")
                            is_ideal = True
                        else:
                            new_val = current_val - delta_mm
                            self.current_mm[self.ac_string - 1][self.ac_note - 1] = new_val
                            self.robot.send_cmd(f"/set_mm;{self.ac_string};{self.ac_note};{new_val:.4f}")
                            self.log_gui(f"-> Отправлено смещение: /set_mm;{self.ac_string};{self.ac_note};{new_val:.4f}")
                            self.robot.send_cmd(f"/sn;{self.ac_string};0")
                            self.log_gui("-> Отпущен палец для обновления физической позиции")
                            self.ac_needs_save = True
                    except Exception as e:
                        self.log_gui(f"[!] Ошибка корректировки: {e}")
                        is_ideal = True
            else:
                self.log_gui("-> Звук не распознан или слишком тихий.")

            if getattr(self, 'ac_mode', 'GAMMA') == 'SINGLE':
                if getattr(self, 'ac_ideal_count', 0) >= 2:
                    self.log_gui(f"\n=== НОТА {self.ac_note} ОТКАЛИБРОВАНА (2 РАЗА ИДЕАЛЬНО) ===")
                    self.log_gui("Ожидание следующих команд...")
                    self.robot.send_cmd(f"/sn;{self.ac_string};0")
                    self.auto_calib_active = False
                    self.ac_state = "IDLE"
                else:
                    self.log_gui("\nЦикл одной ноты: повтор через 1.5 сек...")
                    self.ac_wait_time = time.time() + 1.5
                    self.ac_state = "PLAY"
            else:
                if is_ideal:
                    if getattr(self, 'ac_ideal_count', 0) >= 2:
                        self.log_gui(f"\n=== НОТА {self.ac_note} ОТКАЛИБРОВАНА. ПЕРЕХОД К СЛЕДУЮЩЕЙ ===")
                        self.ac_ideal_count = 0
                        self.ac_note += 1
                        self.ac_wait_time = time.time() + 1.0
                        self.ac_state = "PLAY"
                    else:
                        self.log_gui("\nИдеально (1 раз). Для надежности повторяем еще раз через 1.5 сек...")
                        self.ac_wait_time = time.time() + 1.5
                        self.ac_state = "PLAY"
                else:
                    self.log_gui("\nНота еще не идеальна: повторяем попытку через 1.5 сек...")
                    self.ac_wait_time = time.time() + 1.5
                    self.ac_state = "PLAY"

    def serial_reader_thread(self):
        while True:
            if self.robot.serial_conn and self.robot.serial_conn.is_open:
                try:
                    line = self.robot.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        self.root.after(0, lambda l=line: self.log_gui(f"[EMMA -> PC] {l}"))
                        if "Струна" in line and ":" in line:
                            parts = line.split(":")
                            str_str = parts[0].replace("[FINGERS]", "").replace("Струна", "").strip()
                            if str_str.isdigit():
                                str_idx = int(str_str) - 1
                                vals = parts[1].strip().split()
                                if len(vals) == 7:
                                    for i, v in enumerate(vals):
                                        self.current_mm[str_idx][i] = float(v)
                except Exception:
                    pass
            else:
                time.sleep(0.1)

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
        self.line_var.set(f"Строка {self.editor.index('insert').split('.')[0]}")

    def _insert_starter(self) -> None:
        self.editor.insert("1.0", "New scenario\n/load_test\n/end\n/play\n")
        self.editor.mark_set("insert", "3.0")
        self.editor.edit_reset()
        self.editor.edit_modified(False)

    # --- AUDIO SETUP & LISTENING ---
    def populate_audio_devices(self) -> None:
        try:
            devices = sd.query_devices()
            input_device_strs = []
            default_idx = sd.default.device[0]
            default_str = ""
            for idx, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    label = f"{idx}: {dev['name']}"
                    input_device_strs.append(label)
                    if idx == default_idx:
                        default_str = label
            
            self.device_combo['values'] = input_device_strs
            if default_str:
                self.device_combo.set(default_str)
            elif input_device_strs:
                self.device_combo.set(input_device_strs[0])
        except Exception as exc:
            self.status_var.set("Ошибка получения аудиоустройств")

    def apply_audio_settings(self) -> None:
        was_running = self.running
        if was_running:
            self.stop_detector()
            
        try:
            device_str = self.device_var.get()
            device_id = int(device_str.split(":")[0]) if device_str else None
            
            sample_rate = 48000
            if device_id is not None:
                device_info = sd.query_devices(device_id)
                sample_rate = int(device_info['default_samplerate'])
                
            self.detector = MpmPitchDetector(
                sample_rate=sample_rate,
                min_freq=self.min_freq_var.get(),
                max_freq=self.max_freq_var.get(),
                clarity_threshold=self.clarity_var.get(),
                cutoff_k=self.cutoff_k_var.get(),
                a4=self.a4_var.get()
            )
            
            self.ring = np.zeros(self.buffer_size_var.get(), dtype=np.float32)
            self.audio_queue = queue.Queue(maxsize=16)
            self.status_queue = queue.Queue(maxsize=4)
            self.sample_rate = sample_rate
            self.device_id = device_id
            
            self.status_var.set("Настройки звука применены")
            if was_running:
                self.start_detector()
        except Exception as exc:
            messagebox.showerror("Ошибка аудио", f"Не удалось настроить аудио:\n{exc}")

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

    def start_detector(self) -> None:
        if self.running:
            return
        try:
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.block_size_var.get(),
                callback=self.audio_callback,
            )
            self.stream.start()
            self.running = True
            self.activity_canvas.itemconfigure(self.activity_dot, fill="#b56b00")
            self.status_var.set("Слушаю микрофон")
        except Exception as exc:
            self.status_var.set("Ошибка запуска аудио")
            messagebox.showerror("Ошибка аудиовхода", str(exc))

    def stop_detector(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.running = False
        self.live_note_var.set("--")
        self.live_hz_var.set("-- Hz")
        self.live_conf_var.set("Conf: 0.00")
        self.activity_canvas.itemconfigure(self.activity_dot, fill="#c8ccd2")
        self.status_var.set("Аудиовход остановлен")

    def poll_audio(self) -> None:
        processed = False
        input_status = ""
        try:
            input_status = self.status_queue.get_nowait()
        except queue.Empty:
            pass

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

        if self.running and processed and self.detector:
            result = self.detector.detect(self.ring)
            now = time.time()
            if result is None:
                self.silent_ticks += 1
                if self.silent_ticks >= self.confirm_frames_var.get():
                    self.live_note_var.set("--")
                    self.live_hz_var.set("-- Hz")
                    self.live_conf_var.set("Conf: 0.00")
                    self.activity_canvas.itemconfigure(self.activity_dot, fill="#c8ccd2")
                    
                    if hasattr(self, 'calib_dev_label'):
                        self.calib_dev_label.config(text="Отклонение: 0 цент. 00 коп.")
                        self.calib_dev_bar['value'] = 50
                    
                    if self.recording:
                        self.raw_detections.append((now, None))
            else:
                self.silent_ticks = 0
                self.live_note_var.set(f"{result.note}{result.octave}")
                self.live_hz_var.set(f"{result.frequency:.2f} Hz")
                self.live_conf_var.set(f"Conf: {result.confidence:.2f}")
                
                sign = "+" if result.cents >= 0 else ""
                kopecks = int(abs(result.cents) * 100) % 100
                cents_int = int(result.cents)
                if hasattr(self, 'calib_dev_label'):
                    self.calib_dev_label.config(text=f"Отклонение: {sign}{cents_int} цент. {kopecks:02d} коп.")
                    self.calib_dev_bar['value'] = int(np.clip(result.cents + 50, 0, 100))
                
                color = "#4caf50" if result.confidence > 0.65 else "#ff9800"
                self.activity_canvas.itemconfigure(self.activity_dot, fill=color)
                
                if self.recording:
                    nearest_midi = int(round(69.0 + 12.0 * math.log2(result.frequency / self.detector.a4)))
                    self.raw_detections.append((now, nearest_midi))
                    self.update_recording_status()

        self.root.after(50, self.poll_audio)

    # --- RECORDING PROCESS (OLD TOP PANEL RECORDER) ---
    def start_recording(self) -> None:
        self.raw_detections.clear()
        self.recorded_notes.clear()
        self.recording_start_time = time.time()
        self.recording = True
        
        self.btn_start_rec.state(["disabled"])
        self.btn_stop_rec.state(["!disabled"])
        self.btn_auto_tempo.state(["disabled"])
        self.btn_insert_scenario.state(["disabled"])
        
        self.rec_status_label.configure(text="Запись... Сыграйте мелодию", fg="#d32f2f")
        self.status_var.set("Запись исполнения...")

    def stop_recording(self) -> None:
        self.recording = False
        self.btn_start_rec.state(["!disabled"])
        self.btn_stop_rec.state(["disabled"])
        
        self.status_var.set("Запись завершена. Обработка...")
        
        self.recorded_notes = process_detections_to_notes(self.raw_detections)
        
        if not self.recorded_notes:
            self.rec_status_label.configure(text="Запись пуста (ноты не распознаны)", fg="#66707a")
            self.btn_auto_tempo.state(["disabled"])
            self.btn_insert_scenario.state(["disabled"])
            messagebox.showwarning("Запись пуста", "Не удалось распознать стабильные ноты. Попробуйте сыграть громче или настроить порог чувствительности (dB).")
            return
            
        durations = [d for m, d in self.recorded_notes]
        estimated_tempo = int(round(estimate_human_tempo(durations)))
        self.human_tempo_var.set(estimated_tempo)
        
        self.rec_status_label.configure(
            text=f"Записано нот: {len(self.recorded_notes)}\nОпределенный темп: {estimated_tempo} мс",
            fg="#2e7d32"
        )
        
        self.btn_auto_tempo.state(["!disabled"])
        self.btn_insert_scenario.state(["!disabled"])

    def update_recording_status(self) -> None:
        elapsed = time.time() - self.recording_start_time
        current_notes = process_detections_to_notes(self.raw_detections)
        self.rec_status_label.configure(
            text=f"Запись: {elapsed:.1f} сек\nОбнаружено нот: {len(current_notes)}",
            fg="#d32f2f"
        )

    def auto_detect_tempo(self) -> None:
        if not self.recorded_notes:
            return
        durations = [d for m, d in self.recorded_notes]
        estimated_tempo = int(round(estimate_human_tempo(durations)))
        self.human_tempo_var.set(estimated_tempo)
        self.status_var.set(f"Темп автоматически установлен в {estimated_tempo} мс")

    def apply_and_insert_scenario(self) -> None:
        if not self.recorded_notes:
            messagebox.showinfo("Нет данных", "Сначала запишите исполнение.")
            return
            
        try:
            t_human = float(self.human_tempo_entry.get())
            t_robot = float(self.robot_tempo_var.get())
        except ValueError:
            messagebox.showerror("Ошибка ввода", "Некорректные значения темпа (должны быть числами).")
            return
            
        scaled_notes = scale_and_quantize(self.recorded_notes, t_human, t_robot)
        if not scaled_notes:
            messagebox.showinfo("Пусто", "После квантования не осталось валидных нот.")
            return
            
        previous_string, next_direction = parse_last_state(self.editor.get("1.0", "insert"))
        
        warnings_notes = []
        commands = []
        
        for idx, (midi, duration) in enumerate(scaled_notes):
            fitted_midi = fit_to_violin_range(midi)
            if fitted_midi is None:
                warnings_notes.append(midi_name(midi))
                continue
                
            try:
                string, position = choose_position(fitted_midi, previous_string)
                speed = speed_for_duration(duration)
                commands.append(f"/sn;{string};{position}")
                commands.append(f"/bow;{next_direction};{duration};{speed}")
                
                previous_string = string
                next_direction = 1 - next_direction
            except ValueError:
                warnings_notes.append(midi_name(fitted_midi))
                
        if not commands:
            messagebox.showerror("Ошибка генерации", "Не удалось сгенерировать команды для записанных нот.")
            return
            
        block = "\n".join(commands) + "\n"
        
        if self.editor.compare("insert", ">=", self.footer_start_index()):
            self.editor.mark_set("insert", self.footer_start_index())
            
        if self.editor.compare("insert", "!=", "1.0"):
            previous = self.editor.get("insert -1c", "insert")
            if previous and previous != "\n":
                block = "\n" + block
                
        self.editor.edit_separator()
        self.editor.insert("insert", block)
        self.editor.edit_separator()
        self.editor.see("insert")
        
        status_msg = f"Успешно вставлено нот: {len(commands)//2}"
        if warnings_notes:
            status_msg += f" (Пропущено/не влезло: {', '.join(set(warnings_notes))})"
            messagebox.showwarning(
                "Предупреждение диапазона",
                f"Некоторые ноты ({', '.join(set(warnings_notes))}) выходят за пределы диапазона скрипки G3-B5. Они были транспонированы по октавам или quick пропущены."
            )
            
        self.status_var.set(status_msg)

    # --- KEYBOARD CANVAS ---
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
        if isinstance(event.widget, (ttk.Entry, tk.Entry, tk.Text)):
            return
        if event.widget != self.editor:
            self.root.focus_set()

    def duration_shortcut(self, event: tk.Event, duration_ms: int) -> str | None:
        if isinstance(event.widget, (ttk.Entry, tk.Entry, tk.Text)):
            return None
        if event.widget == self.editor:
            return None
        self.set_duration(duration_ms)
        self.status_var.set(f"Длина {duration_ms} мс")
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
        self.status_var.set("Вставлена нота")

    def undo(self, _event: tk.Event | None = None) -> str:
        try:
            self.editor.edit_undo()
            self.status_var.set("Отмена действия")
        except tk.TclError:
            self.status_var.set("Нечего отменять")
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
            self.status_var.set("Концовка уже есть")
            return
        while lines and lines[-1].strip().lower() in {"/end", "/play"}:
            lines.pop()
        body = "\n".join(lines).rstrip()
        final_text = f"{body}\n/end\n/play\n" if body else "/end\n/play\n"
        self.editor.edit_separator()
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", final_text)
        self.editor.edit_separator()
        self.status_var.set("Концовка добавлена")

    def toggle_line_mark(self) -> None:
        line = int(self.editor.index("insert").split(".")[0])
        start = f"{line}.0"
        end = f"{line}.end"
        if "marked" in self.editor.tag_names(start):
            self.editor.tag_remove("marked", start, end)
            self.status_var.set(f"Строка {line} размаркирована")
        else:
            self.editor.tag_add("marked", start, end)
            self.status_var.set(f"Строка {line} помечена")

    # --- FILE ACTIONS ---
    def new_file(self) -> None:
        if not self._confirm_discard():
            return
        self.path = None
        self.editor.delete("1.0", "end")
        self._insert_starter()
        self.status_var.set("Новый сценарий")

    def open_file(self) -> None:
        if not self._confirm_discard():
            return
        path = filedialog.askopenfilename(
            title="Открыть сценарий",
            filetypes=(("Файлы сценариев", "*.txt *.scenario *.scr"), ("Все файлы", "*.*")),
        )
        if not path:
            return
        self.path = Path(path)
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", self.path.read_text(encoding="utf-8"))
        self.editor.edit_reset()
        self.editor.edit_modified(False)
        self.status_var.set(f"Открыт файл: {self.path.name}")

    def save_file(self) -> None:
        self.ensure_footer()
        if self.path is None:
            default_name = safe_filename_from_title(self.editor.get("1.0", "1.end"))
            path = filedialog.asksaveasfilename(
                title="Сохранить сценарий",
                defaultextension=".txt",
                initialfile=default_name,
                filetypes=(("Файлы сценариев", "*.txt"), ("Все файлы", "*.*")),
            )
            if not path:
                return
            self.path = Path(path)
        self.path.write_text(self.editor.get("1.0", "end-1c"), encoding="utf-8")
        self.editor.edit_modified(False)
        self.status_var.set(f"Сохранено: {self.path.name}")

    # --- PLAYBACK ---
    def play_all(self) -> None:
        self.play_from_line(1)

    def play_from_cursor(self) -> None:
        self.play_from_line(int(self.editor.index("insert").split(".")[0]))

    def play_from_line(self, start_line: int) -> None:
        if self.play_thread and self.play_thread.is_alive():
            return
        events = [event for event in parse_events(self.editor.get("1.0", "end-1c")) if event.line >= start_line]
        if not events:
            messagebox.showinfo("Нечего играть", "Валидные пары нот /sn + /bow не найдены.")
            return
        self.stop_flag.clear()
        self.play_thread = threading.Thread(target=self._play_events, args=(events,), daemon=True)
        self.play_thread.start()

    def _play_events(self, events: list[PlayEvent]) -> None:
        self.root.after(0, lambda: self.status_var.set(f"Воспроизведение {len(events)} нот"))
        try:
            for event in events:
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
            self.root.after(0, lambda error=exc: messagebox.showerror("Ошибка воспроизведения", str(error)))
        finally:
            self.root.after(0, self._clear_playback)

    def _highlight_event(self, event: PlayEvent) -> None:
        self.editor.tag_remove("playing", "1.0", "end")
        start = f"{event.line}.0"
        end = f"{event.line}.end"
        self.editor.tag_add("playing", start, end)
        self.editor.see(start)
        self.status_var.set(f"Играет {event.label} ({event.duration_ms} мс)")

    def stop(self) -> None:
        self.stop_flag.set()
        sd.stop()
        self._clear_playback()

    def _clear_playback(self) -> None:
        self.editor.tag_remove("playing", "1.0", "end")
        if self.stop_flag.is_set():
            self.status_var.set("Остановлено")
        else:
            self.status_var.set("Готов")

    def _confirm_discard(self) -> bool:
        if not self.editor.edit_modified():
            return True
        return messagebox.askyesno("Сбросить изменения?", "Текущий сценарий содержит несохраненные изменения.")

    def close(self) -> None:
        if not self._confirm_discard():
            return
        self.stop()
        self.stop_detector()
        self.stop_network_server()
        self.root.destroy()

# --- SELF TEST SUITE ---
def run_recorder_tests() -> int:
    failures = 0
    # Test 1: process_detections_to_notes
    raw = [
        (0.0, 60), (0.05, 60), (0.10, 60),
        (0.15, None), # gap
        (0.20, 60), (0.25, 60),
        (0.30, 62), (0.35, 62), (0.40, 62),
        (0.45, None), (0.50, None), (0.55, None), # silence
        (0.60, 64), (0.65, 64),
    ]
    notes = process_detections_to_notes(raw, min_note_duration_ms=80)
    if len(notes) != 3:
        failures += 1
        print("Fail: expected 3 notes in process_detections_to_notes")
    else:
        if notes[0][0] != 60 or abs(notes[0][1] - 300) > 10:
            failures += 1
            print(f"Fail: note 0 duration error, got {notes[0]}")
        if notes[1][0] != 62 or abs(notes[1][1] - 150) > 10:
            failures += 1
            print(f"Fail: note 1 duration error, got {notes[1]}")
        if notes[2][0] != 64 or abs(notes[2][1] - 100) > 10:
            failures += 1
            print(f"Fail: note 2 duration error, got {notes[2]}")

    # Test 2: estimate_human_tempo
    durations = [590.0, 1180.0, 610.0, 305.0]
    tempo = estimate_human_tempo(durations)
    if abs(tempo - 600.0) > 15.0:
        failures += 1
        print(f"Fail: estimated tempo should be around 600.0, got {tempo}")

    # Test 3: scale_and_quantize
    notes_to_scale = [(60, 590.0), (62, 1180.0), (64, 305.0)]
    scaled = scale_and_quantize(notes_to_scale, tempo, t_robot=500.0)
    if scaled != [(60, 500), (62, 1000), (64, 250)]:
        failures += 1
        print(f"Fail: quantization output incorrect, got {scaled}")
        
    # Test 4: choose_position
    if choose_position(76, None) != (1, 0):
        failures += 1
        print("Fail: choose_position (76, None) should be (1, 0)")
    if choose_position(76, 2) != (2, 7):
        failures += 1
        print("Fail: choose_position (76, 2) should be (2, 7)")
        
    print(f"Recorder self-tests: {'OK' if not failures else 'FAIL'}")
    return 1 if failures else 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robot violin scenario recorder and editor.")
    parser.add_argument("--self-test", action="store_true", help="run unit tests and exit")
    return parser

def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        return run_recorder_tests()
    root = tk.Tk()
    ViolinRecorderEditorApp(root)
    root.mainloop()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())