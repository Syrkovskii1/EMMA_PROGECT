#!/usr/bin/env python3
"""
Автономный модуль записи, распознавания нот и генерации команд робота EMMA.
Полный интерфейс настройки DSP, живого тюнера, распознавания нот и генератора сценариев.

Зависимости:
    pip install numpy sounddevice

Не требует:
    - pyserial
    - socket / сеть
    - torch / transformers / RealtimeSTT
"""

from __future__ import annotations

import math
import queue
import time
import tkinter as tk
from tkinter import messagebox, ttk
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

# --- МУЗЫКАЛЬНЫЕ КОНСТАНТЫ И ХЕЛПЕРЫ ---
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
OPEN_STRINGS = {4: 55, 3: 62, 2: 69, 1: 76}  # G3, D4, A4, E5
BOW_SPEEDS = {250: 200, 500: 100, 750: 90, 1000: 80, 1250: 75, 1500: 70, 1750: 65, 2000: 60}
MAX_BOW_POS = 100000


def midi_to_name(midi: int) -> str:
    """Преобразует MIDI-номер в название ноты (например, 69 -> A4)."""
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def speed_for_duration(duration_ms: int) -> int:
    """Подбирает скорость смычка под длительность ноты."""
    closest = min(BOW_SPEEDS, key=lambda item: abs(item - duration_ms))
    return BOW_SPEEDS[closest]


def playable_positions(midi: int) -> list[tuple[int, int]]:
    """Находит варианты игры ноты на струнах скрипки (струна 1-4, позиция 0-7)."""
    choices = []
    for string, open_midi in OPEN_STRINGS.items():
        position = midi - open_midi
        if 0 <= position <= 7:
            choices.append((string, position))
    return choices


def choose_position(midi: int, previous_string: int | None) -> tuple[int, int]:
    """Выбирает оптимальную струну и позицию пальца."""
    choices = playable_positions(midi)
    if not choices:
        for alt_midi in (midi - 12, midi + 12):
            choices = playable_positions(alt_midi)
            if choices:
                break
    if not choices:
        raise ValueError(f"Нота {midi_to_name(midi)} ({midi}) вне диапазона скрипки")

    if previous_string is None:
        return min(choices, key=lambda item: (item[1], item[0]))
    return min(choices, key=lambda item: (0 if item[0] == previous_string else 10, item[1], item[0]))


def optimize_bow_directions(notes: list[tuple[int, int]], x_start: float = 0.0, initial_dir: int = 1) -> list[int]:
    """
    Планирует оптимальное распределение смычка (0 или 1) через динамическое программирование
    с контролем границ смычка [0, 100000].
    """
    n = len(notes)
    if n == 0:
        return []

    disps = [duration * speed_for_duration(duration) for _, duration in notes]
    memo = {}

    def search(idx: int, x: float, stroke_dir: int, stroke_disp: float):
        if idx == n:
            return 0.0, []

        state = (idx, int(round(x)), stroke_dir, int(round(stroke_disp)))
        if state in memo:
            return memo[state]

        d = disps[idx]
        duration = notes[idx][1]
        best_cost = float("inf")
        best_path = []

        for dir_choice in (0, 1):
            is_same = (dir_choice == stroke_dir)
            x_new = x + d if dir_choice == 1 else x - d

            overrun = max(0.0, -x_new) + max(0.0, x_new - MAX_BOW_POS)
            new_disp = stroke_disp + d if is_same else d
            stroke_overrun = max(0.0, new_disp - MAX_BOW_POS)

            cost = overrun * 2.0 + stroke_overrun * 2.0
            if not is_same and stroke_disp < 75000 and duration < 1000:
                cost += 5000
            elif is_same and duration >= 1000:
                cost += 20000

            sub_cost, sub_path = search(idx + 1, x_new, dir_choice, new_disp)
            total = cost + sub_cost
            if total < best_cost:
                best_cost = total
                best_path = [dir_choice] + sub_path

        memo[state] = (best_cost, best_path)
        return best_cost, best_path

    _, path = search(0, x_start, 1 - initial_dir, MAX_BOW_POS)
    return path


# --- ДЕТЕКТОР НОТ MPM (Чистый NumPy) ---
@dataclass
class PitchResult:
    frequency: float
    note: str
    octave: int
    cents: float
    confidence: float


class MpmPitchDetector:
    """Pitch detector using McLeod Pitch Method (MPM)."""

    def __init__(
        self,
        sample_rate: int = 48000,
        min_freq: float = 150.0,
        max_freq: float = 4000.0,
        clarity_threshold: float = 0.45,
        cutoff_k: float = 0.85,
        a4: float = 443.0,
        volume_threshold: float = 0.01,
    ):
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.clarity_threshold = clarity_threshold
        self.cutoff_k = cutoff_k
        self.a4 = a4
        self.volume_threshold = volume_threshold

    def detect(self, signal: np.ndarray) -> PitchResult | None:
        n = len(signal)
        if n < 1024:
            return None

        rms = float(np.sqrt(np.mean(signal**2)))
        if rms < self.volume_threshold:
            return None

        fft_len = 2 ** int(np.ceil(np.log2(2 * n)))
        f_sig = np.fft.rfft(signal, n=fft_len)
        autocorr = np.fft.irfft(f_sig * np.conj(f_sig))[:n]

        sq = signal**2
        sum_x_sq = np.zeros(n)
        sum_x_sq[0] = np.sum(sq)
        for tau in range(1, n):
            sum_x_sq[tau] = sum_x_sq[tau - 1] - sq[tau - 1]

        sum_xtau_sq = np.zeros(n)
        sum_xtau_sq[0] = sum_x_sq[0]
        for tau in range(1, n):
            sum_xtau_sq[tau] = sum_xtau_sq[tau - 1] - sq[n - tau]

        m = sum_x_sq + sum_xtau_sq
        m[m == 0] = 1e-12
        nsdf = 2.0 * autocorr / m

        min_lag = max(2, int(self.sample_rate / self.max_freq))
        max_lag = min(n - 2, int(self.sample_rate / self.min_freq))
        if min_lag >= max_lag:
            return None

        peaks = []
        for tau in range(min_lag, max_lag):
            if nsdf[tau] > nsdf[tau - 1] and nsdf[tau] >= nsdf[tau + 1] and nsdf[tau] > 0:
                peaks.append(tau)

        if not peaks:
            return None

        max_peak_val = max(nsdf[p] for p in peaks)
        if max_peak_val < self.clarity_threshold:
            return None

        threshold = max_peak_val * self.cutoff_k
        selected_peak = next((p for p in peaks if nsdf[p] >= threshold), None)
        if selected_peak is None:
            return None

        p = selected_peak
        alpha, beta, gamma = nsdf[p - 1], nsdf[p], nsdf[p + 1]
        denom = alpha - 2 * beta + gamma
        delta = (alpha - gamma) / (2.0 * denom) if abs(denom) > 1e-9 else 0.0

        freq = self.sample_rate / (p + delta)
        if not (self.min_freq <= freq <= self.max_freq):
            return None

        midi_double = 69.0 + 12.0 * math.log2(freq / self.a4)
        midi_int = int(round(midi_double))
        cents = (midi_double - midi_int) * 100.0

        return PitchResult(
            frequency=freq,
            note=NOTE_NAMES[midi_int % 12],
            octave=midi_int // 12 - 1,
            cents=cents,
            confidence=float(beta),
        )


# --- ПОЛНОЦЕННОЕ ГРАФИЧЕСКОЕ ПРИЛОЖЕНИЕ ТЮНЕРА И КОНВЕРТЕРА ---
class StandaloneTunerApp:
    """
    Приложение Тюнера и Конвертера аудио в сценарии EMMA.
    Может работать как самостоятельное окно или встраиваться в код робота через callback.
    """

    def __init__(self, root: tk.Tk, on_scenario_ready: Optional[Callable[[list[str]], None]] = None) -> None:
        self.root = root
        self.on_scenario_ready = on_scenario_ready

        self.root.title("EMMA: Тюнер, Запись нот и Конвертер команд (Локально)")
        self.root.geometry("700x960")
        self.root.minsize(600, 800)
        self.root.configure(bg="#f6f7f9")

        # Переменные аудио
        self.detector: MpmPitchDetector | None = None
        self.stream: sd.InputStream | None = None
        self.running = False
        self.ring = np.zeros(8192, dtype=np.float32)
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=16)

        # Переменные настроек MPM / DSP
        self.device_var = tk.StringVar()
        self.buffer_size_var = tk.IntVar(value=8192)
        self.block_size_var = tk.IntVar(value=512)
        self.clarity_var = tk.DoubleVar(value=0.45)
        self.cutoff_k_var = tk.DoubleVar(value=0.85)
        self.a4_var = tk.DoubleVar(value=443.0)
        self.min_freq_var = tk.DoubleVar(value=150.0)
        self.max_freq_var = tk.DoubleVar(value=4000.0)
        self.volume_thresh_var = tk.DoubleVar(value=0.01)

        # Фильтры стабильности ноты
        self.confirm_frames_var = tk.IntVar(value=3)
        self.smoothing_frames_var = tk.IntVar(value=3)
        self.min_duration_ms_var = tk.IntVar(value=80)

        # Авто-стоп по тишине
        self.silence_stop_sec_var = tk.DoubleVar(value=3.0)
        self.silence_start_time: float | None = None

        # Состояние сессии записи
        self.is_recording_session = False
        self.rec_status_var = tk.StringVar(value="Запись не ведется. Нажмите 'Старт записи'")
        self.recorded_events: list[tuple[int, int]] = []
        self.latest_commands: list[str] = []

        # Дисплей тюнера
        self.live_note_var = tk.StringVar(value="--")
        self.live_hz_var = tk.StringVar(value="-- Hz")
        self.dev_text_var = tk.StringVar(value="Отклонение: 0 цент.")

        # Буферы трекинга нот
        self.current_midi: int | None = None
        self.note_start_time: float = 0.0
        self.pitch_buffer: list[int] = []
        self.candidate_midi: int | None = None
        self.candidate_count: int = 0
        self.candidate_start_time: float = 0.0
        self.silent_ticks = 0

        # История ручных команд
        self.manual_cmd_var = tk.StringVar()
        self.command_history: list[str] = []
        self.history_index = -1

        self._build_styles()
        self._build_ui()

        self.populate_audio_devices()
        self.apply_audio_settings()
        self.start_detector()

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(50, self.poll_audio)

    def _build_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Root.TFrame", background="#f6f7f9")
        style.configure(
            "Primary.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(10, 6),
            background="#1c4f63",
            foreground="#ffffff",
        )
        style.configure(
            "Secondary.TButton",
            font=("Segoe UI", 10),
            padding=(8, 5),
            background="#e8edf1",
            foreground="#121417",
        )
        style.configure(
            "Record.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(10, 6),
            background="#d32f2f",
            foreground="#ffffff",
        )
        style.configure(
            "Convert.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(10, 6),
            background="#2e7d32",
            foreground="#ffffff",
        )

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.root, style="Root.TFrame", padding=15)
        main_frame.pack(fill="both", expand=True)

        # 1. Панель параметров DSP
        settings_frame = ttk.LabelFrame(main_frame, text=" Настройки DSP и фильтров ", padding=10)
        settings_frame.pack(fill="x", pady=(0, 10))
        settings_frame.columnconfigure(1, weight=1)
        settings_frame.columnconfigure(3, weight=1)

        ttk.Label(settings_frame, text="Вход:").grid(row=0, column=0, sticky="w", pady=2)
        self.device_combo = ttk.Combobox(
            settings_frame, textvariable=self.device_var, state="readonly", font=("Segoe UI", 9)
        )
        self.device_combo.grid(row=0, column=1, columnspan=3, sticky="ew", pady=2)

        ttk.Label(settings_frame, text="Окно (samples):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Combobox(
            settings_frame,
            textvariable=self.buffer_size_var,
            values=[2048, 4096, 8192, 16384],
            state="readonly",
            width=8,
        ).grid(row=1, column=1, sticky="w", pady=2)

        ttk.Label(settings_frame, text="Чанк (block):").grid(row=1, column=2, sticky="w", pady=2, padx=(10, 0))
        ttk.Combobox(
            settings_frame,
            textvariable=self.block_size_var,
            values=[256, 512, 1024],
            state="readonly",
            width=8,
        ).grid(row=1, column=3, sticky="w", pady=2)

        ttk.Label(settings_frame, text="Порог (Clarity):").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Spinbox(
            settings_frame, from_=0.1, to=0.9, increment=0.05, textvariable=self.clarity_var, width=8
        ).grid(row=2, column=1, sticky="w", pady=2)

        ttk.Label(settings_frame, text="Коэф. октавы K:").grid(row=2, column=2, sticky="w", pady=2, padx=(10, 0))
        ttk.Spinbox(
            settings_frame, from_=0.5, to=0.98, increment=0.02, textvariable=self.cutoff_k_var, width=8
        ).grid(row=2, column=3, sticky="w", pady=2)

        ttk.Label(settings_frame, text="Стабильность (кадр):").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Spinbox(
            settings_frame, from_=1, to=10, increment=1, textvariable=self.confirm_frames_var, width=8
        ).grid(row=3, column=1, sticky="w", pady=2)

        ttk.Label(settings_frame, text="Мин. время (мс):").grid(row=3, column=2, sticky="w", pady=2, padx=(10, 0))
        ttk.Spinbox(
            settings_frame, from_=20, to=500, increment=20, textvariable=self.min_duration_ms_var, width=8
        ).grid(row=3, column=3, sticky="w", pady=2)

        ttk.Label(settings_frame, text="Авто-стоп тишины (с):").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Spinbox(
            settings_frame, from_=1.0, to=10.0, increment=0.5, textvariable=self.silence_stop_sec_var, width=8
        ).grid(row=4, column=1, sticky="w", pady=2)

        ttk.Label(settings_frame, text="A4 (Гц):").grid(row=4, column=2, sticky="w", pady=2, padx=(10, 0))
        ttk.Spinbox(
            settings_frame, from_=430.0, to=450.0, increment=0.5, textvariable=self.a4_var, width=8
        ).grid(row=4, column=3, sticky="w", pady=2)

        ttk.Label(settings_frame, text="Мин. громкость (RMS):").grid(row=5, column=0, sticky="w", pady=2)
        ttk.Spinbox(
            settings_frame, from_=0.001, to=0.5, increment=0.005, textvariable=self.volume_thresh_var, width=8
        ).grid(row=5, column=1, sticky="w", pady=2)

        ttk.Button(
            settings_frame,
            text="Применить настройки",
            command=self.apply_audio_settings,
            style="Secondary.TButton",
        ).grid(row=6, column=0, columnspan=4, sticky="ew", pady=(8, 2))

        # 2. Дисплей тюнера в реальном времени
        tuner_frame = ttk.LabelFrame(main_frame, text=" Живой Тюнер ", padding=15)
        tuner_frame.pack(fill="x", pady=(0, 10))

        self.lbl_note = tk.Label(
            tuner_frame, textvariable=self.live_note_var, font=("Segoe UI", 48, "bold"), fg="#1c4f63", bg="#ffffff"
        )
        self.lbl_note.pack()

        self.lbl_hz = tk.Label(
            tuner_frame, textvariable=self.live_hz_var, font=("Segoe UI", 14), fg="#3b4148", bg="#ffffff"
        )
        self.lbl_hz.pack()

        self.lbl_dev = ttk.Label(tuner_frame, textvariable=self.dev_text_var, font=("Segoe UI", 10, "bold"))
        self.lbl_dev.pack(pady=(5, 5))

        self.dev_bar = ttk.Progressbar(tuner_frame, orient="horizontal", length=300, mode="determinate")
        self.dev_bar.pack(pady=5)
        self.dev_bar["value"] = 50

        # 3. Окно записи с кнопками управления сессией и генератором
        log_frame = ttk.LabelFrame(main_frame, text=" Запись и генератор команд EMMA ", padding=10)
        log_frame.pack(fill="both", expand=True, pady=(0, 10))

        # Верхний тулбар записи
        rec_ctrl_bar = ttk.Frame(log_frame)
        rec_ctrl_bar.pack(fill="x", pady=(0, 8))

        self.btn_start_rec = ttk.Button(
            rec_ctrl_bar, text="● Старт записи", command=self.start_recording, style="Record.TButton"
        )
        self.btn_start_rec.pack(side="left", padx=(0, 4))

        self.btn_stop_rec = ttk.Button(
            rec_ctrl_bar,
            text="■ Стоп",
            command=lambda: self.stop_recording_session("Ручная остановка"),
            style="Secondary.TButton",
        )
        self.btn_stop_rec.pack(side="left", padx=(0, 6))
        self.btn_stop_rec.state(["disabled"])

        self.btn_convert = ttk.Button(
            rec_ctrl_bar, text="⚡ Сгенерировать сценарий", command=self.convert_to_scenario, style="Convert.TButton"
        )
        self.btn_convert.pack(side="left", padx=(6, 0))

        ttk.Button(
            rec_ctrl_bar, text="📋 Копировать команды", command=self.copy_commands, style="Secondary.TButton"
        ).pack(side="right", padx=(4, 0))

        ttk.Button(rec_ctrl_bar, text="Очистить", command=self.clear_log, style="Secondary.TButton").pack(
            side="right", padx=(0, 4)
        )

        # Строка ручного ввода команд
        manual_frame = ttk.Frame(log_frame)
        manual_frame.pack(fill="x", pady=(0, 8))

        ttk.Label(manual_frame, text="Ручная команда:", background="#ffffff").pack(side="left", padx=(0, 5))
        self.cmd_entry = ttk.Entry(manual_frame, textvariable=self.manual_cmd_var, font=("Cascadia Mono", 10))
        self.cmd_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        ttk.Button(
            manual_frame, text="Добавить", command=self.add_manual_command, style="Secondary.TButton"
        ).pack(side="right")

        self.cmd_entry.bind("<Return>", lambda e: self.add_manual_command())
        self.cmd_entry.bind("<Up>", self._on_up_arrow)
        self.cmd_entry.bind("<Down>", self._on_down_arrow)

        # Индикатор статуса записи
        self.status_label = tk.Label(
            log_frame,
            textvariable=self.rec_status_var,
            font=("Segoe UI", 10, "bold"),
            fg="#66707a",
            bg="#ffffff",
            anchor="w",
            padx=5,
            pady=5,
        )
        self.status_label.pack(fill="x", pady=(0, 5))

        # Текстовые поля: слева ноты, справа команды
        text_container = ttk.PanedWindow(log_frame, orient="horizontal")
        text_container.pack(fill="both", expand=True)

        # Левая панель: ноты
        notes_frame = ttk.Frame(text_container)
        text_container.add(notes_frame, weight=1)
        ttk.Label(notes_frame, text="Записанные ноты:", background="#ffffff").pack(anchor="w")

        self.notes_text = tk.Text(
            notes_frame,
            bg="#fbfcfd",
            fg="#121417",
            font=("Cascadia Mono", 10),
            relief="flat",
            padx=8,
            pady=8,
        )
        self.notes_text.pack(side="left", fill="both", expand=True)
        notes_scroll = ttk.Scrollbar(notes_frame, orient="vertical", command=self.notes_text.yview)
        notes_scroll.pack(side="right", fill="y")
        self.notes_text.configure(yscrollcommand=notes_scroll.set)

        # Правая панель: команды EMMA
        commands_frame = ttk.Frame(text_container)
        text_container.add(commands_frame, weight=1)
        ttk.Label(commands_frame, text="Команды робота EMMA:", background="#ffffff").pack(anchor="w")

        self.commands_text = tk.Text(
            commands_frame,
            bg="#fbfcfd",
            fg="#121417",
            font=("Cascadia Mono", 10),
            relief="flat",
            padx=8,
            pady=8,
        )
        self.commands_text.pack(side="left", fill="both", expand=True)
        commands_scroll = ttk.Scrollbar(commands_frame, orient="vertical", command=self.commands_text.yview)
        commands_scroll.pack(side="right", fill="y")
        self.commands_text.configure(yscrollcommand=commands_scroll.set)

    def _on_up_arrow(self, event) -> None:
        if self.command_history and self.history_index > 0:
            self.history_index -= 1
            self.manual_cmd_var.set(self.command_history[self.history_index])
            self.cmd_entry.icursor(tk.END)

    def _on_down_arrow(self, event) -> None:
        if self.command_history and self.history_index < len(self.command_history) - 1:
            self.history_index += 1
            self.manual_cmd_var.set(self.command_history[self.history_index])
            self.cmd_entry.icursor(tk.END)
        elif self.history_index == len(self.command_history) - 1:
            self.history_index += 1
            self.manual_cmd_var.set("")

    def add_manual_command(self) -> None:
        cmd = self.manual_cmd_var.get().strip()
        if not cmd:
            return

        if not self.command_history or self.command_history[-1] != cmd:
            self.command_history.append(cmd)

        self.history_index = len(self.command_history)
        self.manual_cmd_var.set("")

        self.commands_text.insert(tk.END, f"{cmd}\n")
        self.commands_text.see(tk.END)

    def copy_commands(self) -> None:
        text = self.commands_text.get(1.0, tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.rec_status_var.set("📋 Команды скопированы в буфер обмена!")

    def populate_audio_devices(self) -> None:
        try:
            devices = sd.query_devices()
            input_device_strs = []
            default_idx = sd.default.device[0]
            default_str = ""
            for idx, dev in enumerate(devices):
                if dev["max_input_channels"] > 0:
                    label = f"{idx}: {dev['name']}"
                    input_device_strs.append(label)
                    if idx == default_idx:
                        default_str = label
            self.device_combo["values"] = input_device_strs
            if default_str:
                self.device_combo.set(default_str)
            elif input_device_strs:
                self.device_combo.set(input_device_strs[0])
        except Exception as exc:
            print(f"Ошибка получения аудиоустройств: {exc}")

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
                sample_rate = int(device_info["default_samplerate"])

            self.detector = MpmPitchDetector(
                sample_rate=sample_rate,
                min_freq=self.min_freq_var.get(),
                max_freq=self.max_freq_var.get(),
                clarity_threshold=self.clarity_var.get(),
                cutoff_k=self.cutoff_k_var.get(),
                a4=self.a4_var.get(),
                volume_threshold=self.volume_thresh_var.get(),
            )
            self.ring = np.zeros(self.buffer_size_var.get(), dtype=np.float32)
            self.audio_queue = queue.Queue(maxsize=16)
            self.sample_rate = sample_rate
            self.device_id = device_id

            self._reset_recording_buffers()

            if was_running:
                self.start_detector()
        except Exception as exc:
            messagebox.showerror("Ошибка аудио", f"Не удалось настроить аудио:\n{exc}")

    def _reset_recording_buffers(self) -> None:
        self.pitch_buffer.clear()
        self.candidate_midi = None
        self.candidate_count = 0
        self.current_midi = None
        self.silence_start_time = None

    def start_recording(self) -> None:
        self._reset_recording_buffers()
        self.recorded_events.clear()
        self.is_recording_session = True
        self.silence_start_time = time.time()

        self.btn_start_rec.state(["disabled"])
        self.btn_stop_rec.state(["!disabled"])

        self.status_label.configure(fg="#d32f2f")
        self.rec_status_var.set("🔴 ИДЕТ ЗАПИСЬ | Играйте! (Авто-стоп при тишине)")

    def stop_recording_session(self, reason: str = "Остановлено") -> None:
        if not self.is_recording_session:
            return

        self._finalize_current_note(time.time())
        self.is_recording_session = False
        self.silence_start_time = None

        self.btn_start_rec.state(["!disabled"])
        self.btn_stop_rec.state(["disabled"])
        self.status_label.configure(fg="#66707a")

        # Автоматическая генерация сценария
        self.convert_to_scenario()

    def convert_to_scenario(self) -> list[str]:
        """
        Преобразует список записанных нот в сценарий команд EMMA.
        Выводит результат в текстовое поле и вызывает колбэк on_scenario_ready (если задан).
        """
        if not self.recorded_events:
            self.rec_status_var.set("ℹ️ Запись остановлена (ноты не обнаружены).")
            return []

        lines = [
            "Recorded Scenario",
            "/load_test",
            "/dyn_press;1",
            "/bow_p;1.1",
        ]

        valid_notes = []
        warnings = []
        prev_string = None

        for midi_val, duration_ms in self.recorded_events:
            try:
                string, position = choose_position(midi_val, prev_string)
                valid_notes.append((midi_val, duration_ms, string, position))
                prev_string = string
            except ValueError:
                warnings.append(midi_to_name(midi_val))

        if not valid_notes:
            self.rec_status_var.set("ℹ️ Нет валидных нот в диапазоне скрипки.")
            return []

        notes_for_planning = [(n[0], n[1]) for n in valid_notes]
        planned_dirs = optimize_bow_directions(notes_for_planning, x_start=0.0, initial_dir=1)

        for idx, (_, duration_ms, string, position) in enumerate(valid_notes):
            speed = speed_for_duration(duration_ms)
            direction = planned_dirs[idx] if idx < len(planned_dirs) else 1
            lines.append(f"/sn;{string};{position}")
            lines.append(f"/bow;{direction};{duration_ms};{speed}")

        lines.extend([
            "/sn;2;0",
            "/end",
            "/play",
        ])

        scenario_text = "\n".join(lines) + "\n"
        self.commands_text.delete(1.0, tk.END)
        self.commands_text.insert(1.0, scenario_text)
        self.commands_text.see(1.0)

        self.latest_commands = lines
        status_msg = f"✅ Сценарий готов: {len(valid_notes)} нот, {len(lines)} команд."
        if warnings:
            status_msg += f" (Пропущены: {', '.join(set(warnings))})"
        self.rec_status_var.set(status_msg)

        # Вызов колбэка для кода робота
        if self.on_scenario_ready is not None:
            self.on_scenario_ready(lines)

        return lines

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            pass
        try:
            self.audio_queue.put_nowait(indata.copy())
        except queue.Full:
            pass

    def start_detector(self) -> None:
        if self.detector is None:
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
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось открыть микрофон: {exc}")

    def stop_detector(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.running = False

    def poll_audio(self) -> None:
        processed = False
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
                    self.dev_text_var.set("Отклонение: 0 цент.")
                    self.dev_bar["value"] = 50

                    if self.is_recording_session:
                        self._finalize_current_note(now)
                        self.candidate_midi = None
                        self.candidate_count = 0
                        self.pitch_buffer.clear()

                # Авто-стоп по тишине
                if self.is_recording_session:
                    if self.silence_start_time is None:
                        self.silence_start_time = now

                    silence_elapsed = now - self.silence_start_time
                    target_silence = self.silence_stop_sec_var.get()
                    self.rec_status_var.set(
                        f"🔴 ИДЕТ ЗАПИСЬ | Тишина: {silence_elapsed:.1f} / {target_silence:.1f} сек"
                    )

                    if silence_elapsed >= target_silence:
                        self.stop_recording_session(f"Авто-стоп: {target_silence:.1f} сек тишины")
            else:
                self.silent_ticks = 0
                self.live_note_var.set(f"{result.note}{result.octave}")
                self.live_hz_var.set(f"{result.frequency:.2f} Hz")

                sign = "+" if result.cents >= 0 else ""
                self.dev_text_var.set(f"Отклонение: {sign}{int(result.cents)} цент.")
                self.dev_bar["value"] = int(np.clip(result.cents + 50, 0, 100))

                if self.is_recording_session:
                    self.silence_start_time = None
                    self.rec_status_var.set("🔴 ИДЕТ ЗАПИСЬ | Зафиксирован звук...")

                    raw_midi = int(round(69.0 + 12.0 * math.log2(result.frequency / self.detector.a4)))

                    self.pitch_buffer.append(raw_midi)
                    max_buf = self.smoothing_frames_var.get()
                    if len(self.pitch_buffer) > max_buf:
                        self.pitch_buffer.pop(0)

                    stable_midi = int(np.median(self.pitch_buffer))

                    if stable_midi == self.candidate_midi:
                        self.candidate_count += 1
                    else:
                        self.candidate_midi = stable_midi
                        self.candidate_count = 1
                        self.candidate_start_time = now

                    if self.candidate_count >= self.confirm_frames_var.get():
                        if self.current_midi != self.candidate_midi:
                            self._finalize_current_note(self.candidate_start_time)
                            self.current_midi = self.candidate_midi
                            self.note_start_time = self.candidate_start_time

        self.root.after(50, self.poll_audio)

    def _finalize_current_note(self, end_time: float) -> None:
        if self.current_midi is not None:
            duration_ms = int((end_time - self.note_start_time) * 1000)
            if duration_ms >= self.min_duration_ms_var.get():
                note_name = midi_to_name(self.current_midi)
                line = f"Нота: {note_name:<5} | Длительность: {duration_ms} мс\n"
                self.notes_text.insert(tk.END, line)
                self.notes_text.see(tk.END)
                self.recorded_events.append((self.current_midi, duration_ms))

            self.current_midi = None

    def clear_log(self) -> None:
        self.notes_text.delete(1.0, tk.END)
        self.commands_text.delete(1.0, tk.END)
        self.recorded_events.clear()
        self.latest_commands.clear()
        self.command_history.clear()
        self.history_index = -1
        self.manual_cmd_var.set("")
        self.rec_status_var.set("Очищено.")

    def close(self) -> None:
        self.stop_detector()
        self.root.destroy()


def launch_app(on_scenario_ready: Optional[Callable[[list[str]], None]] = None) -> StandaloneTunerApp:
    """Запуск приложения с возможностью передать функцию-колбэк отправки на робота."""
    root = tk.Tk()
    app = StandaloneTunerApp(root, on_scenario_ready=on_scenario_ready)
    root.mainloop()
    return app


if __name__ == "__main__":
    launch_app()
