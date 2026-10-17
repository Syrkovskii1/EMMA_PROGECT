#!/usr/bin/env python3
"""Standalone Real-time Violin Tuner using McLeod Pitch Method (MPM) with Auto-Stop and EMMA Scenario Converter."""

import math
import queue
import time
import socket
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd

# --- МУЗЫКАЛЬНЫЕ КОНСТАНТЫ И ХЕЛПЕРЫ ---
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
OPEN_STRINGS = {4: 55, 3: 62, 2: 69, 1: 76}  # G3, D4, A4, E5
BOW_SPEEDS = {250: 200, 500: 100, 750: 90, 1000: 80, 1250: 75, 1500: 70, 1750: 65, 2000: 60}

def midi_to_name(midi: int) -> str:
    """Преобразует номер MIDI ноты в текстовое название (например, 69 -> A4)."""
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"

def speed_for_duration(duration_ms: int) -> int:
    """Подбирает скорость смычка под длительность."""
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
        # Если нота выше диапазона, пытаемся транспонировать
        for alt_midi in (midi - 12, midi + 12):
            choices = playable_positions(alt_midi)
            if choices:
                break
    if not choices:
        raise ValueError(f"Нота {midi_to_name(midi)} вне диапазона скрипки")

    if previous_string is None:
        return min(choices, key=lambda item: (item[1], item[0]))
    # Приоритет отдаем текущей струне
    return min(choices, key=lambda item: (0 if item[0] == previous_string else 10, item[1], item[0]))

def optimize_bow_directions(notes: list[tuple[int, int]], X_start: float, initial_dir: int) -> list[int]:
    """Планирует оптимальное распределение смычка (0 или 1) с помощью динамического программирования."""
    n = len(notes)
    if n == 0:
        return []
        
    # Предварительный расчет смещений
    disps = []
    for midi, duration in notes:
        speed = speed_for_duration(duration)
        disps.append(duration * speed)
        
    # --- ЭТАП 1: СТРОГИЙ ПОИСК ---
    memo_strict = {}
    def search_strict(idx, X, stroke_dir, stroke_disp):
        if idx == n:
            return 0, []
            
        state = (idx, int(round(X)), stroke_dir, int(round(stroke_disp)))
        if state in memo_strict:
            return memo_strict[state]
            
        d = disps[idx]
        duration = notes[idx][1]
        
        best_cost = float('inf')
        best_path = []
        
        for dir_choice in (0, 1):
            is_same = (dir_choice == stroke_dir)
            if dir_choice == 1:
                X_new = X + d
            else:
                X_new = X - d
                
            if X_new < -0.1 or X_new > 100000.1:
                continue
                
            if is_same:
                new_disp = stroke_disp + d
            else:
                new_disp = d
                
            if is_same and new_disp > 100000.1:
                continue
            if is_same and duration >= 1000:
                continue
                
            cost = 0
            if not is_same:
                if stroke_disp < 75000 and duration < 1000:
                    cost += 10
            else:
                if duration >= 750:
                    cost += 5
                    
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
        
    # --- ЭТАП 2: МЯГКИЙ ПОИСК (ЗАПАСНОЙ) ---
    memo_soft = {}
    def search_soft(idx, X, stroke_dir, stroke_disp):
        if idx == n:
            return 0, []
            
        X_q = int(round(X / 5000)) * 5000
        disp_q = int(round(stroke_disp / 5000)) * 5000
        state = (idx, X_q, stroke_dir, disp_q)
        
        if state in memo_soft:
            return memo_soft[state]
            
        d = disps[idx]
        duration = notes[idx][1]
        
        best_cost = float('inf')
        best_path = []
        
        for dir_choice in (0, 1):
            is_same = (dir_choice == stroke_dir)
            if dir_choice == 1:
                X_new = X + d
            else:
                X_new = X - d
                
            if is_same:
                new_disp = stroke_disp + d
            else:
                new_disp = d
                
            cost = 0
            if not is_same:
                if stroke_disp < 75000 and duration < 1000:
                    cost += 10000
            else:
                if duration >= 750:
                    cost += 5000
                if duration >= 1000:
                    cost += 50000
                    
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
        a4: float = 443.0
    ):
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.clarity_threshold = clarity_threshold
        self.cutoff_k = cutoff_k
        self.a4 = a4

    def detect(self, signal: np.ndarray) -> PitchResult | None:
        n = len(signal)
        if n < 1024:
            return None

        # 1. Автокорреляция через FFT
        fft_len = 2 ** int(np.ceil(np.log2(2 * n)))
        f_sig = np.fft.rfft(signal, n=fft_len)
        autocorr = np.fft.irfft(f_sig * np.conj(f_sig))[:n]

        # 2. Вычисление знаменателя m(tau) для NSDF
        sq = signal ** 2
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

        # 3. Границы поиска периода
        min_lag = max(2, int(self.sample_rate / self.max_freq))
        max_lag = min(n - 2, int(self.sample_rate / self.min_freq))

        if min_lag >= max_lag:
            return None

        # 4. Поиск локальных пиков
        peaks = []
        for tau in range(min_lag, max_lag):
            if nsdf[tau] > nsdf[tau - 1] and nsdf[tau] >= nsdf[tau + 1]:
                if nsdf[tau] > 0:
                    peaks.append(tau)

        if not peaks:
            return None

        # 5. Выбор пика Маклауда
        max_peak_val = max(nsdf[p] for p in peaks)
        if max_peak_val < self.clarity_threshold:
            return None

        threshold = max_peak_val * self.cutoff_k
        selected_peak = None
        for p in peaks:
            if nsdf[p] >= threshold:
                selected_peak = p
                break

        if selected_peak is None:
            return None

        # 6. Параболическая интерполяция
        p = selected_peak
        alpha, beta, gamma = nsdf[p - 1], nsdf[p], nsdf[p + 1]
        denom = alpha - 2 * beta + gamma
        delta = (alpha - gamma) / (2.0 * denom) if abs(denom) > 1e-9 else 0.0

        refined_lag = p + delta
        freq = self.sample_rate / refined_lag

        if not (self.min_freq <= freq <= self.max_freq):
            return None

        midi_double = 69.0 + 12.0 * math.log2(freq / self.a4)
        midi_int = int(round(midi_double))
        cents = (midi_double - midi_int) * 100.0

        note_str = NOTE_NAMES[midi_int % 12]
        octave = midi_int // 12 - 1

        return PitchResult(
            frequency=freq,
            note=note_str,
            octave=octave,
            cents=cents,
            confidence=float(beta)
        )


class StandaloneTunerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Тюнер EMMA (Авто-генерация и ручное управление)")
        self.root.geometry("680x960")
        self.root.minsize(580, 780)
        self.root.configure(bg="#f6f7f9")

        # Переменные аудио
        self.detector: MpmPitchDetector | None = None
        self.stream: sd.InputStream | None = None
        self.running = False
        self.ring = np.zeros(8192, dtype=np.float32)
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=16)

        # Переменные настроек MPM
        self.device_var = tk.StringVar()
        self.buffer_size_var = tk.IntVar(value=8192)
        self.block_size_var = tk.IntVar(value=512)
        self.clarity_var = tk.DoubleVar(value=0.45)
        self.cutoff_k_var = tk.DoubleVar(value=0.85)
        self.a4_var = tk.DoubleVar(value=443.0)
        self.min_freq_var = tk.DoubleVar(value=150.0)
        self.max_freq_var = tk.DoubleVar(value=4000.0)

        # Сегментация нот (офлайн-анализ записанного фрагмента целиком)
        self.min_bow_gap_ms_var = tk.IntVar(value=20)   # мин. пауза между сменами смычка (Снижено для лучшего отклика)
        self.min_duration_ms_var = tk.IntVar(value=60)  # мин. длительность сегмента звука 

        # Авто-стоп по тишине (завершает ВСЮ сессию записи, не отдельную ноту)
        self.silence_stop_sec_var = tk.DoubleVar(value=3.0)
        self.silence_start_time: float | None = None

        # Состояние сессии записи
        self.is_recording_session = False
        self.rec_status_var = tk.StringVar(value="Запись не ведется. Нажмите 'Старт записи'")
        self.recorded_events: list[tuple[int, int]] = []

        # Сырой звук всей сессии — копится во время записи, разбирается на ноты после стопа
        self.session_audio_chunks: list[np.ndarray] = []
        self.session_sample_rate: int = 48000

        # Дисплей тюнера
        self.live_note_var = tk.StringVar(value="--")
        self.live_hz_var = tk.StringVar(value="-- Hz")
        self.dev_text_var = tk.StringVar(value="Отклонение: 0 цент.")
        
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
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 6), background="#1c4f63", foreground="#ffffff")
        style.configure("Secondary.TButton", font=("Segoe UI", 10), padding=(8, 5), background="#e8edf1", foreground="#121417")
        style.configure("Record.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 6), background="#d32f2f", foreground="#ffffff")
        style.configure("Convert.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 6), background="#2e7d32", foreground="#ffffff")

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.root, style="Root.TFrame", padding=15)
        main_frame.pack(fill="both", expand=True)

        # 1. Панель параметров DSP
        settings_frame = ttk.LabelFrame(main_frame, text=" Настройки DSP и фильтров ", padding=10)
        settings_frame.pack(fill="x", pady=(0, 10))
        settings_frame.columnconfigure(1, weight=1)
        settings_frame.columnconfigure(3, weight=1)

        ttk.Label(settings_frame, text="Вход:").grid(row=0, column=0, sticky="w", pady=2)
        self.device_combo = ttk.Combobox(settings_frame, textvariable=self.device_var, state="readonly", font=("Segoe UI", 9))
        self.device_combo.grid(row=0, column=1, columnspan=3, sticky="ew", pady=2)

        ttk.Label(settings_frame, text="Окно (samples):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Combobox(settings_frame, textvariable=self.buffer_size_var, values=[2048, 4096, 8192, 16384], state="readonly", width=8).grid(row=1, column=1, sticky="w", pady=2)

        ttk.Label(settings_frame, text="Чанк (block):").grid(row=1, column=2, sticky="w", pady=2, padx=(10, 0))
        ttk.Combobox(settings_frame, textvariable=self.block_size_var, values=[256, 512, 1024], state="readonly", width=8).grid(row=1, column=3, sticky="w", pady=2)

        ttk.Label(settings_frame, text="Порог (Clarity):").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Spinbox(settings_frame, from_=0.1, to=0.9, increment=0.05, textvariable=self.clarity_var, width=8).grid(row=2, column=1, sticky="w", pady=2)

        ttk.Label(settings_frame, text="Коэф. октавы K:").grid(row=2, column=2, sticky="w", pady=2, padx=(10, 0))
        ttk.Spinbox(settings_frame, from_=0.5, to=0.98, increment=0.02, textvariable=self.cutoff_k_var, width=8).grid(row=2, column=3, sticky="w", pady=2)

        ttk.Label(settings_frame, text="Пауза смычка (мс):").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Spinbox(settings_frame, from_=10, to=200, increment=5, textvariable=self.min_bow_gap_ms_var, width=8).grid(row=3, column=1, sticky="w", pady=2)

        ttk.Label(settings_frame, text="Мин. время ноты (мс):").grid(row=3, column=2, sticky="w", pady=2, padx=(10, 0))
        ttk.Spinbox(settings_frame, from_=20, to=500, increment=20, textvariable=self.min_duration_ms_var, width=8).grid(row=3, column=3, sticky="w", pady=2)

        ttk.Label(settings_frame, text="Авто-стоп тишины (с):").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Spinbox(settings_frame, from_=1.0, to=10.0, increment=0.5, textvariable=self.silence_stop_sec_var, width=8).grid(row=4, column=1, sticky="w", pady=2)

        ttk.Label(settings_frame, text="A4 (Гц):").grid(row=4, column=2, sticky="w", pady=2, padx=(10, 0))
        ttk.Spinbox(settings_frame, from_=430.0, to=450.0, increment=0.5, textvariable=self.a4_var, width=8).grid(row=4, column=3, sticky="w", pady=2)

        ttk.Button(settings_frame, text="Применить настройки", command=self.apply_audio_settings, style="Secondary.TButton").grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 2))

        # 2. Дисплей тюнера
        tuner_frame = ttk.LabelFrame(main_frame, text=" Тюнер ", padding=15)
        tuner_frame.pack(fill="x", pady=(0, 10))

        self.lbl_note = tk.Label(tuner_frame, textvariable=self.live_note_var, font=("Segoe UI", 48, "bold"), fg="#1c4f63", bg="#ffffff")
        self.lbl_note.pack()

        self.lbl_hz = tk.Label(tuner_frame, textvariable=self.live_hz_var, font=("Segoe UI", 14), fg="#3b4148", bg="#ffffff")
        self.lbl_hz.pack()

        self.lbl_dev = ttk.Label(tuner_frame, textvariable=self.dev_text_var, font=("Segoe UI", 10, "bold"))
        self.lbl_dev.pack(pady=(5, 5))

        self.dev_bar = ttk.Progressbar(tuner_frame, orient="horizontal", length=250, mode="determinate")
        self.dev_bar.pack(pady=5)
        self.dev_bar['value'] = 50

        # 3. Окно записи с кнопками управления сессией и генератором
        log_frame = ttk.LabelFrame(main_frame, text=" Запись и генератор сценария ", padding=10)
        log_frame.pack(fill="both", expand=True, pady=(0, 10))

        # Верхний тулбар записи
        rec_ctrl_bar = ttk.Frame(log_frame)
        rec_ctrl_bar.pack(fill="x", pady=(0, 8))

        self.btn_start_rec = ttk.Button(rec_ctrl_bar, text="● Старт записи", command=self.start_recording, style="Record.TButton")
        self.btn_start_rec.pack(side="left", padx=(0, 4))

        self.btn_stop_rec = ttk.Button(rec_ctrl_bar, text="■ Стоп", command=lambda: self.stop_recording_session("Ручная остановка"), style="Secondary.TButton")
        self.btn_stop_rec.pack(side="left", padx=(0, 6))
        self.btn_stop_rec.state(["disabled"])

        self.btn_convert = ttk.Button(rec_ctrl_bar, text="⚡ Отправить повторно", command=self.convert_to_scenario, style="Convert.TButton")
        self.btn_convert.pack(side="left", padx=(6, 0))

        ttk.Label(rec_ctrl_bar, text="Сеть (IP:Port):", background="#ffffff").pack(side="left", padx=(10, 2))
        self.network_target_var = tk.StringVar(value="172.27.101.1:9000")
        ttk.Entry(rec_ctrl_bar, textvariable=self.network_target_var, width=15).pack(side="left")

        ttk.Button(rec_ctrl_bar, text="Очистить", command=self.clear_log, style="Secondary.TButton").pack(side="right", padx=(0, 4))
        
        # --- НОВАЯ СТРОКА РУЧНОГО ВВОДА КОМАНД ---
        manual_frame = ttk.Frame(log_frame)
        manual_frame.pack(fill="x", pady=(0, 8))
        
        ttk.Label(manual_frame, text="Ручная команда:", background="#ffffff").pack(side="left", padx=(0, 5))
        self.cmd_entry = ttk.Entry(manual_frame, textvariable=self.manual_cmd_var, font=("Cascadia Mono", 10))
        self.cmd_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        ttk.Button(manual_frame, text="➤ Отправить", command=self.send_manual_command, style="Secondary.TButton").pack(side="right")
        
        # Бинды для ручной строки
        self.cmd_entry.bind("<Return>", lambda e: self.send_manual_command())
        self.cmd_entry.bind("<Up>", self._on_up_arrow)
        self.cmd_entry.bind("<Down>", self._on_down_arrow)

        # Индикатор статуса записи
        self.status_label = tk.Label(log_frame, textvariable=self.rec_status_var, font=("Segoe UI", 10, "bold"), fg="#66707a", bg="#ffffff", anchor="w", padx=5, pady=5)
        self.status_label.pack(fill="x", pady=(0, 5))

        # Текстовое поле для вывода нот и итогового сценария
        text_container = ttk.PanedWindow(log_frame, orient="horizontal")
        text_container.pack(fill="both", expand=True)

        # Левая панель: ноты
        notes_frame = ttk.Frame(text_container)
        text_container.add(notes_frame, weight=1)
        ttk.Label(notes_frame, text="Записанные ноты:", background="#ffffff").pack(anchor="w")
        
        self.notes_text = tk.Text(notes_frame, bg="#fbfcfd", fg="#121417", font=("Cascadia Mono", 11), relief="flat", padx=10, pady=10)
        self.notes_text.pack(side="left", fill="both", expand=True)
        notes_scroll = ttk.Scrollbar(notes_frame, orient="vertical", command=self.notes_text.yview)
        notes_scroll.pack(side="right", fill="y")
        self.notes_text.configure(yscrollcommand=notes_scroll.set)

        # Правая панель: команды
        commands_frame = ttk.Frame(text_container)
        text_container.add(commands_frame, weight=1)
        ttk.Label(commands_frame, text="Команды EMMA:", background="#ffffff").pack(anchor="w")

        self.commands_text = tk.Text(commands_frame, bg="#fbfcfd", fg="#121417", font=("Cascadia Mono", 11), relief="flat", padx=10, pady=10)
        self.commands_text.pack(side="left", fill="both", expand=True)
        commands_scroll = ttk.Scrollbar(commands_frame, orient="vertical", command=self.commands_text.yview)
        commands_scroll.pack(side="right", fill="y")
        self.commands_text.configure(yscrollcommand=commands_scroll.set)

    def _on_up_arrow(self, event) -> None:
        """Перемещение назад по истории команд (старые команды)."""
        if self.command_history and self.history_index > 0:
            self.history_index -= 1
            self.manual_cmd_var.set(self.command_history[self.history_index])
            self.cmd_entry.icursor(tk.END)

    def _on_down_arrow(self, event) -> None:
        """Перемещение вперед по истории команд (к новым или пустому полю)."""
        if self.command_history and self.history_index < len(self.command_history) - 1:
            self.history_index += 1
            self.manual_cmd_var.set(self.command_history[self.history_index])
            self.cmd_entry.icursor(tk.END)
        elif self.history_index == len(self.command_history) - 1:
            self.history_index += 1
            self.manual_cmd_var.set("")

    def send_manual_command(self) -> None:
        """Отправка одной ручной команды."""
        cmd = self.manual_cmd_var.get().strip()
        if not cmd:
            return
            
        # Добавляем в историю (если не дубликат последней)
        if not self.command_history or self.command_history[-1] != cmd:
            self.command_history.append(cmd)
        
        # Сбрасываем индекс истории в конец
        self.history_index = len(self.command_history)
        self.manual_cmd_var.set("")
        
        # Отображаем в текстовом поле
        self.commands_text.insert(tk.END, f"{cmd}\n")
        self.commands_text.see(tk.END)
        
        # Отправляем по сети
        target_str = self.network_target_var.get().strip()
        if ":" in target_str:
            ip, port_str = target_str.split(":", 1)
            try:
                port = int(port_str)
                self.send_network_scenario(ip, port, f"{cmd}\n")
            except ValueError:
                self.rec_status_var.set("❌ Ошибка формата порта")
        else:
            self.rec_status_var.set("❌ Ошибка формата IP:Port")

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
            print(f"Ошибка получения устройств: {exc}")

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
            self.sample_rate = sample_rate
            self.device_id = device_id

            self._reset_recording_buffers()

            if was_running:
                self.start_detector()
        except Exception as exc:
            messagebox.showerror("Ошибка аудио", f"Не удалось настроить аудио:\n{exc}")

    def _reset_recording_buffers(self) -> None:
        self.session_audio_chunks = []
        self.silence_start_time = None

    def start_recording(self) -> None:
        """Запуск сессии записи: очищаем буфер сырого звука и начинаем копить фрагмент целиком."""
        self._reset_recording_buffers()
        self.recorded_events.clear()
        self.session_sample_rate = self.sample_rate
        self.is_recording_session = True
        self.silence_start_time = time.time()

        self.btn_start_rec.state(["disabled"])
        self.btn_stop_rec.state(["!disabled"])

        self.status_label.configure(fg="#d32f2f")
        self.rec_status_var.set("🔴 ИДЕТ ЗАПИСЬ | Играйте! (Авто-стоп при тишине)")

    def stop_recording_session(self, reason: str = "Остановлено") -> None:
        """Завершение сессии записи: склеиваем записанный фрагмент целиком и только теперь разбираем его на ноты."""
        if not self.is_recording_session:
            return

        self.is_recording_session = False
        self.silence_start_time = None

        self.btn_start_rec.state(["!disabled"])
        self.btn_stop_rec.state(["disabled"])
        self.status_label.configure(fg="#66707a")

        if self.session_audio_chunks:
            full_audio = np.concatenate(self.session_audio_chunks)
        else:
            full_audio = np.zeros(0, dtype=np.float32)
        self.session_audio_chunks = []

        duration_sec = len(full_audio) / self.session_sample_rate if self.session_sample_rate else 0.0
        self.rec_status_var.set(f"⏳ Анализ записанного фрагмента ({duration_sec:.1f} сек)...")
        self.root.update_idletasks()

        self.recorded_events = self._analyze_recorded_session(full_audio, self.session_sample_rate)

        self.notes_text.delete(1.0, tk.END)
        for midi_val, note_duration_ms in self.recorded_events:
            note_name = midi_to_name(midi_val)
            line = f"Нота: {note_name:<5} | Длительность: {note_duration_ms} мс\n"
            self.notes_text.insert(tk.END, line)
        self.notes_text.see(tk.END)

        self.rec_status_var.set(f"✅ {reason} | Найдено нот: {len(self.recorded_events)}")

        # АВТОМАТИЧЕСКИЙ ВЫЗОВ создания и отправки сценария
        self.convert_to_scenario()

    def _find_bow_segments(self, samples: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
        """
        Ищет сегменты звучания смычка по RMS-огибающей записанного фрагмента.
        Обновленный алгоритм: отслеживает не только глобальную тишину, но и 
        локальные провалы амплитуды (dip) для фиксации быстрой смены смычка.
        """
        frame_len = max(64, int(sample_rate * 0.010))  # 10 мс окно RMS
        hop_len = max(32, int(sample_rate * 0.002))    # 2 мс шаг для микро-пауз
        n = len(samples)
        if n < frame_len:
            return []

        n_frames = 1 + (n - frame_len) // hop_len
        s64 = samples.astype(np.float64)
        rms = np.empty(n_frames, dtype=np.float64)
        for i in range(n_frames):
            s = i * hop_len
            frame = s64[s:s + frame_len]
            rms[i] = math.sqrt(float(np.mean(frame * frame)))

        # Сглаживание скользящим средним
        kernel = np.ones(3) / 3.0
        rms = np.convolve(rms, kernel, mode='same')

        noise_floor = float(np.percentile(rms, 5))
        peak = float(np.percentile(rms, 95))
        span = max(peak - noise_floor, 1e-9)
        
        enter_threshold = noise_floor + 0.20 * span
        exit_threshold = noise_floor + 0.12 * span 

        hop_ms = hop_len / sample_rate * 1000.0
        min_gap_frames = max(1, int(round(self.min_bow_gap_ms_var.get() / hop_ms)))
        min_duration_ms = self.min_duration_ms_var.get()

        raw_segments: list[tuple[int, int]] = []
        in_sound = False
        seg_start_frame = 0
        below_run = 0
        current_note_peak = 0.0

        def close_segment(end_frame: int) -> None:
            start_sample = seg_start_frame * hop_len
            end_sample = min(n, end_frame * hop_len + frame_len)
            seg_duration_ms = (end_sample - start_sample) / sample_rate * 1000.0
            if seg_duration_ms >= min_duration_ms:
                raw_segments.append((start_sample, end_sample))

        for i, level in enumerate(rms):
            if not in_sound:
                if level >= enter_threshold:
                    in_sound = True
                    seg_start_frame = i
                    below_run = 0
                    current_note_peak = level
            else:
                current_note_peak = max(current_note_peak, level)
                
                is_silence = level < exit_threshold
                is_bow_change_dip = level < (current_note_peak * 0.40)  # Спад на 60% от пика
                
                if is_silence or is_bow_change_dip:
                    below_run += 1
                    if below_run >= min_gap_frames:
                        close_segment(i - below_run + 1)
                        in_sound = False
                        below_run = 0
                else:
                    below_run = 0

        if in_sound:
            close_segment(n_frames)

        return raw_segments

    def _detect_segment_pitch(self, samples: np.ndarray, seg_start: int, seg_end: int) -> int | None:
        """Определяет MIDI-номер ноты внутри одного сегмента смычка.

        Скользит несколькими окнами MPM по сегменту (пропуская атаку смычка в начале, если
        сегмент достаточно длинный) и берёт самый частый результат — так один случайный
        всплеск/провал внутри сегмента не портит итоговую ноту.
        """
        if self.detector is None:
            return None

        sample_rate = self.detector.sample_rate
        window = 4096
        hop = 1024
        seg_len = seg_end - seg_start
        if seg_len < 1024:
            return None

        attack_skip = int(sample_rate * 0.02)  # 20 мс переходного процесса атаки смычка
        analysis_start = seg_start + attack_skip if seg_len > window + attack_skip else seg_start

        midi_votes: list[int] = []
        pos = analysis_start
        while pos + window <= seg_end:
            chunk = samples[pos:pos + window]
            result = self.detector.detect(chunk)
            if result is not None:
                midi_votes.append(int(round(69.0 + 12.0 * math.log2(result.frequency / self.detector.a4))))
            pos += hop

        if not midi_votes:
            # Сегмент короче полного окна анализа — пробуем то, что есть, целиком.
            chunk = samples[seg_start:seg_end]
            if len(chunk) >= 1024:
                result = self.detector.detect(chunk)
                if result is not None:
                    return int(round(69.0 + 12.0 * math.log2(result.frequency / self.detector.a4)))
            return None

        values, counts = np.unique(midi_votes, return_counts=True)
        return int(values[np.argmax(counts)])

    def _analyze_recorded_session(self, samples: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
        """Разбирает записанный фрагмент целиком на отдельные ноты (по одной на каждое движение смычка)."""
        if len(samples) == 0 or self.detector is None:
            return []

        segments = self._find_bow_segments(samples, sample_rate)
        events: list[tuple[int, int]] = []
        for start_sample, end_sample in segments:
            midi = self._detect_segment_pitch(samples, start_sample, end_sample)
            if midi is None:
                continue
            duration_ms = int(round((end_sample - start_sample) / sample_rate * 1000.0))
            events.append((midi, duration_ms))
        return events

    def send_network_scenario(self, ip: str, port: int, text_data: str) -> None:
        """Отправка в фоне, чтобы интерфейс не зависал при недоступности сети."""
        self.rec_status_var.set(f"⏳ Отправка на {ip}:{port}...")
        
        def task():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                sock.connect((ip, port))
                sock.sendall(text_data.encode('utf-8'))
                sock.close()
                self.root.after(0, lambda: self.rec_status_var.set(f"✅ Отправка успешна на {ip}:{port}"))
            except ConnectionRefusedError:
                self.root.after(0, lambda: self.rec_status_var.set(f"❌ Приемник на {ip}:{port} не запущен!"))
            except socket.timeout:
                self.root.after(0, lambda: self.rec_status_var.set(f"❌ Превышено время ожидания от {ip}:{port}"))
            except Exception as e:
                self.root.after(0, lambda: self.rec_status_var.set(f"❌ Ошибка сети: {e}"))
                
        threading.Thread(target=task, daemon=True).start()

    def convert_to_scenario(self) -> None:
        """Преобразует список записанных нот в полноценный сценарий команд EMMA и сразу отправляет."""
        if not self.recorded_events:
            self.rec_status_var.set("ℹ️ Запись остановлена (ноты не обнаружены).")
            return

        lines = [
            "Recorded Scenario", 
            "/load_test",
            "/dyn_press;1",
            "/bow_p;1.1"
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
            return

        notes_for_planning = [(n[0], n[1]) for n in valid_notes]
        planned_dirs = optimize_bow_directions(notes_for_planning, X_start=0.0, initial_dir=1)
        
        for idx, (midi_val, duration_ms, string, position) in enumerate(valid_notes):
            speed = speed_for_duration(duration_ms)
            direction = planned_dirs[idx]
            
            lines.append(f"/sn;{string};{position}")
            lines.append(f"/bow;{direction};{duration_ms};{speed}")

        lines.append("/sn;2;0")
        lines.append("/end")
        lines.append("/play")

        scenario_text = "\n".join(lines) + "\n"

        self.commands_text.delete(1.0, tk.END)
        self.commands_text.insert(1.0, scenario_text)
        self.commands_text.see(1.0)

        if warnings:
             print(f"Пропущены ноты вне границ скрипки: {', '.join(set(warnings))}")

        # Отправка по сети
        target_str = self.network_target_var.get().strip()
        if ":" in target_str:
            ip, port_str = target_str.split(":", 1)
            try:
                port = int(port_str)
                self.send_network_scenario(ip, port, scenario_text)
            except ValueError:
                 self.rec_status_var.set("❌ Ошибка формата порта")
        else:
            self.rec_status_var.set("❌ Сценарий сгенерирован (ошибка формата IP:Port)")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice stream."""
        if status:
            print(status)
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

            # Во время записи копим ВЕСЬ фрагмент целиком — ноты будут разобраны
            # одним проходом после остановки записи (см. _analyze_recorded_session).
            if self.is_recording_session:
                self.session_audio_chunks.append(mono.copy())

            if mono.size >= self.ring.size:
                self.ring[:] = mono[-self.ring.size :]
            else:
                self.ring[:-mono.size] = self.ring[mono.size :]
                self.ring[-mono.size :] = mono
            processed = True

        if self.running and processed and self.detector:
            # Этот детект — только для живого дисплея тюнера (высота/отклонение в центах).
            # На сегментацию нот при записи он больше не влияет.
            result = self.detector.detect(self.ring)
            now = time.time()

            if result is None:
                self.live_note_var.set("--")
                self.live_hz_var.set("-- Hz")
                self.dev_text_var.set("Отклонение: 0 цент.")
                self.dev_bar['value'] = 50

                # --- ЛОГИКА АВТО-СТОПА ПО ТИШИНЕ (завершает всю сессию записи) ---
                if self.is_recording_session:
                    if self.silence_start_time is None:
                        self.silence_start_time = now

                    silence_elapsed = now - self.silence_start_time
                    target_silence = self.silence_stop_sec_var.get()

                    self.rec_status_var.set(f"🔴 ИДЕТ ЗАПИСЬ | Тишина: {silence_elapsed:.1f} / {target_silence:.1f} сек")

                    if silence_elapsed >= target_silence:
                        self.stop_recording_session(f"Авто-стоп: {target_silence:.1f} сек тишины")
            else:
                self.live_note_var.set(f"{result.note}{result.octave}")
                self.live_hz_var.set(f"{result.frequency:.2f} Hz")

                sign = "+" if result.cents >= 0 else ""
                self.dev_text_var.set(f"Отклонение: {sign}{int(result.cents)} цент.")
                self.dev_bar['value'] = int(np.clip(result.cents + 50, 0, 100))

                if self.is_recording_session:
                    self.silence_start_time = None
                    self.rec_status_var.set("🔴 ИДЕТ ЗАПИСЬ | Зафиксирован звук...")

        self.root.after(50, self.poll_audio)

    def clear_log(self) -> None:
        self.notes_text.delete(1.0, tk.END)
        self.commands_text.delete(1.0, tk.END)
        self.recorded_events.clear()
        self.command_history.clear()
        self.history_index = -1
        self.manual_cmd_var.set("")

    def close(self) -> None:
        self.stop_detector()
        self.root.destroy()

def main() -> None:
    root = tk.Tk()
    StandaloneTunerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()