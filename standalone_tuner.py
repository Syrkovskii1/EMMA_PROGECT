#!/usr/bin/env python3
"""Standalone Real-time Violin Tuner using McLeod Pitch Method (MPM) with Auto-Stop and EMMA Scenario Converter."""

import math
import queue
import time
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
        self.root.title("Тюнер EMMA (McLeod Pitch Method & Scenario Generator)")
        self.root.geometry("660x920")
        self.root.minsize(560, 740)
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
        self.recorded_events: list[tuple[int, int]] = []  # Хранит пары (midi_note, duration_ms)

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

        # Ряд 0: Вход
        ttk.Label(settings_frame, text="Вход:").grid(row=0, column=0, sticky="w", pady=2)
        self.device_combo = ttk.Combobox(settings_frame, textvariable=self.device_var, state="readonly", font=("Segoe UI", 9))
        self.device_combo.grid(row=0, column=1, columnspan=3, sticky="ew", pady=2)

        # Ряд 1: Окно и Чанк
        ttk.Label(settings_frame, text="Окно (samples):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Combobox(settings_frame, textvariable=self.buffer_size_var, values=[2048, 4096, 8192, 16384], state="readonly", width=8).grid(row=1, column=1, sticky="w", pady=2)

        ttk.Label(settings_frame, text="Чанк (block):").grid(row=1, column=2, sticky="w", pady=2, padx=(10, 0))
        ttk.Combobox(settings_frame, textvariable=self.block_size_var, values=[256, 512, 1024], state="readonly", width=8).grid(row=1, column=3, sticky="w", pady=2)

        # Ряд 2: Clarity и Cutoff K
        ttk.Label(settings_frame, text="Порог (Clarity):").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Spinbox(settings_frame, from_=0.1, to=0.9, increment=0.05, textvariable=self.clarity_var, width=8).grid(row=2, column=1, sticky="w", pady=2)

        ttk.Label(settings_frame, text="Коэф. октавы K:").grid(row=2, column=2, sticky="w", pady=2, padx=(10, 0))
        ttk.Spinbox(settings_frame, from_=0.5, to=0.98, increment=0.02, textvariable=self.cutoff_k_var, width=8).grid(row=2, column=3, sticky="w", pady=2)

        # Ряд 3: Фильтры стабильности
        ttk.Label(settings_frame, text="Стабильность (кадр):").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Spinbox(settings_frame, from_=1, to=10, increment=1, textvariable=self.confirm_frames_var, width=8).grid(row=3, column=1, sticky="w", pady=2)

        ttk.Label(settings_frame, text="Мин. время (мс):").grid(row=3, column=2, sticky="w", pady=2, padx=(10, 0))
        ttk.Spinbox(settings_frame, from_=20, to=500, increment=20, textvariable=self.min_duration_ms_var, width=8).grid(row=3, column=3, sticky="w", pady=2)

        # Ряд 4: Настройка авто-стопа по тишине и A4
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

        # КНОПКА ПРЕОБРАЗОВАНИЯ В СЦЕНАРИЙ EMMA
        self.btn_convert = ttk.Button(rec_ctrl_bar, text="⚡ Сценарий EMMA", command=self.convert_to_scenario, style="Convert.TButton")
        self.btn_convert.pack(side="left", padx=(6, 0))

        ttk.Button(rec_ctrl_bar, text="Сохранить в .txt", command=self.save_to_file, style="Secondary.TButton").pack(side="right")
        ttk.Button(rec_ctrl_bar, text="Очистить", command=self.clear_log, style="Secondary.TButton").pack(side="right", padx=(0, 4))

        # Индикатор статуса записи
        self.status_label = tk.Label(log_frame, textvariable=self.rec_status_var, font=("Segoe UI", 10, "bold"), fg="#66707a", bg="#ffffff", anchor="w", padx=5, pady=5)
        self.status_label.pack(fill="x", pady=(0, 5))

        # Текстовое поле для вывода нот и итогового сценария
        text_container = ttk.Frame(log_frame)
        text_container.pack(fill="both", expand=True)

        self.log_text = tk.Text(text_container, bg="#fbfcfd", fg="#121417", font=("Cascadia Mono", 11), relief="flat", padx=10, pady=10)
        self.log_text.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(text_container, orient="vertical", command=self.log_text.yview)
        scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scroll.set)

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
        self.pitch_buffer.clear()
        self.candidate_midi = None
        self.candidate_count = 0
        self.current_midi = None
        self.silence_start_time = None

    def start_recording(self) -> None:
        """Запуск сессии записи нот."""
        self._reset_recording_buffers()
        self.recorded_events.clear()
        self.is_recording_session = True
        self.silence_start_time = time.time()

        self.btn_start_rec.state(["disabled"])
        self.btn_stop_rec.state(["!disabled"])

        self.status_label.configure(fg="#d32f2f")
        self.rec_status_var.set("🔴 ИДЕТ ЗАПИСЬ | Играйте! (Авто-стоп при тишине)")

    def stop_recording_session(self, reason: str = "Остановлено") -> None:
        """Завершение сессии записи нот."""
        if not self.is_recording_session:
            return

        self._finalize_current_note(time.time())
        self.is_recording_session = False
        self.silence_start_time = None

        self.btn_start_rec.state(["!disabled"])
        self.btn_stop_rec.state(["disabled"])

        self.status_label.configure(fg="#66707a")
        self.rec_status_var.set(f"Запись завершена ({reason}). Готово к конвертации!")

    def convert_to_scenario(self) -> None:
        """Преобразует список записанных нот в полноценный сценарий команд EMMA (/sn и /bow)."""
        if not self.recorded_events:
            messagebox.showinfo("Нет данных", "Сначала запишите исполнение кнопкой '● Старт записи'.")
            return

        # Инициализация сценария с новыми командами подготовки механики
        lines = [
            "Recorded Scenario", 
            "/load_test",
            "/dyn_press;1",
            "/bow_p;1.1"
        ]
        
        # 1. Фильтрация валидных нот и подбор позиций на грифе
        valid_notes = []
        warnings = []
        prev_string = None
        
        for midi_val, duration_ms in self.recorded_events:
            try:
                # Проверяем, можем ли мы физически сыграть ноту
                string, position = choose_position(midi_val, prev_string)
                valid_notes.append((midi_val, duration_ms, string, position))
                prev_string = string
            except ValueError:
                warnings.append(midi_to_name(midi_val))
                
        if not valid_notes:
            messagebox.showinfo("Пусто", "Нет валидных нот в диапазоне скрипки.")
            return

        # 2. Интеллектуальное планирование направлений смычка
        # Подготавливаем список (midi, duration) для алгоритма
        notes_for_planning = [(n[0], n[1]) for n in valid_notes]
        
        # Стартуем с X=0.0 (начало смычка) и initial_dir=1 (вниз/вперед). 
        planned_dirs = optimize_bow_directions(notes_for_planning, X_start=0.0, initial_dir=1)
        
        # 3. Генерация финальных команд
        for idx, (midi_val, duration_ms, string, position) in enumerate(valid_notes):
            speed = speed_for_duration(duration_ms)
            direction = planned_dirs[idx]
            
            lines.append(f"/sn;{string};{position}")
            lines.append(f"/bow;{direction};{duration_ms};{speed}")

        # Отпускаем палец (переход на нулевую позицию) перед завершением программы
        lines.append("/sn;2;0")
        
        lines.append("/end")
        lines.append("/play")

        scenario_text = "\n".join(lines) + "\n"

        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(1.0, scenario_text)
        self.log_text.see(1.0)

        self.rec_status_var.set(f"⚡ Успешно сгенерирован умный сценарий! Сохраните файл в scenarios/")
        
        if warnings:
            messagebox.showwarning(
                "Предупреждение",
                f"Некоторые ноты ({', '.join(set(warnings))}) выходят за границы скрипки G3-B5 и были пропущены."
            )

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
                    self.dev_bar['value'] = 50

                    if self.is_recording_session:
                        self._finalize_current_note(now)
                        self.candidate_midi = None
                        self.candidate_count = 0
                        self.pitch_buffer.clear()

                # --- ЛОГИКА АВТО-СТОПА ПО ТИШИНЕ ---
                if self.is_recording_session:
                    if self.silence_start_time is None:
                        self.silence_start_time = now

                    silence_elapsed = now - self.silence_start_time
                    target_silence = self.silence_stop_sec_var.get()

                    self.rec_status_var.set(f"🔴 ИДЕТ ЗАПИСЬ | Тишина: {silence_elapsed:.1f} / {target_silence:.1f} сек")

                    if silence_elapsed >= target_silence:
                        self.stop_recording_session(f"Авто-стоп: {target_silence:.1f} сек тишины")
            else:
                self.silent_ticks = 0
                self.live_note_var.set(f"{result.note}{result.octave}")
                self.live_hz_var.set(f"{result.frequency:.2f} Hz")

                sign = "+" if result.cents >= 0 else ""
                self.dev_text_var.set(f"Отклонение: {sign}{int(result.cents)} цент.")
                self.dev_bar['value'] = int(np.clip(result.cents + 50, 0, 100))

                if self.is_recording_session:
                    self.silence_start_time = None
                    self.rec_status_var.set("🔴 ИДЕТ ЗАПИСЬ | Зафиксирован звук...")

                    raw_midi = int(round(69.0 + 12.0 * math.log2(result.frequency / self.detector.a4)))

                    # 1. Медианная фильтрация
                    self.pitch_buffer.append(raw_midi)
                    max_buf = self.smoothing_frames_var.get()
                    if len(self.pitch_buffer) > max_buf:
                        self.pitch_buffer.pop(0)

                    stable_midi = int(np.median(self.pitch_buffer))

                    # 2. Подтверждение стабильности
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
                self.log_text.insert(tk.END, line)
                self.log_text.see(tk.END)

                # Сохраняем во внутренний список для генератора сценария
                self.recorded_events.append((self.current_midi, duration_ms))

            self.current_midi = None

    def clear_log(self) -> None:
        self.log_text.delete(1.0, tk.END)
        self.recorded_events.clear()

    def save_to_file(self) -> None:
        content = self.log_text.get(1.0, tk.END).strip()
        if not content:
            messagebox.showinfo("Пусто", "Список нот пуст.")
            return

        path = filedialog.asksaveasfilename(
            title="Сохранить записанные ноты или сценарий",
            defaultextension=".txt",
            filetypes=(("Текстовые файлы", "*.txt"), ("Все файлы", "*.*"))
        )
        if path:
            Path(path).write_text(content, encoding="utf-8")

    def close(self) -> None:
        self.stop_detector()
        self.root.destroy()

def main() -> None:
    root = tk.Tk()
    StandaloneTunerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()