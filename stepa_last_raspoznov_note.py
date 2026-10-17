#!/usr/bin/env python3
"""Hybrid Tuner: Real-time MPM + Offline Micro-pause Segmentation + Smart Bow Planning for EMMA."""

import math
import queue
import socket
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

# --- МУЗЫКАЛЬНЫЕ КОНСТАНТЫ И ХЕЛПЕРЫ ---
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
OPEN_STRINGS = {4: 55, 3: 62, 2: 69, 1: 76}  # G3, D4, A4, E5

def midi_to_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"

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
        for alt_midi in (midi - 12, midi + 12):
            choices = playable_positions(alt_midi)
            if choices: break
    if not choices:
        raise ValueError(f"Нота {midi_to_name(midi)} вне диапазона скрипки")
    if previous_string is None:
        return min(choices, key=lambda item: (item[1], item[0]))
    return min(choices, key=lambda item: (0 if item[0] == previous_string else 10, item[1], item[0]))


# --- УМНОЕ РАСПРЕДЕЛЕНИЕ СМЫЧКА (CANON IN D LOGIC) ---
def optimize_bowing_sequence(durations_ms: list[int]) -> list[tuple[int, int]]:
    """
    Умное распределение смычка (Beam Search), основанное на принципах исполнения Канона:
    - Скорость 100 для коротких и средних нот (<= 1000 мс).
    - Скорость 50 для длинных нот (>= 1500 мс).
    - Для нот <= 250 мс ИСКЛЮЧЕНО искусственное замедление (скорость не ниже 80-100).
    - Короткие и средние ноты (<= 600 мс) объединяются в один смычок до 100 000 пунктов.
    """
    MAX_BOW = 100000
    BEAM_WIDTH = 60
    
    STANDARD_SPEEDS = [50, 60, 67, 75, 80, 90, 100, 120, 150]

    # Состояние: (общий_штраф, текущая_позиция, пред_направление, история_движений)
    beam = [(0.0, 0.0, None, [])]

    for dur in durations_ms:
        next_beam = []
        
        # Подбор естественной целевой скорости
        if dur >= 1500:
            target_speed = 50
        elif dur >= 700:
            target_speed = int(100000 / dur)
            target_speed = max(50, min(100, target_speed))
        else:
            # Для всех коротких нот (500, 250, 180 мс и т.д.) базовая скорость — 100!
            target_speed = 100  

        for cost, pos, prev_dir, history in beam:
            valid_speeds = set(STANDARD_SPEEDS + [target_speed])
            
            # ЗАЩИТА: Для нот <= 250 мс ЗАПРЕЩЕНО замедлять ниже 80!
            if dur <= 250:
                valid_speeds = {s for s in valid_speeds if s >= 80}

            for speed in valid_speeds:
                for direction in (0, 1):
                    delta = dur * speed
                    new_pos = pos + delta if direction == 1 else pos - delta

                    # Проверка границ смычка [0, 100000]
                    if 0 <= new_pos <= MAX_BOW:
                        step_cost = 0.0

                        # 1. Штраф за отклонение от комфортной целевой скорости
                        step_cost += abs(speed - target_speed) * 0.6

                        # 2. Логика объединения смычков (Легато)
                        if prev_dir is not None:
                            if direction == prev_dir:
                                # Тот же смычок (продолжение движения)
                                if dur <= 600:
                                    # Высокий бонус за объединение коротких нот (например, две по 500мс)
                                    step_cost -= 25.0  
                                else:
                                    # Ноты > 600 мс лучше играть с нового смычка
                                    step_cost += 35.0
                            else:
                                # Смена направления смычка
                                if dur <= 600:
                                    rem_space = (MAX_BOW - pos) if prev_dir == 1 else pos
                                    # Если в смычке еще полно места (>= 35 000), штрафуем за ранний разворот
                                    if rem_space >= 35000:
                                        step_cost += 30.0 
                                else:
                                    # Для длинных нот разворот смычка поощряется
                                    step_cost -= 15.0

                        # 3. Штраф за смену смычка посередине (вдали от краев)
                        if prev_dir is not None and direction != prev_dir:
                            if 20000 < pos < 80000:
                                step_cost += 15.0

                        next_beam.append((cost + step_cost, new_pos, direction, history + [(direction, speed)]))

        # Спасательный фоллбек, если обычные варианты не поместились
        if not next_beam:
            for cost, pos, prev_dir, history in beam:
                safe_dir = 1 - prev_dir if prev_dir is not None else 1
                available_space = (MAX_BOW - pos) if safe_dir == 1 else pos
                
                if available_space <= 0:
                    safe_dir = 1 - safe_dir
                    available_space = (MAX_BOW - pos) if safe_dir == 1 else pos

                calc_speed = int(available_space / dur) if dur > 0 else 100
                safe_speed = max(80 if dur <= 250 else 50, min(200, calc_speed))
                
                delta = dur * safe_speed
                new_pos = pos + delta if safe_dir == 1 else pos - delta
                new_pos = max(0, min(MAX_BOW, new_pos))
                
                next_beam.append((cost + 500.0, new_pos, safe_dir, history + [(safe_dir, safe_speed)]))

        # Сортировка и фильтрация луча (Beam)
        next_beam.sort(key=lambda item: item[0])
        
        unique_beam = []
        seen = set()
        for item in next_beam:
            state_key = (round(item[1], -3), item[2])
            if state_key not in seen:
                seen.add(state_key)
                unique_beam.append(item)
                if len(unique_beam) >= BEAM_WIDTH:
                    break
        beam = unique_beam

    return beam[0][3] if beam else []


# --- ДЕТЕКТОР НОТ (MPM) ---
class MpmPitchDetector:
    def __init__(self, sample_rate=48000, a4=443.0, volume_thresh=0.005):
        self.fs = sample_rate
        self.a4 = a4
        self.vol_thresh = volume_thresh

    def detect(self, signal: np.ndarray):
        n = len(signal)
        if n < 1024: return None
        
        rms = np.sqrt(np.mean(signal ** 2))
        if rms < self.vol_thresh:
            return None

        fft_len = 2 ** int(np.ceil(np.log2(2 * n)))
        f_sig = np.fft.rfft(signal, n=fft_len)
        autocorr = np.fft.irfft(f_sig * np.conj(f_sig))[:n]

        sq = signal ** 2
        sum_x_sq = np.zeros(n)
        sum_x_sq[0] = np.sum(sq)
        for tau in range(1, n): sum_x_sq[tau] = sum_x_sq[tau - 1] - sq[tau - 1]

        sum_xtau_sq = np.zeros(n)
        sum_xtau_sq[0] = sum_x_sq[0]
        for tau in range(1, n): sum_xtau_sq[tau] = sum_xtau_sq[tau - 1] - sq[n - tau]

        m = sum_x_sq + sum_xtau_sq
        m[m == 0] = 1e-12
        nsdf = 2.0 * autocorr / m

        min_lag = max(2, int(self.fs / 4000.0))
        max_lag = min(n - 2, int(self.fs / 150.0))

        peaks = [tau for tau in range(min_lag, max_lag) if nsdf[tau] > nsdf[tau - 1] and nsdf[tau] >= nsdf[tau + 1] and nsdf[tau] > 0]
        if not peaks: return None

        max_peak_val = max(nsdf[p] for p in peaks)
        if max_peak_val < 0.45: return None

        selected_peak = next((p for p in peaks if nsdf[p] >= max_peak_val * 0.85), None)
        if selected_peak is None: return None

        p = selected_peak
        alpha, beta, gamma = nsdf[p - 1], nsdf[p], nsdf[p + 1]
        denom = alpha - 2 * beta + gamma
        delta = (alpha - gamma) / (2.0 * denom) if abs(denom) > 1e-9 else 0.0
        
        freq = self.fs / (p + delta)
        if not (150.0 <= freq <= 4000.0): return None
        
        midi = 69.0 + 12.0 * math.log2(freq / self.a4)
        return int(round(midi)), freq


# --- ПОСТ-ПРОЦЕССИНГ С СЕГМЕНТАЦИЕЙ И DIP DETECTION ---
@dataclass
class NoteEvent:
    midi: int
    start_time: float
    end_time: float
    duration_ms: int

class OfflineSegmenter:
    def __init__(self, fs=48000, a4=443.0, vol_thresh=0.005, dip_thresh=0.5):
        self.fs = fs
        self.detector = MpmPitchDetector(sample_rate=fs, a4=a4, volume_thresh=vol_thresh)
        self.dip_thresh = dip_thresh
        
    def analyze(self, audio: np.ndarray) -> list[NoteEvent]:
        if len(audio) == 0: return []
        
        window_size = int(self.fs * 0.04)
        hop_size = int(self.fs * 0.01)
        
        raw_frames = []
        for i in range(0, len(audio) - window_size, hop_size):
            chunk = audio[i:i + window_size]
            time_sec = i / self.fs
            rms = np.sqrt(np.mean(chunk**2))
            
            res = self.detector.detect(chunk)
            midi = res[0] if res else None
            raw_frames.append((time_sec, midi, rms))
            
        smoothed_midis = []
        for i in range(len(raw_frames)):
            window = [f[1] for f in raw_frames[max(0, i-2):min(len(raw_frames), i+3)] if f[1] is not None]
            if len(window) >= 2:
                smoothed_midis.append(int(np.median(window)))
            else:
                smoothed_midis.append(None)
                
        events = []
        current_note = None
        start_t = 0.0
        local_max_rms = 0.0
        
        for i, midi in enumerate(smoothed_midis):
            time_sec, _, rms = raw_frames[i]
            
            if midi is not None:
                if current_note is None:
                    current_note = midi
                    start_t = time_sec
                    local_max_rms = rms
                elif current_note != midi:
                    dur_ms = int((time_sec - start_t) * 1000)
                    if dur_ms >= 100:  # Защита от слишком короткого шума
                        events.append(NoteEvent(current_note, start_t, time_sec, dur_ms))
                    current_note = midi
                    start_t = time_sec
                    local_max_rms = rms
                else:
                    if rms > local_max_rms:
                        local_max_rms = rms
                    elif rms < (local_max_rms * self.dip_thresh):
                        dur_ms = int((time_sec - start_t) * 1000)
                        if dur_ms >= 100:
                            events.append(NoteEvent(current_note, start_t, time_sec, dur_ms))
                        current_note = None
            else:
                if current_note is not None:
                    dur_ms = int((time_sec - start_t) * 1000)
                    if dur_ms >= 100:
                        events.append(NoteEvent(current_note, start_t, time_sec, dur_ms))
                    current_note = None
                    
        if current_note is not None:
            time_sec = raw_frames[-1][0]
            dur_ms = int((time_sec - start_t) * 1000)
            if dur_ms >= 100:
                events.append(NoteEvent(current_note, start_t, time_sec, dur_ms))
                
        return events


# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ ---
class HybridTunerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EMMA Tuner + Pachelbel Bow Planner")
        self.root.geometry("800x950")
        self.root.configure(bg="#f6f7f9")

        self.fs = 48000
        self.stream = None
        self.is_recording = False
        self.recorded_audio = []
        
        self.audio_queue = queue.Queue(maxsize=16)
        self.ring = np.zeros(8192, dtype=np.float32)

        # Настройки
        self.device_var = tk.StringVar()
        self.a4_var = tk.DoubleVar(value=443.0)
        self.vol_thresh_var = tk.DoubleVar(value=0.005)
        self.dip_thresh_var = tk.DoubleVar(value=0.5)

        # UI
        self.live_note_var = tk.StringVar(value="--")
        self.live_hz_var = tk.StringVar(value="-- Hz")
        self.status_var = tk.StringVar(value="Микрофон активен. Готов к записи.")

        self.detector = MpmPitchDetector(sample_rate=self.fs, a4=self.a4_var.get(), volume_thresh=self.vol_thresh_var.get())
        
        self._build_styles()
        self._build_ui()
        self.populate_audio_devices()
        self.start_audio_stream()
        
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(30, self.poll_audio)

    def _build_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Root.TFrame", background="#f6f7f9")
        style.configure("Record.TButton", font=("Segoe UI", 12, "bold"), padding=10, background="#d32f2f", foreground="white")
        style.configure("Stop.TButton", font=("Segoe UI", 12, "bold"), padding=10, background="#f57c00", foreground="white")
        style.configure("Convert.TButton", font=("Segoe UI", 10, "bold"), padding=6, background="#2e7d32", foreground="white")

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, style="Root.TFrame", padding=15)
        main_frame.pack(fill="both", expand=True)

        # 1. Настройки
        settings = ttk.LabelFrame(main_frame, text=" Настройки Чувствительности ", padding=10)
        settings.pack(fill="x", pady=(0, 10))

        ttk.Label(settings, text="Микрофон:").grid(row=0, column=0, sticky="w", pady=5)
        self.device_combo = ttk.Combobox(settings, textvariable=self.device_var, state="readonly", width=40)
        self.device_combo.grid(row=0, column=1, columnspan=3, sticky="w", pady=5)
        self.device_combo.bind("<<ComboboxSelected>>", lambda e: self.restart_audio_stream())

        ttk.Label(settings, text="Абс. порог тишины:").grid(row=1, column=0, sticky="w", pady=5)
        sp1 = ttk.Spinbox(settings, from_=0.001, to=0.2, increment=0.001, textvariable=self.vol_thresh_var, width=8)
        sp1.grid(row=1, column=1, sticky="w", pady=5)
        sp1.bind("<ButtonRelease-1>", lambda e: self.update_detector_settings())

        ttk.Label(settings, text="Разделение один. нот (Провал %):").grid(row=1, column=2, sticky="w", padx=10, pady=5)
        sp_dip = ttk.Spinbox(settings, from_=0.1, to=0.9, increment=0.05, textvariable=self.dip_thresh_var, width=8)
        sp_dip.grid(row=1, column=3, sticky="w", pady=5)
        
        ttk.Label(settings, text="Частота A4 (Hz):").grid(row=2, column=0, sticky="w", pady=5)
        sp2 = ttk.Spinbox(settings, from_=430.0, to=450.0, increment=1.0, textvariable=self.a4_var, width=8)
        sp2.grid(row=2, column=1, sticky="w", pady=5)
        sp2.bind("<ButtonRelease-1>", lambda e: self.update_detector_settings())

        # 2. Дисплей реального времени
        tuner_frame = ttk.LabelFrame(main_frame, text=" Сейчас играет (Real-time) ", padding=15)
        tuner_frame.pack(fill="x", pady=10)

        tk.Label(tuner_frame, textvariable=self.live_note_var, font=("Segoe UI", 48, "bold"), fg="#1c4f63", bg="#ffffff").pack()
        tk.Label(tuner_frame, textvariable=self.live_hz_var, font=("Segoe UI", 14), fg="#555", bg="#ffffff").pack()

        # 3. Управление записью
        ctrl_frame = ttk.Frame(main_frame, padding=10)
        ctrl_frame.pack(fill="x")

        self.btn_rec = ttk.Button(ctrl_frame, text="● Начать запись сценария", command=self.start_recording, style="Record.TButton")
        self.btn_rec.pack(side="left", padx=5)

        self.btn_stop = ttk.Button(ctrl_frame, text="■ Стоп и Анализ", command=self.stop_and_analyze, style="Stop.TButton")
        self.btn_stop.pack(side="left", padx=5)
        self.btn_stop.state(["disabled"])
        
        tk.Label(ctrl_frame, textvariable=self.status_var, font=("Segoe UI", 11, "bold"), fg="#555", bg="#f6f7f9").pack(side="left", padx=20)

        # 4. Сеть
        net_frame = ttk.Frame(main_frame, padding=10)
        net_frame.pack(fill="x")
        ttk.Label(net_frame, text="IP:Port EMMA:").pack(side="left")
        self.net_var = tk.StringVar(value="10.24.245.1:9000")
        ttk.Entry(net_frame, textvariable=self.net_var, width=20).pack(side="left", padx=5)
        ttk.Button(net_frame, text="⚡ Отправить сценарий", command=self.send_scenario, style="Convert.TButton").pack(side="left", padx=10)

        # 5. Вывод
        paned = ttk.PanedWindow(main_frame, orient="horizontal")
        paned.pack(fill="both", expand=True, pady=10)

        log_frame = ttk.Frame(paned)
        paned.add(log_frame, weight=1)
        ttk.Label(log_frame, text="Распознанные ноты:").pack(anchor="w")
        self.txt_notes = tk.Text(log_frame, font=("Cascadia Mono", 10), width=35)
        self.txt_notes.pack(fill="both", expand=True)

        cmd_frame = ttk.Frame(paned)
        paned.add(cmd_frame, weight=1)
        ttk.Label(cmd_frame, text="Сценарий EMMA:").pack(anchor="w")
        self.txt_cmds = tk.Text(cmd_frame, font=("Cascadia Mono", 10), width=35)
        self.txt_cmds.pack(fill="both", expand=True)

    def populate_audio_devices(self):
        devices = sd.query_devices()
        idx_list = []
        default = sd.default.device[0]
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                name = f"{i}: {d['name']}"
                idx_list.append(name)
                if i == default:
                    self.device_var.set(name)
        self.device_combo['values'] = idx_list

    def update_detector_settings(self):
        self.detector = MpmPitchDetector(sample_rate=self.fs, a4=self.a4_var.get(), volume_thresh=self.vol_thresh_var.get())

    def audio_callback(self, indata, frames, time, status):
        mono = indata[:, 0].copy()
        try:
            self.audio_queue.put_nowait(mono)
        except queue.Full:
            pass

    def start_audio_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
        dev_str = self.device_var.get()
        if not dev_str: return
        dev_id = int(dev_str.split(":")[0])
        
        self.stream = sd.InputStream(device=dev_id, channels=1, samplerate=self.fs, blocksize=512, callback=self.audio_callback)
        self.stream.start()

    def restart_audio_stream(self):
        self.start_audio_stream()

    def poll_audio(self):
        processed = False
        while True:
            try:
                block = self.audio_queue.get_nowait()
                self.ring[:-len(block)] = self.ring[len(block):]
                self.ring[-len(block):] = block
                
                if self.is_recording:
                    self.recorded_audio.append(block)
                processed = True
            except queue.Empty:
                break

        if processed:
            result = self.detector.detect(self.ring)
            if result is None:
                self.live_note_var.set("--")
                self.live_hz_var.set("-- Hz")
            else:
                midi_val, hz_val = result
                self.live_note_var.set(midi_to_name(midi_val))
                self.live_hz_var.set(f"{hz_val:.1f} Hz")

        self.root.after(30, self.poll_audio)

    def start_recording(self):
        self.recorded_audio = []
        self.is_recording = True
        
        self.btn_rec.state(["disabled"])
        self.btn_stop.state(["!disabled"])
        self.status_var.set("🔴 Идет запись... Играйте!")
        self.txt_notes.delete(1.0, tk.END)
        self.txt_cmds.delete(1.0, tk.END)

    def stop_and_analyze(self):
        self.is_recording = False
        self.btn_rec.state(["!disabled"])
        self.btn_stop.state(["disabled"])
        self.status_var.set("⏳ Сегментация и анализ нот...")
        self.root.update()

        if not self.recorded_audio:
            self.status_var.set("❌ Ошибка: Нет записанных данных.")
            return
            
        audio_arr = np.concatenate(self.recorded_audio)
        
        analyzer = OfflineSegmenter(fs=self.fs, a4=self.a4_var.get(), 
                                    vol_thresh=self.vol_thresh_var.get(),
                                    dip_thresh=self.dip_thresh_var.get())
        
        try:
            events = analyzer.analyze(audio_arr)
            if not events:
                self.status_var.set("⚠️ Нот не обнаружено. Слишком тихо?")
                self.txt_notes.insert(tk.END, "Сигнал не превысил порог громкости (либо все ноты были короче 100 мс).\n")
            else:
                self.display_and_generate(events)
                self.status_var.set("✅ Анализ завершен! Умный план готов.")
        except Exception as e:
            self.status_var.set("❌ Ошибка анализа.")
            messagebox.showerror("Ошибка", str(e))

    def display_and_generate(self, events: list[NoteEvent]):
        self.txt_notes.delete(1.0, tk.END)
        durations = []
        
        for ev in events:
            line = f"[{ev.start_time:.2f}s] {midi_to_name(ev.midi):<4} | {ev.duration_ms} мс\n"
            self.txt_notes.insert(tk.END, line)
            durations.append(ev.duration_ms)

        # Вызов алгоритма оптимизации распределения смычков
        bow_plan = optimize_bowing_sequence(durations)

        lines = ["Recorded Smart Scenario", "/load_test", "/dyn_press;1", "/bow_p;1.1"]
        prev_string = None
        
        for ev, (bow_dir, speed) in zip(events, bow_plan):
            try:
                string, pos = choose_position(ev.midi, prev_string)
                prev_string = string
                
                lines.append(f"/sn;{string};{pos}")
                lines.append(f"/bow;{bow_dir};{ev.duration_ms};{speed}")
            except ValueError as e:
                lines.append(f"; ОШИБКА: {e}")
                
        lines.extend(["/end", "/play"])

        self.txt_cmds.delete(1.0, tk.END)
        self.txt_cmds.insert(1.0, "\n".join(lines) + "\n")

    def send_scenario(self):
        scenario = self.txt_cmds.get(1.0, tk.END).strip()
        if not scenario: return
            
        target = self.net_var.get().strip()
        if ":" not in target: return
        ip, port_str = target.split(":", 1)
        
        def task():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                sock.connect((ip, int(port_str)))
                sock.sendall((scenario + "\n").encode('utf-8'))
                sock.close()
                self.root.after(0, lambda: self.status_var.set("✅ Успешно отправлено!"))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"❌ Ошибка сети: {e}"))
                
        threading.Thread(target=task, daemon=True).start()

    def close(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HybridTunerApp(root)
    root.mainloop()