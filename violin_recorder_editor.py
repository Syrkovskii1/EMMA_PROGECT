#!/usr/bin/env python3
"""Unified Scenario Recorder and Editor for the robot violin."""

from __future__ import annotations

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

import numpy as np
import sounddevice as sd
from scipy.signal import find_peaks

# --- CONSTANTS & HELPERS FROM ORIGINAL SCENARIO EDITOR ---
OPEN_STRINGS = {4: 55, 3: 62, 2: 69, 1: 76}  # G3, D4, A4, E5
STRING_NAMES = {1: "Mi", 2: "La", 3: "Re", 4: "Sol"}
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
WHITE_NOTES = {0, 2, 4, 5, 7, 9, 11}
DURATIONS = (250, 500, 750, 1000, 1250, 1500, 1750, 2000)
BOW_SPEEDS = {250: 100, 500: 100, 750: 100, 1000: 100, 1250: 80, 1500: 66, 1750: 57, 2000: 50}
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

def parse_last_state_and_position(text: str) -> tuple[int | None, int, float]:
    last_string: int | None = None
    last_direction: int = 0
    X = 0.0
    for line in text.splitlines():
        parts = line.strip().lower().split(";")
        if len(parts) >= 3 and parts[0] == "/sn":
            try:
                last_string = int(parts[1])
            except ValueError:
                pass
        elif len(parts) >= 4 and parts[0] == "/bow":
            try:
                direction = int(parts[1])
                duration = float(parts[2])
                speed = float(parts[3])
                last_direction = direction
                if direction == 1:
                    X += duration * speed
                else:
                    X -= duration * speed
            except ValueError:
                pass
    X = max(0.0, min(100000.0, X))
    return last_string, last_direction, X

def optimize_bow_directions(notes: list[tuple[int, int]], X_start: float, initial_dir: int) -> list[int]:
    n = len(notes)
    if n == 0:
        return []
        
    # Precalculate displacements
    disps = []
    for midi, duration in notes:
        speed = speed_for_duration(duration)
        disps.append(duration * speed)
        
    # --- STAGE 1: STRICT SEARCH (No boundary overrun allowed) ---
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
                
            # Strict boundary check (with 0.1 tolerance for rounding)
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
                # Penalty for playing individually (changing direction on small stroke)
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

    # Try searching starting at X_start
    cost, path = search_strict(0, X_start, 1 - initial_dir, 100000)
    if cost != float('inf'):
        return path
        
    # --- STAGE 2: SOFT FALLBACK SEARCH (If strict search is mathematically impossible) ---
    memo_soft = {}
    def search_soft(idx, X, stroke_dir, stroke_disp):
        if idx == n:
            return 0, []
            
        # Quantize state to reduce state space and speed up memoization
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
                
            # Base cost: prefer to group short notes
            cost = 0
            if not is_same:
                if stroke_disp < 75000 and duration < 1000:
                    cost += 10000
            else:
                if duration >= 750:
                    cost += 5000
                if duration >= 1000:
                    cost += 50000
                    
            # Soft boundaries penalties
            # 1. Overrun penalty (exceeding bow ends 0 or 100,000)
            overrun = max(0.0, -X_new) + max(0.0, X_new - 100000)
            cost += overrun * 2.0
            
            # 2. Stroke capacity overrun
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

# --- DETECTOR FROM DETECTOR MODULE (Re-implemented for self-containment) ---
@dataclass(frozen=True)
class Detection:
    frequency: float
    note: str
    octave: int
    cents: float
    confidence: float

    @property
    def label(self) -> str:
        sign = "+" if self.cents >= 0 else ""
        return f"{self.note}{self.octave} {self.frequency:8.2f} Hz {sign}{self.cents:5.1f} cents conf={self.confidence:4.2f}"

class ViolinPitchDetector:
    def __init__(
        self,
        sample_rate: int = 48_000,
        fft_size: int = 16_384,
        min_freq: float = 150.0,
        max_freq: float = 3_200.0,
        threshold_db: float = 18.0,
        prominence_db: float = 7.0,
        min_harmonic_matches: int = 2,
        harmonics: int = 8,
        a4: float = 440.0,
        smoothing: int = 5,
    ) -> None:
        if fft_size < 2048 or fft_size & (fft_size - 1):
            raise ValueError("fft_size must be a power of two and at least 2048")
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.threshold_db = threshold_db
        self.prominence_db = prominence_db
        self.min_harmonic_matches = min_harmonic_matches
        self.harmonics = harmonics
        self.a4 = a4
        self.history: list[float] = []
        self.smoothing = max(1, smoothing)

        self.window = np.hanning(fft_size).astype(np.float32)
        self.freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        self.search_mask = (self.freqs >= min_freq) & (self.freqs <= max_freq)
        self.bin_hz = sample_rate / fft_size

    def detect(self, samples: np.ndarray) -> Detection | None:
        frame = self._prepare_frame(samples)
        rms = float(np.sqrt(np.mean(frame * frame)))
        if rms < 0.002:
            self.history.clear()
            return None

        spectrum = np.abs(np.fft.rfft(frame * self.window))
        db = 20.0 * np.log10(spectrum + 1e-12)
        noise_floor = float(np.median(db[self.search_mask]))
        min_height = noise_floor + self.threshold_db

        search_db = db[self.search_mask]
        search_freqs = self.freqs[self.search_mask]
        peak_local, props = find_peaks(
            search_db,
            height=min_height,
            prominence=self.prominence_db,
            distance=max(1, int(18.0 / self.bin_hz)),
        )
        if len(peak_local) == 0:
            return None

        peak_indices = np.flatnonzero(self.search_mask)[peak_local]
        candidates: list[tuple[float, float, int, float]] = []
        strongest_db = float(np.max(props["peak_heights"]))

        for idx in peak_indices:
            freq = self._refine_peak_frequency(db, idx)
            if freq < self.min_freq or freq > self.max_freq:
                continue
            peak_db = float(db[idx])
            matches, harmonic_score = self._harmonic_support(db, freq, noise_floor)
            confidence = self._confidence(peak_db, strongest_db, noise_floor, matches, harmonic_score)
            if self.min_harmonic_matches <= 0 or matches >= self.min_harmonic_matches:
                candidates.append((freq, confidence, matches, peak_db))

        if not candidates:
            return None

        # Find the highest confidence among all candidates
        max_conf = max(c[1] for c in candidates)
        if max_conf < 0.25:
            return None
            
        # Keep strong candidate peaks within a 0.25 margin of the strongest peak
        strong_candidates = [c for c in candidates if c[1] >= max_conf - 0.25 and c[1] >= 0.25]
        if not strong_candidates:
            return None
            
        # Choose the lowest frequency fundamental among the strong candidates
        strong_candidates.sort(key=lambda item: item[0])
        best_freq, best_confidence, _, _ = strong_candidates[0]

        smoothed_freq = self._smooth(best_freq)
        return self._to_detection(smoothed_freq, best_confidence)

    def _prepare_frame(self, samples: np.ndarray) -> np.ndarray:
        data = np.asarray(samples, dtype=np.float32)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if data.size >= self.fft_size:
            data = data[-self.fft_size :]
        else:
            data = np.pad(data, (self.fft_size - data.size, 0))
        data = data - np.mean(data)
        return data.astype(np.float32, copy=False)

    def _refine_peak_frequency(self, db: np.ndarray, idx: int) -> float:
        if idx <= 0 or idx >= len(db) - 1:
            return float(self.freqs[idx])
        left, center, right = float(db[idx - 1]), float(db[idx]), float(db[idx + 1])
        denom = left - 2.0 * center + right
        if abs(denom) < 1e-9:
            offset = 0.0
        else:
            offset = 0.5 * (left - right) / denom
        offset = float(np.clip(offset, -0.5, 0.5))
        return float((idx + offset) * self.bin_hz)

    def _harmonic_support(self, db: np.ndarray, freq: float, noise_floor: float) -> tuple[int, float]:
        matches = 0
        score = 0.0
        for harmonic in range(2, self.harmonics + 1):
            target = freq * harmonic
            if target > min(self.sample_rate * 0.48, self.max_freq * self.harmonics):
                break
            center = int(round(target / self.bin_hz))
            radius = max(1, int(round((target * 0.012) / self.bin_hz)))
            lo = max(1, center - radius)
            hi = min(len(db), center + radius + 1)
            local = float(np.max(db[lo:hi]))
            excess = local - noise_floor
            if excess >= self.threshold_db * 0.65:
                matches += 1
                score += min(1.0, excess / 42.0)
        return matches, score

    def _confidence(
        self,
        peak_db: float,
        strongest_db: float,
        noise_floor: float,
        matches: int,
        harmonic_score: float,
    ) -> float:
        level = np.clip((peak_db - noise_floor) / 45.0, 0.0, 1.0)
        relative = np.clip((peak_db - strongest_db + 30.0) / 30.0, 0.0, 1.0)
        harmonic = np.clip((matches + harmonic_score) / max(1.0, self.harmonics - 1), 0.0, 1.0)
        return float(np.clip(0.35 * level + 0.20 * relative + 0.45 * harmonic, 0.0, 1.0))

    def _smooth(self, freq: float) -> float:
        if self.history:
            jump_cents = abs(1200.0 * math.log2(freq / self.history[-1]))
            if jump_cents > 80.0:
                self.history.clear()
        self.history.append(freq)
        self.history = self.history[-self.smoothing :]
        return float(np.median(self.history))

    def _to_detection(self, freq: float, confidence: float) -> Detection:
        midi = 69.0 + 12.0 * math.log2(freq / self.a4)
        nearest = int(round(midi))
        cents = (midi - nearest) * 100.0
        note = NOTE_NAMES[nearest % 12]
        octave = nearest // 12 - 1
        return Detection(freq, note, octave, cents, confidence)

# --- RECORDER PROCESSING & QUANTIZATION LOGIC ---
def fill_gaps(midis: list[int | None], max_gap: int = 4) -> list[int | None]:
    n = len(midis)
    result = list(midis)
    
    # Fill None gaps between identical notes
    i = 0
    while i < n:
        if result[i] is not None:
            j = i + 1
            while j < n and result[j] is None:
                j += 1
            if j < n and j - i - 1 <= max_gap and result[i] == result[j]:
                for k in range(i + 1, j):
                    result[k] = result[i]
            i = j
        else:
            i += 1
            
    # Fill remaining None gaps of length <= max_gap with the preceding note
    for i in range(1, n):
        if result[i] is None:
            j = i
            while j < n and result[j] is None:
                j += 1
            gap_len = j - i
            if gap_len <= max_gap:
                fill_val = result[i - 1]
                if fill_val is not None:
                    for k in range(i, j):
                        result[k] = fill_val
            i = j
    return result

def process_detections_to_notes(raw_detections: list[tuple[float, int | None]], min_note_duration_ms: float = 250) -> list[tuple[int | None, float]]:
    if not raw_detections:
        return []
    
    times = [item[0] for item in raw_detections]
    midis = [item[1] for item in raw_detections]
    n = len(midis)
    
    midis = fill_gaps(midis, max_gap=4)
    
    segments = []
    current_midi = None
    current_start_idx = None
    
    for i in range(n):
        m = midis[i]
        if m != current_midi:
            if current_midi is not None and current_start_idx is not None:
                segments.append({
                    'midi': current_midi,
                    'start_idx': current_start_idx,
                    'end_idx': i - 1,
                    'start_time': times[current_start_idx],
                    'end_time': times[i - 1]
                })
            current_midi = m
            if m is not None:
                current_start_idx = i
            else:
                current_start_idx = None
                
    if current_midi is not None and current_start_idx is not None:
        segments.append({
            'midi': current_midi,
            'start_idx': current_start_idx,
            'end_idx': n - 1,
            'start_time': times[current_start_idx],
            'end_time': times[n - 1]
        })
        
    valid_segments = []
    for seg in segments:
        active_duration = (seg['end_time'] - seg['start_time']) * 1000.0
        if active_duration >= min_note_duration_ms:
            valid_segments.append(seg)
            
    if not valid_segments:
        return []
        
    events = []
    num_valid = len(valid_segments)
    for i in range(num_valid):
        seg = valid_segments[i]
        midi = seg['midi']
        
        if i < num_valid - 1:
            next_seg = valid_segments[i + 1]
            gap = (next_seg['start_time'] - seg['end_time']) * 1000.0
            if gap >= 200.0:
                # Distinct pause
                duration = (seg['end_time'] - seg['start_time']) * 1000.0
                events.append((midi, duration))
                events.append((None, gap))
            else:
                # Small gap: stretch previous note
                duration = (next_seg['start_time'] - seg['start_time']) * 1000.0
                events.append((midi, duration))
        else:
            # Last note
            duration = (seg['end_time'] - seg['start_time']) * 1000.0
            events.append((midi, duration))
            
    return events

def estimate_human_tempo(durations: list[float]) -> float:
    if not durations:
        return 600.0
    best_t = 600.0
    min_error = float('inf')
    # Grid search candidate human quarter note from 300ms to 1200ms
    beats = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    for t in range(300, 1205, 5):
        err = 0.0
        for d in durations:
            best_beat_diff = min(abs(d - b * t) for b in beats)
            err += best_beat_diff ** 2
        if err < min_error:
            min_error = err
            best_t = float(t)
    return best_t

def quantize_duration(duration_ms: float) -> int:
    return int(min(DURATIONS, key=lambda x: abs(x - duration_ms)))

def scale_and_quantize(notes: list[tuple[int | None, float]], t_human: float, t_robot: float = 500.0) -> list[tuple[int | None, int]]:
    ratio = t_robot / t_human
    processed = []
    for midi, duration in notes:
        scaled = duration * ratio
        quantized = quantize_duration(scaled)
        processed.append((midi, quantized))
    return processed

def fit_to_violin_range(midi: int) -> int | None:
    if MIDI_MIN <= midi <= MIDI_MAX:
        return midi
    for octave_shift in [-12, 12, -24, 24]:
        shifted = midi + octave_shift
        if MIDI_MIN <= shifted <= MIDI_MAX:
            return shifted
    return None

# --- MAIN APPLICATION CLASS ---
class ViolinRecorderEditorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.path: Path | None = None
        self.play_thread: threading.Thread | None = None
        self.stop_flag = threading.Event()
        
        # Audio stream state
        self.detector: ViolinPitchDetector | None = None
        self.stream: sd.InputStream | None = None
        self.running = False
        self.silent_ticks = 0
        self.ring = np.zeros(16384, dtype=np.float32)
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=16)
        self.status_queue: queue.Queue[str] = queue.Queue(maxsize=4)
        
        # Recording state
        self.recording = False
        self.raw_detections: list[tuple[float, int | None]] = []
        self.recorded_notes: list[tuple[int, float]] = []
        self.recording_start_time = 0.0
        self.min_confidence_var = tk.DoubleVar(value=0.55)
        self.min_note_duration_var = tk.IntVar(value=250)
        
        # Editor keyboard variables
        self.key_items: dict[int, list[int]] = {}
        self.key_rects: dict[int, int] = {}
        self.key_base_fill: dict[int, str] = {}
        self.duration_buttons: dict[int, ttk.Button] = {}
        
        self.root.title("Violin Scenario Recorder & Editor")
        self.root.minsize(1100, 800)
        self.root.configure(bg="#f6f7f9")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        
        # Tkinter variables
        self.duration_var = tk.IntVar(value=500)
        self.line_var = tk.StringVar(value="Line 1")
        self.status_var = tk.StringVar(value="Ready")
        
        # Tuner parameters variables
        self.device_var = tk.StringVar()
        self.fft_size_var = tk.IntVar(value=16384)
        self.block_size_var = tk.IntVar(value=1024)
        self.threshold_db_var = tk.DoubleVar(value=18.0)
        self.prominence_db_var = tk.DoubleVar(value=7.0)
        self.min_harmonic_matches_var = tk.IntVar(value=2)
        self.harmonics_var = tk.IntVar(value=8)
        self.a4_var = tk.DoubleVar(value=440.0)
        self.smoothing_var = tk.IntVar(value=5)
        self.min_freq_var = tk.DoubleVar(value=150.0)
        self.max_freq_var = tk.DoubleVar(value=3200.0)
        
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
        style.configure("Duration.TButton", font=("Segoe UI", 9), padding=(3, 7), background="#eef2f5", foreground="#26313a")
        style.configure("Selected.Duration.TButton", font=("Segoe UI", 9, "bold"), padding=(3, 7), background="#1c4f63", foreground="#ffffff")
        
        style.map("Primary.TButton", background=[("active", "#173f50"), ("disabled", "#aab7be")])
        style.map("Secondary.TButton", background=[("active", "#d9e1e7"), ("disabled", "#eef1f4")])
        style.map("Record.TButton", background=[("active", "#b71c1c"), ("disabled", "#e0e0e0")], foreground=[("disabled", "#9e9e9e")])
        style.map("Duration.TButton", background=[("active", "#d9e4ea")])
        style.map("Selected.Duration.TButton", background=[("active", "#173f50")], foreground=[("active", "#ffffff")])
        
        style.configure("TNotebook", background="#ffffff", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 10))

    def _build_ui(self) -> None:
        root_frame = ttk.Frame(self.root, style="Root.TFrame", padding=(18, 12, 18, 12))
        root_frame.pack(fill="both", expand=True)
        root_frame.columnconfigure(0, weight=1)
        root_frame.rowconfigure(2, weight=1) # The main editor area takes the expansion
        
        # --- TOP PANEL (TUNER, SETTINGS, RECORDER) ---
        top_frame = ttk.Frame(root_frame, style="Toolbar.TFrame")
        top_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        top_frame.columnconfigure(0, weight=2) # Tuner Display
        top_frame.columnconfigure(1, weight=3) # Audio Settings
        top_frame.columnconfigure(2, weight=2) # Recording Controls
        
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
        
        # Row 0: Device
        ttk.Label(settings_frame, text="Вход:", style="Label.TLabel").grid(row=0, column=0, sticky="w", pady=2)
        self.device_combo = ttk.Combobox(settings_frame, textvariable=self.device_var, state="readonly", font=("Segoe UI", 9))
        self.device_combo.grid(row=0, column=1, columnspan=3, sticky="ew", pady=2)
        
        # Row 1: FFT Size & Block Size
        ttk.Label(settings_frame, text="Размер FFT:", style="Label.TLabel").grid(row=1, column=0, sticky="w", pady=2)
        self.fft_combo = ttk.Combobox(settings_frame, textvariable=self.fft_size_var, values=[4096, 8192, 16384, 32768], state="readonly", width=8)
        self.fft_combo.grid(row=1, column=1, sticky="w", pady=2)
        
        ttk.Label(settings_frame, text="Блок:", style="Label.TLabel").grid(row=1, column=2, sticky="w", pady=2, padx=(10, 0))
        self.block_combo = ttk.Combobox(settings_frame, textvariable=self.block_size_var, values=[512, 1024, 2048], state="readonly", width=8)
        self.block_combo.grid(row=1, column=3, sticky="w", pady=2)
        
        # Row 2: Threshold & Prominence
        ttk.Label(settings_frame, text="Порог (dB):", style="Label.TLabel").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Spinbox(settings_frame, from_=5.0, to=100.0, increment=1.0, textvariable=self.threshold_db_var, width=8).grid(row=2, column=1, sticky="w", pady=2)
        
        ttk.Label(settings_frame, text="Пик (dB):", style="Label.TLabel").grid(row=2, column=2, sticky="w", pady=2, padx=(10, 0))
        ttk.Spinbox(settings_frame, from_=2.0, to=60.0, increment=0.5, textvariable=self.prominence_db_var, width=8).grid(row=2, column=3, sticky="w", pady=2)
        
        # Row 3: Smoothing & A4
        ttk.Label(settings_frame, text="Сглаживание:", style="Label.TLabel").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Spinbox(settings_frame, from_=1, to=20, increment=1, textvariable=self.smoothing_var, width=8).grid(row=3, column=1, sticky="w", pady=2)
        
        ttk.Label(settings_frame, text="A4 (Гц):", style="Label.TLabel").grid(row=3, column=2, sticky="w", pady=2, padx=(10, 0))
        ttk.Spinbox(settings_frame, from_=430.0, to=450.0, increment=0.5, textvariable=self.a4_var, width=8).grid(row=3, column=3, sticky="w", pady=2)
        
        # Row 4: Min/Max Freq
        freq_frame = ttk.Frame(settings_frame)
        freq_frame.grid(row=4, column=0, columnspan=4, sticky="w", pady=2)
        ttk.Label(freq_frame, text="Мин:", style="Label.TLabel").pack(side="left")
        ttk.Entry(freq_frame, textvariable=self.min_freq_var, width=5).pack(side="left", padx=2)
        ttk.Label(freq_frame, text="Гц  Макс:", style="Label.TLabel").pack(side="left", padx=(10, 0))
        ttk.Entry(freq_frame, textvariable=self.max_freq_var, width=5).pack(side="left", padx=2)
        ttk.Label(freq_frame, text="Гц", style="Label.TLabel").pack(side="left")
        
        # Row 5: Min Confidence & Apply Button
        conf_frame = ttk.Frame(settings_frame)
        conf_frame.grid(row=5, column=0, columnspan=3, sticky="w", pady=5)
        ttk.Label(conf_frame, text="Уверенность:", style="Label.TLabel").pack(side="left")
        ttk.Spinbox(conf_frame, from_=0.10, to=0.95, increment=0.05, textvariable=self.min_confidence_var, width=5).pack(side="left", padx=2)
        
        ttk.Button(settings_frame, text="Применить", command=self.apply_audio_settings, style="Secondary.TButton").grid(row=5, column=3, sticky="e", pady=5)
        
        # Populate Audio Devices
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
        
        # Adjustable noise/short note duration filter
        filter_frame = ttk.Frame(rec_frame)
        filter_frame.grid(row=1, column=0, sticky="ew", pady=(5, 2))
        ttk.Label(filter_frame, text="Фильтр шума <", style="Label.TLabel").pack(side="left")
        ttk.Spinbox(filter_frame, from_=50, to=1000, increment=50, textvariable=self.min_note_duration_var, width=5).pack(side="left", padx=2)
        ttk.Label(filter_frame, text="мс", style="Hint.TLabel").pack(side="left")
        
        self.rec_status_label = tk.Label(rec_frame, text="Запись не ведется", font=("Segoe UI", 10), fg="#66707a", bg="#ffffff", height=2)
        self.rec_status_label.grid(row=2, column=0, sticky="ew")
        
        # --- MIDDLE PANEL (TEMPO & QUANTIZATION) ---
        tempo_frame = ttk.LabelFrame(root_frame, text=" Обработка темпа и ритма ", padding=10)
        tempo_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
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
        editor_title_frame = ttk.Frame(root_frame, style="Toolbar.TFrame")
        editor_title_frame.grid(row=2, column=0, sticky="nsew")
        editor_title_frame.columnconfigure(0, weight=1)
        editor_title_frame.rowconfigure(1, weight=1)
        
        # Toolbar
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
        
        # Main editor workspace
        main_workspace = ttk.PanedWindow(editor_title_frame, orient="horizontal")
        main_workspace.grid(row=1, column=0, sticky="nsew")
        
        # Left pane: Text Editor
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
        
        # Right pane: Composer and Virtual Keyboard
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
        
        # --- FOOTER PANEL ---
        footer = ttk.Frame(root_frame, style="Toolbar.TFrame")
        footer.grid(row=3, column=0, sticky="ew", pady=(8, 0))
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
                
            self.detector = ViolinPitchDetector(
                sample_rate=sample_rate,
                fft_size=self.fft_size_var.get(),
                min_freq=self.min_freq_var.get(),
                max_freq=self.max_freq_var.get(),
                threshold_db=self.threshold_db_var.get(),
                prominence_db=self.prominence_db_var.get(),
                min_harmonic_matches=self.min_harmonic_matches_var.get(),
                harmonics=self.harmonics_var.get(),
                a4=self.a4_var.get(),
                smoothing=self.smoothing_var.get()
            )
            
            self.ring = np.zeros(self.fft_size_var.get(), dtype=np.float32)
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
            if result is None:
                self.silent_ticks += 1
                if self.silent_ticks >= 4:
                    self.live_note_var.set("--")
                    self.live_hz_var.set("-- Hz")
                    self.live_conf_var.set("Conf: 0.00")
                    self.activity_canvas.itemconfigure(self.activity_dot, fill="#c8ccd2")
                    if self.recording:
                        self.raw_detections.append((time.time(), None))
            else:
                self.silent_ticks = 0
                self.live_note_var.set(f"{result.note}{result.octave}")
                self.live_hz_var.set(f"{result.frequency:.2f} Hz")
                self.live_conf_var.set(f"Conf: {result.confidence:.2f}")
                
                # Dynamic green/orange feedback based on confidence
                color = "#4caf50" if result.confidence > 0.65 else "#ff9800"
                self.activity_canvas.itemconfigure(self.activity_dot, fill=color)
                
                if self.recording:
                    if result.confidence >= self.min_confidence_var.get():
                        nearest_midi = int(round(69.0 + 12.0 * math.log2(result.frequency / self.detector.a4)))
                        self.raw_detections.append((time.time(), nearest_midi))
                    else:
                        self.raw_detections.append((time.time(), None))
                    
            if self.recording:
                self.update_recording_status()

        self.root.after(50, self.poll_audio)

    # --- RECORDING PROCESS ---
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
        
        # Process raw stream
        self.recorded_notes = process_detections_to_notes(self.raw_detections, min_note_duration_ms=self.min_note_duration_var.get())
        
        if not self.recorded_notes:
            self.rec_status_label.configure(text="Запись пуста (ноты не распознаны)", fg="#66707a")
            self.btn_auto_tempo.state(["disabled"])
            self.btn_insert_scenario.state(["disabled"])
            messagebox.showwarning("Запись пуста", "Не удалось распознать стабильные ноты. Попробуйте сыграть громче или настроить порог чувствительности (dB).")
            return
            
        # Estimate human tempo
        durations = [d for m, d in self.recorded_notes if m is not None]
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
        # Count notes so far in background
        current_notes = process_detections_to_notes(self.raw_detections, min_note_duration_ms=self.min_note_duration_var.get())
        self.rec_status_label.configure(
            text=f"Запись: {elapsed:.1f} сек\nОбнаружено нот: {len(current_notes)}",
            fg="#d32f2f"
        )

    def auto_detect_tempo(self) -> None:
        if not self.recorded_notes:
            return
        durations = [d for m, d in self.recorded_notes if m is not None]
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
            
        previous_string, last_dir, X_start = parse_last_state_and_position(self.editor.get("1.0", "insert"))
        initial_dir = 1 - last_dir
        
        valid_notes_for_planning = []
        warnings_notes = []
        
        for midi, duration in scaled_notes:
            if midi is None:
                valid_notes_for_planning.append((None, duration))
            else:
                fitted_midi = fit_to_violin_range(midi)
                if fitted_midi is None:
                    warnings_notes.append(midi_name(midi))
                else:
                    valid_notes_for_planning.append((fitted_midi, duration))
                
        if not valid_notes_for_planning:
            messagebox.showinfo("Пусто", "Нет валидных нот в диапазоне скрипки.")
            return
            
        actual_notes = [note for note in valid_notes_for_planning if note[0] is not None]
        planned_dirs = optimize_bow_directions(actual_notes, X_start, initial_dir)
        
        commands = []
        note_idx = 0
        for fitted_midi, duration in valid_notes_for_planning:
            if fitted_midi is None:
                commands.append(f"/p;{duration}")
            else:
                try:
                    string, position = choose_position(fitted_midi, previous_string)
                    speed = speed_for_duration(duration)
                    direction = planned_dirs[note_idx]
                    note_idx += 1
                    commands.append(f"/sn;{string};{position}")
                    commands.append(f"/bow;{direction};{duration};{speed}")
                    previous_string = string
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
                f"Некоторые ноты ({', '.join(set(warnings_notes))}) выходят за пределы диапазона скрипки G3-B5. Они были транспонированы по октавам или пропущены."
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
        if event.widget != self.editor:
            self.root.focus_set()

    def duration_shortcut(self, event: tk.Event, duration_ms: int) -> str | None:
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
        previous_string, last_dir, X = parse_last_state_and_position(self.editor.get("1.0", "insert"))
        default_direction = 1 - last_dir
        speed = speed_for_duration(duration)
        d = duration * speed
        
        if default_direction == 1:
            if X + d <= 100000:
                next_direction = 1
            elif X - d >= 0:
                next_direction = 0
            else:
                next_direction = 1
        else:
            if X - d >= 0:
                next_direction = 0
            elif X + d <= 100000:
                next_direction = 1
            else:
                next_direction = 0
                
        string, position = choose_position(midi, previous_string)
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
        (0.45, None), (0.50, None), (0.55, None), (0.60, None), (0.65, None), # silence (250ms gap)
        (0.70, 64), (0.75, 64), (0.80, 64), (0.85, 64),
    ]
    notes = process_detections_to_notes(raw, min_note_duration_ms=80)
    if len(notes) != 4:
        failures += 1
        print(f"Fail: expected 4 events in process_detections_to_notes, got {len(notes)}: {notes}")
    else:
        if notes[0][0] != 60 or abs(notes[0][1] - 300) > 10:
            failures += 1
            print(f"Fail: note 0 duration error, got {notes[0]}")
        if notes[1][0] != 62 or abs(notes[1][1] - 100) > 10:
            failures += 1
            print(f"Fail: note 1 duration error, got {notes[1]}")
        if notes[2][0] is not None or abs(notes[2][1] - 300.0) > 10:
            failures += 1
            print(f"Fail: pause event error, got {notes[2]}")
        if notes[3][0] != 64 or abs(notes[3][1] - 150) > 10:
            failures += 1
            print(f"Fail: note 3 duration error, got {notes[3]}")

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
