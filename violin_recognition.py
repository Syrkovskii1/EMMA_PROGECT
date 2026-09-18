from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks

# These constants should match those in emma_studio.py
OPEN_STRINGS = {4: 55, 3: 62, 2: 69, 1: 76}  # G3, D4, A4, E5
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
DURATIONS = (250, 500, 750, 1000, 1250, 1500, 1750, 2000)
MIDI_MIN = OPEN_STRINGS[4]
MIDI_MAX = OPEN_STRINGS[1] + 7


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

        candidates.sort(key=lambda item: item[0])
        best_freq, best_confidence, _, _ = candidates[0]
        for freq, confidence, _, _ in candidates[1:]:
            if freq > best_freq * 1.35:
                break
            if confidence > best_confidence + 0.35:
                best_freq, best_confidence = freq, confidence

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
def process_detections_to_notes(raw_detections: list[tuple[float, int | None]], min_note_duration_ms: float = 300) -> list[tuple[int, float]]:
    if not raw_detections:
        return []
    
    midis = [item[1] for item in raw_detections]
    times = [item[0] for item in raw_detections]
    n = len(midis)
    
    # Fill brief dropouts (1 frame gap)
    for i in range(1, n - 1):
        if midis[i] is None and midis[i-1] is not None and midis[i-1] == midis[i+1]:
            midis[i] = midis[i-1]
            
    notes: list[tuple[int, float]] = []
    current_midi = None
    current_start_time = None
    
    for i in range(n):
        t = times[i]
        m = midis[i]
        if m != current_midi:
            if current_midi is not None:
                duration = (t - current_start_time) * 1000.0
                if duration >= min_note_duration_ms:
                    notes.append((current_midi, duration))
            current_midi = m
            current_start_time = t
            
    if current_midi is not None and current_start_time is not None:
        dt = (times[-1] - times[0]) / (n - 1) if n > 1 else 0.05
        duration = (times[-1] + dt - current_start_time) * 1000.0
        if duration >= min_note_duration_ms:
            notes.append((current_midi, duration))
            
    return notes

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

def scale_and_quantize(notes: list[tuple[int, float]], t_human: float, t_robot: float = 500.0) -> list[tuple[int, int]]:
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
