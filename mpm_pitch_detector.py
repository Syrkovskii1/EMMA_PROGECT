import math
from dataclasses import dataclass
import numpy as np

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

@dataclass
class PitchResult:
    frequency: float
    note: str
    octave: int
    cents: float
    confidence: float
    
    @property
    def label(self) -> str:
        sign = "+" if self.cents >= 0 else ""
        return f"{self.note}{self.octave} {self.frequency:8.2f} Hz {sign}{self.cents:5.1f} cents conf={self.confidence:4.2f}"

class MpmPitchDetector:
    """Pitch detector using McLeod Pitch Method (MPM)."""
    def __init__(
        self,
        sample_rate: int = 48000,
        min_freq: float = 150.0,
        max_freq: float = 4000.0,
        clarity_threshold: float = 0.45,
        cutoff_k: float = 0.85,
        a4: float = 440.0
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

        # 1. Fast autocorrelation via FFT
        fft_len = 2 ** int(np.ceil(np.log2(2 * n)))
        f_sig = np.fft.rfft(signal, n=fft_len)
        autocorr = np.fft.irfft(f_sig * np.conj(f_sig))[:n]

        # 2. Denominator m(tau) for NSDF
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

        # NSDF (Normalized Square Difference Function)
        nsdf = 2.0 * autocorr / m

        # 3. Lag search bounds
        min_lag = max(2, int(self.sample_rate / self.max_freq))
        max_lag = min(n - 2, int(self.sample_rate / self.min_freq))

        if min_lag >= max_lag:
            return None

        # 4. Find NSDF peaks
        peaks = []
        for tau in range(min_lag, max_lag):
            if nsdf[tau] > nsdf[tau - 1] and nsdf[tau] >= nsdf[tau + 1]:
                if nsdf[tau] > 0:
                    peaks.append(tau)

        if not peaks:
            return None

        # 5. McLeod algorithm for peak selection (combat octave errors)
        max_peak_val = max(nsdf[p] for p in peaks)
        if max_peak_val < self.clarity_threshold:
            return None  # Too noisy / no distinct pitch

        threshold = max_peak_val * self.cutoff_k
        selected_peak = None
        for p in peaks:
            if nsdf[p] >= threshold:
                selected_peak = p
                break

        if selected_peak is None:
            return None

        # 6. Parabolic interpolation of peak vertex
        p = selected_peak
        alpha, beta, gamma = nsdf[p - 1], nsdf[p], nsdf[p + 1]
        denom = alpha - 2 * beta + gamma
        delta = (alpha - gamma) / (2.0 * denom) if abs(denom) > 1e-9 else 0.0

        refined_lag = p + delta
        freq = self.sample_rate / refined_lag

        if not (self.min_freq <= freq <= self.max_freq):
            return None

        # Calculate deviation in cents
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

# --- RECORDER PROCESSING & QUANTIZATION LOGIC ---
OPEN_STRINGS = {4: 55, 3: 62, 2: 69, 1: 76}  # G3, D4, A4, E5
DURATIONS = (250, 500, 750, 1000, 1250, 1500, 1750, 2000)
MIDI_MIN = OPEN_STRINGS[4]
MIDI_MAX = OPEN_STRINGS[1] + 7

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
