#!/usr/bin/env python3
"""Realtime violin note detector.

The detector intentionally does not choose the loudest spectral peak.  It
finds stable low-frequency peaks and prefers the lowest peak that is backed by
harmonic energy, which is better for violin tones with strong overtones.
"""

from __future__ import annotations

import argparse
import math
import queue
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import sounddevice as sd
from scipy.signal import find_peaks


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


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
        fft_size: int = 8192,
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

        # Prefer the lowest plausible spectral peak, but reject very weak lows
        # if a nearby higher candidate has much stronger harmonic evidence.
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


def list_devices() -> None:
    print(sd.query_devices())


def synthetic_violin_note(freq: float, sample_rate: int, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    signal = np.zeros_like(t)
    amplitudes = [0.28, 1.0, 0.72, 0.52, 0.38, 0.28, 0.18, 0.12]
    for harmonic, amplitude in enumerate(amplitudes, start=1):
        signal += amplitude * np.sin(2.0 * np.pi * freq * harmonic * t)
    signal *= np.exp(-1.5 * t) * 0.5
    signal += 0.004 * np.random.default_rng(7).normal(size=signal.size)
    return signal.astype(np.float32)


def run_self_test(args: argparse.Namespace) -> int:
    detector = make_detector(args)
    failures = 0
    tests = [196.0, 293.66, 440.0, 659.25, 880.0]
    for freq in tests:
        samples = synthetic_violin_note(freq, args.sample_rate)
        result = detector.detect(samples)
        if result is None:
            failures += 1
            print(f"{freq:7.2f} Hz -> no detection")
            continue
        error_cents = 1200.0 * math.log2(result.frequency / freq)
        ok = abs(error_cents) < 20.0
        failures += 0 if ok else 1
        status = "OK" if ok else "FAIL"
        print(f"{freq:7.2f} Hz -> {result.label} error={error_cents:+5.1f} cents {status}")
    return 1 if failures else 0


def make_detector(args: argparse.Namespace) -> ViolinPitchDetector:
    return ViolinPitchDetector(
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


def audio_callback(audio_queue: queue.Queue[np.ndarray]):
    def callback(indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags) -> None:
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(indata.copy())

    return callback


def realtime(args: argparse.Namespace) -> int:
    detector = make_detector(args)
    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
    ring = np.zeros(args.fft_size, dtype=np.float32)
    last_print = 0.0

    print("Listening. Press Ctrl+C to stop.")
    with sd.InputStream(
        device=args.device,
        channels=1,
        samplerate=args.sample_rate,
        blocksize=args.block_size,
        callback=audio_callback(audio_queue),
    ):
        while True:
            block = audio_queue.get()
            mono = block[:, 0].astype(np.float32, copy=False)
            if mono.size >= ring.size:
                ring[:] = mono[-ring.size :]
            else:
                ring[:-mono.size] = ring[mono.size :]
                ring[-mono.size :] = mono

            now = time.monotonic()
            if now - last_print < args.update_interval:
                continue
            last_print = now

            result = detector.detect(ring)
            if result is None:
                sys.stdout.write("\r" + " " * 72 + "\r")
                sys.stdout.write("No stable note")
            else:
                sys.stdout.write("\r" + " " * 72 + "\r")
                sys.stdout.write(result.label)
            sys.stdout.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime violin fundamental frequency detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--list-devices", action="store_true", help="show available audio devices and exit")
    parser.add_argument("--device", default=None, help="input device id or name")
    parser.add_argument("--sample-rate", type=int, default=48_000, help="audio sample rate")
    parser.add_argument("--block-size", type=int, default=1024, help="audio callback block size")
    parser.add_argument("--fft-size", type=int, default=16_384, help="analysis window size")
    parser.add_argument("--update-interval", type=float, default=0.08, help="screen update interval in seconds")
    parser.add_argument("--min-freq", type=float, default=150.0, help="lowest expected violin frequency")
    parser.add_argument("--max-freq", type=float, default=3_200.0, help="highest expected fundamental frequency")
    parser.add_argument("--threshold-db", type=float, default=18.0, help="peak height above local noise floor")
    parser.add_argument("--prominence-db", type=float, default=7.0, help="minimum peak prominence")
    parser.add_argument("--min-harmonic-matches", type=int, default=2, help="set 0 to accept the lowest peak directly")
    parser.add_argument("--harmonics", type=int, default=8, help="harmonics checked for support")
    parser.add_argument("--a4", type=float, default=440.0, help="A4 reference frequency for note names")
    parser.add_argument("--smoothing", type=int, default=5, help="median smoothing window")
    parser.add_argument("--self-test", action="store_true", help="run synthetic violin-like tests and exit")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        list_devices()
        return 0
    if args.self_test:
        return run_self_test(args)

    try:
        return realtime(args)
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
