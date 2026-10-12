"""
Audio energy analysis for Module 4.

Computes a frame-level RMS energy profile of the job's audio.wav so cuts
can be placed at the quietest instant inside a word gap (between breaths),
not just at the arithmetic midpoint.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

FRAME_SEC = 0.02  # 20ms frames


class EnergyProfile:
    def __init__(self, rms: np.ndarray, frame_sec: float, duration: float):
        self._rms = rms
        self._frame = frame_sec
        self.duration = duration

    @classmethod
    def from_wav(cls, path: Path) -> "EnergyProfile":
        with wave.open(str(path), "rb") as f:
            sr = f.getframerate()
            n = f.getnframes()
            if f.getsampwidth() != 2 or f.getnchannels() != 1:
                raise ValueError(
                    "EnergyProfile expects 16-bit mono WAV (Module 1 output).")
            audio = np.frombuffer(f.readframes(n), dtype=np.int16)
        duration = n / sr
        hop = int(sr * FRAME_SEC)
        n_frames = max(1, len(audio) // hop)
        trimmed = audio[: n_frames * hop].astype(np.float64)
        frames = trimmed.reshape(n_frames, hop)
        rms = np.sqrt((frames ** 2).mean(axis=1))
        return cls(rms, FRAME_SEC, duration)

    def quietest_point(self, t0: float, t1: float) -> float:
        """Time of minimum RMS energy inside [t0, t1]. Falls back to the
        midpoint for degenerate/out-of-range windows."""
        if t1 <= t0:
            return (t0 + t1) / 2
        i0 = max(0, int(t0 / self._frame))
        i1 = min(len(self._rms), int(np.ceil(t1 / self._frame)))
        if i1 <= i0:
            return (t0 + t1) / 2
        idx = i0 + int(np.argmin(self._rms[i0:i1]))
        # center of the winning frame, clamped into the window
        t = (idx + 0.5) * self._frame
        return float(min(max(t, t0), t1))
