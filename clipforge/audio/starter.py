"""
Built-in starter sound effects, synthesised with numpy.

They make the audio layer work before Diego has added a library, and they
need no licence because ClipForge generates them. They are deliberately
simple; real files from the library (docs/AUDIO_SOURCES.md) win whenever a
tag has one. Generation is deterministic (fixed seeds), so renders and tests
are repeatable. Arrays are float32 stereo, shape (n, 2), at SR.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

SR = 48_000
STARTER_TAGS = ("impact", "whoosh", "pop", "riser")


def _t(sec: float) -> np.ndarray:
    return np.arange(int(round(sec * SR))) / SR


def _onepole_lowpass(x: np.ndarray, cutoff_hz: np.ndarray) -> np.ndarray:
    """One-pole low-pass whose cutoff can change every sample."""
    a = np.exp(-2.0 * np.pi * np.clip(cutoff_hz, 20.0, SR / 2.2) / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):                  # ~1 s of audio: fast enough
        acc = (1.0 - a[i]) * x[i] + a[i] * acc
        y[i] = acc
    return y


def _fade(x: np.ndarray, sec: float = 0.004) -> np.ndarray:
    n = min(len(x) // 2, int(sec * SR))
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n)
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def _stereo(mono: np.ndarray, pan: np.ndarray | None = None) -> np.ndarray:
    if pan is None:
        out = np.stack([mono, mono], axis=1)
    else:            # pan in [-1, 1], constant-power
        ang = (pan + 1.0) * np.pi / 4.0
        out = np.stack([mono * np.cos(ang), mono * np.sin(ang)], axis=1)
    peak = float(np.max(np.abs(out))) or 1.0
    return (out * (0.9 / peak)).astype(np.float32)


def _impact() -> np.ndarray:
    t = _t(0.9)
    freq = 45.0 + 75.0 * np.exp(-t / 0.05)              # 120 Hz -> 45 Hz drop
    phase = 2 * np.pi * np.cumsum(freq) / SR
    body = np.sin(phase) * np.exp(-t / 0.28)
    rng = np.random.default_rng(48001)
    click = rng.standard_normal(len(t)) * np.exp(-t / 0.006) * 0.6
    x = np.tanh(1.8 * (body + click))                   # a little punch
    return _stereo(_fade(x))


def _whoosh() -> np.ndarray:
    t = _t(0.6)
    peak_at = 0.40
    env = np.where(t < peak_at, (t / peak_at) ** 2,
                   np.exp(-(t - peak_at) / 0.06))
    rng = np.random.default_rng(48002)
    noise = rng.standard_normal(len(t))
    cutoff = 300.0 + 4200.0 * env
    x = _onepole_lowpass(noise, cutoff)
    x = _onepole_lowpass(x, cutoff) * env
    pan = np.clip(-0.8 + 1.6 * t / t[-1], -1.0, 1.0)    # sweeps left to right
    return _stereo(_fade(x), pan)


def _pop() -> np.ndarray:
    t = _t(0.12)
    freq = 500.0 + 900.0 * np.exp(-t / 0.02)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    env = np.minimum(1.0, t / 0.002) * np.exp(-t / 0.03)
    return _stereo(_fade(np.sin(phase) * env, 0.002))


def _riser() -> np.ndarray:
    t = _t(1.4)
    prog = t / t[-1]
    env = prog ** 2.2
    rng = np.random.default_rng(48004)
    noise = _onepole_lowpass(rng.standard_normal(len(t)),
                             300.0 + 5700.0 * prog ** 1.5)
    freq = 180.0 * (5.0 ** prog)                        # 180 Hz -> 900 Hz
    tone = np.sin(2 * np.pi * np.cumsum(freq) / SR)
    x = (0.7 * noise / (np.max(np.abs(noise)) or 1.0) + 0.3 * tone) * env
    return _stereo(_fade(x, 0.02))


_MAKERS = {"impact": _impact, "whoosh": _whoosh, "pop": _pop,
           "riser": _riser}


@lru_cache(maxsize=None)
def _cached(tag: str) -> np.ndarray:
    arr = _MAKERS[tag]()
    arr.setflags(write=False)
    return arr


def starter_sound(tag: str) -> np.ndarray:
    """A copy of the generated sound for `tag` (one of STARTER_TAGS)."""
    if tag not in _MAKERS:
        raise KeyError(tag)
    return _cached(tag).copy()
