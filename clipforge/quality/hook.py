"""
First-3-seconds hook score for one rendered clip.

Four sub-scores, each 0-100 with a one-line reason, combined into a weighted
0-100 total:

  speech  how soon the first word is spoken (second 1)
  motion  how much the picture changes in second 1
  face    how much of the first 3 s has a face on screen
  text    how much of the first 3 s has on-screen text

On-screen text comes from what the pipeline itself puts on screen (caption
cards, plus any overlay spans the caller passes in). No OCR.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

from clipforge.transcribe.schema import Word

WINDOW_SEC = 3.0
WEIGHTS = {"speech": 0.30, "motion": 0.20, "face": 0.25, "text": 0.25}

SPEECH_FULL_SEC = 0.3     # first word by here scores 100
SPEECH_ZERO_SEC = 3.0     # first word at/after here scores 0
MOTION_FULL_DIFF = 6.0    # mean abs gray diff (0-255) per frame that scores 100
TEXT_FULL_COVER = 0.7     # text over this share of the window scores 100


def _sub(score: float, reason: str) -> dict:
    return {"score": int(round(max(0.0, min(100.0, score)))),
            "reason": reason}


def score_speech(words: Iterable[Word], clip_start: float,
                 clip_end: float) -> dict:
    first: Optional[float] = None
    for w in words:
        if clip_start <= (w.start + w.end) / 2 < clip_end:
            first = max(0.0, w.start - clip_start)
            break
    if first is None or first >= SPEECH_ZERO_SEC:
        return _sub(0, "no speech in the first 3 s")
    if first <= SPEECH_FULL_SEC:
        return _sub(100, f"first word at {first:.1f} s")
    frac = (first - SPEECH_FULL_SEC) / (SPEECH_ZERO_SEC - SPEECH_FULL_SEC)
    return _sub(100 * (1 - frac), f"first word at {first:.1f} s")


def score_motion(frames_gray: list) -> dict:
    """`frames_gray`: consecutive grayscale frames (numpy arrays) covering
    the first second."""
    import numpy as np
    if len(frames_gray) < 2:
        return _sub(0, "too few frames to measure motion")
    diffs = [float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
             for a, b in zip(frames_gray, frames_gray[1:])]
    d = sum(diffs) / len(diffs)
    if d < 0.05:
        return _sub(0, "picture is static in the first second")
    return _sub(100 * d / MOTION_FULL_DIFF,
                f"average frame change {d:.1f} in the first second")


def score_face(face_flags: list[bool]) -> dict:
    """`face_flags`: one bool per sampled frame of the first 3 s."""
    if not face_flags:
        return _sub(0, "no frames to check for a face")
    n = sum(face_flags)
    if n == 0:
        return _sub(0, "no face in the first 3 s")
    return _sub(100 * n / len(face_flags),
                f"face on screen in {n} of {len(face_flags)} sampled frames")


def _covered(spans: Iterable[tuple[float, float]]) -> float:
    """Seconds of [0, WINDOW_SEC] covered by the union of clip-relative spans."""
    clipped = sorted((max(0.0, a), min(WINDOW_SEC, b)) for a, b in spans
                     if math.isfinite(a) and math.isfinite(b))
    total, cur_end = 0.0, 0.0
    for a, b in clipped:
        if b <= a:
            continue
        a = max(a, cur_end)
        if b > a:
            total += b - a
            cur_end = b
    return total


def score_text(spans: Iterable[tuple[float, float]]) -> dict:
    """`spans`: clip-relative (start, end) seconds of on-screen text
    (captions and overlays)."""
    cover = _covered(spans) / WINDOW_SEC
    if cover <= 0:
        return _sub(0, "no on-screen text in the first 3 s")
    return _sub(100 * cover / TEXT_FULL_COVER,
                f"text on screen for {cover * 100:.0f}% of the first 3 s")


def combine(subs: dict[str, dict]) -> dict:
    total = sum(WEIGHTS[k] * subs[k]["score"] for k in WEIGHTS)
    return {"total": int(round(total)), "sub_scores": subs}


# ------------------------------------------------------------ video access

def read_gray_frames(video_path, seconds: float = 1.0, size=(96, 170)) -> list:
    """The first `seconds` of a video as small grayscale frames."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("cannot open rendered clip")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        out = []
        for _ in range(max(2, int(math.ceil(fps * seconds)))):
            ok, frame = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out.append(cv2.resize(g, size))
        return out
    finally:
        cap.release()


def face_flags(video_path, detector, fps: float = 2.0) -> list[bool]:
    from clipforge.reframe.core import sample_frames
    flags = []
    for t, frame in sample_frames(video_path, 0.0, WINDOW_SEC, fps=fps):
        if hasattr(detector, "set_time"):
            detector.set_time(t)
        flags.append(bool(detector.detect(frame)))
    return flags


def hook_score(video_path, words: list[Word], clip_start: float,
               clip_end: float, detector,
               text_spans: Iterable[tuple[float, float]] = ()) -> dict:
    """Score the first 3 s of one rendered clip. `text_spans` are
    clip-relative (start, end) seconds of on-screen text. A signal that
    cannot be measured scores 0 with a reason rather than failing."""
    subs = {"speech": score_speech(words, clip_start, clip_end)}
    try:
        subs["motion"] = score_motion(read_gray_frames(video_path))
    except Exception:  # noqa: BLE001 - best effort, no paths in the reason
        subs["motion"] = _sub(0, "motion could not be measured")
    try:
        subs["face"] = score_face(face_flags(video_path, detector))
    except Exception:  # noqa: BLE001
        subs["face"] = _sub(0, "face check unavailable")
    subs["text"] = score_text(text_spans)
    return combine(subs)
