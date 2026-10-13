"""
Module 5 schemas + face-detection backends.

Coordinates convention: everything normalized to [0,1] relative to the
frame (x = fraction of width, y = fraction of height). This makes crop
paths computed on the 480p proxy directly applicable to the 1080p
normalized video in Module 6.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------- schemas

@dataclass
class FaceBox:
    """One detected face, normalized coords."""
    cx: float          # center x, 0..1
    cy: float          # center y, 0..1
    w: float           # width fraction
    h: float           # height fraction

    @property
    def area(self) -> float:
        return self.w * self.h


class CropKeyframe(BaseModel):
    t: float                     # seconds, relative to CLIP start
    cx: float = Field(ge=0.0, le=1.0)   # crop-center x, normalized
    cut: bool = False            # True => hard jump, don't interpolate into


class CropPath(BaseModel):
    """9:16 crop path for one clip. Crop is full source height; only the
    horizontal center moves. cy handles headroom inside Module 6's crop."""
    schema_version: int = 1
    clip_id: int
    mode: Literal["tracked", "center", "static"]
    aspect_w: int = 9
    aspect_h: int = 16
    cy: float = 0.5              # vertical center (fixed per clip)
    keyframes: list[CropKeyframe] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ordered(self):
        for a, b in zip(self.keyframes, self.keyframes[1:]):
            if b.t < a.t - 1e-6:
                raise ValueError("crop keyframes out of order")
        return self

    def cx_at(self, t: float) -> float:
        """Sample the path at time t (piecewise-linear, hard cuts honored)."""
        kfs = self.keyframes
        if not kfs:
            return 0.5
        if t <= kfs[0].t:
            return kfs[0].cx
        for a, b in zip(kfs, kfs[1:]):
            if t <= b.t:
                if b.cut or b.t == a.t:
                    return a.cx if t < b.t else b.cx
                frac = (t - a.t) / (b.t - a.t)
                return a.cx + frac * (b.cx - a.cx)
        return kfs[-1].cx

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "CropPath":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- detectors

class FaceDetector(abc.ABC):
    name = "base"

    @abc.abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        """Detect faces in one BGR frame -> normalized FaceBoxes."""


class HaarDetector(FaceDetector):
    """OpenCV Haar cascade. Bundled with opencv — zero downloads. Weaker
    than MediaPipe (frontal faces only) but a solid offline fallback."""
    name = "haar"

    def __init__(self):
        import cv2
        self._cv2 = cv2
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        cv2 = self._cv2
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape[:2]
        rects = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(int(W * 0.05), int(H * 0.05)))
        return [FaceBox(cx=(x + w / 2) / W, cy=(y + h / 2) / H,
                        w=w / W, h=h / H) for (x, y, w, h) in rects]


class MediaPipeDetector(FaceDetector):
    """MediaPipe face detection — the recommended backend on a real machine.
    pip install mediapipe. Lazy import so the pipeline doesn't require it."""
    name = "mediapipe"

    def __init__(self, min_confidence: float = 0.5):
        import mediapipe as mp
        self._mp = mp
        self._det = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=min_confidence)

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        rgb = frame_bgr[:, :, ::-1]
        res = self._det.process(rgb)
        out = []
        for d in (res.detections or []):
            bb = d.location_data.relative_bounding_box
            out.append(FaceBox(cx=bb.xmin + bb.width / 2,
                               cy=bb.ymin + bb.height / 2,
                               w=bb.width, h=bb.height))
        return out


class MockDetector(FaceDetector):
    """Scripted detector for tests: a function t -> list[FaceBox].
    The sampler passes the frame timestamp via `set_time` before detect."""
    name = "mock"

    def __init__(self, script):
        self._script = script
        self._t = 0.0

    def set_time(self, t: float) -> None:
        self._t = t

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        return self._script(self._t)


def get_detector(engine: str) -> FaceDetector:
    if engine == "haar":
        return HaarDetector()
    if engine == "mediapipe":
        return MediaPipeDetector()
    raise ValueError(f"Unknown face detector: {engine}")
