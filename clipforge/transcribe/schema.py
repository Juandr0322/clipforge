"""
Transcript schema — Module 2's output contract.

Consumed by:
  Module 3 (highlight detection) — segments/text
  Module 4 (boundary refinement) — word-level timestamps
  Module 6 (captions)            — word-level timestamps

Units rule (project-wide): ALL timestamps are float SECONDS. Never ms.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Word(BaseModel):
    w: str                      # the word text (may include punctuation)
    start: float                # seconds
    end: float                  # seconds
    speaker: str = "S1"
    confidence: Optional[float] = None

    @model_validator(mode="after")
    def _valid_span(self):
        if self.end < self.start:
            raise ValueError(f"word '{self.w}': end {self.end} < start {self.start}")
        return self


class Segment(BaseModel):
    """A sentence-ish unit as emitted by the ASR engine."""
    id: int
    text: str
    start: float
    end: float
    speaker: str = "S1"

    @model_validator(mode="after")
    def _valid_span(self):
        if self.end < self.start:
            raise ValueError(f"segment {self.id}: end < start")
        return self


class Transcript(BaseModel):
    schema_version: int = 1
    language: str = "en"
    engine: str                     # e.g. "faster-whisper/small", "mock"
    audio_duration_sec: float
    words: list[Word] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)

    @model_validator(mode="after")
    def _invariants(self):
        # words must be time-ordered (allowing exact ties)
        for a, b in zip(self.words, self.words[1:]):
            if b.start < a.start - 1e-6:
                raise ValueError(
                    f"words out of order at '{a.w}'({a.start}) -> '{b.w}'({b.start})")
        # segments must be ordered and ids unique
        ids = [s.id for s in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate segment ids")
        for a, b in zip(self.segments, self.segments[1:]):
            if b.start < a.start - 1e-6:
                raise ValueError(f"segments out of order at id {a.id}->{b.id}")
        # nothing may exceed the audio duration (small tolerance for ASR drift)
        tol = 0.75
        for w in self.words:
            if w.end > self.audio_duration_sec + tol:
                raise ValueError(
                    f"word '{w.w}' ends at {w.end}s beyond audio "
                    f"({self.audio_duration_sec}s)")
        return self

    # ---------- convenience ----------
    @property
    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()

    def words_between(self, start: float, end: float) -> list[Word]:
        """Words whose midpoint falls inside [start, end)."""
        return [w for w in self.words
                if start <= (w.start + w.end) / 2 < end]

    # ---------- persistence ----------
    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(self.model_dump_json(indent=2))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Transcript":
        return cls.model_validate_json(path.read_text())
