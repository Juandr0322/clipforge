"""
Clip-candidate schema — Module 3's output contract.

Consumed by Module 4 (boundary refinement) and the review UI (Module 7).
All timestamps are float seconds, same rule as everywhere.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

MIN_CLIP_SEC = 15.0
MAX_CLIP_SEC = 90.0

ClipType = Literal["opinion", "story", "insight", "howto", "funny",
                   "emotional", "other"]

HookType = Literal["bold_claim", "question", "story_open", "contrarian",
                   "number", "reveal", "emotion", "other"]
HOOK_TYPES = frozenset(HookType.__args__)


class ClipCandidate(BaseModel):
    id: int
    start: float
    end: float
    score: int = Field(ge=0, le=100)     # overall clip-worthiness
    hook: str                            # first-3-seconds hook line
    reason: str                          # why the model picked it
    type: ClipType = "other"
    # v2, all optional. Absolute seconds, trigger <= peak <= payoff in [start, end].
    trigger_t: float | None = None
    reaction_peak_t: float | None = None
    payoff_end_t: float | None = None
    context_setup: str | None = None     # one line of needed background
    hook_type: HookType | None = None

    @model_validator(mode="after")
    def _valid(self):
        if self.end <= self.start:
            raise ValueError(f"candidate {self.id}: end <= start")
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlap_fraction(self, other: "ClipCandidate") -> float:
        """Overlap as a fraction of the SHORTER clip (0..1)."""
        inter = min(self.end, other.end) - max(self.start, other.start)
        if inter <= 0:
            return 0.0
        return inter / min(self.duration, other.duration)


class RejectedCandidate(BaseModel):
    """A moment the model found but failed its own stranger check."""
    start: float
    end: float
    hook: str = ""
    reason: str


class CandidateSet(BaseModel):
    schema_version: int = 2
    engine: str                          # e.g. "anthropic/claude-sonnet-4-6"
    source_duration_sec: float
    candidates: list[ClipCandidate] = Field(default_factory=list)
    rejected: list[RejectedCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _invariants(self):
        ids = [c.id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate candidate ids")
        for c in self.candidates:
            if c.start < -1e-6 or c.end > self.source_duration_sec + 0.75:
                raise ValueError(
                    f"candidate {c.id} [{c.start},{c.end}] outside source "
                    f"duration {self.source_duration_sec}")
        return self

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "CandidateSet":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def dedup(cands: list[ClipCandidate],
          max_overlap: float = 0.5) -> list[ClipCandidate]:
    """
    Greedy non-max suppression: keep highest-scored candidates, drop any
    later candidate overlapping a kept one by more than `max_overlap`.
    Result is re-id'd and sorted by score desc.
    """
    kept: list[ClipCandidate] = []
    for c in sorted(cands, key=lambda c: (-c.score, c.start)):
        if all(c.overlap_fraction(k) <= max_overlap for k in kept):
            kept.append(c)
    return [c.model_copy(update={"id": i}) for i, c in enumerate(kept)]
