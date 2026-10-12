"""
Module 4 — Boundary refinement.

LLM candidates carry approximate timestamps. This module turns them into
professional cut points:

  1. SENTENCE SNAP  — expand the span outward to whole ASR segments, so a
     clip always starts at the beginning of a sentence and ends at the end
     of one.
  2. LENGTH ENFORCE — trim trailing sentences if too long; grow by adjacent
     sentences if too short. If constraints can't be met, reject the
     candidate (recorded, not crashed).
  3. SILENCE-AWARE CUT — the physical cut lands in the word gap around the
     sentence span, at the quietest instant (EnergyProfile) or the gap
     midpoint as fallback, never within GUARD_SEC of a spoken word.

Hard invariant, checked at the end: NO WORD STRADDLES A CUT.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from clipforge.highlight.schema import (
    MAX_CLIP_SEC, MIN_CLIP_SEC, CandidateSet, ClipCandidate,
)
from clipforge.refine.silence import EnergyProfile
from clipforge.transcribe.schema import Segment, Transcript, Word

GUARD_SEC = 0.05     # never cut closer than this to a word edge
MAX_LEAD_SEC = 0.40  # max silence padding before first word / after last


class RefineReject(Exception):
    """Candidate cannot be refined into a valid clip."""


class RefinedClip(BaseModel):
    id: int
    candidate_id: int
    start: float                 # physical cut points (seconds)
    end: float
    speech_start: float          # first word start / last word end
    speech_end: float
    score: int
    hook: str
    type: str
    n_words: int
    segment_ids: list[int]

    @model_validator(mode="after")
    def _valid(self):
        if not (self.start <= self.speech_start < self.speech_end <= self.end):
            raise ValueError(f"clip {self.id}: cut/speech ordering broken")
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start


class RefinedSet(BaseModel):
    schema_version: int = 1
    source_duration_sec: float
    clips: list[RefinedClip] = Field(default_factory=list)
    rejected: list[dict] = Field(default_factory=list)  # audit of drops

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(self.model_dump_json(indent=2))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "RefinedSet":
        return cls.model_validate_json(path.read_text())


# ---------------------------------------------------------------------------

def _overlapping_segments(t: Transcript, start: float, end: float) -> list[Segment]:
    return [s for s in t.segments if s.start < end and s.end > start]


def _enforce_length(segs: list[Segment], all_segs: list[Segment],
                    min_sec: float, max_sec: float) -> list[Segment]:
    """Trim from the tail if too long; grow with neighbors if too short."""
    segs = list(segs)

    def span(ss):
        return ss[-1].end - ss[0].start

    while len(segs) > 1 and span(segs) > max_sec:
        segs.pop()

    if span(segs) > max_sec:
        raise RefineReject(
            f"single sentence spans {span(segs):.1f}s > max {max_sec:.0f}s")

    idx = {s.id: i for i, s in enumerate(all_segs)}
    while span(segs) < min_sec:
        i_last = idx[segs[-1].id]
        i_first = idx[segs[0].id]
        nxt = all_segs[i_last + 1] if i_last + 1 < len(all_segs) else None
        prv = all_segs[i_first - 1] if i_first > 0 else None
        if nxt and (nxt.end - segs[0].start) <= max_sec:
            segs.append(nxt)
        elif prv and (segs[-1].end - prv.start) <= max_sec:
            segs.insert(0, prv)
        else:
            raise RefineReject(
                f"cannot reach min length {min_sec:.0f}s "
                f"(got {span(segs):.1f}s)")
    return segs


def _cut_before(words: list[Word], first_word: Word,
                energy: Optional[EnergyProfile]) -> float:
    """Place the opening cut in the gap before `first_word`."""
    prev_end = 0.0
    for w in words:
        if w.end <= first_word.start and w is not first_word:
            prev_end = max(prev_end, w.end)
        if w.start >= first_word.start:
            break
    lo = prev_end + GUARD_SEC
    hi = first_word.start - GUARD_SEC
    if hi <= lo:                      # words are packed tight: best effort
        return max(prev_end, first_word.start - GUARD_SEC)
    lo = max(lo, first_word.start - MAX_LEAD_SEC)  # don't hang in silence
    if energy is not None and hi > lo:
        return energy.quietest_point(lo, hi)
    return (lo + hi) / 2


def _cut_after(words: list[Word], last_word: Word, duration: float,
               energy: Optional[EnergyProfile]) -> float:
    """Place the closing cut in the gap after `last_word`."""
    next_start = duration
    for w in words:
        if w.start >= last_word.end and w is not last_word:
            next_start = w.start
            break
    lo = last_word.end + GUARD_SEC
    hi = min(next_start - GUARD_SEC, duration)
    if hi <= lo:
        return min(last_word.end + GUARD_SEC, duration)
    hi = min(hi, last_word.end + MAX_LEAD_SEC)
    if energy is not None and hi > lo:
        return energy.quietest_point(lo, hi)
    return (lo + hi) / 2


def _assert_no_straddle(t: Transcript, cut: float) -> None:
    for w in t.words:
        if w.start < cut < w.end:
            raise RefineReject(
                f"cut at {cut:.3f}s lands inside word '{w.w}' "
                f"[{w.start:.3f},{w.end:.3f}]")


def refine_candidate(cand: ClipCandidate, t: Transcript,
                     energy: Optional[EnergyProfile],
                     min_sec: float = MIN_CLIP_SEC,
                     max_sec: float = MAX_CLIP_SEC) -> RefinedClip:
    dur = t.audio_duration_sec
    c_start = max(0.0, min(cand.start, dur))
    c_end = max(0.0, min(cand.end, dur))
    if c_end <= c_start:
        raise RefineReject("empty span after clamping")

    segs = _overlapping_segments(t, c_start, c_end)
    if not segs:
        raise RefineReject("candidate overlaps no transcript segments")

    segs = _enforce_length(segs, t.segments, min_sec, max_sec)

    seg_words = t.words_between(segs[0].start - 1e-3, segs[-1].end + 1e-3)
    if not seg_words:
        raise RefineReject("no words inside refined sentence span")
    first_w, last_w = seg_words[0], seg_words[-1]

    start_cut = _cut_before(t.words, first_w, energy)
    end_cut = _cut_after(t.words, last_w, dur, energy)
    start_cut = max(0.0, min(start_cut, dur))
    end_cut = max(0.0, min(end_cut, dur))

    _assert_no_straddle(t, start_cut)
    _assert_no_straddle(t, end_cut)

    return RefinedClip(
        id=cand.id, candidate_id=cand.id,
        start=round(start_cut, 3), end=round(end_cut, 3),
        speech_start=first_w.start, speech_end=last_w.end,
        score=cand.score, hook=cand.hook, type=cand.type,
        n_words=len(seg_words),
        segment_ids=[s.id for s in segs],
    )


def refine_all(cset: CandidateSet, t: Transcript,
               energy: Optional[EnergyProfile]) -> RefinedSet:
    clips, rejected = [], []
    for cand in cset.candidates:
        try:
            clips.append(refine_candidate(cand, t, energy))
        except RefineReject as e:
            rejected.append({"candidate_id": cand.id, "reason": str(e),
                             "start": cand.start, "end": cand.end})
    # re-id sequentially, keep score order from Module 3
    clips = [c.model_copy(update={"id": i}) for i, c in enumerate(clips)]
    return RefinedSet(source_duration_sec=t.audio_duration_sec,
                      clips=clips, rejected=rejected)
