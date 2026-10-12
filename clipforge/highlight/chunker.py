"""
Transcript chunking for Module 3.

Long transcripts (a 2h podcast is ~25k words) get split into overlapping
time windows. Overlap guarantees a great moment sitting on a chunk boundary
is fully visible in at least one chunk. Dedup handles the double-finds.
"""
from __future__ import annotations

from dataclasses import dataclass

from clipforge.transcribe.schema import Segment, Transcript

CHUNK_SEC = 18 * 60          # 18-minute windows
OVERLAP_SEC = 2 * 60         # 2-minute overlap


@dataclass
class Chunk:
    index: int
    start: float
    end: float
    segments: list[Segment]

    def render(self) -> str:
        """Format for the LLM: one line per segment with timestamps."""
        lines = []
        for s in self.segments:
            lines.append(f"[{s.start:.1f}-{s.end:.1f}] {s.speaker}: {s.text}")
        return "\n".join(lines)


def chunk_transcript(t: Transcript,
                     chunk_sec: float = CHUNK_SEC,
                     overlap_sec: float = OVERLAP_SEC) -> list[Chunk]:
    if overlap_sec >= chunk_sec:
        raise ValueError("overlap must be smaller than chunk size")
    dur = t.audio_duration_sec
    chunks: list[Chunk] = []
    step = chunk_sec - overlap_sec
    start, idx = 0.0, 0
    while start < dur:
        end = min(start + chunk_sec, dur)
        segs = [s for s in t.segments
                if s.start < end and s.end > start]  # any overlap with window
        if segs:
            chunks.append(Chunk(index=idx, start=start, end=end, segments=segs))
            idx += 1
        if end >= dur:
            break
        start += step
    return chunks
