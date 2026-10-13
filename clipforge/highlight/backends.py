"""
Highlighter backends for Module 3.

The LLM is the least trustworthy component in the pipeline: it can return
malformed JSON, hallucinated timestamps, or clips of absurd length. So the
rule here is: parse defensively, clamp/repair what's salvageable, drop what
isn't, and NEVER let a bad LLM response corrupt the candidate file.
"""
from __future__ import annotations

import abc
import json
import os
import re
from pathlib import Path

from clipforge.highlight.chunker import Chunk, chunk_transcript
from clipforge.highlight.schema import (
    MAX_CLIP_SEC, MIN_CLIP_SEC, CandidateSet, ClipCandidate, dedup,
)
from clipforge.transcribe.schema import Transcript

SYSTEM_PROMPT = """You are an expert short-form video editor. You find the \
moments in long-form video transcripts that would perform as standalone \
TikTok/Reels/Shorts clips.

What makes a great clip:
- HOOK: the first line grabs attention (contrarian take, bold claim, \
question, "nobody tells you...", specific number/result)
- SELF-CONTAINED: makes complete sense without the rest of the video
- ONE idea, fully paid off within the clip — no dangling references
- Emotional peak, strong opinion, surprising insight, concrete story, or \
practical how-to
- Ideal length {min_len:.0f}-{max_len:.0f} seconds

You will receive a transcript chunk where each line is:
[start_sec-end_sec] SPEAKER: text

Respond with ONLY a JSON array (no markdown fences, no commentary). Each \
element:
{{"start": <sec>, "end": <sec>, "score": <0-100>, "hook": "<the opening \
line, verbatim or lightly trimmed>", "reason": "<why this works, one \
sentence>", "type": "<opinion|story|insight|howto|funny|emotional|other>"}}

Rules:
- start/end MUST align to segment boundaries from the transcript
- 0-4 candidates per chunk. Return [] if nothing is genuinely strong. Do \
NOT invent weak candidates to fill space.
- score 80+ means you'd bet on it; 60-79 solid; below 60 don't include it
{examples}"""


def build_system_prompt(examples_text: str = "") -> str:
    ex = ""
    if examples_text.strip():
        ex = ("\nHere are examples of clips that performed well, for "
              "calibration:\n" + examples_text.strip())
    return SYSTEM_PROMPT.format(min_len=MIN_CLIP_SEC, max_len=MAX_CLIP_SEC,
                                examples=ex)


def load_examples(examples_dir: Path | None) -> str:
    """Few-shot library: every .txt file in the dir is one example block."""
    if not examples_dir or not examples_dir.exists():
        return ""
    blocks = []
    for f in sorted(examples_dir.glob("*.txt")):
        text = f.read_text(encoding="utf-8").strip()
        blocks.append(f"--- example: {f.stem} ---\n{text}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# defensive parsing of LLM output
# ---------------------------------------------------------------------------

def parse_llm_response(text: str, chunk: Chunk,
                       source_duration: float) -> list[dict]:
    """
    Extract a JSON array from possibly-messy LLM text and sanitize each
    entry. Returns plain dicts (ids assigned later). Unsalvageable entries
    are silently dropped — a partial harvest beats a crashed stage.
    """
    # strip markdown fences and find the outermost array
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if not m:
        return []
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []

    out = []
    valid_types = {"opinion", "story", "insight", "howto", "funny",
                   "emotional", "other"}
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
            score = int(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
        # clamp into chunk and source bounds
        start = max(start, 0.0)
        end = min(end, source_duration)
        if end <= start:
            continue
        dur = end - start
        if dur < MIN_CLIP_SEC * 0.6 or dur > MAX_CLIP_SEC * 1.5:
            continue  # hopeless length — drop rather than mangle
        # soft-clamp toward limits
        if dur > MAX_CLIP_SEC:
            end = start + MAX_CLIP_SEC
        score = max(0, min(100, score))
        ctype = item.get("type", "other")
        out.append({
            "start": round(start, 3), "end": round(end, 3), "score": score,
            "hook": str(item.get("hook", ""))[:300],
            "reason": str(item.get("reason", ""))[:300],
            "type": ctype if ctype in valid_types else "other",
        })
    return out


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

class Highlighter(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def analyze_chunk(self, chunk: Chunk, source_duration: float) -> list[dict]:
        """One transcript chunk -> sanitized candidate dicts."""

    def find_highlights(self, t: Transcript, min_score: int = 55,
                        max_candidates: int = 12) -> CandidateSet:
        """Full pipeline: chunk -> analyze -> merge -> dedup -> rank."""
        all_raw: list[dict] = []
        for chunk in chunk_transcript(t):
            all_raw.extend(self.analyze_chunk(chunk, t.audio_duration_sec))

        cands = [ClipCandidate(id=i, **d) for i, d in enumerate(all_raw)]
        cands = [c for c in cands if c.score >= min_score]
        cands = dedup(cands)[:max_candidates]
        return CandidateSet(
            engine=self.name,
            source_duration_sec=t.audio_duration_sec,
            candidates=cands,
        )


class AnthropicHighlighter(Highlighter):
    """
    Real backend. Needs ANTHROPIC_API_KEY in the environment.
    Cost note: a 2h podcast ≈ 25k words ≈ ~35k input tokens across chunks —
    well under $0.50/video with Sonnet.
    """

    def __init__(self, model: str = "claude-sonnet-4-6",
                 examples_dir: Path | None = None):
        import anthropic
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model
        self.name = f"anthropic/{model}"
        self._system = build_system_prompt(load_examples(examples_dir))

    def analyze_chunk(self, chunk: Chunk, source_duration: float) -> list[dict]:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=self._system,
            messages=[{
                "role": "user",
                "content": (
                    f"Transcript chunk {chunk.index} "
                    f"(covers {chunk.start:.0f}s-{chunk.end:.0f}s of a "
                    f"{source_duration:.0f}s video):\n\n{chunk.render()}"
                ),
            }],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return parse_llm_response(text, chunk, source_duration)


class MockHighlighter(Highlighter):
    """
    Deterministic heuristic backend for tests/offline dev: scores segments
    by hook-ish keywords, grows a window around high-scoring segments to a
    valid clip length. Crude on purpose — it exercises the exact same
    downstream path as the real backend.
    """
    name = "mock"

    KEYWORDS = {"nobody": 30, "wrong": 25, "surprised": 25, "changed": 20,
                "example": 15, "lesson": 15, "honestly": 10, "thing": 5}

    def analyze_chunk(self, chunk: Chunk, source_duration: float) -> list[dict]:
        out = []
        segs = chunk.segments
        for i, seg in enumerate(segs):
            score = 50 + sum(v for k, v in self.KEYWORDS.items()
                             if k in seg.text.lower())
            if score < 65:
                continue
            # grow window forward until >= MIN_CLIP_SEC
            j = i
            while (segs[j].end - seg.start) < MIN_CLIP_SEC and j + 1 < len(segs):
                j += 1
            start, end = seg.start, segs[j].end
            if end - start < MIN_CLIP_SEC * 0.6:
                continue
            out.append({
                "start": round(start, 3),
                "end": round(min(end, start + MAX_CLIP_SEC), 3),
                "score": min(score, 100),
                "hook": seg.text[:80],
                "reason": "keyword-matched hook (mock heuristic)",
                "type": "insight",
            })
        return out[:4]


def get_highlighter(engine: str, examples_dir: Path | None = None) -> Highlighter:
    """Factory: --engine mock | anthropic[:model]"""
    if engine == "mock":
        return MockHighlighter()
    if engine.startswith("anthropic"):
        _, _, model = engine.partition(":")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Export it or use --engine mock.")
        return AnthropicHighlighter(model=model or "claude-sonnet-4-6",
                                    examples_dir=examples_dir)
    raise ValueError(f"Unknown highlight engine: {engine}")
