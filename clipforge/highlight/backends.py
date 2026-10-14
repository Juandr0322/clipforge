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
import sys
from pathlib import Path

from clipforge.highlight.chunker import Chunk, chunk_transcript
from clipforge.highlight.schema import (
    HOOK_TYPES, MAX_CLIP_SEC, MIN_CLIP_SEC, CandidateSet, ClipCandidate,
    RejectedCandidate, dedup,
)
from clipforge.transcribe.schema import Transcript

# Debug file written to the job dir when a chunk's reply yields no clips.
RAW_REPLY_NAME = "highlight_raw_chunk_{index:03d}.txt"
RAW_REPLY_GLOB = "highlight_raw_chunk_*.txt"

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
sentence>", "type": "<opinion|story|insight|howto|funny|emotional|other>", \
"trigger_t": <sec the setup/trigger of the moment begins>, \
"reaction_peak_t": <sec of the peak reaction or punchline>, \
"payoff_end_t": <sec the payoff finishes>, \
"context_setup": "<ONE line: what a viewer must know first>", \
"hook_type": "<bold_claim|question|story_open|contrarian|number|reveal|\
emotion|other>", \
"stranger_check": {{"pass": <true|false>, "reason": "<one sentence>"}}}}

trigger_t <= reaction_peak_t <= payoff_end_t, all inside [start, end].
Stranger check: imagine a viewer who has never seen this channel or the rest \
of the video. Can they follow and enjoy the clip? If it needs backstory, \
inside jokes or earlier context, set pass to false with the reason. Moments \
that need backstory must be rejected.

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
        extra = _parse_v2(item, start, end)
        out.append({
            "start": round(start, 3), "end": round(end, 3), "score": score,
            "hook": str(item.get("hook", ""))[:300],
            "reason": str(item.get("reason", ""))[:300],
            "type": ctype if ctype in valid_types else "other",
            **extra,
        })
    return out


def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if v == v and abs(v) != float("inf") else None


def _parse_v2(item: dict, start: float, end: float) -> dict:
    """Optional v2 fields. Bad values are dropped, never the candidate."""
    out: dict = {}
    times = [_num(item.get(k)) for k in
             ("trigger_t", "reaction_peak_t", "payoff_end_t")]
    ok = (all(t is not None and start <= t <= end for t in times)
          and times[0] <= times[1] <= times[2])
    if ok:
        out.update(trigger_t=round(times[0], 3),
                   reaction_peak_t=round(times[1], 3),
                   payoff_end_t=round(times[2], 3))
    ctx = item.get("context_setup")
    if isinstance(ctx, str) and ctx.strip():
        out["context_setup"] = " ".join(ctx.split("\n")[0].split())[:200] \
            or None
        if out["context_setup"] is None:
            del out["context_setup"]
    if "hook_type" in item:
        ht = item["hook_type"]
        out["hook_type"] = ht if ht in HOOK_TYPES else "other"
    sc = item.get("stranger_check")
    passed, reason = True, ""
    if isinstance(sc, dict):
        passed = sc.get("pass", sc.get("passed", True)) is not False
        reason = str(sc.get("reason", ""))
    elif isinstance(sc, bool):
        passed = sc
    if not passed:
        out["stranger_fail_reason"] = (
            reason.strip()[:300] or "failed stranger check")
    return out


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

class Highlighter(abc.ABC):
    name: str = "base"
    # Set by the stage runner to the job dir; backends may drop debug files
    # there (e.g. raw LLM replies that yielded nothing).
    debug_dir: Path | None = None

    @abc.abstractmethod
    def analyze_chunk(self, chunk: Chunk, source_duration: float) -> list[dict]:
        """One transcript chunk -> sanitized candidate dicts."""

    def find_highlights(self, t: Transcript, min_score: int = 55,
                        max_candidates: int = 12) -> CandidateSet:
        """Full pipeline: chunk -> analyze -> merge -> dedup -> rank."""
        all_raw: list[dict] = []
        rejected: list[RejectedCandidate] = []
        for chunk in chunk_transcript(t):
            for d in self.analyze_chunk(chunk, t.audio_duration_sec):
                why = d.pop("stranger_fail_reason", None)
                if why is None:
                    all_raw.append(d)
                else:
                    rejected.append(RejectedCandidate(
                        start=d["start"], end=d["end"], hook=d["hook"],
                        reason=why))

        cands = [ClipCandidate(id=i, **d) for i, d in enumerate(all_raw)]
        cands = [c for c in cands if c.score >= min_score]
        cands = dedup(cands)[:max_candidates]
        return CandidateSet(
            engine=self.name,
            source_duration_sec=t.audio_duration_sec,
            candidates=cands,
            rejected=rejected,
        )


class AnthropicHighlighter(Highlighter):
    """
    Real backend. Needs ANTHROPIC_API_KEY in the environment.
    Cost note: a 2h podcast ≈ 25k words ≈ ~35k input tokens across chunks —
    well under $0.50/video with Sonnet.
    """

    def __init__(self, model: str = "claude-sonnet-4-6",
                 examples_dir: Path | None = None, client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self._client = client
        # optional; sent as the SDK's native workspace_id argument, never logged
        self._workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID") or None
        self.model = model
        self.name = f"anthropic/{model}"
        self._system = build_system_prompt(load_examples(examples_dir))

    def analyze_chunk(self, chunk: Chunk, source_duration: float) -> list[dict]:
        extra = ({"workspace_id": self._workspace_id}
                 if self._workspace_id else {})
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
            **extra,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        found = parse_llm_response(text, chunk, source_duration)
        if not found:
            self._report_empty(chunk, text)
        return found

    def _report_empty(self, chunk: Chunk, text: str) -> None:
        """Zero clips can be an honest [] or an unparsable reply; keep the
        raw text so the two can be told apart. stderr only, never HTTP."""
        where = "raw reply not saved"
        if self.debug_dir is not None:
            name = RAW_REPLY_NAME.format(index=chunk.index)
            try:
                (self.debug_dir / name).write_text(text, encoding="utf-8")
                where = f"raw reply saved as {name}"
            except OSError:
                where = f"could not save {name}"
        print(f"[highlight] WARNING: chunk {chunk.index} "
              f"({chunk.start:.0f}s-{chunk.end:.0f}s) parsed to 0 clips; "
              f"{where}", file=sys.stderr)


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
            end = min(end, start + MAX_CLIP_SEC)
            span = end - start
            out.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "trigger_t": round(start + span * 0.1, 3),
                "reaction_peak_t": round(start + span * 0.6, 3),
                "payoff_end_t": round(start + span * 0.9, 3),
                "context_setup": "Standalone moment (mock heuristic).",
                "hook_type": "bold_claim",
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
