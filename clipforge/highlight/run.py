"""
Module 3 — Highlight detection stage runner.

Requires TRANSCRIBED. Reads transcript.json, runs the Highlighter,
writes candidates.json, advances manifest to CANDIDATES_READY.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from clipforge.common.manifest import Manifest, Stage
from clipforge.highlight.backends import Highlighter
from clipforge.highlight.schema import CandidateSet
from clipforge.transcribe.schema import Transcript


class HighlightError(Exception):
    pass


def highlight_job(job_dir: str | Path, highlighter: Highlighter,
                  min_score: int = 55, max_candidates: int = 12,
                  force: bool = False) -> Manifest:
    job_dir = Path(job_dir)
    m = Manifest.load(job_dir)

    if not m.is_at_least(Stage.TRANSCRIBED):
        raise HighlightError(
            f"Job is at stage '{m.stage.value}' — run transcribe first.")
    if m.is_at_least(Stage.CANDIDATES_READY) and not force:
        return m

    started = datetime.now(timezone.utc)
    try:
        t = Transcript.load(job_dir / m.transcript_path)
        cset = highlighter.find_highlights(
            t, min_score=min_score, max_candidates=max_candidates)

        # Post-condition: it's VALID for a video to yield zero candidates
        # (the prompt says don't invent weak ones), but we record it loudly.
        out = job_dir / "candidates.json"
        cset.save(out)
        m.candidates_path = out.name
        m.record(Stage.CANDIDATES_READY, started, ok=True)
    except Exception as e:  # noqa: BLE001
        m.record(Stage.CANDIDATES_READY, started, ok=False, error=str(e))
        m.save(job_dir)
        raise
    m.save(job_dir)
    return m
