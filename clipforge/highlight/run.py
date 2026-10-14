"""
Module 3 — Highlight detection stage runner.

Requires TRANSCRIBED. Reads transcript.json, runs the Highlighter,
writes candidates.json, advances manifest to CANDIDATES_READY.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from clipforge.common.manifest import Manifest, Stage
from clipforge.highlight.backends import (
    RAW_REPLY_GLOB, USAGE_NAME, Highlighter,
)
from clipforge.transcribe.schema import Transcript


class HighlightError(Exception):
    pass


def _remove_quietly(path: Path) -> None:
    """Debug and metrics files must never fail the stage."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        print(f"[highlight] WARNING: could not remove {path.name}",
              file=sys.stderr)


def write_usage(job_dir: Path, record: dict) -> bool:
    """Write highlight_usage.json atomically. Returns False (with a warning
    on stderr) instead of raising: usage is a metric, not a stage output."""
    dest = job_dir / USAGE_NAME
    tmp = job_dir / f".{USAGE_NAME}.{uuid.uuid4().hex}.tmp"
    try:
        tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
        os.replace(tmp, dest)
        return True
    except OSError:
        print(f"[highlight] WARNING: could not write {USAGE_NAME}",
              file=sys.stderr)
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


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
    highlighter.usage = None      # no carry-over from another job
    try:
        t = Transcript.load(job_dir / m.transcript_path)
        # raw replies and usage from a previous (forced) run would mislead
        for stale in job_dir.glob(RAW_REPLY_GLOB):
            _remove_quietly(stale)
        _remove_quietly(job_dir / USAGE_NAME)
        highlighter.debug_dir = job_dir
        cset = highlighter.find_highlights(
            t, min_score=min_score, max_candidates=max_candidates)
        write_usage(job_dir, highlighter.usage_record())

        # Post-condition: it's VALID for a video to yield zero candidates
        # (the prompt says don't invent weak ones), but we record it loudly.
        out = job_dir / "candidates.json"
        cset.save(out)
        m.candidates_path = out.name
        m.record(Stage.CANDIDATES_READY, started, ok=True)
    except Exception as e:  # noqa: BLE001
        if highlighter.usage is not None:   # tokens already spent still count
            write_usage(job_dir, highlighter.usage_record())
        m.record(Stage.CANDIDATES_READY, started, ok=False, error=str(e))
        m.save(job_dir)
        raise
    m.save(job_dir)
    return m
