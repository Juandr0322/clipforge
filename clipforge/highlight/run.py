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
    RAW_REPLY_GLOB, USAGE_FIELDS, USAGE_NAME, Highlighter,
)
from clipforge.transcribe.schema import Transcript

# Counters added together when a retry follows a failed attempt.
_SUMMED = ("calls", *USAGE_FIELDS, "cache_write_1h_tokens")


class HighlightError(Exception):
    pass


def _failed_attempt_usage(job_dir: Path, m: Manifest) -> dict | None:
    """Tokens spent by earlier attempts that failed, so a retry keeps
    counting them. A forced re-run after a success starts from zero."""
    if m.failed_stage != Stage.CANDIDATES_READY:
        return None
    path = job_dir / USAGE_NAME
    if not path.is_file():
        return None
    try:
        prior = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — metric only: any damage means "none"
        print(f"[highlight] WARNING: ignored an unreadable {USAGE_NAME} "
              "from an earlier attempt", file=sys.stderr)
        return None
    return prior if isinstance(prior, dict) else None


def merge_usage(record: dict, prior: dict | None) -> dict:
    """record + prior when both used the same model; `attempts` counts the
    runs whose tokens are in the totals. A malformed prior file (hand-edited,
    half-written) is ignored with a warning: usage must never fail the stage."""
    if not prior or prior.get("model") != record.get("model"):
        return {**record, "attempts": 1}
    try:
        merged = dict(record)
        for f in _SUMMED:
            merged[f] = int(record.get(f) or 0) + int(prior.get(f) or 0)
        geos = prior.get("inference_geo") or []
        if not isinstance(geos, list):
            raise TypeError("inference_geo is not a list")
        merged["inference_geo"] = sorted(
            set(record.get("inference_geo") or []) | {str(g) for g in geos})
        merged["attempts"] = int(prior.get("attempts") or 1) + 1
        return merged
    except Exception:  # noqa: BLE001 — metric only; never fail the stage
        print(f"[highlight] WARNING: ignored a malformed {USAGE_NAME} from an "
              "earlier attempt", file=sys.stderr)
        return {**record, "attempts": 1}


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
    prior = None
    try:
        prior = _failed_attempt_usage(job_dir, m)
        t = Transcript.load(job_dir / m.transcript_path)
        # raw replies and usage from a previous (forced) run would mislead;
        # tokens from failed attempts are kept in `prior`
        for stale in job_dir.glob(RAW_REPLY_GLOB):
            _remove_quietly(stale)
        _remove_quietly(job_dir / USAGE_NAME)
        highlighter.debug_dir = job_dir
        cset = highlighter.find_highlights(
            t, min_score=min_score, max_candidates=max_candidates)
        write_usage(job_dir, merge_usage(highlighter.usage_record(), prior))

        # Post-condition: it's VALID for a video to yield zero candidates
        # (the prompt says don't invent weak ones), but we record it loudly.
        out = job_dir / "candidates.json"
        cset.save(out)
        m.candidates_path = out.name
        m.record(Stage.CANDIDATES_READY, started, ok=True)
    except Exception as e:  # noqa: BLE001
        if highlighter.usage is not None:   # tokens already spent still count
            write_usage(job_dir,
                        merge_usage(highlighter.usage_record(), prior))
        m.record(Stage.CANDIDATES_READY, started, ok=False, error=str(e))
        m.save(job_dir)
        raise
    m.save(job_dir)
    return m
