"""
Per-clip quality report: renders/clip_NNN.report.json.

Each later quality feature owns one section and writes it through
`update_section`, which touches only that section. Every section is optional,
so a report with no sections is valid.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from clipforge.common.manifest import Manifest
from clipforge.refine.refine import RefinedSet
from clipforge.review.store import _job_lock, _replace_with_retry

SECTIONS = ("hook_score", "value_adds", "pacing_flags", "variants", "loop",
            "exports", "audio_layer")


class ClipReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: int
    hook_score: Optional[dict[str, Any]] = None
    value_adds: Optional[list[Any]] = None
    pacing_flags: Optional[list[Any]] = None
    variants: Optional[list[Any]] = None
    loop: Optional[dict[str, Any]] = None
    exports: Optional[list[Any]] = None
    audio_layer: Optional[dict[str, Any]] = None


def report_path(job_dir: Path, clip_id: int) -> Path:
    return Path(job_dir) / "renders" / f"clip_{clip_id:03d}.report.json"


def load_report(job_dir: Path, clip_id: int) -> Optional[ClipReport]:
    with _job_lock(job_dir):
        p = report_path(job_dir, clip_id)
        if not p.exists():
            return None
        return ClipReport.model_validate_json(p.read_text(encoding="utf-8"))


def _write(job_dir: Path, report: ClipReport) -> None:
    p = report_path(job_dir, report.clip_id)
    tmp = p.with_name(f"{p.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        # newline="\n": text mode would write CRLF on Windows
        tmp.write_text(report.model_dump_json(indent=2, exclude_none=True),
                       encoding="utf-8", newline="\n")
        _replace_with_retry(tmp, p)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def update_section(job_dir: Path, clip_id: int, section: str,
                   value: Any) -> ClipReport:
    """Set one section of a clip's report, leaving the others untouched.
    Load, modify and atomic save all happen under the per-job lock."""
    if section not in SECTIONS:
        raise ValueError(f"unknown report section: {section}")
    with _job_lock(job_dir):
        report = load_report(job_dir, clip_id) or ClipReport(clip_id=clip_id)
        data = report.model_dump()
        data[section] = value
        report = ClipReport.model_validate(data)  # reject wrongly-typed values
        _write(job_dir, report)
        return report


def quality_job(job_dir: Path) -> None:
    """Create an empty report for every rendered clip that has none.
    Existing reports are never rewritten, so re-runs are no-ops."""
    job_dir = Path(job_dir)
    m = Manifest.load(job_dir)
    rs = RefinedSet.load(job_dir / m.refined_path)
    for clip in rs.clips:
        if not (job_dir / "renders" / f"clip_{clip.id:03d}.mp4").exists():
            continue
        with _job_lock(job_dir):
            if not report_path(job_dir, clip.id).exists():
                _write(job_dir, ClipReport(clip_id=clip.id))
