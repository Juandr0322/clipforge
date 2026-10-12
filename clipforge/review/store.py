"""
Module 7a — Review store and feedback loop.

Every human decision is logged to reviews.json in the job dir. This file is
the system's real training data:
  - approval rate  -> the end-to-end quality number
  - trim distance  -> how good Module 4's boundaries are
  - approved clips -> exported as few-shot examples for Module 3
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from clipforge.common.manifest import Manifest
from clipforge.refine.refine import RefinedSet
from clipforge.transcribe.schema import Transcript


class Review(BaseModel):
    clip_id: int
    decision: Literal["approved", "rejected"]
    adjusted_start: Optional[float] = None   # None = kept as-is
    adjusted_end: Optional[float] = None
    note: str = ""
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ReviewSet(BaseModel):
    reviews: list[Review] = Field(default_factory=list)

    def upsert(self, r: Review) -> None:
        self.reviews = [x for x in self.reviews if x.clip_id != r.clip_id]
        self.reviews.append(r)

    def get(self, clip_id: int) -> Optional[Review]:
        return next((r for r in self.reviews if r.clip_id == clip_id), None)


def reviews_path(job_dir: Path) -> Path:
    return Path(job_dir) / "reviews.json"


def load_reviews(job_dir: Path) -> ReviewSet:
    p = reviews_path(job_dir)
    if not p.exists():
        return ReviewSet()
    return ReviewSet.model_validate_json(p.read_text())


def save_reviews(job_dir: Path, rs: ReviewSet) -> None:
    p = reviews_path(job_dir)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(rs.model_dump_json(indent=2))
    tmp.replace(p)


# ------------------------------------------------------------------ metrics

def job_metrics(job_dir: Path) -> dict:
    """The plan's quality numbers, per job."""
    rs = load_reviews(job_dir)
    refined = RefinedSet.load(Path(job_dir) / "refined.json") \
        if (Path(job_dir) / "refined.json").exists() else None
    n = len(rs.reviews)
    approved = [r for r in rs.reviews if r.decision == "approved"]
    trims = []
    if refined:
        by_id = {c.id: c for c in refined.clips}
        for r in rs.reviews:
            c = by_id.get(r.clip_id)
            if c is None:
                continue
            ds = abs((r.adjusted_start - c.start)) if r.adjusted_start is not None else 0.0
            de = abs((r.adjusted_end - c.end)) if r.adjusted_end is not None else 0.0
            trims.append(ds + de)
    return {
        "reviewed": n,
        "approved": len(approved),
        "approval_rate": round(len(approved) / n, 3) if n else None,
        "untouched_approvals": sum(
            1 for r in approved
            if r.adjusted_start is None and r.adjusted_end is None),
        "avg_trim_adjustment_sec": round(sum(trims) / len(trims), 2) if trims else 0.0,
    }


def workspace_metrics(workspace: Path) -> dict:
    """Aggregate metrics across all jobs — track these over time."""
    per_job, tot_rev, tot_app, trims = {}, 0, 0, []
    for job_dir in sorted(Path(workspace).iterdir()):
        if not (job_dir / "manifest.json").exists():
            continue
        jm = job_metrics(job_dir)
        if jm["reviewed"]:
            per_job[job_dir.name] = jm
            tot_rev += jm["reviewed"]
            tot_app += jm["approved"]
            trims.append(jm["avg_trim_adjustment_sec"])
    return {
        "jobs_with_reviews": len(per_job),
        "total_reviewed": tot_rev,
        "overall_approval_rate": round(tot_app / tot_rev, 3) if tot_rev else None,
        "avg_trim_adjustment_sec": round(sum(trims) / len(trims), 2) if trims else 0.0,
        "per_job": per_job,
    }


# ------------------------------------------------------- few-shot feedback

def export_examples(job_dir: Path, examples_dir: Path,
                    max_words: int = 220) -> list[Path]:
    """Turn APPROVED clips into few-shot example files for Module 3.
    Closes the feedback loop: tomorrow's highlight detection is calibrated
    by what you actually approved today."""
    job_dir, examples_dir = Path(job_dir), Path(examples_dir)
    examples_dir.mkdir(parents=True, exist_ok=True)
    m = Manifest.load(job_dir)
    t = Transcript.load(job_dir / m.transcript_path)
    refined = RefinedSet.load(job_dir / "refined.json")
    by_id = {c.id: c for c in refined.clips}
    written = []
    for r in load_reviews(job_dir).reviews:
        if r.decision != "approved" or r.clip_id not in by_id:
            continue
        c = by_id[r.clip_id]
        words = t.words_between(c.start, c.end)[:max_words]
        text = " ".join(w.w for w in words)
        body = (f"source: {m.source.original_filename} "
                f"({m.source.duration_sec:.0f}s long-form)\n"
                f"clip transcript: {text}\n"
                f"why it worked: hook='{c.hook}' type={c.type} "
                f"(human-approved, score {c.score})\n")
        if r.note:
            body += f"reviewer note: {r.note}\n"
        out = examples_dir / f"{m.job_id}_clip{c.id:03d}.txt"
        out.write_text(body)
        written.append(out)
    return written
