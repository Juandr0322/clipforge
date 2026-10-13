"""
Data contracts for the ClipForge pipeline.

The Manifest is the state machine for one video job. Every module reads it,
does its work, updates its section, and writes it back to disk. Modules never
share in-memory state — this file IS the interface between them.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class Stage(str, Enum):
    """Pipeline stages in order. A job's `stage` is the last COMPLETED stage."""
    CREATED = "created"
    INGESTED = "ingested"
    TRANSCRIBED = "transcribed"
    CANDIDATES_READY = "candidates_ready"
    REFINED = "refined"
    REFRAMED = "reframed"
    RENDERED = "rendered"
    REVIEWED = "reviewed"
    FAILED = "failed"


STAGE_ORDER = [
    Stage.CREATED, Stage.INGESTED, Stage.TRANSCRIBED, Stage.CANDIDATES_READY,
    Stage.REFINED, Stage.REFRAMED, Stage.RENDERED, Stage.REVIEWED,
]


class SourceInfo(BaseModel):
    """Metadata about the original uploaded file, straight from ffprobe."""
    original_filename: str
    duration_sec: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: Optional[str] = None      # None => no audio track
    audio_sample_rate: Optional[int] = None
    audio_channels: Optional[int] = None
    size_bytes: int
    sha256: str


class IngestOutputs(BaseModel):
    """Paths (relative to the job dir) produced by Module 1."""
    normalized_video: str          # H.264 CFR working copy
    audio_wav: str                 # 16 kHz mono WAV for ASR
    proxy_video: str               # 480p proxy for vision analysis
    normalized_fps: float
    normalized_width: int
    normalized_height: int


class StageRecord(BaseModel):
    """Audit record for one completed (or failed) stage run."""
    stage: Stage
    started_at: str
    finished_at: str
    ok: bool
    error: Optional[str] = None
    duration_sec: float


class Manifest(BaseModel):
    job_id: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    stage: Stage = Stage.CREATED          # last SUCCESSFULLY completed stage
    failed_stage: Optional[Stage] = None  # stage that most recently failed
    error: Optional[str] = None

    source: Optional[SourceInfo] = None
    ingest: Optional[IngestOutputs] = None

    # Downstream modules add their sections here as the pipeline grows:
    transcript_path: Optional[str] = None      # Module 2
    candidates_path: Optional[str] = None      # Module 3
    refined_path: Optional[str] = None         # Module 4
    crop_paths_dir: Optional[str] = None       # Module 5
    renders_dir: Optional[str] = None          # Module 6

    history: list[StageRecord] = Field(default_factory=list)

    # ---------- persistence ----------
    @staticmethod
    def path_for(job_dir: Path) -> Path:
        return job_dir / "manifest.json"

    def save(self, job_dir: Path) -> None:
        job_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path_for(job_dir).with_suffix(".json.tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.path_for(job_dir))  # atomic write — never a half-written manifest

    @classmethod
    def load(cls, job_dir: Path) -> "Manifest":
        return cls.model_validate_json(cls.path_for(job_dir).read_text(encoding="utf-8"))

    # ---------- helpers ----------
    @property
    def is_failed(self) -> bool:
        return self.failed_stage is not None

    def is_at_least(self, stage: Stage) -> bool:
        """Progress check based on last successful stage; a later failure
        does not erase earlier completed work (jobs stay resumable)."""
        return STAGE_ORDER.index(self.stage) >= STAGE_ORDER.index(stage)

    def record(self, stage: Stage, started: datetime, ok: bool,
               error: Optional[str] = None) -> None:
        finished = datetime.now(timezone.utc)
        self.history.append(StageRecord(
            stage=stage,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            ok=ok,
            error=error,
            duration_sec=round((finished - started).total_seconds(), 2),
        ))
        if ok:
            self.stage = stage
            self.failed_stage = None
            self.error = None
        else:
            # keep self.stage (last success) so the job can resume
            self.failed_stage = stage
            self.error = error


def new_job_id() -> str:
    """Short, sortable, filesystem-safe job id."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + \
        __import__("uuid").uuid4().hex[:6]
