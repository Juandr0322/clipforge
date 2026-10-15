"""
Module 7b — ClipForge web server.

Single-user prototype server. Run with:
    python -m clipforge.cli serve
Then open http://localhost:8000

Design notes for later scaling (deliberately NOT built yet):
  - jobs run in a background thread pool (fine for one user; swap for a
    real queue like Celery/Redis when multi-user)
  - no auth (add before exposing beyond localhost)
"""
from __future__ import annotations

import math
import shutil
import sys
import threading
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi import Path as PathParam
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from clipforge.common.manifest import (
    Manifest, Stage, is_safe_job_id, new_job_id,
)
from clipforge.highlight.schema import ZERO_CLIPS_WARNING, CandidateSet
from clipforge.ingest.ingest import SUPPORTED_EXTENSIONS, IngestError, ingest
from clipforge.pipeline import PipelineConfig, run_pipeline
from clipforge.refine.refine import RefinedSet
from clipforge.review.store import (
    Review, export_examples, job_metrics, load_reviews, upsert_review,
    workspace_metrics,
)

STATIC = Path(__file__).parent / "static"

MAX_STEM_CHARS = 100
UNREADABLE_VIDEO = ("Could not read this file as a video. Check that it is a "
                    "complete video with an audio track.")


def _clean_filename(raw: str | None) -> str:
    """Last path component of an uploaded filename ('' if there is none).

    The stem is truncated so a very long name can't exceed filesystem limits;
    the extension is kept."""
    name = Path((raw or "").replace("\\", "/")).name
    if name in (".", ".."):
        return ""
    p = Path(name)
    return p.stem[:MAX_STEM_CHARS] + p.suffix[:16]


class ReviewIn(BaseModel):
    decision: str
    adjusted_start: float | None = None
    adjusted_end: float | None = None
    note: str = ""


def create_app(workspace: Path, config: PipelineConfig | None = None,
               sync_pipeline: bool = False) -> FastAPI:
    """sync_pipeline=True runs stages inline (used by tests)."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    cfg = config or PipelineConfig()
    app = FastAPI(title="ClipForge")
    progress: dict[str, str] = {}       # job_id -> live status line
    running: set[str] = set()           # job_ids whose pipeline is executing
    running_lock = threading.Lock()

    # ------------------------------------------------------------- helpers

    def job_dir(job_id: str) -> Path:
        d = workspace / job_id
        if not is_safe_job_id(job_id) or not (d / "manifest.json").exists():
            raise HTTPException(404, "unknown job")
        return d

    def run_bg(jid: str) -> None:
        # caller has already added jid to `running`
        try:
            progress[jid] = "starting…"
            run_pipeline(workspace / jid, cfg,
                         progress=lambda s: progress.__setitem__(jid, s))
        except Exception:  # noqa: BLE001 — error already in manifest
            progress[jid] = "failed"
            traceback.print_exc()
        finally:
            with running_lock:
                running.discard(jid)

    def start_run(jid: str) -> bool:
        """Start the pipeline for jid unless it is already running."""
        with running_lock:
            if jid in running:
                return False
            running.add(jid)
        if sync_pipeline:
            run_bg(jid)
        else:
            threading.Thread(target=run_bg, args=(jid,), daemon=True).start()
        return True

    def _check_clip_exists(d: Path, cid: int) -> None:
        """404 for a clip id the job does not have (once clips exist)."""
        m = Manifest.load(d)
        if m.refined_path and (d / m.refined_path).exists() and \
                all(c.id != cid for c in RefinedSet.load(d / m.refined_path).clips):
            raise HTTPException(404, "unknown clip")

    def _check_trim(d: Path, cid: int, body: ReviewIn) -> None:
        """400 unless the trim lies inside the source and start < end."""
        a, b = body.adjusted_start, body.adjusted_end
        if a is None and b is None:
            return
        m = Manifest.load(d)
        dur = m.source.duration_sec if m.source else None
        for name, v in (("start", a), ("end", b)):
            if v is None:
                continue
            # written so that NaN fails too
            if not (math.isfinite(v) and v >= 0 and (dur is None or v <= dur)):
                lim = f"0 and {dur:g} s" if dur is not None else "0 and the source length"
                raise HTTPException(
                    400, f"trim {name} must be between {lim}")
        # a one-sided trim is compared with the clip's own other boundary
        start, end = a, b
        if (a is None or b is None) and m.refined_path \
                and (d / m.refined_path).exists():
            clip = next((c for c in RefinedSet.load(d / m.refined_path).clips
                         if c.id == cid), None)
            if clip is not None:
                start = clip.start if a is None else a
                end = clip.end if b is None else b
        if start is not None and end is not None and start >= end:
            raise HTTPException(400, "trim start must be before trim end")

    # -------------------------------------------------------------- routes

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    # A plain `def` on purpose: FastAPI runs it in a worker thread, so the
    # blocking copy + ffmpeg ingest never stalls the event loop (and with it
    # every other request).
    @app.post("/api/jobs")
    def create_job(file: UploadFile):
        name = _clean_filename(file.filename)
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                400, f"Unsupported file type '{ext}'. Accepted video types: "
                     f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        # stage under the ORIGINAL filename so the manifest records it
        stage_dir = workspace / ".uploads" / uuid.uuid4().hex[:8]
        stage_dir.mkdir(parents=True, exist_ok=True)
        jid = new_job_id()
        try:
            tmp = stage_dir / name
            try:
                with tmp.open("wb") as f:
                    shutil.copyfileobj(file.file, f)
            except OSError:
                raise HTTPException(400, "Could not store the uploaded file")
            if tmp.stat().st_size == 0:
                raise HTTPException(400, "Uploaded file is empty")
            try:
                m = ingest(tmp, workspace, job_id=jid)
            except IngestError as e:
                shutil.rmtree(workspace / jid, ignore_errors=True)
                # the full text can hold paths and raw ffprobe output: log it
                # here, tell the user only the fixed sentence
                print(f"ingest rejected {name!r}: {e}", file=sys.stderr)
                raise HTTPException(400, UNREADABLE_VIDEO)
            except Exception:  # noqa: BLE001 — don't leak internals
                shutil.rmtree(workspace / jid, ignore_errors=True)
                traceback.print_exc()
                raise HTTPException(500, "Could not process the upload")
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)
        start_run(m.job_id)
        return {"job_id": m.job_id}

    @app.post("/api/jobs/{jid}/rerun")
    def rerun(jid: str):
        job_dir(jid)
        if not start_run(jid):
            raise HTTPException(409, "job is already running")
        return {"ok": True}

    def zero_clip_warning(d: Path, m: Manifest) -> str | None:
        """The fixed sentence for a finished job that has no clips."""
        if m.is_failed or m.job_id in running or \
                not m.is_at_least(Stage.RENDERED):
            return None      # cheap checks first: refined.json only for done jobs
        try:
            if m.refined_path and \
                    RefinedSet.load(d / m.refined_path).clips:
                return None
        except (OSError, ValueError):
            return None            # unreadable: not a proven "no clips"
        return ZERO_CLIPS_WARNING

    @app.get("/api/jobs")
    def list_jobs():
        out = []
        for d in sorted(workspace.iterdir(), reverse=True):
            if not is_safe_job_id(d.name) or \
                    not (d / "manifest.json").exists():
                continue
            m = Manifest.load(d)
            out.append({
                "job_id": m.job_id,
                "stage": m.stage.value,
                "failed": m.is_failed,
                "error": m.error,
                "progress": progress.get(m.job_id, ""),
                "running": m.job_id in running,
                "filename": m.source.original_filename if m.source else "?",
                "duration": m.source.duration_sec if m.source else 0,
                "warning": zero_clip_warning(d, m),
            })
        return out

    @app.get("/api/jobs/{jid}")
    def job_detail(jid: str):
        d = job_dir(jid)
        m = Manifest.load(d)
        detail = {
            "job_id": m.job_id, "stage": m.stage.value,
            "failed": m.is_failed, "error": m.error,
            "progress": progress.get(jid, ""),
            "running": jid in running,
            "source": m.source.model_dump() if m.source else None,
            "history": [h.model_dump() for h in m.history],
            "clips": [], "metrics": None,
            "warning": zero_clip_warning(d, m),
        }
        if m.candidates_path and (d / m.candidates_path).exists():
            cands = {c.id: c for c in
                     CandidateSet.load(d / m.candidates_path).candidates}
        else:
            cands = {}
        if m.refined_path and (d / m.refined_path).exists():
            reviews = load_reviews(d)
            rendered = m.is_at_least(Stage.RENDERED)
            for c in RefinedSet.load(d / m.refined_path).clips:
                r = reviews.get(c.id)
                cand = cands.get(c.candidate_id)
                detail["clips"].append({
                    "id": c.id, "start": c.start, "end": c.end,
                    "duration": round(c.duration, 1),
                    "score": c.score, "hook": c.hook, "type": c.type,
                    "reason": cand.reason if cand else "",
                    "video_ready": rendered,
                    "review": r.model_dump() if r else None,
                })
            detail["metrics"] = job_metrics(d)
        return detail

    @app.get("/api/jobs/{jid}/clips/{cid}/video")
    def clip_video(jid: str, cid: int = PathParam(ge=0)):
        p = job_dir(jid) / "renders" / f"clip_{cid:03d}.mp4"
        if not p.exists():
            raise HTTPException(404, "clip not rendered yet")
        return FileResponse(p, media_type="video/mp4")

    @app.get("/api/jobs/{jid}/clips/{cid}/qa")
    def clip_qa(jid: str, cid: int = PathParam(ge=0)):
        p = job_dir(jid) / "renders" / f"clip_{cid:03d}_qa.png"
        if not p.exists():
            raise HTTPException(404, "no QA sheet")
        return FileResponse(p, media_type="image/png")

    @app.post("/api/jobs/{jid}/clips/{cid}/review")
    def review_clip(jid: str, cid: int = PathParam(ge=0), *, body: ReviewIn):
        if body.decision not in ("approved", "rejected"):
            raise HTTPException(400, "decision must be approved|rejected")
        d = job_dir(jid)
        _check_clip_exists(d, cid)
        _check_trim(d, cid, body)
        upsert_review(d, Review(clip_id=cid, decision=body.decision,
                                adjusted_start=body.adjusted_start,
                                adjusted_end=body.adjusted_end, note=body.note))
        return {"ok": True, "metrics": job_metrics(d)}

    @app.post("/api/jobs/{jid}/export-examples")
    def export_feedback(jid: str):
        d = job_dir(jid)
        ex_dir = Path(cfg.examples_dir) if cfg.examples_dir \
            else workspace.parent / "examples"
        written = export_examples(d, ex_dir)
        return {"written": [p.name for p in written],
                "examples_dir": str(ex_dir)}

    @app.get("/api/metrics")
    def metrics():
        return workspace_metrics(workspace)

    @app.exception_handler(Exception)
    async def unhandled(request, exc):
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "Internal error"})

    return app
