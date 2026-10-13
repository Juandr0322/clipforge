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

import shutil
import threading
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from clipforge.common.manifest import Manifest, Stage
from clipforge.highlight.schema import CandidateSet
from clipforge.ingest.ingest import IngestError, ingest
from clipforge.pipeline import PipelineConfig, run_pipeline
from clipforge.refine.refine import RefinedSet
from clipforge.review.store import (
    Review, export_examples, job_metrics, load_reviews, save_reviews,
    workspace_metrics,
)

STATIC = Path(__file__).parent / "static"


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

    # ------------------------------------------------------------- helpers

    def job_dir(job_id: str) -> Path:
        d = workspace / job_id
        if not (d / "manifest.json").exists():
            raise HTTPException(404, f"unknown job: {job_id}")
        return d

    def run_bg(jid: str) -> None:
        try:
            progress[jid] = "starting…"
            run_pipeline(workspace / jid, cfg,
                         progress=lambda s: progress.__setitem__(jid, s))
        except Exception:  # noqa: BLE001 — error already in manifest
            progress[jid] = "failed"
            traceback.print_exc()

    # -------------------------------------------------------------- routes

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.post("/api/jobs")
    async def create_job(file: UploadFile):
        # stage under the ORIGINAL filename so the manifest records it
        import uuid
        stage_dir = workspace / ".uploads" / uuid.uuid4().hex[:8]
        stage_dir.mkdir(parents=True, exist_ok=True)
        tmp = stage_dir / Path(file.filename).name
        with tmp.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        try:
            m = ingest(tmp, workspace)
        except IngestError as e:
            raise HTTPException(400, str(e))
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)
        if sync_pipeline:
            run_bg(m.job_id)
        else:
            threading.Thread(target=run_bg, args=(m.job_id,),
                             daemon=True).start()
        return {"job_id": m.job_id}

    @app.post("/api/jobs/{jid}/rerun")
    def rerun(jid: str):
        job_dir(jid)
        if sync_pipeline:
            run_bg(jid)
        else:
            threading.Thread(target=run_bg, args=(jid,), daemon=True).start()
        return {"ok": True}

    @app.get("/api/jobs")
    def list_jobs():
        out = []
        for d in sorted(workspace.iterdir(), reverse=True):
            if not (d / "manifest.json").exists():
                continue
            m = Manifest.load(d)
            out.append({
                "job_id": m.job_id,
                "stage": m.stage.value,
                "failed": m.is_failed,
                "error": m.error,
                "progress": progress.get(m.job_id, ""),
                "filename": m.source.original_filename if m.source else "?",
                "duration": m.source.duration_sec if m.source else 0,
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
            "source": m.source.model_dump() if m.source else None,
            "history": [h.model_dump() for h in m.history],
            "clips": [], "metrics": None,
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
    def clip_video(jid: str, cid: int):
        p = job_dir(jid) / "renders" / f"clip_{cid:03d}.mp4"
        if not p.exists():
            raise HTTPException(404, "clip not rendered yet")
        return FileResponse(p, media_type="video/mp4")

    @app.get("/api/jobs/{jid}/clips/{cid}/qa")
    def clip_qa(jid: str, cid: int):
        p = job_dir(jid) / "renders" / f"clip_{cid:03d}_qa.png"
        if not p.exists():
            raise HTTPException(404, "no QA sheet")
        return FileResponse(p, media_type="image/png")

    @app.post("/api/jobs/{jid}/clips/{cid}/review")
    def review_clip(jid: str, cid: int, body: ReviewIn):
        if body.decision not in ("approved", "rejected"):
            raise HTTPException(400, "decision must be approved|rejected")
        d = job_dir(jid)
        rs = load_reviews(d)
        rs.upsert(Review(clip_id=cid, decision=body.decision,
                         adjusted_start=body.adjusted_start,
                         adjusted_end=body.adjusted_end, note=body.note))
        save_reviews(d, rs)
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
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return app
