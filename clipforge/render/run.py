"""
Module 6 — Captioning & rendering stage runner.

Requires REFRAMED. For each refined clip: build captions from the
transcript, load its crop path, render renders/clip_<id>.mp4, verify it,
and write a QA contact sheet. Advances manifest to RENDERED.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from clipforge.common.manifest import Manifest, Stage
from clipforge.refine.refine import RefinedSet
from clipforge.reframe.detect import CropPath
from clipforge.render.captions import CaptionStyle, group_words, to_ass
from clipforge.render.render import (
    RenderError, make_contact_sheet, render_clip, verify_render,
)
from clipforge.transcribe.schema import Transcript


def render_job(job_dir: str | Path, force: bool = False,
               captions: bool = True,
               style: CaptionStyle | None = None,
               contact_sheets: bool = True) -> Manifest:
    job_dir = Path(job_dir)
    m = Manifest.load(job_dir)

    if not m.is_at_least(Stage.REFRAMED):
        raise RenderError(
            f"Job is at stage '{m.stage.value}' — run reframe first.")
    if m.is_at_least(Stage.RENDERED) and not force:
        return m

    started = datetime.now(timezone.utc)
    try:
        t = Transcript.load(job_dir / m.transcript_path)
        rs = RefinedSet.load(job_dir / m.refined_path)
        crops_dir = job_dir / m.crop_paths_dir
        renders = job_dir / "renders"
        renders.mkdir(exist_ok=True)
        norm = job_dir / m.ingest.normalized_video

        for clip in rs.clips:
            out = renders / f"clip_{clip.id:03d}.mp4"
            ass_path = None
            if captions:
                cards = group_words(t.words, clip.start, clip.end)
                ass_path = renders / f"clip_{clip.id:03d}.ass"
                ass_path.write_text(to_ass(cards, style), encoding="utf-8")
            crop = CropPath.load(crops_dir / f"clip_{clip.id:03d}.json")
            render_clip(
                norm, out, clip.start, clip.end, crop,
                src_w=m.ingest.normalized_width,
                src_h=m.ingest.normalized_height,
                fps=m.ingest.normalized_fps,
                ass_path=ass_path,
            )
            verify_render(out, clip.end - clip.start)
            if contact_sheets:
                make_contact_sheet(out, renders / f"clip_{clip.id:03d}_qa.png")

        m.renders_dir = renders.name
        m.record(Stage.RENDERED, started, ok=True)
    except Exception as e:  # noqa: BLE001
        m.record(Stage.RENDERED, started, ok=False, error=str(e))
        m.save(job_dir)
        raise
    m.save(job_dir)
    return m
