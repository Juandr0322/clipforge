"""
Module 5 — Visual reframing stage runner.

Requires REFINED. For each refined clip, computes a smoothed 9:16 crop
path from the 480p proxy and writes crops/clip_<id>.json. Advances the
manifest to REFRAMED.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from clipforge.common.manifest import Manifest, Stage
from clipforge.refine.refine import RefinedSet
from clipforge.reframe.core import compute_crop_path
from clipforge.reframe.detect import FaceDetector


class ReframeError(Exception):
    pass


def reframe_job(job_dir: str | Path, detector: FaceDetector,
                force: bool = False) -> Manifest:
    job_dir = Path(job_dir)
    m = Manifest.load(job_dir)

    if not m.is_at_least(Stage.REFINED):
        raise ReframeError(
            f"Job is at stage '{m.stage.value}' — run refine first.")
    if m.is_at_least(Stage.REFRAMED) and not force:
        return m

    started = datetime.now(timezone.utc)
    try:
        rs = RefinedSet.load(job_dir / m.refined_path)
        proxy = job_dir / m.ingest.proxy_video
        crops_dir = job_dir / "crops"
        crops_dir.mkdir(exist_ok=True)

        for clip in rs.clips:
            path = compute_crop_path(proxy, clip.id, clip.start, clip.end,
                                     detector)
            path.save(crops_dir / f"clip_{clip.id:03d}.json")

        # post-condition: one crop file per clip
        produced = sorted(p.name for p in crops_dir.glob("clip_*.json"))
        expected = sorted(f"clip_{c.id:03d}.json" for c in rs.clips)
        missing = set(expected) - set(produced)
        if missing:
            raise ReframeError(f"missing crop paths: {sorted(missing)}")

        m.crop_paths_dir = crops_dir.name
        m.record(Stage.REFRAMED, started, ok=True)
    except Exception as e:  # noqa: BLE001
        m.record(Stage.REFRAMED, started, ok=False, error=str(e))
        m.save(job_dir)
        raise
    m.save(job_dir)
    return m
