"""
Module 4 — Boundary refinement stage runner.

Requires CANDIDATES_READY. Reads transcript.json + candidates.json,
builds an EnergyProfile from audio.wav, writes refined.json, advances
manifest to REFINED.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from clipforge.common.manifest import Manifest, Stage
from clipforge.highlight.schema import CandidateSet
from clipforge.refine.refine import refine_all
from clipforge.refine.silence import EnergyProfile
from clipforge.transcribe.schema import Transcript


class RefineError(Exception):
    pass


def refine_job(job_dir: str | Path, force: bool = False,
               use_energy: bool = True) -> Manifest:
    job_dir = Path(job_dir)
    m = Manifest.load(job_dir)

    if not m.is_at_least(Stage.CANDIDATES_READY):
        raise RefineError(
            f"Job is at stage '{m.stage.value}' — run highlight first.")
    if m.is_at_least(Stage.REFINED) and not force:
        return m

    started = datetime.now(timezone.utc)
    try:
        t = Transcript.load(job_dir / m.transcript_path)
        cset = CandidateSet.load(job_dir / m.candidates_path)
        energy = None
        if use_energy:
            energy = EnergyProfile.from_wav(job_dir / m.ingest.audio_wav)

        refined = refine_all(cset, t, energy)

        if cset.candidates and not refined.clips:
            raise RefineError(
                "All candidates were rejected during refinement: "
                + "; ".join(r["reason"] for r in refined.rejected))

        out = job_dir / "refined.json"
        refined.save(out)
        m.refined_path = out.name
        m.record(Stage.REFINED, started, ok=True)
    except Exception as e:  # noqa: BLE001
        m.record(Stage.REFINED, started, ok=False, error=str(e))
        m.save(job_dir)
        raise
    m.save(job_dir)
    return m
