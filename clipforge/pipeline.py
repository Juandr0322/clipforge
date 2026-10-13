"""
Pipeline orchestrator — runs all stages on a job, in order, resumably.

Engine selection comes from PipelineConfig (env-driven by default), with
graceful fallbacks so the prototype always runs:
  ASR:       whisper:small  (mock if CLIPFORGE_ASR=mock)
  Highlight: anthropic      (auto-falls back to mock if no API key)
  Detector:  mediapipe      (auto-falls back to haar if not installed)

Because every stage is idempotent, run_pipeline() is safely re-runnable:
completed stages no-op, the failed/remaining ones execute.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from clipforge.common.manifest import Manifest, Stage


@dataclass
class PipelineConfig:
    asr_engine: str = field(
        default_factory=lambda: os.environ.get("CLIPFORGE_ASR", "whisper:small"))
    highlight_engine: str = field(
        default_factory=lambda: os.environ.get("CLIPFORGE_HL", "anthropic"))
    detector: str = field(
        default_factory=lambda: os.environ.get("CLIPFORGE_DETECTOR", "mediapipe"))
    examples_dir: Optional[str] = field(
        default_factory=lambda: os.environ.get("CLIPFORGE_EXAMPLES") or None)
    min_score: int = 55
    max_candidates: int = 12
    captions: bool = True

    def resolve(self) -> "PipelineConfig":
        """Apply graceful fallbacks; log what changed."""
        cfg = PipelineConfig(**self.__dict__)
        if cfg.highlight_engine.startswith("anthropic") and \
                not os.environ.get("ANTHROPIC_API_KEY"):
            print("[pipeline] no ANTHROPIC_API_KEY -> highlight engine: mock")
            cfg.highlight_engine = "mock"
        if cfg.detector == "mediapipe":
            try:
                importlib.import_module("mediapipe")
            except Exception:  # missing OR installed-but-broken
                print("[pipeline] mediapipe unavailable -> detector: haar")
                cfg.detector = "haar"
        return cfg


ProgressFn = Callable[[str], None]


def run_pipeline(job_dir: Path, cfg: PipelineConfig,
                 progress: ProgressFn = print) -> Manifest:
    """Run every remaining stage on an already-ingested job."""
    cfg = cfg.resolve()
    job_dir = Path(job_dir)
    m = Manifest.load(job_dir)

    if not m.is_at_least(Stage.TRANSCRIBED):
        progress("transcribing…")
        from clipforge.transcribe.backends import get_transcriber
        from clipforge.transcribe.run import transcribe_job
        m = transcribe_job(job_dir, get_transcriber(cfg.asr_engine))

    if not m.is_at_least(Stage.CANDIDATES_READY):
        progress("finding highlights…")
        from clipforge.highlight.backends import get_highlighter
        from clipforge.highlight.run import highlight_job
        hl = get_highlighter(
            cfg.highlight_engine,
            examples_dir=Path(cfg.examples_dir) if cfg.examples_dir else None)
        m = highlight_job(job_dir, hl, min_score=cfg.min_score,
                          max_candidates=cfg.max_candidates)

    if not m.is_at_least(Stage.REFINED):
        progress("refining boundaries…")
        from clipforge.refine.run import refine_job
        m = refine_job(job_dir)

    if not m.is_at_least(Stage.REFRAMED):
        progress("computing crop paths…")
        from clipforge.reframe.detect import get_detector
        from clipforge.reframe.run import reframe_job
        m = reframe_job(job_dir, get_detector(cfg.detector))

    if not m.is_at_least(Stage.RENDERED):
        progress("rendering clips…")
        from clipforge.render.run import render_job
        m = render_job(job_dir, captions=cfg.captions)

    from clipforge.quality.report import quality_job
    quality_job(job_dir)  # idempotent: only creates missing reports

    progress("done")
    return m
