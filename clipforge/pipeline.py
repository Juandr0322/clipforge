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
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from clipforge.common.manifest import STAGE_ORDER, Manifest, Stage
from clipforge.recipes import (DEFAULT_RECIPE, Recipe, RecipeError, load_recipe,
                               read_record, write_record)


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
    # None = take the value from the recipe (clipforge/recipes/files/<name>.json)
    recipe: Optional[str] = field(
        default_factory=lambda: os.environ.get("CLIPFORGE_RECIPE") or None)
    min_score: Optional[int] = None
    max_candidates: Optional[int] = None
    captions: Optional[bool] = None

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


RECIPE_FAILED = ("The recipe could not be loaded. Check CLIPFORGE_RECIPE and the "
                 "recipe files, then run the job again.")


def _job_recipe(job_dir: Path, m: Manifest, cfg: PipelineConfig,
                progress: ProgressFn) -> Optional[Recipe]:
    """The recipe a job runs with: the one recorded on the job, else the requested one.

    The first run that has a stage to run records name and version in recipe.json;
    later runs keep the recorded recipe so a re-run never mixes settings. If the
    recipe file was edited to a new version since, the record is updated and a
    warning is shown. A finished job is never given a record it did not run with
    (returns None when it has none), and a broken file cannot fail its no-op re-run.
    """
    recorded = read_record(job_dir)
    if m.is_at_least(Stage.RENDERED):
        if recorded is None:
            return None
        try:
            return load_recipe(recorded["name"])
        except RecipeError as exc:
            print(f"[pipeline] recipe unreadable on a finished job: {exc}",
                  file=sys.stderr)
            return None
    if recorded is not None:
        if cfg.recipe and cfg.recipe != recorded["name"]:
            progress(f"job already uses recipe {recorded['name']}; "
                     f"ignoring {cfg.recipe}")
        recipe = load_recipe(recorded["name"])
        if recipe.version != recorded["version"]:
            progress(f"recipe {recipe.name} changed from version "
                     f"{recorded['version']} to {recipe.version}; "
                     "using the new version")
            write_record(job_dir, recipe)
        return recipe
    recipe = load_recipe(cfg.recipe or DEFAULT_RECIPE)
    write_record(job_dir, recipe)
    return recipe


def _recipe_failed(job_dir: Path, m: Manifest, exc: RecipeError) -> None:
    """Put a fixed sentence in the manifest (the page shows it); details go to stderr."""
    print(f"[pipeline] recipe error: {exc}", file=sys.stderr)
    i = STAGE_ORDER.index(m.stage) if m.stage in STAGE_ORDER else 0
    stage = STAGE_ORDER[min(i + 1, len(STAGE_ORDER) - 1)]
    m.record(stage, datetime.now(timezone.utc), ok=False, error=RECIPE_FAILED)
    m.save(job_dir)


def run_pipeline(job_dir: Path, cfg: PipelineConfig,
                 progress: ProgressFn = print) -> Manifest:
    """Run every remaining stage on an already-ingested job."""
    cfg = cfg.resolve()
    job_dir = Path(job_dir)
    m = Manifest.load(job_dir)
    try:
        recipe = _job_recipe(job_dir, m, cfg, progress)
    except RecipeError as exc:
        _recipe_failed(job_dir, m, exc)
        raise
    min_score = cfg.min_score if cfg.min_score is not None else (
        recipe.min_score if recipe else 55)
    max_candidates = cfg.max_candidates if cfg.max_candidates is not None else (
        recipe.max_candidates if recipe else 12)
    captions = cfg.captions if cfg.captions is not None else (
        recipe.captions if recipe else True)

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
        m = highlight_job(job_dir, hl, min_score=min_score,
                          max_candidates=max_candidates)

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
        m = render_job(job_dir, captions=captions,
                       audio=recipe.audio_config() if recipe else None)

    from clipforge.quality.report import quality_job
    try:
        quality_job(job_dir)  # idempotent: only creates missing reports
    except Exception:  # noqa: BLE001 - reports must not fail a rendered job
        print("quality reports could not be written; "
              "re-run the pipeline to retry", file=sys.stderr)
    try:
        from clipforge.quality.run import hook_scores_job
        from clipforge.reframe.detect import get_detector
        hook_scores_job(job_dir, get_detector(cfg.detector),
                        captions=captions)
    except Exception:  # noqa: BLE001 - best effort, like the reports
        print("hook scores could not be computed; "
              "re-run the pipeline to retry", file=sys.stderr)

    progress("done")
    return m
