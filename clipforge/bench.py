"""
Benchmark runner (F25).

Runs a fixed set of Diego's own videos through the full pipeline and records
the numbers every later change is judged on: clips found, approval rate,
processing minutes, tokens and $ per video.

Everything lives in the bench folder (default ~/clipforge-bench), never in the
repo, because video names can carry client names:

    clipforge-bench/
      bench.json          which videos to run (written by `bench init`)
      README.txt
      runs/<run_id>.json  one file per `bench run`
      results.csv         rebuilt from runs/ by `bench report`

Jobs go into the normal workspace as bench-<run_id>-<name>, so they show up
on the review website. Approval rates come from reviewing them there.
"""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, ConfigDict, ValidationError

from clipforge.common.manifest import Manifest, is_safe_job_id
from clipforge.highlight.backends import USAGE_NAME

BENCH_ENV = "CLIPFORGE_BENCH_DIR"
DEFAULT_BENCH_DIR = Path.home() / "clipforge-bench"
CONFIG_NAME = "bench.json"
README_NAME = "README.txt"
RUNS_DIR = "runs"
RESULTS_NAME = "results.csv"
REPO_ROOT = Path(__file__).resolve().parent.parent

# Same vocabulary as the recipes planned in docs/SPEC.md, Group D.
CONTENT_TYPES = ("talk", "gaming", "storytime", "talking_head", "cinematic",
                 "day_in_life")
MAX_NAME_LEN = 40

# USD per million tokens: (input, output, 5-minute cache write, cache read).
# Source: https://platform.claude.com/docs/en/about-claude/pricing
# Checked 2026-10-13. Standard (not batch) prices. Haiku 5.5 is its tier for
# prompts up to 100K tokens; a highlight chunk is far below that.
PRICES_CHECKED = "2026-10-13"
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-sonnet-4-6": (3.00, 15.00, 3.75, 0.30),
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 0.10),
    "claude-haiku-4-5": (1.00, 5.00, 1.25, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01),
    "claude-opus-5-5": (4.00, 20.00, 5.00, 0.20),
}

# Stage names as recorded in manifest.history, in pipeline order.
STAGE_COLUMNS = ("ingested", "transcribed", "candidates_ready", "refined",
                 "reframed", "rendered")

CSV_COLUMNS = (
    "run_id", "started_at", "commit", "video", "type", "status", "job_id",
    "source_min", "process_min",
    *(f"{s}_sec" for s in STAGE_COLUMNS),
    "candidates", "clips", "reviewed", "approved", "approval_rate",
    "asr", "highlight_engine", "hl_calls", "input_tokens", "output_tokens",
    "cache_write_tokens", "cache_read_tokens", "cost_usd", "error",
)

TEMPLATE = {
    "videos": [
        {"name": "podcast", "type": "talk", "file": "podcast.mp4",
         "notes": "30-60 min podcast or talk stream"},
        {"name": "gaming", "type": "gaming", "file": "gaming.mp4",
         "notes": "30-60 min gaming stream with facecam"},
        {"name": "storytime", "type": "storytime", "file": "storytime.mp4",
         "notes": "10-20 min storytime or talking-head video"},
        {"name": "cinematic", "type": "cinematic", "file": "cinematic.mp4",
         "notes": "5-15 min of cinematic or travel footage"},
        {"name": "day-in-life", "type": "day_in_life",
         "files": ["day-in-life/clip01.mp4", "day-in-life/clip02.mp4"],
         "notes": "10-30 phone clips in order; runs once multi-file "
                  "editing (F35) exists"},
    ]
}

README = """ClipForge benchmark folder
==========================

Put the benchmark videos in this folder and list them in bench.json.
Only use videos you recorded yourself or have the creator's written OK for.
Once a video is in the set, never edit or replace it: every before/after
comparison depends on the set staying the same.

bench.json fields, per video:
  name   short id: letters, digits, '-' and '_' (becomes part of the job id)
  type   talk | gaming | storytime | talking_head | cinematic | day_in_life
  file   one video, relative to this folder (or a full path)
  files  several clips for one edited video (day_in_life); skipped until
         multi-file editing (F35) is built
  notes  anything you want to remember about it

Commands (run from the clipforge folder):
  python -m clipforge.cli bench run       process every video (takes a while)
  python -m clipforge.cli bench run --only podcast gaming
  python -m clipforge.cli bench report    rebuild results.csv and compare
                                          the last two runs

After a run, review the bench-<run>-<name> jobs on the review website, then
run `bench report` again so results.csv picks up your approvals.
"""


class BenchConfigError(Exception):
    pass


class BenchEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    type: str
    file: Optional[str] = None
    files: Optional[list[str]] = None
    notes: str = ""


class BenchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    videos: list[BenchEntry]


# ------------------------------------------------------------------ helpers

def resolve_bench_dir(arg: str | os.PathLike | None = None) -> Path:
    """--bench-dir wins, then $CLIPFORGE_BENCH_DIR, then ~/clipforge-bench."""
    raw = arg or os.environ.get(BENCH_ENV)
    return Path(raw).expanduser() if raw else DEFAULT_BENCH_DIR


def _write_atomic(dest: Path, text: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}.{uuid.uuid4().hex}.tmp"
    try:
        with tmp.open("w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def _short(err: BaseException | str, limit: int = 200) -> str:
    text = str(err).strip().splitlines()[0] if str(err).strip() else \
        type(err).__name__
    return text if len(text) <= limit else text[:limit - 1] + "…"


def git_commit() -> str | None:
    """Short commit of the code that ran, '+dirty' when there are local
    changes; None when git isn't available."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, encoding="utf-8", timeout=10)
        if head.returncode != 0:
            return None
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
            timeout=10)
        suffix = "+dirty" if dirty.returncode == 0 and dirty.stdout.strip() \
            else ""
        return head.stdout.strip() + suffix
    except (OSError, subprocess.SubprocessError):
        return None


def cost_usd(usage: dict | None) -> float | None:
    """$ for one highlight run. Mock engines cost 0; an unknown model gives
    None (blank), never a guess."""
    if not usage:
        return None
    if usage.get("engine") == "mock":
        return 0.0
    price = PRICES.get(usage.get("model") or "")
    if price is None:
        return None
    p_in, p_out, p_write, p_read = price
    total = (usage.get("input_tokens", 0) * p_in
             + usage.get("output_tokens", 0) * p_out
             + usage.get("cache_creation_input_tokens", 0) * p_write
             + usage.get("cache_read_input_tokens", 0) * p_read) / 1e6
    return round(total, 4)


# ------------------------------------------------------------------- config

def init_bench(bench_dir: Path) -> tuple[Path, bool]:
    """Create the folder with a template bench.json and README.txt.
    Returns (bench.json path, created). Never overwrites bench.json."""
    bench_dir = Path(bench_dir)
    bench_dir.mkdir(parents=True, exist_ok=True)
    readme = bench_dir / README_NAME
    if not readme.exists():
        _write_atomic(readme, README)
    cfg = bench_dir / CONFIG_NAME
    if cfg.exists():
        return cfg, False
    _write_atomic(cfg, json.dumps(TEMPLATE, indent=2) + "\n")
    return cfg, True


def load_config(bench_dir: Path) -> BenchConfig:
    path = Path(bench_dir) / CONFIG_NAME
    if not path.is_file():
        raise BenchConfigError(
            f"no {CONFIG_NAME} in {bench_dir}: run `bench init` first")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise BenchConfigError(
            f"{CONFIG_NAME} is not valid JSON (line {e.lineno}: {e.msg})")
    try:
        conf = BenchConfig.model_validate(raw)
    except ValidationError as e:
        first = e.errors()[0]
        where = ".".join(str(x) for x in first["loc"])
        raise BenchConfigError(f"{CONFIG_NAME}: {where}: {first['msg']}")

    if not conf.videos:
        raise BenchConfigError(f"{CONFIG_NAME} lists no videos")
    seen: set[str] = set()
    for i, e in enumerate(conf.videos, start=1):
        label = f"{CONFIG_NAME} entry {i} ({e.name!r})"
        if not is_safe_job_id(e.name) or len(e.name) > MAX_NAME_LEN:
            raise BenchConfigError(
                f"{label}: name must be letters, digits, '-' and '_', "
                f"start with a letter or digit, and be at most "
                f"{MAX_NAME_LEN} characters")
        if e.name in seen:
            raise BenchConfigError(f"{label}: duplicate name")
        seen.add(e.name)
        if e.type not in CONTENT_TYPES:
            raise BenchConfigError(
                f"{label}: unknown type {e.type!r}; use one of "
                f"{', '.join(CONTENT_TYPES)}")
        if (e.file is None) == (e.files is None):
            raise BenchConfigError(
                f"{label}: give exactly one of 'file' or 'files'")
        if e.files is not None and not e.files:
            raise BenchConfigError(f"{label}: 'files' is empty")
    return conf


# ---------------------------------------------------------------------- run

def _new_run_id(runs_dir: Path) -> str:
    base = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id, n = base, 2
    while (runs_dir / f"{run_id}.json").exists():
        run_id, n = f"{base}-{n}", n + 1
    return run_id


def _collect(job_dir: Path, row: dict) -> None:
    """Fill a result row from whatever the job left on disk."""
    try:
        m = Manifest.load(job_dir)
    except (OSError, ValueError):
        return
    if m.source:
        row["source_min"] = round(m.source.duration_sec / 60, 2)
    stage_sec: dict[str, float] = {}
    for rec in m.history:
        if rec.ok:
            stage_sec[rec.stage.value] = round(
                stage_sec.get(rec.stage.value, 0.0) + rec.duration_sec, 2)
    row["stage_sec"] = stage_sec
    cands = job_dir / "candidates.json"
    if cands.is_file():
        try:
            data = json.loads(cands.read_text(encoding="utf-8"))
            row["candidates"] = len(data.get("candidates", []))
        except (OSError, ValueError):
            pass
    renders = job_dir / (m.renders_dir or "renders")
    if renders.is_dir():
        row["clips"] = len(list(renders.glob("clip_*.mp4")))
    usage_file = job_dir / USAGE_NAME
    if usage_file.is_file():
        try:
            row["usage"] = json.loads(usage_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    row["cost_usd"] = cost_usd(row.get("usage"))


def _run_entry(e: BenchEntry, bench_dir: Path, workspace: Path, run_id: str,
               cfg, progress: Callable[[str], None]) -> dict:
    from clipforge.ingest.ingest import ingest
    from clipforge.pipeline import run_pipeline

    row: dict = {"name": e.name, "type": e.type, "job_id": None,
                 "status": None, "error": None, "source_min": None,
                 "process_min": None, "stage_sec": {}, "candidates": None,
                 "clips": None, "usage": None, "cost_usd": None}
    if e.files is not None:
        row["status"] = "skipped"
        row["error"] = "multi-file entries need multi-file editing (F35)"
        return row
    src = Path(e.file).expanduser()
    if not src.is_absolute():
        src = bench_dir / src
    if not src.is_file():
        row["status"] = "missing"
        row["error"] = f"file not found: {e.file}"
        return row

    job_id = f"bench-{run_id}-{e.name}"
    row["job_id"] = job_id
    started = time.monotonic()
    try:
        progress(f"[{e.name}] ingesting…")
        ingest(src, workspace, job_id=job_id)
        run_pipeline(workspace / job_id, cfg,
                     progress=lambda msg: progress(f"[{e.name}] {msg}"))
        row["status"] = "ok"
    except Exception as ex:  # noqa: BLE001 — one bad video must not stop the run
        row["status"] = "failed"
        row["error"] = _short(ex)
    row["process_min"] = round((time.monotonic() - started) / 60, 2)
    _collect(workspace / job_id, row)
    return row


def run_bench(bench_dir: Path, workspace: Path, cfg,
              only: list[str] | None = None,
              progress: Callable[[str], None] = print,
              run_id: str | None = None) -> dict:
    """Run every entry (or only the named ones). Returns the run record,
    which is also saved as runs/<run_id>.json after each entry."""
    bench_dir, workspace = Path(bench_dir), Path(workspace)
    conf = load_config(bench_dir)
    entries = conf.videos
    if only:
        known = {e.name for e in entries}
        unknown = [n for n in only if n not in known]
        if unknown:
            raise BenchConfigError(
                f"not in {CONFIG_NAME}: {', '.join(unknown)}")
        entries = [e for e in entries if e.name in set(only)]

    runs_dir = bench_dir / RUNS_DIR
    run_id = run_id or _new_run_id(runs_dir)
    if not is_safe_job_id(run_id):
        raise BenchConfigError(f"invalid run id {run_id!r}")
    resolved = cfg.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "commit": git_commit(),
        "engines": {"asr": resolved.asr_engine,
                    "highlight": resolved.highlight_engine,
                    "detector": resolved.detector},
        "prices_checked": PRICES_CHECKED,
        "videos": [],
    }
    out = runs_dir / f"{run_id}.json"
    for e in entries:
        record["videos"].append(
            _run_entry(e, bench_dir, workspace, run_id, resolved, progress))
        _write_atomic(out, json.dumps(record, indent=2) + "\n")  # keep partials
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_atomic(out, json.dumps(record, indent=2) + "\n")
    write_results(bench_dir, workspace)
    return record


# ------------------------------------------------------------------- report

def _load_runs(bench_dir: Path) -> list[dict]:
    runs = []
    for p in sorted((Path(bench_dir) / RUNS_DIR).glob("*.json")):
        try:
            runs.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    runs.sort(key=lambda r: str(r.get("run_id", "")))
    return runs


def build_rows(bench_dir: Path, workspace: Path) -> list[dict]:
    """One row per (run, video), approvals read fresh from reviews.json."""
    from clipforge.review.store import job_metrics

    rows = []
    for run in _load_runs(bench_dir):
        engines = run.get("engines") or {}
        for v in run.get("videos", []):
            usage = v.get("usage") or {}
            stage_sec = v.get("stage_sec") or {}
            row = {
                "run_id": run.get("run_id"),
                "started_at": run.get("started_at"),
                "commit": run.get("commit"),
                "video": v.get("name"),
                "type": v.get("type"),
                "status": v.get("status"),
                "job_id": v.get("job_id"),
                "source_min": v.get("source_min"),
                "process_min": v.get("process_min"),
                **{f"{s}_sec": stage_sec.get(s) for s in STAGE_COLUMNS},
                "candidates": v.get("candidates"),
                "clips": v.get("clips"),
                "reviewed": None, "approved": None, "approval_rate": None,
                "asr": engines.get("asr"),
                "highlight_engine": usage.get("engine")
                or engines.get("highlight"),
                "hl_calls": usage.get("calls"),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "cache_write_tokens": usage.get("cache_creation_input_tokens"),
                "cache_read_tokens": usage.get("cache_read_input_tokens"),
                "cost_usd": v.get("cost_usd"),
                "error": v.get("error"),
            }
            job_id = v.get("job_id")
            if job_id and is_safe_job_id(job_id) and \
                    (Path(workspace) / job_id / "manifest.json").is_file():
                jm = job_metrics(Path(workspace) / job_id)
                row["reviewed"] = jm["reviewed"]
                row["approved"] = jm["approved"]
                row["approval_rate"] = jm["approval_rate"]
            rows.append(row)
    return rows


def _cell(v) -> str:
    return "" if v is None else str(v)


def write_results(bench_dir: Path, workspace: Path) -> Path:
    """Rebuild results.csv from runs/. Same inputs -> same bytes."""
    rows = build_rows(bench_dir, workspace)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for r in rows:
        w.writerow([_cell(r.get(c)) for c in CSV_COLUMNS])
    dest = Path(bench_dir) / RESULTS_NAME
    _write_atomic(dest, buf.getvalue())
    return dest


def _fmt(v, kind: str) -> str:
    if v is None:
        return "-"
    if kind == "pct":
        return f"{v * 100:.0f}%"
    if kind == "usd":
        return f"${v:.4f}"
    if kind == "min":
        return f"{v:.1f}m"
    return str(v)


def compare_latest(rows: list[dict]) -> list[str]:
    """Lines comparing the latest run with the previous one, per video."""
    run_ids = sorted({r["run_id"] for r in rows if r.get("run_id")})
    if not run_ids:
        return ["No benchmark runs yet: run `bench run` first."]
    latest = run_ids[-1]
    prev = run_ids[-2] if len(run_ids) > 1 else None
    by_key = {(r["run_id"], r["video"]): r for r in rows}
    head = f"Latest run {latest}" + (f" vs previous {prev}" if prev else
                                     " (no earlier run to compare)")
    lines = [head]
    fields = (("status", "status", "str"), ("clips", "clips", "str"),
              ("approval_rate", "approved", "pct"),
              ("process_min", "time", "min"), ("cost_usd", "cost", "usd"))
    for r in [r for r in rows if r["run_id"] == latest]:
        before = by_key.get((prev, r["video"])) if prev else None
        parts = []
        for key, label, kind in fields:
            now = _fmt(r.get(key), kind)
            if before is not None:
                parts.append(f"{label} {_fmt(before.get(key), kind)} -> {now}")
            else:
                parts.append(f"{label} {now}")
        lines.append(f"  {r['video']:<14} " + " | ".join(parts))
    return lines
