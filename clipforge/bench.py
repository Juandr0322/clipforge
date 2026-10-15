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
from clipforge.review.store import replace_with_retry

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

# USD per million tokens:
#   (input, output, 5-minute cache write, 1-hour cache write, cache read).
# Source: https://platform.claude.com/docs/en/about-claude/pricing
# Checked 2026-10-13. Standard (not batch) prices. Haiku 5.5 is its tier for
# prompts up to 100K tokens; a highlight chunk is far below that.
PRICES_CHECKED = "2026-10-13"
PRICES: dict[str, tuple[float, float, float, float, float]] = {
    "claude-sonnet-4-6": (3.00, 15.00, 3.75, 6.00, 0.30),
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 4.00, 0.10),
    "claude-haiku-4-5": (1.00, 5.00, 1.25, 2.00, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.20, 0.01),
    "claude-opus-5-5": (4.00, 20.00, 5.00, 8.00, 0.20),
}
# US-only inference (inference_geo "us") costs 1.1x on Claude 4.6 and later.
US_GEO_MULTIPLIER = 1.1
US_GEO_MODELS = {"claude-sonnet-4-6", "claude-sonnet-5-5", "claude-haiku-5-5",
                 "claude-opus-5-5"}
STANDARD_GEOS = {"global"}

# `clipforge bench` exit codes
EXIT_OK, EXIT_VIDEO_PROBLEM, EXIT_SETUP, EXIT_RESULTS_LOCKED = 0, 1, 2, 3
RESULTS_LOCKED_HINT = (f"Could not update {RESULTS_NAME}: is it open in Excel "
                       "or another program? Close it, then run "
                       "`python -m clipforge.cli bench report`.")

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
run `bench report` again so results.csv picks up your approvals. Each video
is compared with the last earlier run that included it.

Exit codes: 0 all good, 1 a video failed or is missing, 2 setup problem
(bench.json, ffmpeg, no bench folder), 3 results.csv is open in another
program (close it and run `bench report`). When results.csv is locked AND a
video failed, the exit code is 3; the failed video is still listed above.
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
    """Temp file + rename, retried while Windows has dest open briefly."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}.{uuid.uuid4().hex}.tmp"
    try:
        with tmp.open("w", encoding="utf-8", newline="") as f:
            f.write(text)
        replace_with_retry(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


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
        dirty = subprocess.run(   # --no-optional-locks: never touch .git/index
            ["git", "--no-optional-locks", "status", "--porcelain",
             "--untracked-files=no"],
            cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
            timeout=10)
        suffix = "+dirty" if dirty.returncode == 0 and dirty.stdout.strip() \
            else ""
        return head.stdout.strip() + suffix
    except (OSError, subprocess.SubprocessError):
        return None


def cost_usd(usage: dict | None) -> float | None:
    """$ for one highlight run. Mock engines cost 0. Anything we can't price
    exactly (unknown model, unknown or mixed inference_geo) gives None
    (blank), never a guess."""
    if not usage:
        return None
    if usage.get("engine") == "mock":
        return 0.0
    model = usage.get("model") or ""
    price = PRICES.get(model)
    if price is None:
        return None
    geos = set(usage.get("inference_geo") or [])
    if geos - STANDARD_GEOS - {"us"} or (len(geos) > 1 and "us" in geos):
        return None           # an unknown geo, or a mix we can't split
    p_in, p_out, p_write_5m, p_write_1h, p_read = price

    def n(key: str) -> int:
        return int(usage.get(key) or 0)

    write_1h = n("cache_write_1h_tokens")
    write_5m = max(n("cache_creation_input_tokens") - write_1h, 0)
    total = (n("input_tokens") * p_in + n("output_tokens") * p_out
             + write_5m * p_write_5m + write_1h * p_write_1h
             + n("cache_read_input_tokens") * p_read) / 1e6
    if "us" in geos and model in US_GEO_MODELS:
        total *= US_GEO_MULTIPLIER
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
        loc = list(first["loc"])
        if len(loc) >= 2 and loc[0] == "videos" and isinstance(loc[1], int):
            # same 1-based numbering as the checks below
            where = f"entry {loc[1] + 1}" + "".join(f", {x}" for x in loc[2:])
        else:
            where = ".".join(str(x) for x in loc)
        raise BenchConfigError(f"{CONFIG_NAME} {where}: {first['msg']}")

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
        # Windows folders ignore case: "Podcast" and "podcast" would share
        # one job folder
        if e.name.casefold() in seen:
            raise BenchConfigError(
                f"{label}: duplicate name (names are compared ignoring case)")
        seen.add(e.name.casefold())
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
    try:
        src = Path(e.file).expanduser()     # "~nosuchuser/x" raises
        if not src.is_absolute():
            src = bench_dir / src
        found = src.is_file()
    except (RuntimeError, OSError, ValueError) as ex:
        row["status"] = "failed"
        row["error"] = f"bad file path {e.file!r}: {_short(ex)}"
        return row
    if not found:
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
    runs_dir.mkdir(parents=True, exist_ok=True)
    try:   # reserve the id: an earlier run's file is never overwritten
        with out.open("x", encoding="utf-8") as f:
            f.write(json.dumps(record, indent=2) + "\n")
    except FileExistsError:
        raise BenchConfigError(f"run id {run_id!r} was already used")
    workspace.mkdir(parents=True, exist_ok=True)
    for e in entries:
        record["videos"].append(
            _run_entry(e, bench_dir, workspace, run_id, resolved, progress))
        _write_atomic(out, json.dumps(record, indent=2) + "\n")  # keep partials
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_atomic(out, json.dumps(record, indent=2) + "\n")
    try:
        write_results(bench_dir, workspace)
    except OSError as ex:   # e.g. results.csv open in Excel; the run is saved
        record["results_error"] = _short(ex)
    return record


# ------------------------------------------------------------------- report

def _load_runs(bench_dir: Path) -> list[dict]:
    runs = []
    for p in sorted((Path(bench_dir) / RUNS_DIR).glob("*.json")):
        try:
            runs.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    # Oldest first by the UTC start time: run ids are local time and their
    # "-2", "-10" suffixes don't sort as numbers.
    runs.sort(key=lambda r: (str(r.get("started_at") or ""),
                             str(r.get("run_id", ""))))
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
    """Lines comparing each video in the latest run with the last earlier
    run that included it (partial runs with --only are common).
    `rows` must be in run order, as build_rows returns them."""
    order: list[str] = []
    for r in rows:
        if r.get("run_id") and r["run_id"] not in order:
            order.append(r["run_id"])
    if not order:
        return ["No benchmark runs yet: run `bench run` first."]
    latest = order[-1]
    rank = {run_id: i for i, run_id in enumerate(order)}
    lines = [f"Latest run {latest}; each video vs its last earlier run"]
    fields = (("status", "status", "str"), ("clips", "clips", "str"),
              ("approval_rate", "approved", "pct"),
              ("process_min", "time", "min"), ("cost_usd", "cost", "usd"))
    for r in [r for r in rows if r["run_id"] == latest]:
        earlier = [x for x in rows if x["video"] == r["video"]
                   and rank.get(x.get("run_id"), len(order)) < rank[latest]]
        before = earlier[-1] if earlier else None
        parts = []
        for key, label, kind in fields:
            now = _fmt(r.get(key), kind)
            if before is not None:
                parts.append(f"{label} {_fmt(before.get(key), kind)} -> {now}")
            else:
                parts.append(f"{label} {now}")
        vs = f"vs {before['run_id']}" if before else "first run"
        lines.append(f"  {r['video']:<14} ({vs}) " + " | ".join(parts))
    return lines
