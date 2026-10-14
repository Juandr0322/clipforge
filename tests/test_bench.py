"""F25 tests: benchmark config, run, usage/cost and report.

One module-scoped run with mock engines covers a missing file, a corrupt
file, a good video and a multi-file entry. Everything is written under
pytest's temp folders, never inside the repo.
"""
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from clipforge import bench
from clipforge.highlight.backends import AnthropicHighlighter
from clipforge.pipeline import PipelineConfig
from clipforge.refine.refine import RefinedSet
from clipforge.review.store import Review, upsert_review

MOCK_CFG = PipelineConfig(asr_engine="mock", highlight_engine="mock",
                          detector="haar")
ENTRIES = [  # bad entries first: later entries must still run
    {"name": "gone", "type": "gaming", "file": "nope.mp4"},
    {"name": "bad", "type": "cinematic", "file": "bad.mp4"},
    {"name": "talk1", "type": "talk", "file": "talk.mp4"},
    {"name": "multi", "type": "day_in_life", "files": ["a.mp4", "b.mp4"]},
]


def _write_config(bench_dir: Path, videos) -> None:
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "bench.json").write_text(
        json.dumps({"videos": videos}), encoding="utf-8")


@pytest.fixture(scope="module")
def bench_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("bench")
    bench_dir, ws = root / "bench", root / "ws"
    _write_config(bench_dir, ENTRIES)
    # 18 s is the shortest synthetic video that yields a clip with the mock
    # engines (MIN_CLIP_SEC is 15); small frames keep render time down.
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=duration=18:size=320x180:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=18",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(bench_dir / "talk.mp4")],
        check=True, capture_output=True)
    (bench_dir / "bad.mp4").write_bytes(b"not a video")
    record = bench.run_bench(bench_dir, ws, MOCK_CFG,
                             progress=lambda m: None, run_id="r1")
    return {"bench": bench_dir, "ws": ws, "record": record}


def _row(record, name):
    (row,) = [v for v in record["videos"] if v["name"] == name]
    return row


# ---------------------------------------------------------------------- run

def test_statuses_and_later_entries_still_run(bench_run):
    rec = bench_run["record"]
    assert [v["name"] for v in rec["videos"]] == [e["name"] for e in ENTRIES]
    status = {v["name"]: v["status"] for v in rec["videos"]}
    assert status == {"gone": "missing", "bad": "failed", "talk1": "ok",
                      "multi": "skipped"}
    assert _row(rec, "gone")["error"] == "file not found: nope.mp4"
    bad = _row(rec, "bad")
    assert bad["error"] and len(bad["error"]) <= 200 and "\n" not in bad["error"]
    assert "F35" in _row(rec, "multi")["error"]


def test_ok_entry_job_and_metrics(bench_run):
    rec, ws = bench_run["record"], bench_run["ws"]
    row = _row(rec, "talk1")
    assert row["job_id"] == "bench-r1-talk1"
    job = ws / row["job_id"]
    assert (job / "manifest.json").is_file()
    assert row["source_min"] == pytest.approx(0.3, abs=0.02)
    assert row["process_min"] > 0
    assert {"ingested", "transcribed", "candidates_ready", "refined",
            "reframed", "rendered"} <= set(row["stage_sec"])
    cands = json.loads((job / "candidates.json").read_text(encoding="utf-8"))
    assert row["candidates"] == len(cands["candidates"]) >= 1
    assert row["clips"] == len(list((job / "renders").glob("clip_*.mp4"))) >= 1
    assert row["usage"]["engine"] == "mock" and row["usage"]["calls"] >= 1
    assert row["cost_usd"] == 0.0


def test_run_file_has_commit_and_engines(bench_run):
    path = bench_run["bench"] / "runs" / "r1.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["run_id"] == "r1" and saved["finished_at"]
    assert "commit" in saved            # None only when git is unavailable
    assert saved["engines"] == {"asr": "mock", "highlight": "mock",
                                "detector": "haar"}
    assert saved["prices_checked"] == bench.PRICES_CHECKED
    assert saved["videos"] == bench_run["record"]["videos"]


def test_results_csv_written_and_byte_identical(bench_run):
    b, ws = bench_run["bench"], bench_run["ws"]
    first = (b / "results.csv").read_bytes()
    bench.write_results(b, ws)
    assert (b / "results.csv").read_bytes() == first
    lines = first.decode("utf-8").splitlines()
    assert lines[0].split(",") == list(bench.CSV_COLUMNS)
    assert len(lines) == 1 + len(ENTRIES)


def test_report_adds_approvals_from_reviews(bench_run, tmp_path):
    # Work on copies so the shared run stays unreviewed for other tests.
    b, ws = tmp_path / "bench", tmp_path / "ws"
    shutil.copytree(bench_run["bench"], b)
    shutil.copytree(bench_run["ws"], ws)
    job = ws / "bench-r1-talk1"
    row = {r["video"]: r for r in bench.build_rows(b, ws)}["talk1"]
    assert row["reviewed"] == 0 and row["approval_rate"] is None
    clip = RefinedSet.load(job / "refined.json").clips[0]
    upsert_review(job, Review(clip_id=clip.id, decision="approved"))
    bench.write_results(b, ws)
    once = (b / "results.csv").read_bytes()
    bench.write_results(b, ws)
    assert (b / "results.csv").read_bytes() == once
    row = {r["video"]: r for r in bench.build_rows(b, ws)}["talk1"]
    assert (row["reviewed"], row["approved"], row["approval_rate"]) == (1, 1, 1.0)


def test_compare_latest_against_previous_run(bench_run, tmp_path):
    b = tmp_path / "bench"
    shutil.copytree(bench_run["bench"], b)
    r2 = json.loads((b / "runs" / "r1.json").read_text(encoding="utf-8"))
    r2["run_id"] = "r2"
    for v in r2["videos"]:
        if v["name"] == "talk1":
            v["clips"], v["cost_usd"] = 3, 0.25
    (b / "runs" / "r2.json").write_text(json.dumps(r2), encoding="utf-8")
    lines = bench.compare_latest(bench.build_rows(b, bench_run["ws"]))
    assert lines[0] == "Latest run r2 vs previous r1"
    (talk,) = [ln for ln in lines if "talk1" in ln]
    clips_before = _row(bench_run["record"], "talk1")["clips"]
    assert f"clips {clips_before} -> 3" in talk
    assert "cost $0.0000 -> $0.2500" in talk


def test_compare_with_no_runs(tmp_path):
    assert bench.compare_latest(bench.build_rows(tmp_path, tmp_path)) == [
        "No benchmark runs yet: run `bench run` first."]


def test_only_unknown_name_runs_nothing(bench_run, tmp_path):
    b = tmp_path / "bench"
    _write_config(b, ENTRIES)
    with pytest.raises(bench.BenchConfigError, match="not in bench.json: nope"):
        bench.run_bench(b, tmp_path / "ws", MOCK_CFG, only=["nope"],
                        progress=lambda m: None)
    assert not (b / "runs").exists()


def test_only_runs_named_entries(tmp_path):
    b = tmp_path / "bench"
    _write_config(b, ENTRIES)
    rec = bench.run_bench(b, tmp_path / "ws", MOCK_CFG, only=["gone", "multi"],
                          progress=lambda m: None, run_id="r9")
    assert [v["name"] for v in rec["videos"]] == ["gone", "multi"]


def test_run_ids_never_overwrite(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    first = bench._new_run_id(runs)
    (runs / f"{first}.json").write_text("{}", encoding="utf-8")
    second = bench._new_run_id(runs)
    assert second != first and second.startswith(first)


# ------------------------------------------------------------------- config

@pytest.mark.parametrize("videos, message", [
    ([{"name": "a", "type": "vlog", "file": "a.mp4"}], "unknown type 'vlog'"),
    ([{"name": "a", "type": "talk", "file": "a.mp4"},
      {"name": "a", "type": "talk", "file": "b.mp4"}], "entry 2 ('a'): duplicate"),
    ([{"name": "../x", "type": "talk", "file": "a.mp4"}], "entry 1 ('../x'): name"),
    ([{"name": "x" * 41, "type": "talk", "file": "a.mp4"}], "at most 40"),
    ([{"name": "a", "type": "talk", "file": "a.mp4", "files": ["b.mp4"]}],
     "exactly one of 'file' or 'files'"),
    ([{"name": "a", "type": "talk"}], "exactly one of 'file' or 'files'"),
    ([{"name": "a", "type": "day_in_life", "files": []}], "'files' is empty"),
    ([{"name": "a", "type": "talk", "file": "a.mp4", "colour": "red"}],
     "videos.0.colour"),
    ([], "lists no videos"),
])
def test_bad_config_refused(tmp_path, videos, message):
    _write_config(tmp_path, videos)
    with pytest.raises(bench.BenchConfigError, match=message.replace("(", r"\(")
                       .replace(")", r"\)").replace(".", r"\.")):
        bench.load_config(tmp_path)


def test_invalid_json_and_missing_config(tmp_path):
    with pytest.raises(bench.BenchConfigError, match="run `bench init` first"):
        bench.load_config(tmp_path)
    (tmp_path / "bench.json").write_text("{oops", encoding="utf-8")
    with pytest.raises(bench.BenchConfigError, match="not valid JSON"):
        bench.load_config(tmp_path)


def test_init_creates_template_and_never_overwrites(tmp_path):
    b = tmp_path / "new" / "bench"
    path, created = bench.init_bench(b)
    assert created and path == b / "bench.json"
    assert (b / "README.txt").is_file()
    conf = bench.load_config(b)                 # the template is valid
    assert {v.type for v in conf.videos} <= set(bench.CONTENT_TYPES)
    path.write_text(json.dumps({"videos": [
        {"name": "mine", "type": "talk", "file": "x.mp4"}]}), encoding="utf-8")
    _, created = bench.init_bench(b)
    assert not created
    assert [v.name for v in bench.load_config(b).videos] == ["mine"]


def test_resolve_bench_dir(monkeypatch, tmp_path):
    monkeypatch.delenv(bench.BENCH_ENV, raising=False)
    assert bench.resolve_bench_dir() == Path.home() / "clipforge-bench"
    monkeypatch.setenv(bench.BENCH_ENV, str(tmp_path / "env"))
    assert bench.resolve_bench_dir() == tmp_path / "env"
    assert bench.resolve_bench_dir(tmp_path / "arg") == tmp_path / "arg"
    monkeypatch.setenv(bench.BENCH_ENV, "~/b")
    assert bench.resolve_bench_dir() == Path.home() / "b"


# --------------------------------------------------------------------- cost

def test_cost_known_model():
    usage = {"engine": "anthropic/claude-sonnet-5-5",
             "model": "claude-sonnet-5-5", "input_tokens": 1_000_000,
             "output_tokens": 100_000, "cache_creation_input_tokens": 200_000,
             "cache_read_input_tokens": 1_000_000}
    # 2.00 + 0.1 * 10.00 + 0.2 * 2.50 + 1.0 * 0.10
    assert bench.cost_usd(usage) == pytest.approx(3.60)


def test_cost_unknown_model_is_blank_not_a_guess():
    assert bench.cost_usd({"engine": "anthropic/x", "model": "claude-new-9",
                           "input_tokens": 10}) is None
    assert bench.cost_usd(None) is None


def test_cost_mock_is_zero():
    assert bench.cost_usd({"engine": "mock", "model": None, "calls": 3}) == 0.0


def test_default_highlight_model_has_a_price():
    default = inspect.signature(AnthropicHighlighter.__init__) \
        .parameters["model"].default
    assert default in bench.PRICES


# ---------------------------------------------------------------------- CLI

def _cli(monkeypatch, *argv):
    from clipforge import cli
    monkeypatch.setattr(cli, "load_env", lambda *a, **k: False)  # never real .env
    monkeypatch.setattr(sys, "argv", ["clipforge", *argv])
    return cli.main()


def test_cli_init_report_and_bad_config(monkeypatch, tmp_path, capsys):
    b = tmp_path / "bench"
    assert _cli(monkeypatch, "bench", "init", "--bench-dir", str(b)) == 0
    assert (b / "bench.json").is_file()
    assert _cli(monkeypatch, "bench", "init", "--bench-dir", str(b)) == 0
    assert "already exists" in capsys.readouterr().out

    assert _cli(monkeypatch, "bench", "report", "--bench-dir", str(b),
                "--workspace", str(tmp_path / "ws")) == 0
    assert "No benchmark runs yet" in capsys.readouterr().out

    (b / "bench.json").write_text('{"videos": []}', encoding="utf-8")
    assert _cli(monkeypatch, "bench", "run", "--bench-dir", str(b),
                "--workspace", str(tmp_path / "ws")) == 2
    assert "BENCH FAILED" in capsys.readouterr().err


def test_cli_uses_env_bench_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(bench.BENCH_ENV, str(tmp_path / "from-env"))
    assert _cli(monkeypatch, "bench", "init") == 0
    assert (tmp_path / "from-env" / "bench.json").is_file()
