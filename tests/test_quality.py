"""F09: per-clip quality report files and the post-render quality pass."""
import json
import shutil
import subprocess
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clipforge.ingest.ingest import ingest
from clipforge.pipeline import PipelineConfig, run_pipeline
from clipforge.quality.report import (
    ClipReport, SECTIONS, load_report, report_path, update_section,
)
from clipforge.server.app import create_app

FIX = Path(__file__).parent / "fixtures"
CFG = PipelineConfig(asr_engine="mock", highlight_engine="mock",
                     detector="haar")


@pytest.fixture(scope="module")
def source_video():
    FIX.mkdir(exist_ok=True)
    src = FIX / "speech_60s.mp4"
    if not src.exists():
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=60:size=640x360:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=300:duration=60",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(src)],
            check=True, capture_output=True)
    return src


@pytest.fixture(scope="module")
def job(tmp_path_factory, source_video):
    ws = tmp_path_factory.mktemp("qws")
    ingest(source_video, ws, job_id="q1")
    run_pipeline(ws / "q1", CFG, progress=lambda s: None)
    return ws / "q1"


def test_every_clip_has_valid_report(job):
    clips = sorted((job / "renders").glob("clip_???.mp4"))
    assert clips
    for mp4 in clips:
        rp = mp4.with_suffix(".report.json")
        assert rp.exists()
        rep = ClipReport.model_validate_json(rp.read_text(encoding="utf-8"))
        assert rep.clip_id == int(mp4.stem.split("_")[1])


def test_unknown_section_and_bad_type_rejected(tmp_path):
    (tmp_path / "renders").mkdir()
    with pytest.raises(ValueError):
        update_section(tmp_path, 1, "bogus", {})
    with pytest.raises(ValueError):
        update_section(tmp_path, 1, "hook_score", "not a dict")
    assert not list((tmp_path / "renders").iterdir())


def test_update_section_keeps_other_sections(tmp_path):
    (tmp_path / "renders").mkdir()
    update_section(tmp_path, 3, "hook_score", {"score": 7})
    update_section(tmp_path, 3, "loop", {"ok": True})
    update_section(tmp_path, 3, "hook_score", {"score": 9})
    rep = load_report(tmp_path, 3)
    assert rep.hook_score == {"score": 9} and rep.loop == {"ok": True}
    assert not list((tmp_path / "renders").glob("*.tmp"))


def test_concurrent_writers_different_sections(tmp_path):
    (tmp_path / "renders").mkdir()
    errs = []

    def w(section, value):
        try:
            for _ in range(10):
                update_section(tmp_path, 1, section, value)
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    vals = {"hook_score": {"a": 1}, "value_adds": [1], "pacing_flags": [2],
            "variants": [3], "loop": {"b": 2}, "exports": [4],
            "audio_layer": {"c": 5}}
    ts = [threading.Thread(target=w, args=kv) for kv in vals.items()]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs
    data = json.loads(report_path(tmp_path, 1).read_text(encoding="utf-8"))
    for s in SECTIONS:
        assert data[s] == vals[s]


def test_rerun_does_not_rewrite_reports(job):
    reports = list((job / "renders").glob("*.report.json"))
    assert reports
    before = {p: p.stat().st_mtime_ns for p in reports}
    run_pipeline(job, CFG, progress=lambda s: None)
    assert {p: p.stat().st_mtime_ns for p in reports} == before


def test_report_written_with_lf_only(tmp_path):
    (tmp_path / "renders").mkdir()
    update_section(tmp_path, 1, "hook_score", {"score": 1})
    assert b"\r\n" not in report_path(tmp_path, 1).read_bytes()


def test_quality_failure_does_not_fail_pipeline(job, monkeypatch, capsys):
    import clipforge.quality.report as qr

    def boom(job_dir):
        raise OSError("/secret/path exploded")

    monkeypatch.setattr(qr, "quality_job", boom)
    m = run_pipeline(job, CFG, progress=lambda s: None)
    assert m.stage.value == "rendered"
    err = capsys.readouterr().err
    assert "quality reports could not be written" in err
    assert "/secret" not in err


def test_job_without_reports_still_loads(job, tmp_path):
    ws = tmp_path / "ws"
    shutil.copytree(job.parent, ws)
    for p in (ws / "q1" / "renders").glob("*.report.json"):
        p.unlink()
    c = TestClient(create_app(ws, config=CFG, sync_pipeline=True),
                   raise_server_exceptions=False)
    r = c.get("/api/jobs/q1")
    assert r.status_code == 200
    assert r.json()["clips"]
