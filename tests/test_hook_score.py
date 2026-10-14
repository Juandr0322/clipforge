"""F11: first-3-seconds hook score."""
import subprocess
from pathlib import Path

import pytest

from clipforge.ingest.ingest import ingest
from clipforge.pipeline import PipelineConfig, run_pipeline
from clipforge.quality.hook import (
    WEIGHTS, combine, hook_score, score_face, score_motion, score_speech,
    score_text,
)
from clipforge.quality.report import load_report
from clipforge.reframe.detect import FaceBox, MockDetector
from clipforge.transcribe.schema import Word

FIX = Path(__file__).parent / "fixtures"
CFG = PipelineConfig(asr_engine="mock", highlight_engine="mock",
                     detector="haar")


def _ffmpeg(out, src):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", src, "-t", "4",
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", str(out)],
                   check=True, capture_output=True)


@pytest.fixture(scope="module")
def moving(tmp_path_factory):
    p = tmp_path_factory.mktemp("hv") / "moving.mp4"
    _ffmpeg(p, "testsrc=duration=4:size=320x568:rate=30")
    return p


@pytest.fixture(scope="module")
def static(tmp_path_factory):
    p = tmp_path_factory.mktemp("hs") / "static.mp4"
    _ffmpeg(p, "color=c=0x336699:duration=4:size=320x568:rate=30")
    return p


def W(start):
    return [Word(w="hi", start=start, end=start + 0.4),
            Word(w="there", start=start + 0.5, end=start + 0.9)]


def test_early_first_word_beats_late_first_word():
    early = score_speech(W(10.2), 10.0, 40.0)
    late = score_speech(W(12.5), 10.0, 40.0)
    assert early["score"] > late["score"]
    assert early["score"] == 100 and 0 < late["score"] < 20
    assert score_speech([], 10.0, 40.0)["score"] == 0
    assert score_speech(W(20.0), 10.0, 40.0)["score"] == 0


def test_moving_video_scores_higher_motion_than_static(moving, static):
    from clipforge.quality.hook import read_gray_frames
    m = score_motion(read_gray_frames(moving))
    s = score_motion(read_gray_frames(static))
    assert m["score"] > s["score"]
    assert s["score"] == 0 and m["score"] > 0


def test_face_scores_follow_mock_detector(moving):
    face = MockDetector(lambda t: [FaceBox(cx=0.5, cy=0.4, w=0.3, h=0.3)])
    none = MockDetector(lambda t: [])
    a = hook_score(moving, W(0.1), 0.0, 4.0, face)
    b = hook_score(moving, W(0.1), 0.0, 4.0, none)
    assert a["sub_scores"]["face"]["score"] == 100
    assert b["sub_scores"]["face"]["score"] == 0
    assert score_face([])["score"] == 0


def test_text_zero_without_captions_or_overlay_and_positive_with():
    assert score_text([])["score"] == 0
    on = score_text([(0.0, 1.0), (0.8, 2.5)])
    assert on["score"] > 0
    assert score_text([(0.0, 3.0)])["score"] == 100
    assert score_text([(5.0, 9.0)])["score"] == 0     # outside first 3 s
    assert score_text([(float("nan"), 2.0)])["score"] == 0


def test_hook_score_shape_and_total(moving):
    det = MockDetector(lambda t: [])
    hs = hook_score(moving, W(0.2), 0.0, 4.0, det, [(0.0, 3.0)])
    assert set(hs["sub_scores"]) == set(WEIGHTS)
    for sub in hs["sub_scores"].values():
        assert 0 <= sub["score"] <= 100
        assert isinstance(sub["reason"], str) and sub["reason"]
        assert "/" not in sub["reason"]
    assert 0 <= hs["total"] <= 100
    assert hs["total"] == combine(hs["sub_scores"])["total"]


def test_unreadable_video_still_gives_a_score(tmp_path):
    bad = tmp_path / "nope.mp4"
    bad.write_bytes(b"not a video")
    hs = hook_score(bad, W(0.2), 0.0, 4.0, MockDetector(lambda t: []))
    assert hs["sub_scores"]["motion"]["score"] == 0
    assert str(tmp_path) not in str(hs)


@pytest.fixture(scope="module")
def job(tmp_path_factory):
    src = FIX / "speech_30s.mp4"
    if not src.exists():
        FIX.mkdir(exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=30:size=640x360:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=300:duration=30",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(src)],
            check=True, capture_output=True)
    ws = tmp_path_factory.mktemp("hws")
    ingest(src, ws, job_id="h1")
    run_pipeline(ws / "h1", CFG, progress=lambda s: None)
    return ws / "h1"


def test_every_rendered_clip_has_hook_score(job):
    clips = sorted((job / "renders").glob("clip_???.mp4"))
    assert clips
    for mp4 in clips:
        rep = load_report(job, int(mp4.stem.split("_")[1]))
        hs = rep.hook_score
        assert 0 <= hs["total"] <= 100
        assert set(hs["sub_scores"]) == {"speech", "motion", "face", "text"}
        for sub in hs["sub_scores"].values():
            assert 0 <= sub["score"] <= 100 and sub["reason"]
        assert hs["sub_scores"]["text"]["score"] > 0   # captions are on


def test_rerun_keeps_hook_scores(job):
    paths = list((job / "renders").glob("*.report.json"))
    before = {p: p.stat().st_mtime_ns for p in paths}
    run_pipeline(job, CFG, progress=lambda s: None)
    assert {p: p.stat().st_mtime_ns for p in paths} == before


def test_pipeline_survives_hook_failure(job, monkeypatch, capsys):
    import clipforge.quality.run as qr

    def boom(*a, **k):
        raise OSError("/secret/path exploded")

    monkeypatch.setattr(qr, "hook_scores_job", boom)
    m = run_pipeline(job, CFG, progress=lambda s: None)
    assert m.stage.value == "rendered"
    err = capsys.readouterr().err
    assert "hook scores could not be computed" in err and "/secret" not in err


def test_captions_off_gives_text_zero(job, tmp_path):
    import shutil
    from clipforge.quality.run import hook_scores_job
    jd = tmp_path / "h1"
    shutil.copytree(job, jd)
    for p in (jd / "renders").glob("*.report.json"):
        p.unlink()
    n = hook_scores_job(jd, MockDetector(lambda t: []), captions=False)
    assert n > 0
    for mp4 in (jd / "renders").glob("clip_???.mp4"):
        rep = load_report(jd, int(mp4.stem.split("_")[1]))
        assert rep.hook_score["sub_scores"]["text"]["score"] == 0
