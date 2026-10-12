"""
Module 1 tests. All fixtures are synthesized with ffmpeg (testsrc + sine)
so the suite is fully self-contained — no real videos needed.
"""
import subprocess
from pathlib import Path

import pytest

from clipforge.common.manifest import Manifest, Stage
from clipforge.ingest.ingest import (
    AUDIO_SR, IngestError, ingest, parse_probe, parse_probe_audio_only, probe,
)

FIX = Path(__file__).parent / "fixtures"


def mk(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


@pytest.fixture(scope="session", autouse=True)
def fixtures():
    FIX.mkdir(exist_ok=True)

    def gen(name, dur, w, h, fps, audio=True, extra=None):
        out = FIX / name
        if out.exists():
            return
        cmd = ["ffmpeg", "-y",
               "-f", "lavfi", "-i", f"testsrc=duration={dur}:size={w}x{h}:rate={fps}"]
        if audio:
            cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}"]
        cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
        if audio:
            cmd += ["-c:a", "aac", "-shortest"]
        cmd += (extra or []) + [str(out)]
        mk(cmd)

    gen("normal_720p.mp4", 12, 1280, 720, 30)
    gen("uhd_4k.mp4", 8, 3840, 2160, 30)
    gen("vertical_source.mp4", 8, 720, 1280, 30)
    gen("weird_fps.mp4", 8, 640, 360, 7)           # 7 fps -> should force 30
    gen("high_fps.mp4", 6, 640, 360, 120)          # 120 fps -> should force 30
    gen("no_audio.mp4", 10, 640, 360, 30, audio=False)
    gen("too_short.mp4", 2, 640, 360, 30)
    gen("odd_dims.mp4", 8, 641, 361, 30, extra=["-vf", "pad=642:362"])
    # corrupt file: just garbage bytes with an mp4 extension
    (FIX / "corrupt.mp4").write_bytes(b"\x00\x01garbage" * 500)


# ---------------------------------------------------------------- happy path

def test_normal_video_full_pipeline(tmp_path):
    m = ingest(FIX / "normal_720p.mp4", tmp_path)
    assert m.stage == Stage.INGESTED
    job = tmp_path / m.job_id

    # outputs exist
    for f in ("normalized.mp4", "audio.wav", "proxy.mp4", "manifest.json"):
        assert (job / f).exists(), f"missing {f}"

    # invariants: audio format
    wav = parse_probe_audio_only(job / "audio.wav")
    assert wav["sample_rate"] == AUDIO_SR
    assert wav["channels"] == 1

    # invariants: duration preserved
    assert abs(wav["duration"] - m.source.duration_sec) < 0.5
    norm = parse_probe(probe(job / "normalized.mp4"))
    assert abs(norm.duration - m.source.duration_sec) < 0.5

    # manifest audit trail
    assert m.history[-1].ok and m.history[-1].stage == Stage.INGESTED
    assert m.source.sha256 and m.source.size_bytes > 0


def test_4k_downscaled_to_1080(tmp_path):
    m = ingest(FIX / "uhd_4k.mp4", tmp_path)
    assert m.ingest.normalized_height == 1080
    assert m.ingest.normalized_width == 1920


def test_vertical_source_preserved(tmp_path):
    m = ingest(FIX / "vertical_source.mp4", tmp_path)
    # vertical stays vertical; nothing exceeds 1080 height
    assert m.ingest.normalized_width < m.ingest.normalized_height
    assert m.ingest.normalized_height <= 1080


def test_weird_fps_forced_to_30(tmp_path):
    m = ingest(FIX / "weird_fps.mp4", tmp_path)
    assert m.ingest.normalized_fps == 30.0


def test_high_fps_forced_to_30(tmp_path):
    m = ingest(FIX / "high_fps.mp4", tmp_path)
    assert m.ingest.normalized_fps == 30.0


def test_odd_dimensions_made_even(tmp_path):
    m = ingest(FIX / "odd_dims.mp4", tmp_path)
    assert m.ingest.normalized_width % 2 == 0
    assert m.ingest.normalized_height % 2 == 0


def test_proxy_is_small(tmp_path):
    m = ingest(FIX / "normal_720p.mp4", tmp_path)
    job = tmp_path / m.job_id
    pr = parse_probe(probe(job / "proxy.mp4"))
    assert pr.height == 480
    assert (job / "proxy.mp4").stat().st_size < \
        (job / "normalized.mp4").stat().st_size


# ---------------------------------------------------------------- rejections

def test_no_audio_rejected(tmp_path):
    with pytest.raises(IngestError, match="no audio"):
        ingest(FIX / "no_audio.mp4", tmp_path, job_id="noaudio")
    # failure must be recorded in the manifest, not just raised
    m = Manifest.load(tmp_path / "noaudio")
    assert m.is_failed and m.failed_stage == Stage.INGESTED
    assert "no audio" in (m.error or "")


def test_too_short_rejected(tmp_path):
    with pytest.raises(IngestError, match="too short"):
        ingest(FIX / "too_short.mp4", tmp_path)


def test_corrupt_file_rejected_cleanly(tmp_path):
    with pytest.raises(IngestError):
        ingest(FIX / "corrupt.mp4", tmp_path, job_id="corrupt")
    m = Manifest.load(tmp_path / "corrupt")
    assert m.is_failed and m.failed_stage == Stage.INGESTED


def test_unsupported_extension(tmp_path):
    bad = tmp_path / "file.txt"
    bad.write_text("hi")
    with pytest.raises(IngestError, match="Unsupported extension"):
        ingest(bad, tmp_path)


def test_missing_file(tmp_path):
    with pytest.raises(IngestError, match="not found"):
        ingest(tmp_path / "nope.mp4", tmp_path)


# ---------------------------------------------------------------- idempotency

def test_rerun_is_noop(tmp_path):
    m1 = ingest(FIX / "normal_720p.mp4", tmp_path, job_id="idem")
    mtime = (tmp_path / "idem" / "normalized.mp4").stat().st_mtime
    m2 = ingest(FIX / "normal_720p.mp4", tmp_path, job_id="idem")
    assert m2.job_id == m1.job_id
    assert (tmp_path / "idem" / "normalized.mp4").stat().st_mtime == mtime
    assert len(m2.history) == len(m1.history)  # no extra stage runs
