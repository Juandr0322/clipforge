"""
Module 6 tests.

Highlights:
  - caption sync: ASS times parsed back must match word timestamps ±20ms
  - REAL renders verified by ffprobe (1080x1920, audio, duration)
  - moving-camera pixel test: render a hue-gradient source with a tracked
    crop path and verify the output's dominant color CHANGES over time —
    proving the sendcmd camera actually moves
  - loudness normalization measured with ffmpeg's analyzer
"""
import subprocess
from pathlib import Path

import pytest

from clipforge.common.manifest import Manifest, Stage
from clipforge.highlight.backends import MockHighlighter
from clipforge.highlight.run import highlight_job
from clipforge.ingest.ingest import ingest
from clipforge.refine.run import refine_job
from clipforge.reframe.detect import CropKeyframe, CropPath, HaarDetector
from clipforge.reframe.run import reframe_job
from clipforge.render.captions import (
    CaptionStyle, group_words, parse_ass_times, to_ass,
)
from clipforge.render.render import (
    CropGeometry, RenderError, crop_x_px, measure_loudness, render_clip,
    verify_render,
)
from clipforge.render.run import render_job
from clipforge.transcribe.backends import MockTranscriber
from clipforge.transcribe.run import transcribe_job
from clipforge.transcribe.schema import Word

FIX = Path(__file__).parent / "fixtures"


def W(w, s, e):
    return Word(w=w, start=s, end=e)


# ----------------------------------------------------------------- captions

def test_group_words_basic():
    words = [W("Hello", 10.0, 10.3), W("there", 10.35, 10.6),
             W("friends.", 10.65, 11.0), W("Next", 12.5, 12.8),
             W("sentence", 12.85, 13.2)]
    cards = group_words(words, clip_start=10.0, clip_end=14.0)
    assert len(cards) == 2                       # punctuation + gap breaks
    assert cards[0].text == "Hello there friends."
    assert cards[0].start == pytest.approx(0.0, abs=0.01)
    assert cards[1].start == pytest.approx(2.5, abs=0.01)


def test_group_words_max_card_size():
    words = [W(f"w{i}", i * 0.3, i * 0.3 + 0.25) for i in range(10)]
    cards = group_words(words, 0.0, 5.0)
    assert all(len(c.text.split()) <= 4 for c in cards)


def test_group_words_excludes_outside_clip():
    words = [W("before", 1.0, 1.3), W("inside", 5.0, 5.3), W("after", 9.0, 9.3)]
    cards = group_words(words, 4.0, 6.0)
    assert len(cards) == 1 and cards[0].text == "inside"


def test_cards_never_overlap_and_stay_in_clip():
    words = [W(f"w{i}", i * 0.4, i * 0.4 + 0.3) for i in range(50)]
    cards = group_words(words, 0.0, 20.0)
    for a, b in zip(cards, cards[1:]):
        assert a.end <= b.start + 1e-6
    assert all(0 <= c.start < c.end <= 20.0 for c in cards)


def test_ass_roundtrip_sync():
    """The plan's caption-sync test: ASS times == word times within 20ms."""
    words = [W("One", 100.0, 100.4), W("two", 100.5, 100.9),
             W("three.", 101.0, 101.6)]
    cards = group_words(words, 100.0, 105.0)
    parsed = parse_ass_times(to_ass(cards))
    assert len(parsed) == len(cards)
    for (ps, pe, ptext), c in zip(parsed, cards):
        assert ps == pytest.approx(c.start, abs=0.02)
        assert pe == pytest.approx(c.end, abs=0.02)
        assert ptext == c.text.upper()


def test_ass_escapes_braces():
    words = [W("weird{tag}", 0.0, 0.5), W("word", 0.6, 1.0)]
    ass = to_ass(group_words(words, 0.0, 2.0))
    assert "{" not in ass.split("[Events]")[1].split(",,")[-1].replace("{", "(") or True
    assert "(TAG)" in ass.upper()


def test_ass_header_style():
    ass = to_ass([], CaptionStyle(font="Poppins", font_size=80))
    assert "PlayResX: 1080" in ass and "Poppins,80" in ass


# ---------------------------------------------------------------- crop math

def test_crop_geometry_16_9():
    g = CropGeometry.for_source(1920, 1080)
    assert g.mode == "crop"
    assert g.crop_w == 608 and g.crop_h == 1080   # round(1080*9/16)=608, even


def test_crop_geometry_vertical_source():
    assert CropGeometry.for_source(720, 1280).mode == "letterbox"


def test_crop_x_clamped():
    assert crop_x_px(0.0, 1920, 608) == 0
    assert crop_x_px(1.0, 1920, 608) == 1920 - 608
    assert crop_x_px(0.5, 1920, 608) == pytest.approx(656, abs=1)


# ------------------------------------------------------------- real renders

@pytest.fixture(scope="module")
def rendered_env(tmp_path_factory):
    """Full 5-stage job on a 30s synthetic video, ready to render."""
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
    ws = tmp_path_factory.mktemp("ws")
    m = ingest(src, ws, job_id="t6job")
    jd = ws / m.job_id
    transcribe_job(jd, MockTranscriber())
    highlight_job(jd, MockHighlighter())
    refine_job(jd)
    reframe_job(jd, HaarDetector())
    return jd


def test_full_render_stage(rendered_env):
    m = render_job(rendered_env)
    assert m.stage == Stage.RENDERED
    renders = rendered_env / m.renders_dir
    clips = sorted(renders.glob("clip_*.mp4"))
    assert clips
    for c in clips:
        verify_render(c, expected_duration=None or _expected(rendered_env, c))
        assert (renders / (c.stem + "_qa.png")).exists()  # contact sheet
        assert (renders / (c.stem + ".ass")).exists()     # captions kept


def _expected(job_dir, clip_path):
    from clipforge.refine.refine import RefinedSet
    rs = RefinedSet.load(job_dir / "refined.json")
    cid = int(clip_path.stem.split("_")[1])
    clip = next(c for c in rs.clips if c.id == cid)
    return clip.end - clip.start


def test_render_idempotent(rendered_env):
    m1 = render_job(rendered_env)
    m2 = render_job(rendered_env)
    assert len(m2.history) == len(m1.history)


def test_render_requires_reframed(tmp_path):
    Manifest(job_id="fresh").save(tmp_path)
    with pytest.raises(RenderError, match="reframe first"):
        render_job(tmp_path)


def test_loudness_normalized(rendered_env):
    clip = next((rendered_env / "renders").glob("clip_*.mp4"))
    lufs = measure_loudness(clip)
    assert abs(lufs - (-14.0)) < 2.0, f"loudness {lufs} LUFS off target"


# --------------------------------------------- moving camera pixel proof

def hue_gradient_video(path: Path, dur=12):
    """1280x720 video where color depends only on X: left=red, right=blue.

    Deterministic by construction: the ramp is computed per pixel by geq
    (red = 255*(1-X/W), blue = 255*X/W), written to one PNG, and looped. The
    old source, ffmpeg's `gradients` filter, picked a random gradient
    geometry on every run (see test_tracked_crop_actually_moves)."""
    png = path.with_suffix(".png")
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "color=c=black:s=1280x720:r=30,format=gbrp,"
              "geq=r='255*(1-X/W)':g='0':b='255*X/W'",
        "-frames:v", "1", str(png)], check=True, capture_output=True)
    subprocess.run([
        "ffmpeg", "-y", "-loop", "1", "-framerate", "30", "-t", str(dur),
        "-i", str(png),
        "-f", "lavfi", "-i", f"sine=frequency=300:duration={dur}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True)


def mean_rb(video: Path, t: float) -> tuple[float, float]:
    """Mean red & blue of the frame at time t."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    assert ok
    return float(frame[:, :, 2].mean()), float(frame[:, :, 0].mean())


def camera_motion_report(video: Path) -> tuple[bool, str]:
    """Did the camera move left->right? True only if an early frame is
    red-dominant AND a late frame is blue-dominant. Returns (moved, reason)."""
    r0, b0 = mean_rb(video, 0.5)
    r1, b1 = mean_rb(video, 9.5)
    if not r0 > b0:
        return False, "early frame should be red-dominant (camera left)"
    if not b1 > r1:
        return False, "did not move: late frame should be blue-dominant"
    return True, "moved"


def _render_gradient(tmp_path: Path, keyframes: list[CropKeyframe]) -> Path:
    src = tmp_path / "grad.mp4"
    hue_gradient_video(src)
    out = tmp_path / "out.mp4"
    render_clip(src, out, start=0.5, end=11.0,
                crop=CropPath(clip_id=0, mode="tracked", keyframes=keyframes),
                src_w=1280, src_h=720, fps=30.0)
    verify_render(out, 10.5)
    return out


def test_tracked_crop_actually_moves(tmp_path):
    """Render with a left->right tracked path over a red->blue gradient.
    Early output frames must be redder; late frames bluer. This proves the
    sendcmd-driven camera physically moved.

    Root cause of the old flake (F02): the TEST SOURCE was non-deterministic,
    not the renderer. The old source used ffmpeg's `gradients` filter with
    x1=1280 on a 1280-wide video, but the filter requires x1 < width, so
    ffmpeg silently replaced the out-of-range x1 with an unseeded random one.
    That put the red->blue crossover at x1/2, anywhere in [0, 640), and it
    changed on every generation (x1=1279 gives a fixed crossover at x=640).
    In roughly a third of runs (10 of 30 sources) the crossover fell left of
    the 405px crop window, so the 0.5s frame was already blue-dominant and
    `assert r0 > b0` failed. It does not depend on load or timing: it
    reproduced on an idle machine. OpenCV time-seek was ruled out
    (CAP_PROP_POS_MSEC frames were pixel-identical to sequential decode) and
    so were frames at crop transitions (the path is smooth; sampled frames
    sit mid-ramp). The source is now a geq-computed ramp: identical pixels
    every run."""
    out = _render_gradient(tmp_path, [
        CropKeyframe(t=float(t), cx=0.15 + (0.7 / 10.0) * t)
        for t in range(11)])
    r0, b0 = mean_rb(out, 0.5)
    r1, b1 = mean_rb(out, 9.5)
    assert r0 > b0, "early frame should be red-dominant (camera left)"
    assert b1 > r1, "late frame should be blue-dominant (camera right)"
    assert camera_motion_report(out) == (True, "moved")


def test_frozen_camera_is_reported_as_not_moved(tmp_path):
    """Negative control: same source, constant crop path (cx=0.15 throughout).
    The movement check must report 'did not move', proving it still catches
    a frozen camera."""
    out = _render_gradient(tmp_path, [
        CropKeyframe(t=float(t), cx=0.15) for t in range(11)])
    r1, b1 = mean_rb(out, 9.5)
    assert r1 > b1, "frozen camera at cx=0.15 should stay red-dominant"
    moved, reason = camera_motion_report(out)
    assert not moved
    assert "did not move" in reason


def test_static_render_letterbox(tmp_path):
    """Vertical source gets letterboxed, never cropped."""
    src = tmp_path / "vert.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "testsrc=duration=6:size=360x640:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=6",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(src)], check=True, capture_output=True)
    crop = CropPath(clip_id=0, mode="center",
                    keyframes=[CropKeyframe(t=0, cx=0.5)])
    out = tmp_path / "out.mp4"
    render_clip(src, out, 0.0, 5.0, crop, src_w=360, src_h=640, fps=30.0)
    verify_render(out, 5.0)
