"""
Module 5 tests — the quality metrics from the plan, as executable checks:
  - jitter metric: noisy detections must NOT become camera shake
  - velocity limit: pans are slow and deliberate
  - hard cuts: teleporting subjects produce cuts, not mad pans
  - hysteresis: one bad detection never switches the subject
  - fallbacks: no face -> center; face lost -> hold
"""
import subprocess
from pathlib import Path

import numpy as np
import pytest

from clipforge.common.manifest import Manifest, Stage
from clipforge.highlight.backends import MockHighlighter
from clipforge.highlight.run import highlight_job
from clipforge.ingest.ingest import ingest
from clipforge.refine.run import refine_job
from clipforge.reframe.core import (
    Observation, compute_crop_path, path_jitter, sample_frames,
    select_subject, smooth_path,
)
from clipforge.reframe.detect import (
    CropKeyframe, CropPath, FaceBox, HaarDetector, MockDetector, get_detector,
)
from clipforge.reframe.run import ReframeError, reframe_job
from clipforge.transcribe.backends import MockTranscriber
from clipforge.transcribe.run import transcribe_job

FIX = Path(__file__).parent / "fixtures"


def F(cx, cy=0.4, w=0.15, h=0.25):
    return FaceBox(cx=cx, cy=cy, w=w, h=h)


def obs_series(centers, fps=5.0):
    return [Observation(t=i / fps, face=(F(c) if c is not None else None))
            for i, c in enumerate(centers)]


# ------------------------------------------------------------- crop path math

def test_croppath_interpolation():
    p = CropPath(clip_id=0, mode="tracked", keyframes=[
        CropKeyframe(t=0, cx=0.2), CropKeyframe(t=1, cx=0.4)])
    assert p.cx_at(-1) == 0.2
    assert p.cx_at(0.5) == pytest.approx(0.3)
    assert p.cx_at(5) == 0.4


def test_croppath_hard_cut_no_interpolation():
    p = CropPath(clip_id=0, mode="tracked", keyframes=[
        CropKeyframe(t=0, cx=0.2), CropKeyframe(t=1, cx=0.8, cut=True)])
    assert p.cx_at(0.99) == 0.2     # holds until the cut
    assert p.cx_at(1.0) == 0.8      # jumps at the cut


# ---------------------------------------------------------------- smoothing

def test_static_face_static_path():
    p = smooth_path(obs_series([0.3] * 50), clip_id=0)
    assert p.mode == "static"
    assert p.keyframes[0].cx == pytest.approx(0.3, abs=0.01)


def test_noise_absorbed_by_dead_zone():
    """Detector noise (±1.5% around a fixed point) must produce ~zero motion."""
    rng = np.random.default_rng(42)
    centers = 0.5 + rng.uniform(-0.015, 0.015, size=100)
    p = smooth_path(obs_series(list(centers)), clip_id=0)
    assert path_jitter(p) < 1e-6
    assert p.mode == "static"


def test_jitter_metric_flags_raw_noise():
    """Sanity: the SAME noisy input, if used raw, has high jitter.
    This proves the metric can tell good from bad."""
    rng = np.random.default_rng(42)
    centers = 0.5 + rng.uniform(-0.03, 0.03, size=100)
    raw = CropPath(clip_id=0, mode="tracked", keyframes=[
        CropKeyframe(t=i / 5, cx=float(np.clip(c, 0, 1)))
        for i, c in enumerate(centers)])
    smoothed = smooth_path(obs_series(list(centers)), clip_id=0)
    assert path_jitter(raw) > 10 * max(path_jitter(smoothed), 1e-9)


def test_velocity_limit_respected():
    """Face walks across frame; camera must never exceed MAX_VEL."""
    centers = list(np.linspace(0.2, 0.8, 60))  # 12s walk at 5fps
    p = smooth_path(obs_series(centers), clip_id=0)
    for a, b in zip(p.keyframes, p.keyframes[1:]):
        if b.cut:
            continue
        dt = b.t - a.t
        assert abs(b.cx - a.cx) <= 0.10 * dt + 1e-6


def test_slow_walk_is_followed():
    centers = list(np.linspace(0.3, 0.6, 100))  # 20s slow walk
    p = smooth_path(obs_series(centers), clip_id=0)
    assert p.keyframes[-1].cx == pytest.approx(0.6, abs=0.05)


def test_teleport_becomes_hard_cut():
    centers = [0.2] * 20 + [0.8] * 20
    p = smooth_path(obs_series(centers), clip_id=0)
    cuts = [k for k in p.keyframes if k.cut]
    assert len(cuts) == 1
    assert cuts[0].cx == pytest.approx(0.8, abs=0.01)
    # after the cut, camera is settled at the new position (no long pan)
    assert p.keyframes[-1].cx == pytest.approx(0.8, abs=0.02)


def test_single_frame_glitch_is_not_a_cut():
    """One bogus far-away detection must be ignored, not cause a cut."""
    centers = [0.3] * 20 + [0.9] + [0.3] * 20
    p = smooth_path(obs_series(centers), clip_id=0)
    assert not any(k.cut for k in p.keyframes)
    assert p.keyframes[-1].cx == pytest.approx(0.3, abs=0.02)


def test_no_faces_center_crop():
    p = smooth_path(obs_series([None] * 30), clip_id=0)
    assert p.mode == "center"
    assert p.keyframes[0].cx == 0.5


def test_face_lost_mid_clip_holds():
    centers = [0.35] * 20 + [None] * 20 + [0.35] * 10
    p = smooth_path(obs_series(centers), clip_id=0)
    assert p.mode == "static"
    assert p.keyframes[0].cx == pytest.approx(0.35, abs=0.01)


def test_path_stays_in_bounds():
    centers = [0.02] * 10 + [0.98] * 30
    p = smooth_path(obs_series(centers), clip_id=0)
    for k in p.keyframes:
        assert 0.0 <= k.cx <= 1.0


# ---------------------------------------------------------------- hysteresis

def test_hysteresis_ignores_single_bigger_face():
    """A bigger face flashing for 1-2 samples must not steal the camera."""
    a, b = F(0.3, w=0.15), F(0.7, w=0.30)
    samples = [(i / 5, [a]) for i in range(10)]
    samples += [(2.0, [a, b]), (2.2, [a, b])]       # only 2 samples
    samples += [(i / 5, [a]) for i in range(12, 20)]
    obs = select_subject(samples)
    chosen = [o.face.cx for o in obs if o.face]
    assert all(abs(c - 0.3) < 0.01 for c in chosen)


def test_hysteresis_sustained_switch_happens():
    a, b = F(0.3, w=0.15), F(0.7, w=0.30)
    samples = [(i / 5, [a]) for i in range(10)]
    samples += [((10 + i) / 5, [a, b]) for i in range(6)]  # sustained
    obs = select_subject(samples)
    assert obs[-1].face.cx == pytest.approx(0.7)


# ------------------------------------------------------------------ sampler

@pytest.fixture(scope="module")
def tiny_video():
    FIX.mkdir(exist_ok=True)
    src = FIX / "gradient_10s.mp4"
    if not src.exists():
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc=duration=10:size=320x180:rate=30",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(src)], check=True, capture_output=True)
    return src


def test_sampler_rate_and_shape(tiny_video):
    frames = list(sample_frames(tiny_video, 2.0, 6.0, fps=5))
    assert 15 <= len(frames) <= 21          # ~4s * 5fps
    t0, f0 = frames[0]
    assert abs(t0 - 2.0) < 0.05
    assert f0.shape == (180, 320, 3)


def test_haar_on_faceless_video_falls_back_to_center(tiny_video):
    """Plumbing test: real detector, real video, no faces -> center crop."""
    p = compute_crop_path(tiny_video, 0, 0.0, 4.0, HaarDetector())
    assert p.mode == "center"


def test_mock_detector_full_path(tiny_video):
    det = MockDetector(lambda t: [F(0.3 + 0.04 * t)])  # slow drift
    p = compute_crop_path(tiny_video, 7, 0.0, 8.0, det)
    assert p.clip_id == 7
    assert p.mode in ("tracked", "static")
    assert path_jitter(p) < 0.01


# ------------------------------------------------------------- stage runner

@pytest.fixture(scope="module")
def refined_job(tmp_path_factory):
    src = FIX / "speech_30s.mp4"
    if not src.exists():
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=30:size=640x360:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=300:duration=30",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(src)],
            check=True, capture_output=True)
    ws = tmp_path_factory.mktemp("ws")
    m = ingest(src, ws, job_id="t5job")
    jd = ws / m.job_id
    transcribe_job(jd, MockTranscriber())
    highlight_job(jd, MockHighlighter())
    refine_job(jd)
    return jd


def test_stage_runner_happy_path(refined_job):
    m = reframe_job(refined_job, HaarDetector())
    assert m.stage == Stage.REFRAMED
    crops = list((refined_job / m.crop_paths_dir).glob("clip_*.json"))
    assert crops
    for c in crops:
        CropPath.load(c)  # schema-valid


def test_stage_runner_requires_refined(tmp_path):
    Manifest(job_id="fresh").save(tmp_path)
    with pytest.raises(ReframeError, match="refine first"):
        reframe_job(tmp_path, HaarDetector())


def test_stage_runner_idempotent(refined_job):
    m1 = reframe_job(refined_job, HaarDetector())
    m2 = reframe_job(refined_job, HaarDetector())
    assert len(m2.history) == len(m1.history)


def test_factory():
    assert isinstance(get_detector("haar"), HaarDetector)
    with pytest.raises(ValueError):
        get_detector("nope")
