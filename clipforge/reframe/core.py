"""
Module 5 core — subject selection and camera-path smoothing.

This is where "professional vs drunk-camera" is decided:

  SUBJECT SELECTION (per sampled frame)
    - largest face wins by default
    - HYSTERESIS: we only switch subjects if a challenger dominates for
      SWITCH_FRAMES consecutive samples — one noisy detection must never
      yank the camera
    - no face: hold last known position (HOLD), fall back to center if
      we never saw one

  PATH SMOOTHING (over the whole clip)
    - DEAD ZONE: ignore target moves smaller than DEAD_ZONE (breathing,
      detector noise)
    - VELOCITY LIMIT: pan at most MAX_VEL per second — slow, deliberate
    - HARD CUT: if the target jumps farther than CUT_DIST and stays there,
      jump instantly (marked cut=True) instead of a long ugly pan
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from clipforge.reframe.detect import (
    CropKeyframe, CropPath, FaceBox, FaceDetector, MockDetector,
)

SAMPLE_FPS = 5.0          # analysis rate on the proxy
DEAD_ZONE = 0.03          # normalized x; smaller errors don't wake the camera
SETTLE_EPS = 0.005        # camera goes back to sleep within this error
MAX_VEL = 0.10            # max pan speed: 10% of frame width per second
FOLLOW_GAIN = 2.0         # P-controller gain (1/s): higher = snappier
EMA_TAU = 0.6             # target low-pass time constant (seconds)
CUT_DIST = 0.25           # jumps larger than this become hard cuts
SWITCH_FRAMES = 3         # consecutive samples before subject switch
HEADROOM = 0.18           # face center sits this far below crop top


# ---------------------------------------------------------------- sampling

def sample_frames(video_path, start: float, end: float,
                  fps: float = SAMPLE_FPS) -> Iterator[tuple[float, "np.ndarray"]]:
    """Yield (t_seconds_absolute, BGR frame) at ~fps within [start, end]."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        t = start
        step = 1.0 / fps
        while t < end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            yield t, frame
            t += step
    finally:
        cap.release()


# ------------------------------------------------------- subject selection

@dataclass
class Observation:
    t: float                       # relative to clip start
    face: Optional[FaceBox]        # chosen subject at this sample (or None)


def select_subject(samples: list[tuple[float, list[FaceBox]]],
                   switch_frames: int = SWITCH_FRAMES) -> list[Observation]:
    """Pick one subject per sample with hysteresis against flicker."""
    obs: list[Observation] = []
    current: Optional[FaceBox] = None
    challenger_count = 0

    for t, faces in samples:
        if not faces:
            obs.append(Observation(t=t, face=None))
            continue
        biggest = max(faces, key=lambda f: f.area)
        if current is None:
            current = biggest
            challenger_count = 0
        else:
            # nearest face to current position = same person (simple tracker)
            nearest = min(faces, key=lambda f: abs(f.cx - current.cx))
            if abs(nearest.cx - current.cx) < 0.15:
                same = nearest
            else:
                same = None
            if same is not None and biggest is same:
                current = same
                challenger_count = 0
            elif same is not None:
                # same person still visible but someone bigger appeared
                challenger_count += 1
                if challenger_count >= switch_frames:
                    current = biggest
                    challenger_count = 0
                else:
                    current = same
            else:
                # lost the current person entirely
                challenger_count += 1
                if challenger_count >= switch_frames:
                    current = biggest
                    challenger_count = 0
                else:
                    obs.append(Observation(t=t, face=None))  # hold
                    continue
        obs.append(Observation(t=t, face=current))
    return obs


# ---------------------------------------------------------- path smoothing

def _median3(xs: list[float]) -> list[float]:
    """3-tap median filter: annihilates single-sample detector glitches
    while passing genuine (sustained) jumps with 1-sample delay."""
    if len(xs) < 3:
        return list(xs)
    out = [xs[0]]
    for i in range(1, len(xs) - 1):
        out.append(sorted(xs[i - 1:i + 2])[1])
    out.append(xs[-1])
    return out


def smooth_path(obs: list[Observation], clip_id: int,
                dead_zone: float = DEAD_ZONE, max_vel: float = MAX_VEL,
                cut_dist: float = CUT_DIST, gain: float = FOLLOW_GAIN,
                ema_tau: float = EMA_TAU) -> CropPath:
    """
    Three-layer smoother:
      median3  -> kills single-frame glitches
      EMA      -> absorbs detector noise
      follower -> idle/tracking state machine; while tracking, a
                  velocity-limited P-controller chases the EMA. Constant
                  drift is followed at constant velocity (no stair-steps);
                  the camera sleeps again once settled.
    Hard cuts fire on sustained jumps > cut_dist, checked BEFORE the EMA
    so a real scene change isn't smeared into a pan.
    """
    if not obs or all(o.face is None for o in obs):
        return CropPath(clip_id=clip_id, mode="center",
                        keyframes=[CropKeyframe(t=0.0, cx=0.5)])

    # fill Nones by holding last (or first) known target
    raw: list[float] = []
    last = next(o.face.cx for o in obs if o.face is not None)
    cys = []
    for o in obs:
        if o.face is not None:
            last = o.face.cx
            cys.append(o.face.cy)
        raw.append(last)
    targets = _median3(raw)

    times = [o.t for o in obs]
    pos = targets[0]
    ema = targets[0]
    tracking = False
    pending_cut = 0
    kfs = [CropKeyframe(t=times[0], cx=round(pos, 4))]

    for i in range(1, len(targets)):
        dt = max(times[i] - times[i - 1], 1e-6)
        target = targets[i]

        # hard-cut check on the (median-filtered) raw target
        if abs(target - pos) > cut_dist:
            pending_cut += 1
            if pending_cut >= 2:            # sustained -> jump
                pos = ema = target
                tracking = False
                kfs.append(CropKeyframe(t=times[i], cx=round(pos, 4), cut=True))
                pending_cut = 0
                continue
            kfs.append(CropKeyframe(t=times[i], cx=round(pos, 4)))
            continue
        pending_cut = 0

        # low-pass the target
        alpha = 1.0 - float(np.exp(-dt / ema_tau))
        ema += alpha * (target - ema)
        err = ema - pos

        if not tracking and abs(err) > dead_zone:
            tracking = True
        if tracking:
            step = float(np.clip(gain * err * dt, -max_vel * dt, max_vel * dt))
            pos = float(np.clip(pos + step, 0.0, 1.0))
            if abs(ema - pos) < SETTLE_EPS:
                tracking = False
        kfs.append(CropKeyframe(t=times[i], cx=round(pos, 4)))

    # vertical center: median face cy (recorded for future partial-height
    # crops; the full-height renderer in Module 6 doesn't use it yet)
    med_cy = float(np.median(cys)) if cys else 0.5
    cy = float(np.clip(med_cy, 0.0, 1.0))

    # static optimization: if the camera never actually moved, collapse it
    if all(abs(k.cx - kfs[0].cx) < 1e-6 for k in kfs):
        return CropPath(clip_id=clip_id, mode="static", cy=cy,
                        keyframes=[CropKeyframe(t=times[0], cx=kfs[0].cx)])
    return CropPath(clip_id=clip_id, mode="tracked", cy=cy, keyframes=kfs)


# ----------------------------------------------------------- full analysis

def compute_crop_path(video_path, clip_id: int, start: float, end: float,
                      detector: FaceDetector,
                      sample_fps: float = SAMPLE_FPS) -> CropPath:
    """Sample proxy frames in [start,end], detect, select, smooth."""
    samples: list[tuple[float, list[FaceBox]]] = []
    for t_abs, frame in sample_frames(video_path, start, end, sample_fps):
        if isinstance(detector, MockDetector):
            detector.set_time(t_abs)
        faces = detector.detect(frame)
        samples.append((t_abs - start, faces))
    obs = select_subject(samples)
    return smooth_path(obs, clip_id=clip_id)


# ------------------------------------------------------------ quality math

def path_jitter(path: CropPath, sample_fps: float = SAMPLE_FPS) -> float:
    """Mean absolute high-frequency movement between consecutive keyframes,
    excluding hard cuts. This is the 'drunk camera' metric — lower = better."""
    kfs = path.keyframes
    if len(kfs) < 3:
        return 0.0
    diffs = []
    for a, b in zip(kfs, kfs[1:]):
        if b.cut:
            continue
        diffs.append(b.cx - a.cx)
    if len(diffs) < 2:
        return 0.0
    # jitter = mean |second difference| (acceleration-ish)
    dd = np.diff(np.array(diffs))
    return float(np.mean(np.abs(dd)))
