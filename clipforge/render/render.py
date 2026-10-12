"""
Module 6b — Renderer.

For each refined clip:
  1. Trim normalized.mp4 to [start, end] (frame-accurate re-encode).
  2. Apply the 9:16 crop. Static/center paths use a fixed crop; tracked
     paths are upsampled to per-frame positions and driven through
     ffmpeg's sendcmd filter (frame-accurate camera motion, no giant
     filter expressions).
  3. Scale to 1080x1920. Sources narrower than 9:16 get letterboxed.
  4. Burn ASS captions (libass).
  5. Normalize loudness to -14 LUFS (platform standard), encode H.264+AAC.

Then verify_render() re-probes the output: correct resolution, audio
present, duration within tolerance — same trust-nothing policy as
every other module.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from clipforge.ingest.ingest import parse_probe, probe
from clipforge.reframe.detect import CropPath

OUT_W, OUT_H = 1080, 1920
TARGET_LUFS = -14.0


class RenderError(Exception):
    pass


def _run(cmd: list[str], timeout: int = 1800) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RenderError(f"ffmpeg failed: {p.stderr.strip()[-800:]}")


def _even(n: int) -> int:
    return int(n) - (int(n) % 2)


@dataclass
class CropGeometry:
    mode: str            # "crop" or "letterbox"
    crop_w: int = 0      # source-pixel crop window (full height)
    crop_h: int = 0

    @classmethod
    def for_source(cls, src_w: int, src_h: int) -> "CropGeometry":
        target = 9 / 16
        if src_w / src_h > target + 1e-6:
            cw = _even(round(src_h * target))
            return cls(mode="crop", crop_w=cw, crop_h=_even(src_h))
        return cls(mode="letterbox")


def crop_x_px(cx: float, src_w: int, crop_w: int) -> int:
    """Normalized crop-center -> clamped integer left-edge in source px."""
    x = cx * src_w - crop_w / 2
    return int(max(0, min(round(x), src_w - crop_w)))


def write_sendcmd(path: Path, crop: CropPath, geom: CropGeometry,
                  src_w: int, fps: float, duration: float) -> None:
    """Per-frame crop-x commands for ffmpeg's sendcmd filter."""
    lines = []
    n = int(duration * fps) + 1
    for i in range(n):
        t = i / fps
        x = crop_x_px(crop.cx_at(t), src_w, geom.crop_w)
        lines.append(f"{t:.4f} crop@dyn x {x};")
    path.write_text("\n".join(lines) + "\n")


def build_filtergraph(geom: CropGeometry, crop: CropPath, src_w: int,
                      ass_path: Path | None, sendcmd_path: Path | None) -> str:
    parts = []
    if geom.mode == "crop":
        if crop.mode in ("static", "center") or sendcmd_path is None:
            cx = crop.keyframes[0].cx if crop.keyframes else 0.5
            x0 = crop_x_px(cx, src_w, geom.crop_w)
            parts.append(f"crop=w={geom.crop_w}:h={geom.crop_h}:x={x0}:y=0")
        else:
            x0 = crop_x_px(crop.keyframes[0].cx, src_w, geom.crop_w)
            parts.append(f"sendcmd=f='{sendcmd_path.as_posix()}'")
            parts.append(f"crop@dyn=w={geom.crop_w}:h={geom.crop_h}:x={x0}:y=0")
        parts.append(f"scale={OUT_W}:{OUT_H}")
    else:  # letterbox narrow sources
        parts.append(f"scale={OUT_W}:-2")
        parts.append(f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2:color=black")
    if ass_path is not None:
        parts.append(f"ass='{ass_path.as_posix()}'")
    return ",".join(parts)


def render_clip(normalized_video: Path, out_path: Path, start: float,
                end: float, crop: CropPath, src_w: int, src_h: int,
                fps: float, ass_path: Path | None = None,
                workdir: Path | None = None) -> None:
    duration = end - start
    geom = CropGeometry.for_source(src_w, src_h)

    sendcmd_path = None
    if geom.mode == "crop" and crop.mode == "tracked":
        sendcmd_path = (workdir or out_path.parent) / (out_path.stem + ".cmds")
        write_sendcmd(sendcmd_path, crop, geom, src_w, fps, duration)

    vf = build_filtergraph(geom, crop, src_w, ass_path, sendcmd_path)
    _run([
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", str(normalized_video),
        "-vf", vf,
        "-af", f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart",
        str(out_path),
    ])
    if sendcmd_path and sendcmd_path.exists():
        sendcmd_path.unlink()


def verify_render(out_path: Path, expected_duration: float) -> None:
    pr = parse_probe(probe(out_path))
    if (pr.width, pr.height) != (OUT_W, OUT_H):
        raise RenderError(
            f"{out_path.name}: wrong resolution {pr.width}x{pr.height}")
    if pr.audio_codec is None:
        raise RenderError(f"{out_path.name}: no audio track")
    if abs(pr.duration - expected_duration) > max(0.5, expected_duration * 0.03):
        raise RenderError(
            f"{out_path.name}: duration {pr.duration:.2f}s != "
            f"expected {expected_duration:.2f}s")


def measure_loudness(path: Path) -> float:
    """Integrated LUFS via ffmpeg loudnorm analysis pass (for QA/tests)."""
    p = subprocess.run([
        "ffmpeg", "-i", str(path), "-af",
        "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
        "-f", "null", "-",
    ], capture_output=True, text=True, timeout=600)
    tail = p.stderr[p.stderr.rfind("{"):p.stderr.rfind("}") + 1]
    return float(json.loads(tail)["input_i"])


def make_contact_sheet(clip_path: Path, out_png: Path,
                       cols: int = 6, rows: int = 4) -> None:
    """1 frame/sec QA grid — review framing of a whole clip at a glance."""
    _run([
        "ffmpeg", "-y", "-i", str(clip_path),
        "-vf", f"fps=1,scale=180:-1,tile={cols}x{rows}",
        "-frames:v", "1", str(out_png),
    ])
