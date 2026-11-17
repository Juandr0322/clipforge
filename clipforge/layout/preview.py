"""
`clipforge layout preview`: one frame of a creator's video, shown twice so
Diego can measure the boxes before a launch night.

  <out>_boxes.png   the source frame with the facecam box (green) and the
                    game box (red) drawn on it, to check the measurements
  <out>.png         the stacked 1080x1920 result for that frame
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from clipforge.layout import Layout


class PreviewError(Exception):
    pass


def _ffmpeg(args: list[str]) -> None:
    p = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=300)
    if p.returncode != 0:
        raise PreviewError("ffmpeg could not read a frame at that time; "
                           "try an earlier --at")


def video_size(video: Path) -> tuple[int, int]:
    from clipforge.ingest.ingest import parse_probe, probe
    pr = parse_probe(probe(Path(video)))
    return pr.width, pr.height


def write_preview(video: Path, layout: Layout, out_png: Path,
                  at: float = 0.0) -> tuple[Path, Path]:
    """Write the stacked frame and the boxes frame. Returns both paths."""
    video, out_png = Path(video), Path(out_png)
    w, h = video_size(video)
    if not layout.fits(w, h):
        raise PreviewError(
            f"the layout was measured on {layout.frame_w}x{layout.frame_h}, "
            f"but this video is {w}x{h} (different aspect ratio)")
    cam, game = layout.boxes_for(w, h)
    boxes_png = out_png.with_name(out_png.stem + "_boxes.png")
    seek = ["-ss", f"{max(0.0, at):.3f}", "-i", str(video), "-frames:v", "1"]
    _ffmpeg([*seek, "-vf", layout.filter(w, h), str(out_png)])
    draw = (f"drawbox=x={game.x}:y={game.y}:w={game.w}:h={game.h}"
            ":color=red@0.9:t=6,"
            f"drawbox=x={cam.x}:y={cam.y}:w={cam.w}:h={cam.h}"
            ":color=green@0.9:t=6")
    _ffmpeg([*seek, "-vf", draw, str(boxes_png)])
    return out_png, boxes_png
