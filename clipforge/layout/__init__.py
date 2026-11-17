"""
Stacked layout (F56, the first slice of F52).

A creator's layout file, creators/<slug>/layout.json in the workspace, holds
two boxes measured by hand on one frame of their stream: the facecam and the
gameplay. The renderer stacks the facecam (top, about 35% of the height) over
the gameplay at 1080x1920, so a gaming clip keeps both instead of a 9:16
window that follows the face and drops the game.

    {
      "name": "stacked",
      "frame": {"w": 1920, "h": 1080},
      "facecam": {"x": 1460, "y": 40, "w": 420, "h": 300},
      "game": {"x": 0, "y": 0, "w": 1920, "h": 1080},
      "cam_share": 0.35
    }

`frame` is the size of the frame the boxes were measured on, so a layout
measured on a 1440p VOD still fits the 1080p normalized video. Boxes are in
those source pixels. Each box is scaled to cover its band (no stretching) and
centre-cropped to it. No auto-detection: Diego measures the boxes once per
streamer with `clipforge layout preview`.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

LAYOUT_FILE = "layout.json"
CREATORS_DIR = "creators"
LAYOUT_NAMES = ("stacked",)
OUT_W, OUT_H = 1080, 1920
CAM_SHARE_MIN, CAM_SHARE_MAX = 0.2, 0.5
DEFAULT_CAM_SHARE = 0.35
# a layout measured on a different aspect ratio than the video is refused
ASPECT_TOLERANCE = 0.02

_SLUG_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_KEYS = {"name", "frame", "facecam", "game", "cam_share"}


class LayoutError(ValueError):
    """A layout file is missing fields, has bad values or cannot be read."""


def is_safe_slug(slug) -> bool:
    return isinstance(slug, str) and _SLUG_RE.fullmatch(slug) is not None


def creator_from_env() -> str | None:
    """CLIPFORGE_CREATOR, or None when empty, whitespace or not a valid slug."""
    raw = (os.environ.get("CLIPFORGE_CREATOR") or "").strip().lower()
    return raw if is_safe_slug(raw) else None


def layout_path(workspace: Path, slug: str) -> Path:
    if not is_safe_slug(slug):
        raise LayoutError("creator name must be lowercase letters, digits, "
                          "'-' or '_' (at most 64 characters)")
    return Path(workspace) / CREATORS_DIR / slug / LAYOUT_FILE


def _even(n: float) -> int:
    n = int(round(n))
    return n - (n % 2)


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    def scaled(self, sx: float, sy: float) -> "Box":
        return Box(_even(self.x * sx), _even(self.y * sy),
                   max(2, _even(self.w * sx)), max(2, _even(self.h * sy)))


@dataclass(frozen=True)
class Layout:
    name: str
    frame_w: int
    frame_h: int
    facecam: Box
    game: Box
    cam_share: float = DEFAULT_CAM_SHARE

    @property
    def cam_h(self) -> int:
        return _even(OUT_H * self.cam_share)

    @property
    def game_h(self) -> int:
        return OUT_H - self.cam_h

    @property
    def gameplay_share(self) -> float:
        """Share of the output frame that shows gameplay (report metric)."""
        return round(self.game_h / OUT_H, 3)

    def fits(self, src_w: int, src_h: int) -> bool:
        """True when the video has the aspect ratio the boxes were measured on."""
        if src_w <= 0 or src_h <= 0:
            return False
        a, b = self.frame_w / self.frame_h, src_w / src_h
        return abs(a - b) / b <= ASPECT_TOLERANCE

    def boxes_for(self, src_w: int, src_h: int) -> tuple[Box, Box]:
        """Facecam and game boxes in the video's own pixels, kept inside it."""
        sx, sy = src_w / self.frame_w, src_h / self.frame_h
        out = []
        for b in (self.facecam, self.game):
            s = b.scaled(sx, sy)
            w, h = min(s.w, _even(src_w)), min(s.h, _even(src_h))
            x = max(0, min(s.x, _even(src_w - w)))
            y = max(0, min(s.y, _even(src_h - h)))
            out.append(Box(x, y, w, h))
        return out[0], out[1]

    def filter(self, src_w: int, src_h: int) -> str:
        """A -vf graph (one input, one output) that builds the 1080x1920 stack.

        Each band: crop the box, scale it to cover the band keeping its aspect
        ratio, centre-crop to the band. Labels stay inside the graph, so this
        works as a plain -vf value next to the existing filters."""
        cam, game = self.boxes_for(src_w, src_h)

        def band(b: Box, h: int) -> str:
            return (f"crop={b.w}:{b.h}:{b.x}:{b.y},"
                    f"scale={OUT_W}:{h}:force_original_aspect_ratio=increase,"
                    f"crop={OUT_W}:{h},setsar=1")
        return (f"split=2[lc][lg];[lc]{band(cam, self.cam_h)}[top];"
                f"[lg]{band(game, self.game_h)}[bot];[top][bot]vstack=inputs=2")

    def caption_margin_v(self, font_size: int = 72) -> int:
        """ASS bottom margin that puts a caption card across the seam."""
        return OUT_H - self.cam_h - font_size // 2 - 20


def _fail(path: Path, field: str, msg: str):
    raise LayoutError(f"{path.name}: field '{field}' {msg}")


def _int(path: Path, data: dict, key: str, label: str, minimum: int) -> int:
    if key not in data:
        _fail(path, label, "is missing")
    v = data[key]
    if isinstance(v, bool) or not isinstance(v, int):
        _fail(path, label, "must be a whole number of pixels")
    if v < minimum:
        _fail(path, label, f"must be {minimum} or more")
    return v


def _box(path: Path, data: dict, key: str, fw: int, fh: int) -> Box:
    if key not in data:
        _fail(path, key, "is missing")
    raw = data[key]
    if not isinstance(raw, dict):
        _fail(path, key, "must be an object with x, y, w and h")
    for k in raw:
        if k not in ("x", "y", "w", "h"):
            _fail(path, f"{key}.{k}", "is not a box field (use x, y, w, h)")
    b = Box(_int(path, raw, "x", f"{key}.x", 0),
            _int(path, raw, "y", f"{key}.y", 0),
            _int(path, raw, "w", f"{key}.w", 16),
            _int(path, raw, "h", f"{key}.h", 16))
    if b.x + b.w > fw:
        _fail(path, f"{key}.w", f"runs past the frame width {fw} (x + w)")
    if b.y + b.h > fh:
        _fail(path, f"{key}.h", f"runs past the frame height {fh} (y + h)")
    return b


def load_layout_file(path: Path) -> Layout:
    """Load and validate a layout file. Every error names the file and field."""
    path = Path(path)
    try:
        # utf-8-sig: PowerShell 5 adds a BOM to files Diego edits by hand
        data = json.loads(path.read_text(encoding="utf-8-sig"),
                          parse_constant=lambda c: _fail(
                              path, "(file)", f"contains {c}, which is not a number"))
    except LayoutError:
        raise
    except (OSError, ValueError, RecursionError) as exc:
        raise LayoutError(
            f"{path.name}: cannot read layout ({exc.__class__.__name__})") from None
    if not isinstance(data, dict):
        raise LayoutError(f"{path.name}: top level must be an object")
    for k in data:
        if k not in _KEYS:
            _fail(path, k, "is not a layout field")
    name = data.get("name")
    if name not in LAYOUT_NAMES:
        _fail(path, "name", f"must be one of: {', '.join(LAYOUT_NAMES)}")
    if not isinstance(data.get("frame"), dict):
        _fail(path, "frame", "must be an object with w and h "
              "(the size of the frame the boxes were measured on)")
    for k in data["frame"]:
        if k not in ("w", "h"):
            _fail(path, f"frame.{k}", "is not a frame field (use w, h)")
    fw = _int(path, data["frame"], "w", "frame.w", 16)
    fh = _int(path, data["frame"], "h", "frame.h", 16)
    share = data.get("cam_share", DEFAULT_CAM_SHARE)
    if isinstance(share, bool) or not isinstance(share, (int, float)) \
            or not math.isfinite(share):
        _fail(path, "cam_share", "must be a number")
    if not CAM_SHARE_MIN <= share <= CAM_SHARE_MAX:
        _fail(path, "cam_share",
              f"must be between {CAM_SHARE_MIN} and {CAM_SHARE_MAX}")
    return Layout(name=name, frame_w=fw, frame_h=fh,
                  facecam=_box(path, data, "facecam", fw, fh),
                  game=_box(path, data, "game", fw, fh),
                  cam_share=float(share))


def template(frame_w: int = 1920, frame_h: int = 1080) -> dict:
    """A starting layout: facecam top-right corner, whole frame as game."""
    cw, ch = _even(frame_w * 0.22), _even(frame_h * 0.28)
    return {"name": "stacked", "frame": {"w": frame_w, "h": frame_h},
            "facecam": {"x": frame_w - cw - 20, "y": 20, "w": cw, "h": ch},
            "game": {"x": 0, "y": 0, "w": frame_w, "h": frame_h},
            "cam_share": DEFAULT_CAM_SHARE}


@dataclass
class LayoutChoice:
    """What the renderer uses for a job, and why (for the clip report)."""
    layout: Layout | None
    reason: str
    creator: str | None = None

    def report(self) -> dict:
        if self.layout is None:
            return {"applied": False, "name": "face_crop",
                    "reason": self.reason, "creator": self.creator}
        return {"applied": True, "name": self.layout.name,
                "reason": self.reason, "creator": self.creator,
                "cam_share": self.layout.cam_share,
                "gameplay_share": self.layout.gameplay_share}


def choose_layout(recipe_layout: str | None, workspace: Path,
                  creator: str | None, src_w: int, src_h: int) -> LayoutChoice:
    """Pick the stacked layout when the recipe asks for it and a usable
    layout file exists; otherwise the face-tracked crop, with the reason."""
    if recipe_layout != "stacked":
        return LayoutChoice(None, "the recipe uses the face-tracked crop",
                            creator)
    if not creator:
        return LayoutChoice(None, "no creator set (CLIPFORGE_CREATOR), so no "
                            "layout file; face-tracked crop used", None)
    try:
        path = layout_path(workspace, creator)
    except LayoutError as exc:
        return LayoutChoice(None, str(exc), None)
    if not path.is_file():
        return LayoutChoice(None, f"no layout file for creator '{creator}'; "
                            "face-tracked crop used", creator)
    try:
        lay = load_layout_file(path)
    except LayoutError as exc:
        return LayoutChoice(None, f"layout file invalid ({exc}); face-tracked "
                            "crop used", creator)
    if not lay.fits(src_w, src_h):
        return LayoutChoice(None, f"layout measured on {lay.frame_w}x"
                            f"{lay.frame_h} does not match the video's "
                            "aspect ratio; face-tracked crop used", creator)
    return LayoutChoice(lay, "stacked: facecam over gameplay", creator)
