"""Platform profiles: one hand-editable JSON file per platform.

Later steps read these files through this module and hard-code none of the values.
If the project is ever packaged, the profiles/*.json files must be declared as package data.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

PROFILES_DIR = Path(__file__).parent / "profiles"
TIERS = ("main", "side")
FRAME_W, FRAME_H = 1080, 1920


class ProfileError(ValueError):
    """A profile file is invalid or a platform name is unknown."""


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Profile:
    name: str
    tier: str
    max_length_s: float
    target_min_s: float
    target_max_s: float
    width: int
    height: int
    safe_zone: Rect
    checked: str
    source: str


def _fail(path: Path, field: str, msg: str):
    raise ProfileError(f"{path.name}: field '{field}' {msg}")


def _obj(path: Path, data: dict, key: str) -> dict:
    if key not in data:
        _fail(path, key, "is missing")
    if not isinstance(data[key], dict):
        _fail(path, key, "must be an object")
    return data[key]


def _num(path: Path, data: dict, key: str, parent: str = "", integer: bool = False):
    label = f"{parent}.{key}" if parent else key
    if key not in data:
        _fail(path, label, "is missing")
    v = data[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        _fail(path, label, "must be a number")
    if integer and not isinstance(v, int):
        _fail(path, label, "must be a whole number")
    if not math.isfinite(v):
        _fail(path, label, "must be a finite number")
    return v


def _text(path: Path, data: dict, key: str) -> str:
    if key not in data:
        _fail(path, key, "is missing")
    if not isinstance(data[key], str) or not data[key].strip():
        _fail(path, key, "must be a non-empty string")
    return data[key]


def load_profile_file(path: Path) -> Profile:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProfileError(f"{path.name}: cannot read profile ({exc.__class__.__name__})") from None
    if not isinstance(data, dict):
        raise ProfileError(f"{path.name}: top level must be an object")

    name = _text(path, data, "name")
    if name != path.stem:
        _fail(path, "name", f"must match the file name '{path.stem}'")
    tier = _text(path, data, "tier")
    if tier not in TIERS:
        _fail(path, "tier", f"must be one of {', '.join(TIERS)}")
    max_len = _num(path, data, "max_length_s")
    if max_len <= 0:
        _fail(path, "max_length_s", "must be positive")
    target = _obj(path, data, "target_length_s")
    tmin = _num(path, target, "min", "target_length_s")
    tmax = _num(path, target, "max", "target_length_s")
    if tmin <= 0 or tmin > tmax:
        _fail(path, "target_length_s", "needs 0 < min <= max")
    if tmax > max_len:
        _fail(path, "target_length_s.max", "is above max_length_s")
    size = _obj(path, data, "output_size")
    w = _num(path, size, "width", "output_size", integer=True)
    h = _num(path, size, "height", "output_size", integer=True)
    if (w, h) != (FRAME_W, FRAME_H):
        _fail(path, "output_size", f"must be {FRAME_W}x{FRAME_H}")
    sz = _obj(path, data, "safe_zone")
    x, y, sw, sh = (_num(path, sz, k, "safe_zone", integer=True) for k in ("x", "y", "width", "height"))
    if sw <= 0 or sh <= 0:
        _fail(path, "safe_zone", "must have positive width and height")
    if x < 0 or y < 0 or x + sw > w or y + sh > h:
        _fail(path, "safe_zone", f"is outside the {w}x{h} frame")
    checked = _text(path, data, "checked")
    if checked != "unverified" and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", checked):
        _fail(path, "checked", "must be a YYYY-MM-DD date or 'unverified'")
    return Profile(
        name=name, tier=tier, max_length_s=max_len, target_min_s=tmin, target_max_s=tmax,
        width=w, height=h, safe_zone=Rect(x, y, sw, sh),
        checked=checked, source=_text(path, data, "source"),
    )


def platform_names(directory: Path | None = None) -> list[str]:
    d = Path(directory) if directory else PROFILES_DIR
    return sorted(p.stem for p in d.glob("*.json"))


def load_profile(name: str, directory: Path | None = None) -> Profile:
    d = Path(directory) if directory else PROFILES_DIR
    valid = platform_names(d)
    if not isinstance(name, str) or name not in valid:
        raise ProfileError(f"Unknown platform {name!r}. Valid platforms: {', '.join(valid)}")
    return load_profile_file(d / f"{name}.json")


def load_all(directory: Path | None = None) -> dict[str, Profile]:
    d = Path(directory) if directory else PROFILES_DIR
    return {n: load_profile(n, d) for n in platform_names(d)}


def shared_safe_zone(directory: Path | None = None) -> Rect:
    """Intersection of the safe zones of all main-tier platforms."""
    zones = [p.safe_zone for p in load_all(directory).values() if p.tier == "main"]
    if not zones:
        raise ProfileError("No main-tier platform profiles found")
    left = max(z.x for z in zones)
    top = max(z.y for z in zones)
    right = min(z.x + z.width for z in zones)
    bottom = min(z.y + z.height for z in zones)
    if right <= left or bottom <= top:
        raise ProfileError("Main-tier safe zones do not overlap")
    return Rect(left, top, right - left, bottom - top)
