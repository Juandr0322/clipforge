"""Recipes: one hand-editable, versioned JSON file per content type.

A recipe only carries values that already exist as settings (pipeline thresholds,
captions on/off, audio-layer levels). Env vars still override recipe values.
The files in files/*.json must be declared as package data if the project is packaged.
"""
from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from clipforge.audio.layer import (MUSIC_MAX_DB, SFX_MAX_DB, AudioLayerConfig,
                                   _FALSE, _TRUE)
from clipforge.review.store import _replace_with_retry

RECIPES_DIR = Path(__file__).parent / "files"
RECORD_NAME = "recipe.json"
DEFAULT_RECIPE = "talk"
_FLAG_WORDS = _TRUE | _FALSE
# (key, env var, lower bound, upper bound) for the dB settings
_DB = (("sfx_level_db", "CLIPFORGE_SFX_LEVEL_DB", -40.0, SFX_MAX_DB),
       ("music_level_db", "CLIPFORGE_MUSIC_LEVEL_DB", -40.0, MUSIC_MAX_DB),
       ("music_duck_db", "CLIPFORGE_MUSIC_DUCK_DB", -30.0, 0.0))
_AUDIO_KEYS = {"sfx", "music", "music_tags"} | {k for k, *_ in _DB}


class RecipeError(ValueError):
    """A recipe file is invalid or a recipe name is unknown."""


@dataclass(frozen=True)
class Recipe:
    name: str
    version: int
    description: str
    min_score: int
    max_candidates: int
    captions: bool
    audio: dict | None = None   # validated overrides of the F48 audio settings

    def audio_config(self) -> AudioLayerConfig | None:
        """Audio settings: env var if set, else the recipe value, else the default.

        None when the recipe has no audio section (render then reads the env as before).
        """
        if self.audio is None:
            return None
        cfg = AudioLayerConfig.from_env()
        a = self.audio
        if "sfx" in a and not _flag_set("CLIPFORGE_SFX"):
            cfg.sfx = a["sfx"]
        if "music" in a and not _flag_set("CLIPFORGE_MUSIC"):
            cfg.music = a["music"]
        if "music_tags" in a and not any(
                t.strip() for t in os.environ.get("CLIPFORGE_MUSIC_TAGS", "").split(",")):
            cfg.music_tags = list(a["music_tags"])
        for key, env, _lo, _hi in _DB:
            if key in a and not _db_set(env):
                setattr(cfg, key, a[key])
        return cfg


def _flag_set(name: str) -> bool:
    """True when the env var holds a value AudioLayerConfig.from_env understands."""
    raw = os.environ.get(name)
    return raw is not None and raw.strip().lower() in _FLAG_WORDS


def _db_set(name: str) -> bool:
    try:
        return math.isfinite(float(os.environ.get(name, "")))
    except ValueError:
        return False


def recipe_problem() -> str | None:
    """Why CLIPFORGE_RECIPE cannot be used (unknown name or broken file), else None."""
    name = os.environ.get("CLIPFORGE_RECIPE") or DEFAULT_RECIPE
    try:
        load_recipe(name)
    except RecipeError as exc:
        return str(exc)
    return None


def _fail(path: Path, field: str, msg: str):
    raise RecipeError(f"{path.name}: field '{field}' {msg}")


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


def _bool(path: Path, data: dict, key: str, parent: str = "") -> bool:
    label = f"{parent}.{key}" if parent else key
    if key not in data:
        _fail(path, label, "is missing")
    if not isinstance(data[key], bool):
        _fail(path, label, "must be true or false")
    return data[key]


def _text(path: Path, data: dict, key: str) -> str:
    if key not in data:
        _fail(path, key, "is missing")
    if not isinstance(data[key], str) or not data[key].strip():
        _fail(path, key, "must be a non-empty string")
    return data[key]


def _audio(path: Path, data) -> dict:
    if not isinstance(data, dict):
        _fail(path, "audio", "must be an object")
    for k in data:
        if k not in _AUDIO_KEYS:
            _fail(path, f"audio.{k}", "is not a known audio setting")
    out: dict = {}
    for k in ("sfx", "music"):
        if k in data:
            out[k] = _bool(path, data, k, "audio")
    for key, _env, lo, hi in _DB:
        if key in data:
            v = _num(path, data, key, "audio")
            if v > hi:
                _fail(path, f"audio.{key}", f"is above the maximum of {hi:g} dB")
            if v < lo:
                _fail(path, f"audio.{key}", f"is below the minimum of {lo:g} dB")
            out[key] = float(v)
    if "music_tags" in data:
        tags = data["music_tags"]
        if (not isinstance(tags, list) or not tags
                or not all(isinstance(t, str) and t.strip() for t in tags)):
            _fail(path, "audio.music_tags", "must be a non-empty list of strings")
        out["music_tags"] = [t.strip().lower() for t in tags]
    return out


def load_recipe_file(path: Path) -> Recipe:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, RecursionError) as exc:
        raise RecipeError(
            f"{path.name}: cannot read recipe ({exc.__class__.__name__})") from None
    if not isinstance(data, dict):
        raise RecipeError(f"{path.name}: top level must be an object")
    name = _text(path, data, "name")
    if name != path.stem:
        _fail(path, "name", f"must match the file name '{path.stem}'")
    version = _num(path, data, "version", integer=True)
    if version < 1:
        _fail(path, "version", "must be 1 or more")
    min_score = _num(path, data, "min_score", integer=True)
    if not 0 <= min_score <= 100:
        _fail(path, "min_score", "must be between 0 and 100")
    max_cand = _num(path, data, "max_candidates", integer=True)
    if max_cand < 1:
        _fail(path, "max_candidates", "must be 1 or more")
    audio = _audio(path, data["audio"]) if "audio" in data else None
    return Recipe(name=name, version=version,
                  description=_text(path, data, "description"),
                  min_score=min_score, max_candidates=max_cand,
                  captions=_bool(path, data, "captions"), audio=audio)


def recipe_names(directory: Path | None = None) -> list[str]:
    d = Path(directory) if directory else RECIPES_DIR
    return sorted(p.stem for p in d.glob("*.json"))


def load_recipe(name: str, directory: Path | None = None) -> Recipe:
    d = Path(directory) if directory else RECIPES_DIR
    valid = recipe_names(d)
    if not isinstance(name, str) or name not in valid:
        raise RecipeError(f"Unknown recipe {name!r}. Valid recipes: {', '.join(valid)}")
    return load_recipe_file(d / f"{name}.json")


def load_all(directory: Path | None = None) -> dict[str, Recipe]:
    d = Path(directory) if directory else RECIPES_DIR
    return {n: load_recipe(n, d) for n in recipe_names(d)}


def read_record(job_dir: Path) -> dict | None:
    """The recipe a job ran with ({name, version}), or None if not recorded yet."""
    try:
        data = json.loads((Path(job_dir) / RECORD_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (isinstance(data, dict) and isinstance(data.get("name"), str)
            and isinstance(data.get("version"), int)
            and not isinstance(data["version"], bool)):
        return {"name": data["name"], "version": data["version"]}
    return None


def write_record(job_dir: Path, recipe: Recipe) -> None:
    path = Path(job_dir) / RECORD_NAME
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"name": recipe.name, "version": recipe.version}, f, indent=2)
            f.write("\n")
        _replace_with_retry(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
