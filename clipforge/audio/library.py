"""
Audio library: a folder Diego fills with sound effects and music, plus a
hand-edited `library.json` that records each file's licence and source.

    {
      "schema_version": 1,
      "items": [
        {"file": "sfx/boom.wav", "kind": "sfx", "tags": ["impact"],
         "license": "CC0", "source": "https://kenney.nl/assets/impact-sounds"},
        {"file": "music/track.mp3", "kind": "music", "tags": ["gaming"],
         "license": "Pixabay Content License", "source": "https://...",
         "attribution": "", "platforms": ["all"], "gain_db": 0, "start": 0}
      ]
    }

An item with any problem is skipped with a reason and never used, so a typo
in one entry can't silence the whole library or slip in an unlicensed file.
Problems name the file and field (``library.json: items[2].license: ...``)
and never contain absolute paths.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

LIBRARY_FILE = "library.json"
SCHEMA_VERSION = 1
KINDS = ("sfx", "music")
PLATFORMS = ("all", "tiktok", "reels", "shorts", "x", "facebook")
MAIN_PLATFORMS = ("tiktok", "reels", "shorts")
AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".opus"}
TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
MAX_TEXT = 500
GAIN_RANGE = (-30.0, 12.0)
MAX_ITEMS = 5000


def default_library_dir() -> Path:
    raw = os.environ.get("CLIPFORGE_AUDIO_LIBRARY", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / "clipforge-audio"


@dataclass
class AudioItem:
    file: str                    # posix path relative to the library folder
    kind: str
    tags: list[str]
    license: str
    source: str
    attribution: str = ""
    platforms: list[str] = field(default_factory=lambda: ["all"])
    gain_db: float = 0.0
    start: float = 0.0           # music: seconds into the track to start from
    path: Path | None = None     # resolved file on disk

    def cleared_for(self, platforms) -> bool:
        return "all" in self.platforms or all(p in self.platforms
                                              for p in platforms)


@dataclass
class Library:
    root: Path
    items: list[AudioItem] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    found: bool = False          # library.json exists

    def of_kind(self, kind: str) -> list[AudioItem]:
        return [i for i in self.items if i.kind == kind]


def _reject_constant(name):
    raise ValueError(f"{name} is not allowed")


def _num(v) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v))


def _text(v, where: str, required: bool) -> str:
    if v is None and not required:
        return ""
    if not isinstance(v, str):
        raise ValueError(f"{where}: must be text")
    s = " ".join(v.split())
    if required and not s:
        raise ValueError(f"{where}: required")
    if len(s) > MAX_TEXT:
        raise ValueError(f"{where}: longer than {MAX_TEXT} characters")
    return s


def clean_relpath(raw, where: str) -> str:
    """A library-relative path in posix form. Windows separators are fine;
    absolute paths, drive letters and '..' are refused."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{where}: required")
    s = raw.strip().replace("\\", "/")
    if s.startswith("/") or re.match(r"^[A-Za-z]:", s):
        raise ValueError(f"{where}: must be relative to the library folder")
    parts = [p for p in PurePosixPath(s).parts if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        raise ValueError(f"{where}: must stay inside the library folder")
    return "/".join(parts)


def parse_item(raw, idx: int, root: Path,
               check_file: bool = True) -> AudioItem:
    where = f"{LIBRARY_FILE}: items[{idx}]"
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: must be an object")
    known = {"file", "kind", "tags", "license", "source", "attribution",
             "platforms", "gain_db", "start"}
    extra = sorted(k for k in raw if k not in known)
    if extra:
        raise ValueError(f"{where}.{extra[0]}: unknown field")

    rel = clean_relpath(raw.get("file"), f"{where}.file")
    kind = raw.get("kind")
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError(f"{where}.kind: must be one of {', '.join(KINDS)}")

    tags = raw.get("tags")
    if not isinstance(tags, list) or not tags:
        raise ValueError(f"{where}.tags: must be a non-empty list")
    clean_tags = []
    for j, t in enumerate(tags):
        if not isinstance(t, str) or not TAG_RE.match(t.strip().lower()):
            raise ValueError(f"{where}.tags[{j}]: use 1-32 letters, digits,"
                             " '-' or '_'")
        if t.strip().lower() not in clean_tags:
            clean_tags.append(t.strip().lower())

    lic = _text(raw.get("license"), f"{where}.license", required=True)
    src = _text(raw.get("source"), f"{where}.source", required=True)
    attr = _text(raw.get("attribution"), f"{where}.attribution",
                 required=False)

    plats = raw.get("platforms", ["all"])
    if not isinstance(plats, list) or not plats:
        raise ValueError(f"{where}.platforms: must be a non-empty list")
    clean_plats = []
    for j, p in enumerate(plats):
        if not isinstance(p, str) or p.strip().lower() not in PLATFORMS:
            raise ValueError(f"{where}.platforms[{j}]: must be one of "
                             f"{', '.join(PLATFORMS)}")
        clean_plats.append(p.strip().lower())

    gain = raw.get("gain_db", 0)
    if not _num(gain) or not GAIN_RANGE[0] <= gain <= GAIN_RANGE[1]:
        raise ValueError(f"{where}.gain_db: must be a number from "
                         f"{GAIN_RANGE[0]:g} to {GAIN_RANGE[1]:g}")
    start = raw.get("start", 0)
    if not _num(start) or start < 0:
        raise ValueError(f"{where}.start: must be a number of seconds >= 0")

    path = None
    if check_file:
        root_r = root.resolve()
        path = (root_r / rel).resolve()
        if not path.is_relative_to(root_r):
            raise ValueError(f"{where}.file: must stay inside the library "
                             "folder")
        if path.suffix.lower() not in AUDIO_EXTS:
            raise ValueError(f"{where}.file: not an audio file type "
                             f"({', '.join(sorted(AUDIO_EXTS))})")
        if not path.is_file():
            raise ValueError(f"{where}.file: {rel} not found")
    return AudioItem(file=rel, kind=kind, tags=clean_tags, license=lic,
                     source=src, attribution=attr, platforms=clean_plats,
                     gain_db=float(gain), start=float(start), path=path)


def _read_raw(root: Path):
    """(data, problem): the parsed library.json, or a problem sentence."""
    p = root / LIBRARY_FILE
    try:
        # utf-8-sig: PowerShell 5 writes a BOM
        text = p.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None, None
    except (OSError, UnicodeDecodeError):
        return None, f"{LIBRARY_FILE}: could not be read as UTF-8 text"
    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        return None, f"{LIBRARY_FILE}: not valid JSON (NaN/Infinity are not allowed)"
    if not isinstance(data, dict):
        return None, f"{LIBRARY_FILE}: must be an object with an items list"
    sv = data.get("schema_version", SCHEMA_VERSION)
    if not _num(sv) or sv != SCHEMA_VERSION or isinstance(sv, float):
        return None, f"{LIBRARY_FILE}: schema_version must be {SCHEMA_VERSION}"
    items = data.get("items")
    if not isinstance(items, list):
        return None, f"{LIBRARY_FILE}: items must be a list"
    if len(items) > MAX_ITEMS:
        return None, f"{LIBRARY_FILE}: more than {MAX_ITEMS} items"
    return data, None


def load_library(root: Path | None = None) -> Library:
    """Load and validate the library. Never raises for bad content."""
    root = Path(root) if root is not None else default_library_dir()
    lib = Library(root=root)
    data, problem = _read_raw(root)
    if problem:
        lib.found = True
        lib.problems.append(problem)
        return lib
    if data is None:
        return lib
    lib.found = True
    seen = set()
    for idx, raw in enumerate(data["items"]):
        try:
            item = parse_item(raw, idx, root)
        except ValueError as e:
            lib.problems.append(str(e))
            continue
        except OSError:
            lib.problems.append(f"{LIBRARY_FILE}: items[{idx}].file: "
                                "could not be checked")
            continue
        if item.file in seen:
            lib.problems.append(f"{LIBRARY_FILE}: items[{idx}].file: "
                                f"{item.file} is listed twice")
            continue
        seen.add(item.file)
        lib.items.append(item)
    return lib


def _safe_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._")
    return (stem or "sound")[:60] + Path(name).suffix.lower()


def add_file(src: Path, kind: str, tags: list[str], license: str,
             source: str, attribution: str = "",
             platforms: list[str] | None = None,
             root: Path | None = None) -> AudioItem:
    """Copy `src` into <library>/<kind>/ and record it in library.json.
    Raises ValueError with a readable sentence on any problem."""
    root = Path(root) if root is not None else default_library_dir()
    src = Path(src)
    if not src.is_file():
        raise ValueError(f"{src.name}: file not found")
    if src.suffix.lower() not in AUDIO_EXTS:
        raise ValueError(f"{src.name}: not an audio file type "
                         f"({', '.join(sorted(AUDIO_EXTS))})")
    data, problem = _read_raw(root)
    if problem:
        raise ValueError(f"{problem}; fix it before adding files")
    if data is None:
        data = {"schema_version": SCHEMA_VERSION, "items": []}

    name = _safe_name(src.name)
    rel = f"{kind}/{name}"
    taken = {str(i.get("file", "")).replace("\\", "/")
             for i in data["items"] if isinstance(i, dict)}
    n = 2
    while rel in taken or (root / rel).exists():
        rel = f"{kind}/{Path(name).stem}_{n}{Path(name).suffix}"
        n += 1
    raw = {"file": rel, "kind": kind, "tags": tags, "license": license,
           "source": source}
    if attribution:
        raw["attribution"] = attribution
    if platforms:
        raw["platforms"] = platforms
    # validate everything except the file (not copied yet)
    item = parse_item(raw, len(data["items"]), root, check_file=False)

    dest = root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    try:
        data["items"].append(raw)
        lib_path = root / LIBRARY_FILE
        tmp = lib_path.with_name(f"{LIBRARY_FILE}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8", newline="\n")
        os.replace(tmp, lib_path)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    item.path = dest
    return item
