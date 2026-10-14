"""
Audio layer (F48): sound effects and a quiet music bed, built as ONE extra
audio track that the renderer mixes on top of the original clip audio.

The original game audio and voice stay the main sound. Every level here is
relative to the clip's own measured level (a gated RMS of the original
audio), so the layer sits the same distance under a loud stream as under a
quiet podcast; after mixing, the whole clip is normalised to -14 LUFS as
before. Music levels are capped below the original (MUSIC_MAX_DB).

Sound effects land on beats the pipeline already found:
  reaction_peak_t   impact (plus a whoosh peaking just before it when the
                    punch-in zoom is applied, and a riser into it when the
                    peak is far enough in)
  context overlay   pop as the text appears
  end card          whoosh into the card
Cue groups keep MIN_GROUP_GAP seconds apart and there is at most one group
per started SEC_PER_GROUP seconds, so edits don't turn into noise.

The layer is written as raw float32 little-endian stereo at SR, which ffmpeg
reads with `-f f32le -ar 48000 -ac 2`. Times are clip-relative seconds.
"""
from __future__ import annotations

import hashlib
import math
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from clipforge.audio.library import (
    MAIN_PLATFORMS, AudioItem, Library, default_library_dir,
)
from clipforge.audio.starter import SR, STARTER_TAGS, starter_sound

SFX_LEVEL_DEFAULT = -6.0       # loudest 100 ms of an effect vs the clip level
MUSIC_LEVEL_DEFAULT = -20.0    # music bed vs the clip level
MUSIC_DUCK_DEFAULT = -6.0      # extra drop while someone talks
MUSIC_MAX_DB = -6.0            # music can never be set louder than this
SFX_MAX_DB = 0.0
LEVEL_FLOOR_DB = -60.0         # original quieter than this = silent clip

TAG_GAIN_DB = {"impact": 0.0, "whoosh": -2.0, "pop": -4.0, "riser": -6.0}
MIN_GROUP_GAP = 2.0
SEC_PER_GROUP = 8.0
RISER_MIN_PEAK = 3.0           # a riser needs room to build
SFX_MAX_SEC = 5.0              # longer library effects are cut (with a fade)

DUCK_ATTACK = 0.12
DUCK_RELEASE = 0.30
SPEECH_MERGE_GAP = 0.35
SPEECH_PAD_BEFORE = 0.05
SPEECH_PAD_AFTER = 0.15
MUSIC_FADE_IN = 0.2
MUSIC_FADE_OUT = 1.5
LOOP_XFADE = 0.5
MAX_MUSIC_DECODE = 900.0

TRUE_WORDS = {"1", "true", "yes", "on"}
FALSE_WORDS = {"0", "false", "no", "off", ""}


class LayerError(Exception):
    pass


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    return True if raw in TRUE_WORDS else False if raw in FALSE_WORDS else default


def _db_setting(name: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(os.environ.get(name, ""))
    except ValueError:
        return default
    if not math.isfinite(v):
        return default
    return min(max(v, lo), hi)


@dataclass
class AudioLayerConfig:
    sfx: bool = True
    music: bool = True
    library_dir: Path | None = None        # None = default_library_dir()
    sfx_level_db: float = SFX_LEVEL_DEFAULT
    music_level_db: float = MUSIC_LEVEL_DEFAULT
    music_duck_db: float = MUSIC_DUCK_DEFAULT
    music_tags: list[str] = field(default_factory=lambda: ["gaming"])

    @classmethod
    def from_env(cls) -> "AudioLayerConfig":
        tags = [t.strip().lower() for t in
                os.environ.get("CLIPFORGE_MUSIC_TAGS", "gaming").split(",")
                if t.strip()]
        return cls(
            sfx=_flag("CLIPFORGE_SFX", True),
            music=_flag("CLIPFORGE_MUSIC", True),
            library_dir=default_library_dir(),
            sfx_level_db=_db_setting("CLIPFORGE_SFX_LEVEL_DB",
                                     SFX_LEVEL_DEFAULT, -40.0, SFX_MAX_DB),
            music_level_db=_db_setting("CLIPFORGE_MUSIC_LEVEL_DB",
                                       MUSIC_LEVEL_DEFAULT, -40.0,
                                       MUSIC_MAX_DB),
            music_duck_db=_db_setting("CLIPFORGE_MUSIC_DUCK_DB",
                                      MUSIC_DUCK_DEFAULT, -30.0, 0.0),
            music_tags=tags or ["gaming"],
        )


# ---------------------------------------------------------------- cues

@dataclass
class Cue:
    t: float          # clip-relative time the sound's anchor lands on
    tag: str
    event: str
    align: str        # "onset" | "peak" | "end": which point of the sound
    offset: float = 0.0


@dataclass
class Beats:
    """Clip-relative moments the rest of the pipeline already decided."""
    duration: float                       # clip length without end card
    total: float                          # with end card
    reaction_peak_t: float | None = None
    zoom_applied: bool = False
    overlay_t: float | None = None        # context overlay appears
    end_card_t: float | None = None       # end card starts


def plan_cues(b: Beats) -> list[list[Cue]]:
    """Cue groups in priority order, already spaced and capped."""
    wanted: list[list[Cue]] = []
    if b.reaction_peak_t is not None:
        p = b.reaction_peak_t
        g = [Cue(p, "impact", "reaction_peak", "onset")]
        if b.zoom_applied:
            g.insert(0, Cue(p, "whoosh", "punch_in", "peak", -0.05))
        if p >= RISER_MIN_PEAK:
            g.insert(0, Cue(p, "riser", "build_up", "end", -0.08))
        wanted.append(g)
    if b.end_card_t is not None:
        wanted.append([Cue(b.end_card_t, "whoosh", "end_card", "peak")])
    if b.overlay_t is not None:
        wanted.append([Cue(b.overlay_t + 0.05, "pop", "context_overlay",
                           "onset")])

    cap = max(1, math.ceil(b.total / SEC_PER_GROUP)) if b.total > 0 else 0
    kept: list[list[Cue]] = []
    for g in wanted:
        anchor = g[-1].t
        if not 0.0 <= anchor <= b.total:
            continue
        if any(abs(anchor - k[-1].t) < MIN_GROUP_GAP for k in kept):
            continue
        if len(kept) >= cap:
            break
        kept.append(g)
    return sorted(kept, key=lambda g: g[-1].t)


# ---------------------------------------------------------------- audio io

def decode_audio(path: Path, start: float = 0.0,
                 duration: float | None = None) -> np.ndarray:
    """Decode any audio/video file to float32 stereo (n, 2) at SR."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(path), "-vn", "-f", "f32le", "-ac", "2",
            "-ar", str(SR), "-"]
    p = subprocess.run(cmd, capture_output=True, timeout=600)
    if p.returncode != 0:
        raise LayerError(f"could not decode {Path(path).name}")
    data = np.frombuffer(p.stdout, dtype="<f4")
    n = len(data) // 2
    return data[: n * 2].reshape(n, 2).astype(np.float32)


def write_layer(path: Path, layer: np.ndarray) -> None:
    np.ascontiguousarray(layer, dtype="<f4").tofile(path)


def gated_level_db(x: np.ndarray, block: float = 0.4) -> float | None:
    """Mean power of the 'active' 400 ms blocks in dBFS (an absolute gate at
    LEVEL_FLOOR_DB, then a relative gate 10 dB under the mean, like
    BS.1770 without the K-weighting). None when everything is below the
    floor."""
    if x.size == 0:
        return None
    mono_pow = np.mean(np.square(x.astype(np.float64)), axis=1)
    hop = max(1, int(block * SR))
    nb = len(mono_pow) // hop
    if nb == 0:
        blocks = np.array([mono_pow.mean()])
    else:
        blocks = mono_pow[: nb * hop].reshape(nb, hop).mean(axis=1)
    floor = 10 ** (LEVEL_FLOOR_DB / 10)
    active = blocks[blocks > floor]
    if active.size == 0:
        return None
    rel = active[active >= active.mean() * 10 ** (-10 / 10)]
    return float(10 * np.log10(rel.mean()))


def loudest_window_db(x: np.ndarray, win: float = 0.1) -> float | None:
    if x.size == 0:
        return None
    pw = np.mean(np.square(x.astype(np.float64)), axis=1)
    n = max(1, min(len(pw), int(win * SR)))
    c = np.concatenate([[0.0], np.cumsum(pw)])
    best = float(np.max((c[n:] - c[:-n]) / n))
    return None if best <= 0 else 10 * math.log10(best)


def anchor_index(x: np.ndarray, align: str) -> int:
    env = np.max(np.abs(x), axis=1)
    if align == "end":
        return len(x)
    if align == "peak":
        k = max(1, int(0.02 * SR))
        sm = np.convolve(env, np.ones(k) / k, mode="same")
        return int(np.argmax(sm))
    peak = float(env.max()) if env.size else 0.0
    hit = np.nonzero(env >= 0.1 * peak)[0] if peak > 0 else []
    return int(hit[0]) if len(hit) else 0


def _gain(db: float) -> float:
    return 10 ** (db / 20)


def place(layer: np.ndarray, snd: np.ndarray, t: float, align: str,
          offset: float = 0.0) -> float:
    """Add `snd` into `layer` so its anchor lands on t+offset. Returns the
    clip time where the sound starts (may be < 0 when the head is cut)."""
    start = int(round((t + offset) * SR)) - anchor_index(snd, align)
    s0, s1 = max(0, start), min(len(layer), start + len(snd))
    if s1 > s0:
        piece = snd[s0 - start: s1 - start].copy()
        n = min(len(piece) // 2, int(0.01 * SR))
        if start < 0 and n:              # cut head: short fade-in
            piece[:n] *= np.linspace(0, 1, n)[:, None]
        if start + len(snd) > len(layer) and n:      # cut tail: fade-out
            piece[-n:] *= np.linspace(1, 0, n)[:, None]
        layer[s0:s1] += piece
    return start / SR


# ---------------------------------------------------------------- music

def speech_spans(words, clip_start: float, clip_end: float
                 ) -> list[tuple[float, float]]:
    spans: list[list[float]] = []
    for w in words:
        if w.end <= clip_start or w.start >= clip_end:
            continue
        a = max(0.0, w.start - clip_start - SPEECH_PAD_BEFORE)
        b = min(clip_end - clip_start, w.end - clip_start + SPEECH_PAD_AFTER)
        if spans and a - spans[-1][1] <= SPEECH_MERGE_GAP:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    return [(a, b) for a, b in spans]


def duck_envelope(n: int, spans, duck_db: float) -> np.ndarray:
    """Per-sample gain: 1 outside speech, duck_db inside, linear ramps."""
    g = np.ones(n, dtype=np.float64)
    if duck_db >= 0 or not spans:
        return g
    low = _gain(duck_db)
    t = np.arange(n) / SR
    depth = np.zeros(n)
    for a, b in spans:
        up = np.clip((t - (a - DUCK_ATTACK)) / DUCK_ATTACK, 0, 1)
        down = np.clip(((b + DUCK_RELEASE) - t) / DUCK_RELEASE, 0, 1)
        depth = np.maximum(depth, np.minimum(up, down))
    return g - depth * (1.0 - low)


def loop_to(x: np.ndarray, n: int) -> np.ndarray:
    """Repeat a music section to n samples with short crossfades."""
    if len(x) >= n:
        return x[:n]
    xf = min(int(LOOP_XFADE * SR), len(x) // 4)
    if xf < 1 or len(x) < 2:
        reps = math.ceil(n / max(1, len(x)))
        return np.tile(x, (reps, 1))[:n]
    out = x.copy()
    ramp = np.linspace(0, 1, xf)[:, None]
    while len(out) < n:
        head = out[:-xf]
        mix = out[-xf:] * (1 - ramp) + x[:xf] * ramp
        out = np.concatenate([head, mix, x[xf:]])
    return out[:n]


def _choose(items: list[AudioItem], key: str) -> AudioItem:
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    return sorted(items, key=lambda i: i.file)[h % len(items)]


def pick_music(lib: Library, tags: list[str], key: str):
    """(item, reason): a track cleared for every main platform, preferring
    `tags`; or (None, why not)."""
    music = lib.of_kind("music")
    if not music:
        return None, "no music in the audio library"
    cleared = [i for i in music if i.cleared_for(MAIN_PLATFORMS)]
    if not cleared:
        return None, ("no music cleared for tiktok, reels and shorts "
                      "(one master render is posted on all three)")
    tagged = [i for i in cleared if set(i.tags) & set(tags)]
    return _choose(tagged or cleared, key), ""


# ---------------------------------------------------------------- build

def _source(item: AudioItem | None, tag: str) -> dict:
    if item is None:
        return {"file": f"starter:{tag}", "license": "generated by ClipForge",
                "source": "built in", "attribution": ""}
    return {"file": item.file, "license": item.license,
            "source": item.source, "attribution": item.attribution}


def _sfx_for(tag: str, lib: Library, key: str, cache: dict):
    """(array, item or None). Library effects with the tag win."""
    cands = [i for i in lib.of_kind("sfx") if tag in i.tags]
    item = _choose(cands, f"{key}:{tag}") if cands else None
    ck = item.file if item else f"starter:{tag}"
    if ck not in cache:
        if item is None:
            if tag not in STARTER_TAGS:
                raise LayerError(f"no sound for tag {tag}")
            cache[ck] = starter_sound(tag)
        else:
            x = decode_audio(item.path, 0.0, SFX_MAX_SEC)
            if x.size == 0:
                raise LayerError(f"{item.file} decoded to nothing")
            cache[ck] = x
    return cache[ck], item


def build_layer(original: np.ndarray, total: float, beats: Beats,
                spans, cfg: AudioLayerConfig, lib: Library, key: str
                ) -> tuple[np.ndarray | None, dict]:
    """The layer array (n, 2) for one clip and its report section. Returns
    (None, report) when nothing is added. `original` is the clip's own
    audio (without end card), used only to measure its level."""
    report: dict = {"applied": False, "reason": ""}
    if lib.problems:
        report["library_problems"] = lib.problems[:20]
    ref = gated_level_db(original)
    report["reference_db"] = None if ref is None else round(ref, 2)
    sfx_rep: dict = {"applied": False, "level_db": cfg.sfx_level_db}
    mus_rep: dict = {"applied": False, "level_db": cfg.music_level_db,
                     "duck_db": cfg.music_duck_db}
    report["sound_effects"], report["music"] = sfx_rep, mus_rep
    if ref is None:
        reason = "the original audio is silent here, nothing to sit under"
        sfx_rep["reason"] = mus_rep["reason"] = reason
        report["reason"] = reason
        return None, report

    n = int(round(total * SR))
    layer = np.zeros((n, 2), dtype=np.float32)
    credits: list[str] = []

    # sound effects
    if not cfg.sfx:
        sfx_rep["reason"] = "disabled by CLIPFORGE_SFX"
    else:
        groups = plan_cues(beats)
        cues_out, cache = [], {}
        for g in groups:
            for c in g:
                snd, item = _sfx_for(c.tag, lib, key, cache)
                lvl = loudest_window_db(snd)
                if lvl is None:
                    continue
                target = (ref + cfg.sfx_level_db + TAG_GAIN_DB.get(c.tag, 0.0)
                          + (item.gain_db if item else 0.0))
                placed = place(layer, snd * _gain(target - lvl), c.t,
                               c.align, c.offset)
                src = _source(item, c.tag)
                cues_out.append({"t": round(c.t + c.offset, 3),
                                 "starts_at": round(placed, 3),
                                 "event": c.event, "tag": c.tag, **src})
                if src["attribution"] and src["attribution"] not in credits:
                    credits.append(src["attribution"])
        if cues_out:
            sfx_rep.update(applied=True, reason="placed on the clip's beats",
                           cues=cues_out)
        else:
            sfx_rep["reason"] = ("no beats to place sounds on (no reaction "
                                 "peak, context overlay or end card)")

    # music bed
    if not cfg.music:
        mus_rep["reason"] = "disabled by CLIPFORGE_MUSIC"
    else:
        item, why = pick_music(lib, cfg.music_tags, key)
        if item is None:
            mus_rep["reason"] = why
        else:
            need = min(total + LOOP_XFADE, MAX_MUSIC_DECODE)
            x = decode_audio(item.path, item.start, need)
            if x.size == 0 and item.start > 0:
                x = decode_audio(item.path, 0.0, need)
            lvl = gated_level_db(x)
            if lvl is None:
                mus_rep["reason"] = f"{item.file} is silent"
            else:
                x = loop_to(x, n) * _gain(ref + cfg.music_level_db
                                          + item.gain_db - lvl)
                env = duck_envelope(n, spans, cfg.music_duck_db)
                t = np.arange(n) / SR
                env *= np.clip(t / MUSIC_FADE_IN, 0, 1)
                env *= np.clip((total - t) / MUSIC_FADE_OUT, 0, 1)
                layer += (x * env[:, None]).astype(np.float32)
                mus_rep.update(applied=True, reason="quiet bed under the "
                               "original audio, ducked while someone talks",
                               **_source(item, "music"),
                               start=item.start, speech_spans=len(spans))
                if item.attribution and item.attribution not in credits:
                    credits.append(item.attribution)

    if credits:
        report["credits"] = credits
    if not (sfx_rep["applied"] or mus_rep["applied"]):
        report["reason"] = "nothing added"
        return None, report
    report["applied"] = True
    report["reason"] = "mixed on top of the original audio"
    return layer, report
