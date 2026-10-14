"""
Module 6 — Captioning & rendering stage runner.

Requires REFRAMED. For each refined clip: build captions from the
transcript, load its crop path, render renders/clip_<id>.mp4, verify it,
and write a QA contact sheet. Advances manifest to RENDERED.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from clipforge.common.manifest import Manifest, Stage
from clipforge.refine.refine import RefinedSet
from clipforge.reframe.detect import CropPath
from clipforge.render.captions import CaptionStyle, group_words, to_ass
from clipforge.render.overlays import (
    OverlayConfig, ZoomSpec, build_overlay_ass, context_event, valid_rel_time,
)
from clipforge.render.render import (
    RenderError, make_contact_sheet, render_clip, verify_render,
)
from clipforge.audio.layer import AudioLayerConfig
from clipforge.transcribe.schema import Transcript


_TIMES = ("trigger_t", "reaction_peak_t", "payoff_end_t")


def _num(v) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v))


def _pick_time(own, cand, start: float, end: float):
    """(value, seen). The refined clip's own value wins when it is a finite
    number inside the clip. When it is None/absent/out of range, the
    candidate's value is used only if finite and strictly inside the clip.
    `seen` says some source had a value, i.e. a miss means 'outside'."""
    if _num(own) and start <= own <= end:
        return float(own), True
    if _num(cand) and start < cand < end:
        return float(cand), True
    return None, own is not None or cand is not None


def _v2_fields(job_dir: Path, m: Manifest, rs: RefinedSet) -> dict[int, dict]:
    """Per refined clip: the optional highlight v2 fields (absolute times),
    resolved as in _pick_time. Unreadable or v1 files give no fields."""
    cands = {}
    try:
        if m.candidates_path:
            from clipforge.highlight.schema import CandidateSet
            cs = CandidateSet.load(job_dir / m.candidates_path)
            cands = {c.id: c for c in cs.candidates}
    except Exception:  # noqa: BLE001 - extras are best-effort
        cands = {}
    out = {}
    for clip in rs.clips:
        src = cands.get(clip.candidate_id)
        f = {}
        for name in _TIMES:
            f[name], f[name + "_seen"] = _pick_time(
                getattr(clip, name, None), getattr(src, name, None),
                clip.start, clip.end)
        own = getattr(clip, "context_setup", None)
        f["context_setup"] = (own if own is not None
                              else getattr(src, "context_setup", None))
        out[clip.id] = f
    return out


def _plan_extras(clip, fields: dict, cfg: OverlayConfig, renders: Path):
    """Decide the value-adds for one clip. Returns (zoom, overlay_ass_path,
    end_card_sec, entries) where entries are the report value_adds rows
    (without captions); overlay_ass_path is None when nothing is drawn."""
    dur = clip.end - clip.start
    entries, events, zoom = [], [], None

    ev = None
    if not cfg.context_overlay:
        reason = "disabled by CLIPFORGE_CONTEXT_OVERLAY"
    else:
        ev = context_event(fields.get("context_setup"), dur)
        reason = "" if ev else "no context_setup from highlight"
    if ev:
        events.append((ev[0], ev[1], "Ctx", ev[2]))
        entries.append({"name": "context_overlay", "applied": True,
                        "reason": "context_setup shown at the top",
                        "start": ev[0], "end": ev[1], "text": ev[2]})
    else:
        entries.append({"name": "context_overlay", "applied": False,
                        "reason": reason})

    peak = None
    if not cfg.punch_in:
        reason = "disabled by CLIPFORGE_PUNCH_IN"
    else:
        raw = fields.get("reaction_peak_t")
        peak = valid_rel_time(None if raw is None else raw - clip.start, dur)
        reason = ("" if peak is not None else
                  "reaction_peak_t outside the clip"
                  if fields.get("reaction_peak_t_seen")
                  else "no reaction_peak_t from highlight")
    if peak is not None:
        zoom = ZoomSpec(peak_t=peak)
        entries.append({"name": "punch_in", "applied": True,
                        "reason": "zoom on the reaction peak, follows the "
                                  "tracked crop", "peak_t": round(peak, 3)})
    else:
        entries.append({"name": "punch_in", "applied": False,
                        "reason": reason})

    card = 0.0
    if cfg.end_card:
        card = cfg.end_card_sec
        events.append((dur, dur + card, "Card", cfg.end_card_text))
        entries.append({"name": "end_card", "applied": True,
                        "reason": "end card appended",
                        "start": round(dur, 3), "end": round(dur + card, 3)})
    else:
        entries.append({"name": "end_card", "applied": False,
                        "reason": "disabled (set CLIPFORGE_END_CARD=1)"})

    ass = renders / f"clip_{clip.id:03d}.overlay.ass"
    if not events:
        ass.unlink(missing_ok=True)      # stale file from an earlier render
        ass = None
    else:
        ass.write_text(build_overlay_ass(events), encoding="utf-8",
                       newline="\n")
    return zoom, ass, card, entries


def _record_value_adds(job_dir: Path, clip_id: int, entries: list) -> None:
    _record(job_dir, clip_id, "value_adds", entries)


def _record(job_dir: Path, clip_id: int, section: str, value) -> None:
    try:
        from clipforge.quality.report import update_section
        update_section(job_dir, clip_id, section, value)
    except Exception:  # noqa: BLE001 - a report must not fail a render
        print(f"clip {clip_id}: {section} report could not be written",
              file=sys.stderr)


def _plan_audio(clip, fields: dict, entries: list, card: float,
                cfg: AudioLayerConfig, lib, norm: Path, words, key: str,
                renders: Path):
    """Build the F48 audio layer for one clip. Returns (layer_path or None,
    report section). The beats come from the value-adds actually planned,
    so a sound never marks something that isn't on screen."""
    from clipforge.audio.layer import (
        Beats, build_layer, decode_audio, speech_spans, write_layer,
    )
    dur = clip.end - clip.start
    va = {e["name"]: e for e in entries}
    raw = fields.get("reaction_peak_t")
    peak = valid_rel_time(None if raw is None else raw - clip.start, dur)
    ov = va.get("context_overlay", {})
    beats = Beats(
        duration=dur, total=dur + card, reaction_peak_t=peak,
        zoom_applied=bool(va.get("punch_in", {}).get("applied")),
        overlay_t=ov.get("start") if ov.get("applied") else None,
        end_card_t=dur if card > 0 else None)
    original = decode_audio(norm, clip.start, dur)
    spans = speech_spans(words, clip.start, clip.end)
    layer, rep = build_layer(original, dur + card, beats, spans, cfg, lib,
                             key)
    if layer is None:
        return None, rep
    path = renders / f"clip_{clip.id:03d}.layer.f32"
    write_layer(path, layer)
    return path, rep


def render_job(job_dir: str | Path, force: bool = False,
               captions: bool = True,
               style: CaptionStyle | None = None,
               contact_sheets: bool = True,
               overlays: OverlayConfig | None = None,
               audio: AudioLayerConfig | None = None) -> Manifest:
    job_dir = Path(job_dir)
    m = Manifest.load(job_dir)

    if not m.is_at_least(Stage.REFRAMED):
        raise RenderError(
            f"Job is at stage '{m.stage.value}' — run reframe first.")
    if m.is_at_least(Stage.RENDERED) and not force:
        return m

    started = datetime.now(timezone.utc)
    try:
        t = Transcript.load(job_dir / m.transcript_path)
        rs = RefinedSet.load(job_dir / m.refined_path)
        crops_dir = job_dir / m.crop_paths_dir
        renders = job_dir / "renders"
        renders.mkdir(exist_ok=True)
        norm = job_dir / m.ingest.normalized_video
        cfg = overlays or OverlayConfig.from_env()
        acfg = audio or AudioLayerConfig.from_env()
        v2 = _v2_fields(job_dir, m, rs)
        lib = None
        if acfg.sfx or acfg.music:
            from clipforge.audio.library import load_library
            lib = load_library(acfg.library_dir)

        for clip in rs.clips:
            out = renders / f"clip_{clip.id:03d}.mp4"
            ass_path = None
            cards = []
            if captions:
                cards = group_words(t.words, clip.start, clip.end)
                ass_path = renders / f"clip_{clip.id:03d}.ass"
                ass_path.write_text(to_ass(cards, style), encoding="utf-8",
                                    newline="\n")
            crop = CropPath.load(crops_dir / f"clip_{clip.id:03d}.json")
            kw = dict(src_w=m.ingest.normalized_width,
                      src_h=m.ingest.normalized_height,
                      fps=m.ingest.normalized_fps, ass_path=ass_path)
            try:
                zoom, ov_ass, card, entries = _plan_extras(
                    clip, v2.get(clip.id, {}), cfg, renders)
            except Exception:  # noqa: BLE001 - extras are best-effort
                zoom, ov_ass, card = None, None, 0.0
                entries = [{"name": n, "applied": False,
                            "reason": "could not be planned"}
                           for n in ("context_overlay", "punch_in",
                                     "end_card")]
            expected = clip.end - clip.start + card
            plain = zoom is None and ov_ass is None and not card

            layer, arep = None, None
            if lib is not None:
                try:
                    layer, arep = _plan_audio(
                        clip, v2.get(clip.id, {}), entries, card, acfg, lib,
                        norm, t.words, f"{job_dir.name}:{clip.id}", renders)
                except Exception:  # noqa: BLE001 - the layer is best-effort
                    layer, arep = None, {"applied": False,
                                         "reason": "could not be built"}
            else:
                arep = {"applied": False, "reason": "disabled by "
                        "CLIPFORGE_SFX and CLIPFORGE_MUSIC"}

            # Attempts, best first: everything; then without the added
            # audio; then the plain cut (which never gets the layer, since
            # its sounds mark extras that would be missing).
            attempts = []
            if layer is not None:
                attempts.append(("extras" if not plain else "plain", layer))
            attempts.append(("extras" if not plain else "plain", None))
            if not plain:
                attempts.append(("plain", None))
            try:
                for i, (kind, lay) in enumerate(attempts):
                    try:
                        if kind == "plain":
                            render_clip(norm, out, clip.start, clip.end,
                                        crop, audio_layer=lay, **kw)
                            verify_render(out, clip.end - clip.start)
                        else:
                            render_clip(norm, out, clip.start, clip.end,
                                        crop, zoom=zoom, overlay_ass=ov_ass,
                                        end_card_sec=card, audio_layer=lay,
                                        **kw)
                            verify_render(out, expected if not plain
                                          else clip.end - clip.start)
                    except RenderError:
                        if i == len(attempts) - 1:
                            raise
                        continue
                    if layer is not None and lay is None:
                        # its sounds mark on-screen extras: never play
                        # them over a cut that lost those extras
                        arep = {**arep, "applied": False, "reason":
                                "mixing failed, rendered without added audio"
                                if kind != "plain" or plain else
                                "the render with extras failed; plain cut "
                                "used without added audio"}
                    if kind == "plain" and not plain:
                        # extras must never fail the stage: plain cut used
                        entries = [{"name": e["name"], "applied": False,
                                    "reason": "extras failed, rendered "
                                              "without them"}
                                   if e["applied"] else e for e in entries]
                        if ov_ass is not None:
                            ov_ass.unlink(missing_ok=True)
                    break
            finally:
                if layer is not None:
                    layer.unlink(missing_ok=True)
            entries.append({
                "name": "captions", "applied": bool(cards),
                "reason": ("burned in from the transcript" if cards else
                           "captions disabled" if not captions
                           else "no words in this clip")})
            _record_value_adds(job_dir, clip.id, entries)
            _record(job_dir, clip.id, "audio_layer", arep)
            if contact_sheets:
                make_contact_sheet(out, renders / f"clip_{clip.id:03d}_qa.png")

        m.renders_dir = renders.name
        m.record(Stage.RENDERED, started, ok=True)
    except Exception as e:  # noqa: BLE001
        m.record(Stage.RENDERED, started, ok=False, error=str(e))
        m.save(job_dir)
        raise
    m.save(job_dir)
    return m
