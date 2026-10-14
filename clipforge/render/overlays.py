"""
Module 6c — Originality layer (all optional, all best-effort).

Three value-adds on top of the plain cut, each switchable by a CLIPFORGE_*
setting and each skipped (with a recorded reason) when its input is missing:

  context_overlay  one line of background (candidate.context_setup) shown at
                   the top of the frame from 0 s, gone by CONTEXT_END_SEC.
  punch_in         a smooth zoom that peaks on reaction_peak_t. It scales the
                   already-cropped frame about its centre, and that centre IS
                   the face-tracking crop path, so the zoom follows the face.
  end_card         a short title card appended after the clip.

Text from the model only ever reaches ffmpeg inside an ASS file (never inside
the filter graph), and is sanitised so it cannot inject ASS override tags.
Times are clip-relative seconds; the runner converts from absolute.
"""
from __future__ import annotations

import math
import os
import re
import textwrap
import unicodedata
from dataclasses import dataclass

from clipforge.render.captions import CaptionStyle, _ass_time

CONTEXT_END_SEC = 3.5        # overlay is gone before 4 s
CONTEXT_MAX_CHARS = 110
CONTEXT_WRAP = 24
CONTEXT_MAX_LINES = 4

ZOOM_AMOUNT = 0.25           # peak scale is 1 + ZOOM_AMOUNT
ZOOM_RAMP_SEC = 0.4
ZOOM_HOLD_SEC = 1.2

END_CARD_DEFAULT_SEC = 2.0
END_CARD_MAX_SEC = 10.0
END_CARD_DEFAULT_TEXT = "Follow for more"
END_CARD_MAX_CHARS = 60

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return default


@dataclass
class OverlayConfig:
    context_overlay: bool = True
    punch_in: bool = True
    end_card: bool = False
    end_card_sec: float = END_CARD_DEFAULT_SEC
    end_card_text: str = END_CARD_DEFAULT_TEXT

    @classmethod
    def from_env(cls) -> "OverlayConfig":
        try:
            sec = float(os.environ.get("CLIPFORGE_END_CARD_SEC", ""))
        except ValueError:
            sec = END_CARD_DEFAULT_SEC
        if not math.isfinite(sec) or sec <= 0:
            sec = END_CARD_DEFAULT_SEC
        text = sanitize_text(os.environ.get("CLIPFORGE_END_CARD_TEXT", ""),
                             END_CARD_MAX_CHARS) or END_CARD_DEFAULT_TEXT
        return cls(
            context_overlay=_flag("CLIPFORGE_CONTEXT_OVERLAY", True),
            punch_in=_flag("CLIPFORGE_PUNCH_IN", True),
            end_card=_flag("CLIPFORGE_END_CARD", False),
            end_card_sec=min(sec, END_CARD_MAX_SEC),
            end_card_text=text,
        )


def sanitize_text(text, max_chars: int) -> str:
    """Plain one-line text safe to place in an ASS Dialogue line: control
    and format characters (bidi overrides, zero-width), backslashes and braces (ASS override syntax) are removed."""
    if not isinstance(text, str):
        return ""
    s = re.sub(r"[\x00-\x1f\x7f-\x9f  ]", " ", text)
    s = re.sub(r"[\\{}]", "", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Cf")
    s = " ".join(s.split())
    if len(s) > max_chars:
        s = s[:max_chars - 1].rstrip() + "…"
    return s


def _wrap(text: str) -> str:
    lines = textwrap.wrap(text, CONTEXT_WRAP, break_long_words=True) or [text]
    if len(lines) > CONTEXT_MAX_LINES:
        lines = lines[:CONTEXT_MAX_LINES]
        lines[-1] = lines[-1].rstrip(" .,") + "…"
    return "\\N".join(lines)


def valid_rel_time(t, duration: float):
    """A clip-relative time inside the clip, else None (junk is ignored)."""
    if isinstance(t, bool) or not isinstance(t, (int, float)):
        return None
    if not math.isfinite(t) or t < 0 or t > duration:
        return None
    return float(t)


def context_event(context_setup, duration: float):
    """(start, end, text) for the context overlay, or None when unusable."""
    text = sanitize_text(context_setup, CONTEXT_MAX_CHARS)
    if not text or duration <= 0:
        return None
    return 0.0, round(min(CONTEXT_END_SEC, duration), 3), text


def build_overlay_ass(events: list[tuple[float, float, str, str]],
                      style: CaptionStyle | None = None) -> str:
    """ASS file for overlay events: (start, end, style_name, text) where
    style_name is 'Ctx' (top box) or 'Card' (centre, end card)."""
    st = style or CaptionStyle()
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {st.play_res_x}
PlayResY: {st.play_res_y}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Ctx,{st.font},54,&H00FFFFFF,&H000000FF,&H00000000,&H99000000,-1,0,0,0,100,100,0,0,3,14,0,8,70,70,260,1
Style: Card,{st.font},84,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,5,1,5,80,80,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [f"Dialogue: 0,{_ass_time(a)},{_ass_time(b)},{name},,0,0,0,,"
             f"{_wrap(text)}" for a, b, name, text in events]
    return head + "\n".join(lines) + "\n"


@dataclass
class ZoomSpec:
    peak_t: float                      # clip-relative
    amount: float = ZOOM_AMOUNT
    ramp: float = ZOOM_RAMP_SEC
    hold: float = ZOOM_HOLD_SEC

    def scale_expr(self) -> str:
        """ffmpeg expression for the zoom factor at time t."""
        t0 = self.peak_t - self.ramp
        t3 = self.peak_t + self.hold + self.ramp
        return (f"(1+{self.amount:.4f}*clip(min((t-{t0:.4f})/{self.ramp:.4f},"
                f"({t3:.4f}-t)/{self.ramp:.4f}),0,1))")

    def filter(self, out_w: int, out_h: int) -> str:
        """Scale about the frame centre, then crop back to the output size.
        The centre of the frame is the tracked crop centre, so the punch-in
        stays on the face. scale(eval=frame) resizes frames but not the
        filter link, so crop cannot use iw/ih: its offset is computed from
        the same zoom expression instead. Quoted so commas stay inside."""
        z = self.scale_expr()
        sw = f"trunc({out_w}*{z}/2)*2"
        sh = f"trunc({out_h}*{z}/2)*2"
        return (f"scale=w='{sw}':h='{sh}':eval=frame,"
                f"crop=w={out_w}:h={out_h}:x='({sw}-{out_w})/2'"
                f":y='({sh}-{out_h})/2'")
