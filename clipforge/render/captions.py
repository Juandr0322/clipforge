"""
Module 6a — Captions.

Turns word-level transcript timestamps into short-form style caption
"cards" (1-4 words each), rendered as an ASS subtitle file that ffmpeg
burns in via libass.

Units rule: input times are absolute seconds; card times are converted to
CLIP-RELATIVE seconds here (single conversion point, to keep the
seconds-vs-anything-else bug class impossible to reintroduce).
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from clipforge.transcribe.schema import Word

MAX_WORDS_PER_CARD = 4
CARD_GAP_BREAK = 0.6         # a pause longer than this starts a new card
PUNCT_BREAK = ".!?,;:"


class CaptionCard(BaseModel):
    text: str
    start: float               # clip-relative seconds
    end: float


@dataclass
class CaptionStyle:
    font: str = "Poppins"
    font_size: int = 72                  # for 1080x1920 canvas
    primary_color: str = "&H00FFFFFF"    # white (ASS BGR order)
    outline_color: str = "&H00000000"    # black
    outline: int = 4
    shadow: int = 1
    bold: bool = True
    margin_v: int = 640                  # px from bottom: lower-center,
    #                                      clear of platform UI overlays
    play_res_x: int = 1080
    play_res_y: int = 1920


def group_words(words: list[Word], clip_start: float,
                clip_end: float) -> list[CaptionCard]:
    """Group words inside [clip_start, clip_end] into caption cards.
    Break on: max card size, long pauses, or trailing punctuation."""
    inside = [w for w in words
              if clip_start <= (w.start + w.end) / 2 < clip_end]
    cards: list[CaptionCard] = []
    cur: list[Word] = []

    def flush():
        if not cur:
            return
        cards.append(CaptionCard(
            text=" ".join(w.w for w in cur),
            start=round(cur[0].start - clip_start, 3),
            end=round(cur[-1].end - clip_start, 3),
        ))
        cur.clear()

    for w in inside:
        if cur:
            gap = w.start - cur[-1].end
            if len(cur) >= MAX_WORDS_PER_CARD or gap > CARD_GAP_BREAK:
                flush()
        cur.append(w)
        if w.w and w.w[-1] in PUNCT_BREAK and len(cur) >= 2:
            flush()
    flush()

    # extend each card's end to the next card's start (no dead air between
    # captions), capped at +0.8s hold on the final card
    for a, b in zip(cards, cards[1:]):
        a.end = round(min(b.start, a.end + 0.8), 3)
    if cards:
        cards[-1].end = round(min(cards[-1].end + 0.8,
                                  clip_end - clip_start), 3)
    # clamp starts to clip
    for c in cards:
        c.start = max(0.0, c.start)
        c.end = max(c.start + 0.05, c.end)
    return cards


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    if cs == 100:
        s, cs = s + 1, 0
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def to_ass(cards: list[CaptionCard], style: CaptionStyle | None = None) -> str:
    st = style or CaptionStyle()
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {st.play_res_x}
PlayResY: {st.play_res_y}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{st.font},{st.font_size},{st.primary_color},&H000000FF,{st.outline_color},&H80000000,{-1 if st.bold else 0},0,0,0,100,100,0,0,1,{st.outline},{st.shadow},2,60,60,{st.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for c in cards:
        text = c.text.replace("\\", "").replace("{", "(").replace("}", ")")
        lines.append(f"Dialogue: 0,{_ass_time(c.start)},{_ass_time(c.end)},"
                     f"Cap,,0,0,0,,{text.upper()}")
    return header + "\n".join(lines) + "\n"


def parse_ass_times(ass_text: str) -> list[tuple[float, float, str]]:
    """Parse Dialogue times back out (used by tests to verify sync)."""
    out = []
    for line in ass_text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        def sec(ts):
            h, m, s = ts.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
        out.append((sec(parts[1]), sec(parts[2]), parts[9]))
    return out
