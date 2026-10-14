"""F12: originality layer (context overlay, punch-in, end card, report)."""
import json
import subprocess
from pathlib import Path

import pytest

from clipforge.ingest.ingest import ingest, parse_probe, probe
from clipforge.pipeline import PipelineConfig, run_pipeline
from clipforge.quality.report import load_report
from clipforge.reframe.detect import CropKeyframe, CropPath
from clipforge.render.captions import parse_ass_times
from clipforge.render.overlays import (
    OverlayConfig, ZoomSpec, build_overlay_ass, context_event, sanitize_text,
)
from clipforge.render.render import render_clip, verify_render
from clipforge.render.run import render_job

CFG = PipelineConfig(asr_engine="mock", highlight_engine="mock",
                     detector="haar")
SQ_X = 192            # square centre = 0.3 * 640


@pytest.fixture(scope="module")
def square_video(tmp_path_factory):
    """12 s, 640x360: a 40 px white square at x=0.3 on black, with audio."""
    p = tmp_path_factory.mktemp("sq") / "square.mp4"
    geq = ("geq=lum='if(between(X,172,211)*between(Y,160,199),255,0)'"
           ":cb=128:cr=128")
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "color=c=black:s=640x360:r=30:d=12,format=yuv420p," + geq,
        "-f", "lavfi", "-i", "sine=frequency=300:duration=12",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(p)], check=True, capture_output=True)
    return p


def _tracked(cx=0.3):
    return CropPath(clip_id=1, mode="tracked", keyframes=[
        CropKeyframe(t=0.0, cx=cx), CropKeyframe(t=12.0, cx=cx)])


def _gray(video: Path, t: float, w=1080, h=1920):
    p = subprocess.run([
        "ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
        "-frames:v", "1", "-vf", "format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True)
    assert len(p.stdout) == w * h
    return p.stdout


def _white(buf: bytes, w=1080):
    idx = [i for i, v in enumerate(buf) if v > 200]
    xs = [i % w for i in idx]
    return len(idx) / len(buf), (min(xs), max(xs)) if xs else None


def _render(src, out, **kw):
    render_clip(src, out, 0.0, 6.0, _tracked(), 640, 360, 30.0, **kw)


def test_punch_in_grows_square_and_stays_on_tracked_path(square_video,
                                                         tmp_path):
    out = tmp_path / "z.mp4"
    _render(square_video, out, zoom=ZoomSpec(peak_t=3.0))
    verify_render(out, 6.0)
    f0, span0 = _white(_gray(out, 0.0))
    fp, spanp = _white(_gray(out, 3.0))
    assert f0 > 0 and fp > f0 * 1.3
    lo, hi = 1080 / 3, 2 * 1080 / 3
    assert lo <= spanp[0] and spanp[1] <= hi      # middle third at the peak
    fm, spanm = _white(_gray(out, 2.8))           # mid zoom, still tracked
    assert lo <= spanm[0] and spanm[1] <= hi
    flate, _ = _white(_gray(out, 5.5))            # zoom is gone again
    assert flate == pytest.approx(f0, rel=0.05)


def test_context_overlay_events_and_pixels(square_video, tmp_path):
    ev = context_event("Streamer just lost a 3 hour run", 6.0)
    assert ev and ev[0] == 0.0 and ev[1] < 4.0
    ass = tmp_path / "o.ass"
    ass.write_text(build_overlay_ass([(ev[0], ev[1], "Ctx", ev[2])]),
                   encoding="utf-8", newline="\n")
    times = parse_ass_times(ass.read_text(encoding="utf-8"))
    assert times[0][0] == 0.0 and 0 < times[0][1] < 4.0
    out = tmp_path / "c.mp4"
    _render(square_video, out, overlay_ass=ass)
    top = lambda b: sum(b[y * 1080 + x] > 128 for y in range(150, 500)
                        for x in range(0, 1080, 3))
    assert top(_gray(out, 1.0)) > 200          # text drawn at 1 s
    assert top(_gray(out, 4.5)) == 0           # and gone by 4 s


def test_hostile_context_text_is_sanitised(square_video, tmp_path):
    nasty = "{\\an5\\fs400}a'b:c;[x],\n\x00Dialogue: 0,0:00:00.00" + "z" * 400
    s = sanitize_text(nasty, 110)
    assert "{" not in s and "\\" not in s and "\n" not in s and len(s) <= 110
    ass = tmp_path / "h.ass"
    ass.write_text(build_overlay_ass([(0, 3, "Ctx", s)]), encoding="utf-8",
                   newline="\n")
    out = tmp_path / "h.mp4"
    _render(square_video, out, overlay_ass=ass)
    verify_render(out, 6.0)
    assert sanitize_text("a\u202eb\u200bc\u2066d", 50) == "abcd"
    assert sanitize_text(None, 10) == "" and sanitize_text(5, 10) == ""
    assert context_event("  \n ", 5.0) is None


def test_end_card_adds_duration(square_video, tmp_path):
    ass = tmp_path / "e.ass"
    ass.write_text(build_overlay_ass([(6.0, 8.0, "Card", "Follow")]),
                   encoding="utf-8", newline="\n")
    out = tmp_path / "e.mp4"
    _render(square_video, out, overlay_ass=ass, end_card_sec=2.0)
    verify_render(out, 8.0)
    assert parse_probe(probe(out)).duration == pytest.approx(8.0, abs=0.1)
    plain = tmp_path / "p.mp4"
    _render(square_video, plain)
    assert parse_probe(probe(plain)).duration == pytest.approx(6.0, abs=0.1)


def test_config_from_env(monkeypatch):
    for n in ("CONTEXT_OVERLAY", "PUNCH_IN", "END_CARD", "END_CARD_SEC",
              "END_CARD_TEXT"):
        monkeypatch.delenv("CLIPFORGE_" + n, raising=False)
    c = OverlayConfig.from_env()
    assert (c.context_overlay, c.punch_in, c.end_card) == (True, True, False)
    monkeypatch.setenv("CLIPFORGE_END_CARD", "1")
    monkeypatch.setenv("CLIPFORGE_PUNCH_IN", "0")
    monkeypatch.setenv("CLIPFORGE_END_CARD_SEC", "nan")
    c = OverlayConfig.from_env()
    assert c.end_card and not c.punch_in and c.end_card_sec == 2.0


@pytest.fixture(scope="module")
def job(tmp_path_factory):
    """Mock pipeline job. The mock highlighter needs >= 15 s clips
    (MIN_CLIP_SEC), so this one source is 40 s; the pixel tests above use
    12 s sources."""
    ws = tmp_path_factory.mktemp("ows")
    src = ws / "long.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "testsrc=duration=40:size=640x360:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=40",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(src)], check=True, capture_output=True)
    ingest(src, ws, job_id="o1")
    run_pipeline(ws / "o1", CFG, progress=lambda s: None)
    return ws / "o1"


V2 = ("trigger_t", "reaction_peak_t", "payoff_end_t", "context_setup",
      "hook_type")


def _set_candidates(job_dir, refined=None, **fields):
    """Rewrite candidates.json (mock's v2 fields dropped, then `fields`
    added) and refined.json (times dropped, then `refined` added), as the
    refine stage would have left them, and re-render. Callable values get
    the candidate / refined clip dict."""
    def rewrite(name, key, drop, add):
        path = job_dir / name
        data = json.loads(path.read_text(encoding="utf-8"))
        for c in data[key]:
            for k in drop:
                c.pop(k, None)
            for k, v in add.items():
                c[k] = v(c) if callable(v) else v
        path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
    rewrite("candidates.json", "candidates", V2, fields)
    rewrite("refined.json", "clips", V2[:3], refined or {})
    return render_job(job_dir, force=True)


def _clip_ids(job_dir):
    ids = [int(p.stem.split("_")[1])
           for p in sorted((job_dir / "renders").glob("clip_???.mp4"))]
    assert ids
    return ids


def _value_adds(job_dir, clip_id):
    rep = load_report(job_dir, clip_id)
    assert rep is not None and rep.value_adds
    return {v["name"]: v for v in rep.value_adds}


def test_v1_candidates_render_as_before(job, monkeypatch):
    monkeypatch.delenv("CLIPFORGE_END_CARD", raising=False)
    _set_candidates(job)
    for cid in _clip_ids(job):
        va = _value_adds(job, cid)
        assert set(va) == {"context_overlay", "punch_in", "end_card",
                           "captions"}
        assert all(isinstance(v["applied"], bool) and v["reason"]
                   for v in va.values())
        assert not va["context_overlay"]["applied"]
        assert "reaction_peak_t" in va["punch_in"]["reason"]
        assert not va["end_card"]["applied"]
        assert va["captions"]["applied"]
        mp4 = job / "renders" / f"clip_{cid:03d}.mp4"
        assert not mp4.with_suffix(".overlay.ass").exists()


def test_v2_fields_and_end_card_in_pipeline_job(job, monkeypatch):
    monkeypatch.setenv("CLIPFORGE_END_CARD", "1")
    m = _set_candidates(
        job, context_setup="He had never lost here before",
        refined={"reaction_peak_t": lambda c: c["start"] + 3.0})
    for cid in _clip_ids(job):
        va = _value_adds(job, cid)
        assert va["context_overlay"]["applied"]
        assert va["context_overlay"]["start"] == 0.0
        assert va["context_overlay"]["end"] < 4.0
        assert va["punch_in"]["applied"] and va["end_card"]["applied"]
        ass = job / "renders" / f"clip_{cid:03d}.overlay.ass"
        ev = parse_ass_times(ass.read_text(encoding="utf-8"))
        assert ev[0][0] == 0.0 and ev[0][1] == va["context_overlay"]["end"]
        mp4 = job / "renders" / f"clip_{cid:03d}.mp4"
        assert parse_probe(probe(mp4)).duration == pytest.approx(
            va["end_card"]["start"] + 2.0, abs=0.1)
    assert m.renders_dir == "renders"


def test_junk_peak_is_not_applied(job, monkeypatch):
    monkeypatch.delenv("CLIPFORGE_END_CARD", raising=False)
    _set_candidates(job, refined={"reaction_peak_t": None},
                    reaction_peak_t=9999.0)
    for cid in _clip_ids(job):
        va = _value_adds(job, cid)
        assert not va["punch_in"]["applied"]
        assert va["punch_in"]["reason"] == "reaction_peak_t outside the clip"


def test_peak_cleared_by_refine_falls_back_to_candidate(job, monkeypatch):
    monkeypatch.delenv("CLIPFORGE_END_CARD", raising=False)
    _set_candidates(job, refined={"reaction_peak_t": None},
                    reaction_peak_t=lambda c: c["start"] + 3.0)
    for cid in _clip_ids(job):
        assert _value_adds(job, cid)["punch_in"]["applied"]


def test_refined_peak_wins_over_candidate(job, monkeypatch):
    monkeypatch.delenv("CLIPFORGE_END_CARD", raising=False)
    _set_candidates(job, refined={"reaction_peak_t":
                                  lambda c: c["start"] + 5.0},
                    reaction_peak_t=lambda c: c["start"] + 3.0)
    for cid in _clip_ids(job):
        assert _value_adds(job, cid)["punch_in"]["peak_t"] == 5.0


def test_cleared_peak_and_candidate_outside_not_applied(job, monkeypatch):
    monkeypatch.delenv("CLIPFORGE_END_CARD", raising=False)
    _set_candidates(job, refined={"reaction_peak_t": None},
                    reaction_peak_t=-1.0)
    for cid in _clip_ids(job):
        va = _value_adds(job, cid)
        assert va["punch_in"]["reason"] == "reaction_peak_t outside the clip"


def test_plain_render_command_has_no_t_option(job, monkeypatch):
    import clipforge.render.render as rr
    cmds = []
    monkeypatch.setattr(rr, "_run", lambda cmd, timeout=0: cmds.append(cmd))
    monkeypatch.delenv("CLIPFORGE_END_CARD", raising=False)
    from clipforge.reframe.detect import CropPath
    crop = CropPath(clip_id=1, mode="center")
    rr.render_clip(job / "x.mp4", job / "y.mp4", 0.0, 5.0, crop, 640, 360,
                   30.0)
    rr.render_clip(job / "x.mp4", job / "y.mp4", 0.0, 5.0, crop, 640, 360,
                   30.0, end_card_sec=2.0)
    assert "-t" in cmds[0] and cmds[0].count("-t") == 1   # input -t only
    assert cmds[0].index("-t") < cmds[0].index("-i")
    assert cmds[1].count("-t") == 2


def test_extras_failure_falls_back_to_plain_render(job, monkeypatch):
    import clipforge.render.run as run
    real = run.render_clip

    def flaky(*a, **kw):
        if kw.get("zoom") is not None:
            raise run.RenderError("boom /secret/path")
        return real(*a, **kw)

    monkeypatch.setattr(run, "render_clip", flaky)
    monkeypatch.delenv("CLIPFORGE_END_CARD", raising=False)
    _set_candidates(job, context_setup="Some setup",
                    refined={"reaction_peak_t": lambda c: c["start"] + 3.0})
    for cid in _clip_ids(job):
        va = _value_adds(job, cid)
        assert not va["punch_in"]["applied"]
        assert "/" not in va["punch_in"]["reason"]
        assert set(va["context_overlay"]) == {"name", "applied", "reason"}
        assert not va["context_overlay"]["applied"]
        assert not (job / "renders" / f"clip_{cid:03d}.overlay.ass").exists()


def test_planning_failure_still_lists_all_four_rows(job, monkeypatch):
    import clipforge.render.run as run

    def boom(*a, **k):
        raise ValueError("/secret")

    monkeypatch.setattr(run, "_plan_extras", boom)
    _set_candidates(job)
    for cid in _clip_ids(job):
        va = _value_adds(job, cid)
        assert set(va) == {"context_overlay", "punch_in", "end_card",
                           "captions"}
        assert not va["punch_in"]["applied"] and va["captions"]["applied"]
