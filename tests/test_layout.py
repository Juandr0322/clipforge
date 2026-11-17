"""
F56 stacked layout: layout file validation, layout choice, the -vf graph,
pixel tests on real renders, the job-level wiring and the CLI.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from clipforge.layout import (
    Layout, LayoutError, choose_layout, creator_from_env, layout_path,
    load_layout_file, template,
)
from clipforge.recipes import RecipeError, load_recipe, load_recipe_file
from clipforge.reframe.detect import CropKeyframe, CropPath
from clipforge.render.render import (
    CropGeometry, build_filtergraph, render_clip, verify_render,
)

REPO = Path(__file__).resolve().parents[1]

GOOD = {"name": "stacked", "frame": {"w": 1920, "h": 1080},
        "facecam": {"x": 1440, "y": 0, "w": 480, "h": 360},
        "game": {"x": 0, "y": 360, "w": 1440, "h": 720},
        "cam_share": 0.35}


def _write(tmp_path, data, bom=False, name="layout.json"):
    p = tmp_path / name
    text = data if isinstance(data, str) else json.dumps(data)
    p.write_text(("﻿" if bom else "") + text, encoding="utf-8")
    return p


# ---------------------------------------------------------------- loading
def test_load_good_layout(tmp_path):
    lay = load_layout_file(_write(tmp_path, GOOD))
    assert lay.facecam.w == 480 and lay.game.y == 360
    assert lay.cam_h == 672 and lay.game_h == 1248
    assert lay.gameplay_share == 0.65


def test_bom_is_accepted(tmp_path):
    assert load_layout_file(_write(tmp_path, GOOD, bom=True)).name == "stacked"


def test_cam_share_defaults(tmp_path):
    d = {k: v for k, v in GOOD.items() if k != "cam_share"}
    assert load_layout_file(_write(tmp_path, d)).cam_share == 0.35


@pytest.mark.parametrize("change, field", [
    ({"name": "split"}, "name"),
    ({"frame": [1920, 1080]}, "frame"),
    ({"frame": {"w": 1920}}, "frame.h"),
    ({"frame": {"w": 1920, "h": 1080, "d": 1}}, "frame.d"),
    ({"facecam": {"x": 0, "y": 0, "w": 100}}, "facecam.h"),
    ({"facecam": {"x": 0, "y": 0, "w": 100, "h": True}}, "facecam.h"),
    ({"facecam": {"x": 0, "y": 0, "w": 100.5, "h": 100}}, "facecam.w"),
    ({"facecam": {"x": -2, "y": 0, "w": 100, "h": 100}}, "facecam.x"),
    ({"facecam": {"x": 1900, "y": 0, "w": 100, "h": 100}}, "facecam.w"),
    ({"game": {"x": 0, "y": 600, "w": 100, "h": 600}}, "game.h"),
    ({"game": "all"}, "game"),
    ({"game": {"x": 0, "y": 0, "w": 100, "h": 100, "z": 1}}, "game.z"),
    ({"cam_share": 0.9}, "cam_share"),
    ({"cam_share": "0.3"}, "cam_share"),
    ({"cam_share": True}, "cam_share"),
    ({"extra": 1}, "extra"),
])
def test_bad_fields_name_file_and_field(tmp_path, change, field):
    with pytest.raises(LayoutError) as ei:
        load_layout_file(_write(tmp_path, {**GOOD, **change}))
    assert "layout.json" in str(ei.value) and f"'{field}'" in str(ei.value)


def test_missing_box_named(tmp_path):
    d = {k: v for k, v in GOOD.items() if k != "game"}
    with pytest.raises(LayoutError, match="'game' is missing"):
        load_layout_file(_write(tmp_path, d))


@pytest.mark.parametrize("text", [
    '{"name": "stacked", "cam_share": NaN}',
    '{"name": "stacked", "cam_share": Infinity}',
    "not json", "[1, 2]", "[" * 5000,
])
def test_junk_files_raise_layout_error(tmp_path, text):
    with pytest.raises(LayoutError):
        load_layout_file(_write(tmp_path, text))


def test_template_is_valid(tmp_path):
    lay = load_layout_file(_write(tmp_path, template(2560, 1440)))
    assert (lay.frame_w, lay.frame_h) == (2560, 1440)


# ---------------------------------------------------------------- geometry
def test_boxes_scale_to_the_video(tmp_path):
    d = {**GOOD, "frame": {"w": 2560, "h": 1440},
         "facecam": {"x": 1920, "y": 0, "w": 640, "h": 480}}
    lay = load_layout_file(_write(tmp_path, d))
    cam, _ = lay.boxes_for(1920, 1080)
    assert (cam.x, cam.y, cam.w, cam.h) == (1440, 0, 480, 360)
    assert lay.fits(1920, 1080) and lay.fits(1280, 720)
    assert not lay.fits(1440, 1080) and not lay.fits(0, 0)


def test_filter_is_one_vf_graph(tmp_path):
    vf = load_layout_file(_write(tmp_path, GOOD)).filter(1920, 1080)
    assert vf.startswith("split=2") and vf.endswith("vstack=inputs=2")
    assert "crop=480:360:1440:0" in vf and "crop=1080:672" in vf
    assert "crop=1080:1248" in vf


def test_caption_margin_sits_on_the_seam(tmp_path):
    lay = load_layout_file(_write(tmp_path, GOOD))
    bottom_of_text = 1920 - lay.caption_margin_v(72)
    assert lay.cam_h <= bottom_of_text <= lay.cam_h + 72


# ---------------------------------------------------------------- choice
def _ws_with_layout(tmp_path, data=GOOD, slug="ron"):
    p = layout_path(tmp_path, slug)
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def test_choice_stacked(tmp_path):
    c = choose_layout("stacked", _ws_with_layout(tmp_path), "ron", 1920, 1080)
    assert c.layout is not None
    rep = c.report()
    assert rep["applied"] and rep["gameplay_share"] == 0.65
    assert rep["creator"] == "ron"


@pytest.mark.parametrize("recipe_layout, creator, w, h, reason", [
    ("face", "ron", 1920, 1080, "recipe uses the face-tracked crop"),
    ("stacked", None, 1920, 1080, "no creator set"),
    ("stacked", "nobody", 1920, 1080, "no layout file"),
    ("stacked", "ron", 1080, 1920, "aspect ratio"),
    ("stacked", "../x", 1920, 1080, "creator name"),
])
def test_choice_falls_back_with_reason(tmp_path, recipe_layout, creator, w,
                                       h, reason):
    c = choose_layout(recipe_layout, _ws_with_layout(tmp_path), creator, w, h)
    assert c.layout is None and reason in c.reason
    assert c.report()["applied"] is False
    assert c.report()["name"] == "face_crop"


def test_choice_invalid_file_falls_back(tmp_path):
    ws = _ws_with_layout(tmp_path, {**GOOD, "cam_share": 2})
    c = choose_layout("stacked", ws, "ron", 1920, 1080)
    assert c.layout is None and "cam_share" in c.reason
    assert str(tmp_path) not in c.reason


@pytest.mark.parametrize("raw, want", [
    ("Ron", "ron"), (" ron_2 ", "ron_2"), ("", None), ("   ", None),
    ("../etc", None), ("a/b", None), ("x" * 65, None),
])
def test_creator_env(monkeypatch, raw, want):
    monkeypatch.setenv("CLIPFORGE_CREATOR", raw)
    assert creator_from_env() == want


def test_creator_env_unset(monkeypatch):
    monkeypatch.delenv("CLIPFORGE_CREATOR", raising=False)
    assert creator_from_env() is None


# ---------------------------------------------------------------- recipes
def test_gaming_recipe_selects_stacked_and_others_do_not():
    assert load_recipe("gaming").layout == "stacked"
    for name in ("talk", "storytime", "talking_head", "cinematic",
                 "day_in_life"):
        assert load_recipe(name).layout == "face"


def test_recipe_layout_value_checked(tmp_path):
    d = json.loads((REPO / "clipforge/recipes/files/talk.json")
                   .read_text(encoding="utf-8"))
    p = tmp_path / "talk.json"
    for bad in ("split", 1, None, True):
        p.write_text(json.dumps({**d, "layout": bad}), encoding="utf-8")
        with pytest.raises(RecipeError, match="'layout'"):
            load_recipe_file(p)


# ------------------------------------------- filtergraph same as main
def _main_render_module():
    """render.py as it is on main, to compare against (LESSONS: same as
    before is checked against main, not the new code)."""
    src = subprocess.run(["git", "show", "origin/main:clipforge/render/render.py"],
                         cwd=REPO, capture_output=True, text=True,
                         encoding="utf-8")
    if src.returncode != 0:
        pytest.skip("origin/main not available in this checkout")
    spec = importlib.util.spec_from_loader("render_main", loader=None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["render_main"] = mod
    exec(compile(src.stdout, "render_main.py", "exec"), mod.__dict__)
    return mod


@pytest.mark.parametrize("w, h, mode", [(1920, 1080, "static"),
                                        (1920, 1080, "tracked"),
                                        (720, 1280, "center")])
def test_filtergraph_without_layout_unchanged(tmp_path, w, h, mode):
    main = _main_render_module()
    crop = CropPath(clip_id=1, mode=mode,
                    keyframes=[CropKeyframe(t=0, cx=0.3)])
    ass, cmds = tmp_path / "c.ass", tmp_path / "c.cmds"
    for kw in ({}, {"end_card_sec": 2.0}, {"overlay_ass": tmp_path / "o.ass"}):
        new = build_filtergraph(CropGeometry.for_source(w, h), crop, w, ass,
                                cmds, **kw)
        old = main.build_filtergraph(main.CropGeometry.for_source(w, h), crop,
                                     w, ass, cmds, **kw)
        assert new == old


# ---------------------------------------------------------------- pixels
def _stream_video(path: Path, dur=3):
    """1920x1080: red game, green facecam box at the top right, tone."""
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=red:size=1920x1080:rate=30:d={dur}",
        "-f", "lavfi", "-i", f"sine=frequency=300:duration={dur}",
        "-vf", "drawbox=x=1440:y=0:w=480:h=360:color=lime:t=fill",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


def _pixel(video: Path, x: int, y: int, t=1.0):
    out = subprocess.run([
        "ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", str(video),
        "-frames:v", "1", "-vf", f"crop=2:2:{x}:{y}", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return tuple(out[:3])


@pytest.fixture(scope="module")
def stream_src(tmp_path_factory):
    return _stream_video(tmp_path_factory.mktemp("lay") / "stream.mp4")


def test_stacked_render_pixels(tmp_path, stream_src):
    lay = load_layout_file(_write(tmp_path, GOOD))
    out = tmp_path / "clip.mp4"
    crop = CropPath(clip_id=1, mode="tracked",
                    keyframes=[CropKeyframe(t=0, cx=0.2),
                               CropKeyframe(t=2, cx=0.8)])
    render_clip(stream_src, out, 0.0, 2.5, crop, 1920, 1080, 30,
                workdir=tmp_path, stack_vf=lay.filter(1920, 1080))
    verify_render(out, 2.5)
    r, g, b = _pixel(out, 540, 300)          # facecam band
    assert g > 180 and r < 80
    r, g, b = _pixel(out, 540, 1500)         # gameplay band
    assert r > 180 and g < 80
    assert not list(tmp_path.glob("*.cmds"))  # no sendcmd for a stack


def test_face_crop_render_shows_no_facecam_band(tmp_path, stream_src):
    """Control: without the layout the old crop (left side) has no green."""
    out = tmp_path / "clip.mp4"
    crop = CropPath(clip_id=1, mode="static",
                    keyframes=[CropKeyframe(t=0, cx=0.2)])
    render_clip(stream_src, out, 0.0, 2.5, crop, 1920, 1080, 30)
    r, g, b = _pixel(out, 540, 300)
    assert r > 180 and g < 80


# ---------------------------------------------------------------- job wiring
@pytest.fixture(scope="module")
def stream_job(tmp_path_factory, stream_src):
    from clipforge.highlight.backends import MockHighlighter
    from clipforge.highlight.run import highlight_job
    from clipforge.ingest.ingest import ingest
    from clipforge.refine.run import refine_job
    from clipforge.reframe.detect import HaarDetector
    from clipforge.reframe.run import reframe_job
    from clipforge.transcribe.backends import MockTranscriber
    from clipforge.transcribe.run import transcribe_job
    ws = tmp_path_factory.mktemp("ws")
    src = _stream_video(ws / "long.mp4", dur=30)
    m = ingest(src, ws, job_id="layjob")
    jd = ws / m.job_id
    transcribe_job(jd, MockTranscriber())
    highlight_job(jd, MockHighlighter())
    refine_job(jd)
    reframe_job(jd, HaarDetector())
    return jd


def _reports(jd):
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((jd / "renders").glob("clip_*.report.json"))]


def test_render_job_with_layout(stream_job):
    from clipforge.render.overlays import OverlayConfig
    from clipforge.render.run import render_job
    from clipforge.audio.layer import AudioLayerConfig
    ws = stream_job.parent
    p = layout_path(ws, "ron")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(GOOD), encoding="utf-8")
    m = __import__("clipforge.common.manifest", fromlist=["Manifest"]) \
        .Manifest.load(stream_job)
    choice = choose_layout("stacked", ws, "ron", m.ingest.normalized_width,
                           m.ingest.normalized_height)
    assert choice.layout is not None
    cfg = OverlayConfig(context_overlay=False, punch_in=True, end_card=False)
    render_job(stream_job, force=True, contact_sheets=False, overlays=cfg,
               audio=AudioLayerConfig(sfx=False, music=False),
               layout=choice)
    clips = sorted((stream_job / "renders").glob("clip_*.mp4"))
    assert clips
    for c in clips:
        r, g, b = _pixel(c, 540, 300, t=0.5)
        assert g > 180 and r < 80
        ass = (stream_job / "renders" / (c.stem + ".ass")).read_text(
            encoding="utf-8")
        assert f",{choice.layout.caption_margin_v(72)},1" in ass
    for rep in _reports(stream_job):
        assert rep["layout"]["applied"] is True
        assert rep["layout"]["gameplay_share"] == 0.65
        pi = [e for e in rep["value_adds"] if e["name"] == "punch_in"][0]
        assert pi["applied"] is False


def test_render_job_without_layout_writes_no_layout_section(stream_job):
    from clipforge.render.run import render_job
    from clipforge.audio.layer import AudioLayerConfig
    for p in (stream_job / "renders").glob("clip_*.report.json"):
        p.unlink()
    render_job(stream_job, force=True, contact_sheets=False,
               audio=AudioLayerConfig(sfx=False, music=False))
    for rep in _reports(stream_job):
        assert "layout" not in rep


def test_render_job_falls_back_to_face_crop(stream_job, monkeypatch):
    """A broken layout graph never fails the stage."""
    from clipforge.render.run import render_job
    from clipforge.audio.layer import AudioLayerConfig
    from clipforge.layout import LayoutChoice
    ws = stream_job.parent
    lay = load_layout_file(layout_path(ws, "ron"))
    monkeypatch.setattr(Layout, "filter",
                        lambda self, w, h: "crop=99999:99999:0:0")
    for p in (stream_job / "renders").glob("clip_*.report.json"):
        p.unlink()
    m = render_job(stream_job, force=True, contact_sheets=False,
                   audio=AudioLayerConfig(sfx=False, music=False),
                   layout=LayoutChoice(lay, "stacked", "ron"))
    assert m.stage.value == "rendered"
    for rep in _reports(stream_job):
        assert rep["layout"]["applied"] is False
        assert "stacked render failed" in rep["layout"]["reason"]


def test_pipeline_passes_layout_for_gaming(stream_job, monkeypatch):
    """run_pipeline picks the layout from the recipe and CLIPFORGE_CREATOR."""
    from clipforge import pipeline
    from clipforge.common.manifest import Manifest, Stage
    seen = {}

    def fake_render(job_dir, **kw):
        seen.update(kw)
        return Manifest.load(job_dir)
    monkeypatch.setattr("clipforge.render.run.render_job", fake_render)
    monkeypatch.setenv("CLIPFORGE_CREATOR", "ron")
    m = Manifest.load(stream_job)
    m.stage = Stage.REFRAMED
    m.save(stream_job)
    (stream_job / "recipe.json").unlink(missing_ok=True)
    cfg = pipeline.PipelineConfig(asr_engine="mock", highlight_engine="mock",
                                  detector="haar", recipe="gaming")
    pipeline.run_pipeline(stream_job, cfg, progress=lambda s: None)
    assert seen["layout"].layout is not None
    assert seen["layout"].creator == "ron"


# ---------------------------------------------------------------- CLI
def _cli(*args, ws):
    return subprocess.run(
        [sys.executable, "-m", "clipforge.cli", *args, "--workspace", str(ws)],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8")


def test_cli_init_check_preview(tmp_path, stream_src):
    p = _cli("layout", "init", "--creator", "ron", "--video", str(stream_src),
             ws=tmp_path)
    assert p.returncode == 0, p.stderr
    path = layout_path(tmp_path, "ron")
    assert load_layout_file(path).frame_w == 1920
    assert _cli("layout", "init", "--creator", "ron", "--video",
                str(stream_src), ws=tmp_path).returncode == 0  # left alone
    path.write_text(json.dumps(GOOD), encoding="utf-8")
    p = _cli("layout", "check", "--creator", "ron", ws=tmp_path)
    assert p.returncode == 0 and "ok" in p.stdout
    out = tmp_path / "prev.png"
    p = _cli("layout", "preview", str(stream_src), "--creator", "ron",
             "--at", "1", "--out", str(out), ws=tmp_path)
    assert p.returncode == 0, p.stderr
    assert out.exists() and (tmp_path / "prev_boxes.png").exists()
    r, g, b = _pixel(out, 540, 300, t=0)
    assert g > 180 and r < 80


def test_cli_errors(tmp_path):
    p = _cli("layout", "check", "--creator", "../x", ws=tmp_path)
    assert p.returncode == 2 and "LAYOUT FAILED" in p.stderr
    p = _cli("layout", "check", "--creator", "nobody", ws=tmp_path)
    assert p.returncode == 1 and "layout init" in p.stderr
