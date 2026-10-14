"""F48: audio layer (sound effects + quiet music bed on top of the original)."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from clipforge.audio import library as L
from clipforge.audio.layer import (
    DUCK_ATTACK, AudioLayerConfig, Beats, build_layer, decode_audio,
    duck_envelope, gated_level_db, pick_music, plan_cues, speech_spans,
    write_layer,
)
from clipforge.audio.starter import SR, STARTER_TAGS, starter_sound
from clipforge.ingest.ingest import ingest
from clipforge.pipeline import PipelineConfig, run_pipeline
from clipforge.quality.report import load_report
from clipforge.reframe.detect import CropPath
from clipforge.render.render import measure_loudness, render_clip, verify_render
from clipforge.render.run import render_job
from clipforge.transcribe.schema import Word

REPO = Path(__file__).resolve().parents[1]
CFG = PipelineConfig(asr_engine="mock", highlight_engine="mock",
                     detector="haar")


# ---------------------------------------------------------------- helpers

def _tone(path: Path, freq: float, sec: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"sine=frequency={freq}:duration={sec}:sample_rate=48000",
                    str(path)], check=True)
    return path


def _write_lib(root: Path, items, raw: str | None = None, bom=False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    text = raw if raw is not None else json.dumps(
        {"schema_version": 1, "items": items})
    (root / "library.json").write_text(("\ufeff" if bom else "") + text,
                                       encoding="utf-8", newline="\n")
    return root


def _item(file="music/a.wav", kind="music", **kw):
    d = {"file": file, "kind": kind, "tags": ["gaming"], "license": "CC0",
         "source": "https://example.com/a"}
    d.update(kw)
    return d


def _music_lib(root: Path, freq=1500.0, sec=10.0, **kw) -> Path:
    _tone(root / "music" / "a.wav", freq, sec)
    return _write_lib(root, [_item(**kw)])


def _sine(freq: float, sec: float, amp=0.3) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    m = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([m, m], axis=1)


def _band_db(x: np.ndarray, f: float, width=15.0) -> float:
    mono = x.mean(axis=1).astype(np.float64)
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2
    freqs = np.fft.rfftfreq(len(mono), 1 / SR)
    return 10 * math.log10(spec[(freqs > f - width) & (freqs < f + width)]
                           .sum() + 1e-30)


def _rms_db(x: np.ndarray) -> float:
    return 10 * math.log10(np.mean(np.square(x.astype(np.float64))) + 1e-30)


def _seg(x, a, b):
    return x[int(a * SR):int(b * SR)]


# ---------------------------------------------------------------- render

@pytest.fixture(scope="module")
def tone_video(tmp_path_factory):
    """12 s black 640x360 video whose audio is a 440 Hz tone."""
    p = tmp_path_factory.mktemp("av") / "tone.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=640x360:r=30:d=12",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(p)], check=True, capture_output=True)
    return p


def test_original_stays_the_main_sound(tone_video, tmp_path):
    lib = L.load_library(_music_lib(tmp_path / "lib"))
    cfg = AudioLayerConfig(sfx=False, music=True)
    orig = decode_audio(tone_video, 0.0, 6.0)
    layer, rep = build_layer(orig, 6.0, Beats(6.0, 6.0), [], cfg, lib, "k")
    assert layer is not None and rep["music"]["applied"]
    lp = tmp_path / "l.f32"
    write_layer(lp, layer)
    out = tmp_path / "out.mp4"
    render_clip(tone_video, out, 0.0, 6.0, CropPath(clip_id=1, mode="center"),
                640, 360, 30.0, audio_layer=lp)
    verify_render(out, 6.0)
    y = _seg(decode_audio(out), 1.0, 4.0)           # clear of the fades
    diff = _band_db(y, 1500) - _band_db(y, 440)
    assert diff == pytest.approx(cfg.music_level_db, abs=3.0)
    assert measure_loudness(out) == pytest.approx(-14.0, abs=1.0)

    plain = tmp_path / "plain.mp4"                  # 440 Hz barely moves
    render_clip(tone_video, plain, 0.0, 6.0,
                CropPath(clip_id=1, mode="center"), 640, 360, 30.0)
    yp = _seg(decode_audio(plain), 1.0, 4.0)
    assert _band_db(y, 440) == pytest.approx(_band_db(yp, 440), abs=1.0)
    assert _band_db(yp, 1500) < _band_db(yp, 440) - 40


def test_layer_covers_end_card(tone_video, tmp_path):
    cfg = AudioLayerConfig(sfx=True, music=False)
    orig = decode_audio(tone_video, 0.0, 6.0)
    beats = Beats(6.0, 8.0, end_card_t=6.0)
    layer, rep = build_layer(orig, 8.0, beats, [], cfg, L.Library(tmp_path),
                             "k")
    assert len(layer) == 8 * SR
    assert [c["event"] for c in rep["sound_effects"]["cues"]] == ["end_card"]
    lp = tmp_path / "l.f32"
    write_layer(lp, layer)
    out = tmp_path / "card.mp4"
    render_clip(tone_video, out, 0.0, 6.0, CropPath(clip_id=1, mode="center"),
                640, 360, 30.0, end_card_sec=2.0, audio_layer=lp)
    verify_render(out, 8.0)


# ---------------------------------------------------------------- music

def test_music_ducks_while_someone_talks(tmp_path):
    lib = L.load_library(_music_lib(tmp_path / "lib", freq=1000, sec=3.0))
    cfg = AudioLayerConfig(sfx=False, music=True, music_duck_db=-6.0)
    spans = [(2.0, 4.0)]
    layer, rep = build_layer(_sine(440, 8.0), 8.0, Beats(8.0, 8.0), spans,
                             cfg, lib, "k")
    talk = _rms_db(_seg(layer, 2.6, 3.6))
    quiet = _rms_db(_seg(layer, 5.0, 6.0))
    assert talk - quiet == pytest.approx(-6.0, abs=1.0)
    assert rep["music"]["speech_spans"] == 1
    # the 3 s track was looped to fill the clip: no silent hole at the end
    assert _rms_db(_seg(layer, 5.5, 6.0)) > quiet - 1.0


def test_duck_envelope_ramps_instead_of_steps():
    n = 4 * SR
    env = duck_envelope(n, [(1.0, 2.0)], -6.0)
    low = 10 ** (-6 / 20)
    assert env[0] == 1.0 and env[int(1.5 * SR)] == pytest.approx(low)
    step = (1 - low) / (DUCK_ATTACK * SR)
    assert np.max(np.abs(np.diff(env))) <= step * 1.01
    assert duck_envelope(n, [], -6.0).min() == 1.0


def test_speech_spans_merge_short_gaps():
    words = [Word(w="a", start=10.0, end=10.3), Word(w="b", start=10.5,
             end=10.8), Word(w="c", start=13.0, end=13.4),
             Word(w="out", start=30.0, end=30.5)]
    spans = speech_spans(words, 10.0, 20.0)
    assert len(spans) == 2
    assert spans[0][0] == 0.0 and spans[0][1] == pytest.approx(0.95)
    assert spans[1][0] == pytest.approx(2.95)


def test_music_level_is_capped_below_the_original(monkeypatch):
    monkeypatch.setenv("CLIPFORGE_MUSIC_LEVEL_DB", "3")
    monkeypatch.setenv("CLIPFORGE_SFX_LEVEL_DB", "12")
    monkeypatch.setenv("CLIPFORGE_MUSIC_DUCK_DB", "nan")
    monkeypatch.setenv("CLIPFORGE_SFX", "off")
    monkeypatch.setenv("CLIPFORGE_MUSIC_TAGS", " Chill , ,lofi")
    c = AudioLayerConfig.from_env()
    assert c.music_level_db == -6.0 and c.sfx_level_db == 0.0
    assert c.music_duck_db == -6.0 and c.sfx is False and c.music is True
    assert c.music_tags == ["chill", "lofi"]
    monkeypatch.setenv("CLIPFORGE_MUSIC_LEVEL_DB", "junk")
    assert AudioLayerConfig.from_env().music_level_db == -20.0


def test_music_must_be_cleared_for_every_main_platform(tmp_path):
    root = tmp_path / "lib"
    _tone(root / "music" / "a.wav", 1000, 2)
    _tone(root / "music" / "b.wav", 1000, 2)
    _write_lib(root, [_item("music/a.wav", platforms=["shorts"])])
    item, why = pick_music(L.load_library(root), ["gaming"], "k")
    assert item is None and "tiktok, reels and shorts" in why
    _write_lib(root, [_item("music/a.wav", platforms=["shorts"]),
                      _item("music/b.wav",
                            platforms=["tiktok", "reels", "shorts"])])
    item, _ = pick_music(L.load_library(root), ["gaming"], "k")
    assert item.file == "music/b.wav"


def test_music_prefers_tags_and_is_deterministic(tmp_path):
    root = tmp_path / "lib"
    for n in "abc":
        _tone(root / "music" / f"{n}.wav", 1000, 1)
    _write_lib(root, [_item("music/a.wav", tags=["chill"]),
                      _item("music/b.wav", tags=["gaming"]),
                      _item("music/c.wav", tags=["gaming", "hype"])])
    lib = L.load_library(root)
    picks = {pick_music(lib, ["gaming"], f"job:{i}")[0].file
             for i in range(20)}
    assert picks <= {"music/b.wav", "music/c.wav"} and len(picks) == 2
    assert (pick_music(lib, ["gaming"], "x")[0].file
            == pick_music(lib, ["gaming"], "x")[0].file)
    assert pick_music(lib, ["none"], "x")[0] is not None   # any cleared


# ---------------------------------------------------------------- cues

def _tags(groups):
    return [[(c.tag, c.event) for c in g] for g in groups]


def test_cues_land_on_the_clips_beats():
    groups = plan_cues(Beats(20.0, 22.0, reaction_peak_t=6.0,
                             zoom_applied=True, overlay_t=0.0,
                             end_card_t=20.0))
    assert _tags(groups) == [
        [("pop", "context_overlay")],
        [("riser", "build_up"), ("whoosh", "punch_in"),
         ("impact", "reaction_peak")],
        [("whoosh", "end_card")]]
    peak = groups[1]
    assert peak[-1].t == 6.0 and peak[-1].align == "onset"
    assert peak[1].align == "peak" and peak[1].t + peak[1].offset < 6.0
    assert peak[0].align == "end"


def test_cue_spacing_cap_and_options():
    # overlay pop too close to an early peak: the peak wins
    g = plan_cues(Beats(20.0, 20.0, reaction_peak_t=1.0, overlay_t=0.0))
    assert _tags(g) == [[("impact", "reaction_peak")]]   # no zoom, no riser
    # 7 s clip: one group at most
    g = plan_cues(Beats(7.0, 7.0, reaction_peak_t=4.0, overlay_t=0.0,
                        zoom_applied=True))
    assert len(g) == 1 and g[0][-1].event == "reaction_peak"
    assert plan_cues(Beats(10.0, 10.0)) == []


def test_impact_onset_lands_on_the_reaction_peak(tmp_path):
    cfg = AudioLayerConfig(sfx=True, music=False)
    layer, rep = build_layer(_sine(440, 8.0), 8.0,
                             Beats(8.0, 8.0, reaction_peak_t=2.5),
                             [], cfg, L.Library(tmp_path), "k")
    env = np.max(np.abs(layer), axis=1)
    onset = np.nonzero(env >= 0.1 * env.max())[0][0] / SR
    assert onset == pytest.approx(2.5, abs=0.005)
    cue = rep["sound_effects"]["cues"][0]
    assert cue["file"] == "starter:impact"
    assert cue["license"] == "generated by ClipForge"
    # level: loudest 100 ms of the effect sits sfx_level_db under the clip
    win = _seg(layer, 2.5, 2.6)
    assert _rms_db(win) - gated_level_db(_sine(440, 8.0)) == pytest.approx(
        cfg.sfx_level_db, abs=1.5)


def test_library_effect_beats_the_starter(tmp_path):
    root = tmp_path / "lib"
    _tone(root / "sfx" / "boom.wav", 80, 0.5)
    _write_lib(root, [_item("sfx/boom.wav", kind="sfx", tags=["impact"],
                            attribution="Boom by someone (CC-BY 4.0)")])
    layer, rep = build_layer(_sine(440, 6.0), 6.0,
                             Beats(6.0, 6.0, reaction_peak_t=2.0), [],
                             AudioLayerConfig(music=False),
                             L.load_library(root), "k")
    cue = rep["sound_effects"]["cues"][0]
    assert cue["file"] == "sfx/boom.wav" and cue["license"] == "CC0"
    assert rep["credits"] == ["Boom by someone (CC-BY 4.0)"]


def test_silent_original_gets_nothing(tmp_path):
    layer, rep = build_layer(np.zeros((6 * SR, 2), np.float32), 6.0,
                             Beats(6.0, 6.0, reaction_peak_t=2.0), [],
                             AudioLayerConfig(), L.Library(tmp_path), "k")
    assert layer is None and not rep["applied"]
    assert "silent" in rep["sound_effects"]["reason"]


def test_starter_sounds_are_generated_not_committed():
    for tag in STARTER_TAGS:
        a, b = starter_sound(tag), starter_sound(tag)
        assert a.shape[1] == 2 and a.dtype == np.float32 and len(a) > SR / 20
        assert np.array_equal(a, b) and 0.5 < np.abs(a).max() <= 0.91
    files = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                           text=True, encoding="utf-8").stdout.split()
    assert not [f for f in files if Path(f).suffix.lower() in L.AUDIO_EXTS]


# ---------------------------------------------------------------- library

def test_library_loads_and_reads_a_bom(tmp_path):
    root = tmp_path / "lib"
    _tone(root / "sfx" / "a.wav", 300, 0.2)
    _write_lib(root, [_item("sfx\\a.wav", kind="sfx", tags=["Impact"],
                            platforms=["ALL"], gain_db=-3, start=0)],
               bom=True)
    lib = L.load_library(root)
    assert lib.found and not lib.problems
    it = lib.items[0]
    assert it.file == "sfx/a.wav" and it.tags == ["impact"]
    assert it.platforms == ["all"] and it.gain_db == -3.0 and it.path.exists()


@pytest.mark.parametrize("change,field", [
    ({"license": ""}, "license"),
    ({"license": None}, "license"),
    ({"source": "  "}, "source"),
    ({"kind": "voice"}, "kind"),
    ({"tags": []}, "tags"),
    ({"tags": ["ok", "bad tag!"]}, "tags[1]"),
    ({"gain_db": True}, "gain_db"),
    ({"gain_db": 40}, "gain_db"),
    ({"start": -1}, "start"),
    ({"platforms": ["myspace"]}, "platforms[0]"),
    ({"file": "/etc/passwd.wav"}, "file"),
    ({"file": "C:\\sounds\\a.wav"}, "file"),
    ({"file": "../outside.wav"}, "file"),
    ({"file": "music/missing.wav"}, "file"),
    ({"file": "music/notes.txt"}, "file"),
    ({"surprise": 1}, "surprise"),
])
def test_bad_items_are_skipped_with_file_and_field(tmp_path, change, field):
    root = tmp_path / "lib"
    _tone(root / "music" / "a.wav", 300, 0.2)
    (root / "music" / "notes.txt").write_text("x", encoding="utf-8")
    _tone(tmp_path / "outside.wav", 300, 0.2)
    good = _item("music/a.wav")
    bad = {**_item("music/a.wav"), **change}
    _write_lib(root, [good, bad])
    lib = L.load_library(root)
    assert [i.file for i in lib.items] == ["music/a.wav"]
    assert len(lib.problems) == 1
    assert lib.problems[0].startswith(f"library.json: items[1].{field}")
    assert str(tmp_path) not in lib.problems[0]


@pytest.mark.parametrize("raw,word", [
    ('{"schema_version": 1, "items": [{"gain_db": NaN}]}', "NaN"),
    ("{not json", "JSON"),
    ("[]", "object"),
    ('{"schema_version": 2, "items": []}', "schema_version"),
    ('{"schema_version": true, "items": []}', "schema_version"),
    ('{"schema_version": 1, "items": {}}', "items"),
    pytest.param('{"schema_version": 1, "items": [' + "[" * 100_000
                 + "]" * 100_000 + "]}", "library.json", id="deep-nesting"),
])
def test_broken_library_file_is_reported_not_raised(tmp_path, raw, word):
    lib = L.load_library(_write_lib(tmp_path / "lib", None, raw=raw))
    assert lib.found and not lib.items
    assert lib.problems and word in lib.problems[0]


def test_duplicate_items_and_missing_library(tmp_path):
    root = tmp_path / "lib"
    _tone(root / "music" / "a.wav", 300, 0.2)
    _write_lib(root, [_item(), _item("music\\a.wav")])
    lib = L.load_library(root)
    assert len(lib.items) == 1 and "listed twice" in lib.problems[0]
    none = L.load_library(tmp_path / "nothing-here")
    assert not none.found and not none.items and not none.problems


def test_add_file_copies_and_records(tmp_path):
    root = tmp_path / "lib"
    src = _tone(tmp_path / "My Boom!.wav", 80, 0.3)
    it = L.add_file(src, "sfx", ["impact"], "CC0", "https://kenney.nl",
                    root=root)
    it2 = L.add_file(src, "sfx", ["impact"], "CC0", "https://kenney.nl",
                     root=root)
    assert it.file == "sfx/My_Boom.wav" and it2.file == "sfx/My_Boom_2.wav"
    lib = L.load_library(root)
    assert len(lib.items) == 2 and not lib.problems
    assert b"\r\n" not in (root / "library.json").read_bytes()
    with pytest.raises(ValueError, match="license"):
        L.add_file(src, "sfx", ["impact"], "", "x", root=root)
    with pytest.raises(ValueError, match="not an audio file"):
        L.add_file(tmp_path / "lib" / "library.json", "sfx", ["a"], "CC0",
                   "x", root=root)
    assert len(L.load_library(root).items) == 2     # failures added nothing


def _cli(monkeypatch, *argv):
    import clipforge.cli as cli
    monkeypatch.setattr(cli, "load_env", lambda: None)
    monkeypatch.setattr(sys, "argv", ["clipforge", *argv])
    return cli.main()


def test_cli_audio_add_and_check(tmp_path, monkeypatch, capsys):
    root = tmp_path / "lib"
    src = _tone(tmp_path / "beat.wav", 1000, 1.0)
    assert _cli(monkeypatch, "audio", "check", "--library", str(root)) == 0
    assert "No library.json" in capsys.readouterr().out
    assert _cli(monkeypatch, "audio", "add", str(src), "--kind", "music",
                "--tags", "gaming,hype", "--license", "Pixabay Content License",
                "--source", "https://pixabay.com/music/x",
                "--platforms", "tiktok,reels,shorts",
                "--library", str(root)) == 0
    assert "added music/beat.wav" in capsys.readouterr().out
    assert _cli(monkeypatch, "audio", "check", "--library", str(root)) == 0
    assert "1 music tracks usable" in capsys.readouterr().out
    assert _cli(monkeypatch, "audio", "add", str(src), "--kind", "sfx",
                "--tags", "x", "--license", "CC0", "--source", "s",
                "--platforms", "myspace", "--library", str(root)) == 1
    assert "platforms" in capsys.readouterr().err
    data = json.loads((root / "library.json").read_text(encoding="utf-8"))
    data["items"].append({"file": "music/gone.wav"})
    (root / "library.json").write_text(json.dumps(data), encoding="utf-8")
    assert _cli(monkeypatch, "audio", "check", "--library", str(root)) == 1
    assert "problem: library.json: items[1]" in capsys.readouterr().out


# ---------------------------------------------------------------- pipeline

@pytest.fixture(scope="module")
def job(tmp_path_factory):
    """Mock pipeline job (40 s; the mock highlighter gives every clip a
    reaction peak and a context line)."""
    ws = tmp_path_factory.mktemp("aws")
    src = ws / "long.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "testsrc=duration=40:size=640x360:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=40",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(src)], check=True, capture_output=True)
    ingest(src, ws, job_id="a1")
    run_pipeline(ws / "a1", CFG, progress=lambda s: None)
    return ws / "a1"


def _reports(job_dir):
    ids = [int(p.stem.split("_")[1])
           for p in sorted((job_dir / "renders").glob("clip_???.mp4"))]
    assert ids
    return [load_report(job_dir, i) for i in ids]


def test_pipeline_adds_starter_effects_by_default(job):
    for rep in _reports(job):
        al = rep.audio_layer
        assert al["applied"] and al["reference_db"] < 0
        events = {c["event"] for c in al["sound_effects"]["cues"]}
        assert "reaction_peak" in events
        assert not al["music"]["applied"]
        assert al["music"]["reason"] == "no music in the audio library"
        va = {v["name"]: v for v in rep.value_adds}
        assert set(va) == {"context_overlay", "punch_in", "end_card",
                           "captions"}
    assert not list((job / "renders").glob("*.layer.f32"))


def test_pipeline_music_from_library_with_credits(job, tmp_path, monkeypatch):
    root = _music_lib(tmp_path / "lib", attribution="Track by Artist (CC-BY)")
    monkeypatch.setenv("CLIPFORGE_AUDIO_LIBRARY", str(root))
    render_job(job, force=True, contact_sheets=False)
    for rep in _reports(job):
        mus = rep.audio_layer["music"]
        assert mus["applied"] and mus["file"] == "music/a.wav"
        assert mus["license"] == "CC0" and mus["source"]
        assert rep.audio_layer["credits"] == ["Track by Artist (CC-BY)"]


def test_pipeline_switches_and_shorts_only_music(job, tmp_path, monkeypatch):
    root = _music_lib(tmp_path / "lib", platforms=["shorts"])
    monkeypatch.setenv("CLIPFORGE_AUDIO_LIBRARY", str(root))
    monkeypatch.setenv("CLIPFORGE_SFX", "0")
    render_job(job, force=True, contact_sheets=False)
    for rep in _reports(job):
        al = rep.audio_layer
        assert not al["applied"]
        assert al["sound_effects"]["reason"] == "disabled by CLIPFORGE_SFX"
        assert "tiktok, reels and shorts" in al["music"]["reason"]
    monkeypatch.setenv("CLIPFORGE_MUSIC", "0")
    render_job(job, force=True, contact_sheets=False)
    for rep in _reports(job):
        assert "CLIPFORGE_SFX and CLIPFORGE_MUSIC" in rep.audio_layer["reason"]


def test_mixing_failure_keeps_the_visual_extras(job, monkeypatch):
    import clipforge.render.run as run
    real = run.render_clip

    def flaky(*a, **kw):
        if kw.get("audio_layer") is not None:
            raise run.RenderError("boom /secret/path")
        return real(*a, **kw)

    monkeypatch.setattr(run, "render_clip", flaky)
    render_job(job, force=True, contact_sheets=False)
    for rep in _reports(job):
        assert not rep.audio_layer["applied"]
        assert rep.audio_layer["reason"].startswith("mixing failed")
        assert "/" not in rep.audio_layer["reason"]
        va = {v["name"]: v for v in rep.value_adds}
        assert va["punch_in"]["applied"]
    assert not list((job / "renders").glob("*.layer.f32"))


def test_layer_build_failure_never_fails_the_render(job, monkeypatch):
    import clipforge.render.run as run

    def boom(*a, **k):
        raise ValueError("/secret")

    monkeypatch.setattr(run, "_plan_audio", boom)
    render_job(job, force=True, contact_sheets=False)
    for rep in _reports(job):
        assert rep.audio_layer == {"applied": False,
                                   "reason": "could not be built"}
