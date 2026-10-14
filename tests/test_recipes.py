"""F31: recipe files (versioned, validated, recorded on each job)."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import clipforge.render.run as render_run
from clipforge.audio.layer import (MUSIC_MAX_DB, SFX_MAX_DB, AudioLayerConfig,
                                   decode_audio)
from clipforge.audio.starter import SR
from clipforge.common.manifest import Manifest
from clipforge.ingest.ingest import ingest
from clipforge.pipeline import PipelineConfig, run_pipeline
from clipforge.quality.report import load_report
from clipforge.recipes import (RECIPES_DIR, RecipeError, load_all, load_recipe,
                               load_recipe_file, read_record, recipe_names,
                               recipe_problem)
from clipforge.pipeline import RECIPE_FAILED

NAMES = {"talk", "gaming", "storytime", "talking_head", "cinematic",
         "day_in_life"}
MOCK = dict(asr_engine="mock", highlight_engine="mock", detector="haar")
AUDIO_ENV = ("CLIPFORGE_SFX", "CLIPFORGE_MUSIC", "CLIPFORGE_MUSIC_TAGS",
             "CLIPFORGE_SFX_LEVEL_DB", "CLIPFORGE_MUSIC_LEVEL_DB",
             "CLIPFORGE_MUSIC_DUCK_DB")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in AUDIO_ENV + ("CLIPFORGE_RECIPE",):
        monkeypatch.delenv(k, raising=False)


def _base(name="x", **kw):
    d = {"name": name, "version": 1, "description": "d", "min_score": 55,
         "max_candidates": 12, "captions": True}
    d.update(kw)
    return d


def _write(tmp_path, data, name="x", raw=None, bom=False):
    p = tmp_path / f"{name}.json"
    text = raw if raw is not None else json.dumps(data)
    p.write_text(("﻿" if bom else "") + text, encoding="utf-8",
                 newline="\n")
    return p


# ---------------------------------------------------------------- files

def test_all_six_recipes_exist_and_load():
    assert set(recipe_names()) == NAMES
    recipes = load_all()
    for name, r in recipes.items():
        assert r.name == name and isinstance(r.version, int) and r.version >= 1
        assert (RECIPES_DIR / f"{name}.json").is_file()


def test_gaming_carries_the_f48_audio_defaults():
    a = load_recipe("gaming").audio
    assert a["sfx_level_db"] == -6 and a["music_level_db"] == -20
    assert a["music_duck_db"] == -6
    assert a["music_level_db"] <= MUSIC_MAX_DB == -6


@pytest.mark.parametrize("data, field", [
    ({"name": "y"}, "name"),
    ({"version": 0}, "version"),
    ({"version": True}, "version"),
    ({"version": 1.5}, "version"),
    ({"version": "1"}, "version"),
    ({"min_score": float("nan")}, "min_score"),
    ({"min_score": 101}, "min_score"),
    ({"min_score": 55.5}, "min_score"),
    ({"max_candidates": 0}, "max_candidates"),
    ({"max_candidates": True}, "max_candidates"),
    ({"captions": "yes"}, "captions"),
    ({"description": " "}, "description"),
    ({"audio": []}, "audio"),
    ({"audio": {"volume": 1}}, "audio.volume"),
    ({"audio": {"sfx": 1}}, "audio.sfx"),
    ({"audio": {"music_duck_db": float("inf")}}, "audio.music_duck_db"),
    ({"audio": {"music_level_db": True}}, "audio.music_level_db"),
    ({"audio": {"music_tags": []}}, "audio.music_tags"),
    ({"audio": {"music_tags": [3]}}, "audio.music_tags"),
])
def test_bad_value_names_file_and_field(tmp_path, data, field):
    p = _write(tmp_path, _base(**data))
    with pytest.raises(RecipeError) as e:
        load_recipe_file(p)
    assert str(e.value).startswith("x.json: ") and f"'{field}'" in str(e.value)


@pytest.mark.parametrize("key", ["name", "version", "description", "min_score",
                                 "max_candidates", "captions"])
def test_missing_field_is_named(tmp_path, key):
    d = _base()
    del d[key]
    with pytest.raises(RecipeError, match=f"x.json: field '{key}' is missing"):
        load_recipe_file(_write(tmp_path, d))


@pytest.mark.parametrize("raw", ["{nope", "[]", "", "[" * 100000])
def test_unreadable_or_wrong_shape(tmp_path, raw):
    with pytest.raises(RecipeError, match="x.json"):
        load_recipe_file(_write(tmp_path, None, raw=raw))


def test_nan_literal_in_json_is_rejected(tmp_path):
    raw = json.dumps(_base()).replace('"min_score": 55', '"min_score": NaN')
    with pytest.raises(RecipeError, match="'min_score'"):
        load_recipe_file(_write(tmp_path, None, raw=raw))


def test_bom_file_loads_and_error_has_no_path(tmp_path):
    assert load_recipe_file(_write(tmp_path, _base(), bom=True)).name == "x"
    with pytest.raises(RecipeError) as e:
        load_recipe_file(tmp_path / "missing.json")
    assert str(tmp_path) not in str(e.value)


def test_unknown_recipe_lists_valid_names():
    with pytest.raises(RecipeError, match="Unknown recipe 'nope'.*talk"):
        load_recipe("nope")
    with pytest.raises(RecipeError):
        load_recipe("../talk")
    with pytest.raises(RecipeError):
        load_recipe(None)


# ------------------------------------------------- audio caps, env override

@pytest.mark.parametrize("key, bad", [
    ("music_level_db", -5.9), ("music_level_db", 0), ("music_level_db", 6),
    ("sfx_level_db", 0.5), ("sfx_level_db", 6),
    ("music_duck_db", 3),
])
def test_recipe_cannot_exceed_the_audio_caps(tmp_path, key, bad):
    p = _write(tmp_path, _base(audio={key: bad}))
    with pytest.raises(RecipeError) as e:
        load_recipe_file(p)
    assert str(e.value).startswith("x.json: ") and f"'audio.{key}'" in str(e.value)


def test_caps_themselves_are_allowed(tmp_path):
    r = load_recipe_file(_write(tmp_path, _base(
        audio={"music_level_db": MUSIC_MAX_DB, "sfx_level_db": SFX_MAX_DB})))
    assert r.audio["music_level_db"] == MUSIC_MAX_DB


def test_gaming_audio_config_matches_the_f48_defaults():
    cfg = load_recipe("gaming").audio_config()
    ref = AudioLayerConfig.from_env()
    assert (cfg.sfx_level_db, cfg.music_level_db, cfg.music_duck_db) == (
        ref.sfx_level_db, ref.music_level_db, ref.music_duck_db) == (
        -6.0, -20.0, -6.0)
    assert cfg.sfx and cfg.music and cfg.music_tags == ["gaming"]


def test_recipe_values_feed_the_settings_and_env_wins(tmp_path, monkeypatch):
    r = load_recipe_file(_write(tmp_path, _base(audio={
        "sfx": False, "music": False, "sfx_level_db": -12,
        "music_level_db": -30, "music_duck_db": -9,
        "music_tags": ["Chill"]})))
    cfg = r.audio_config()
    assert (cfg.sfx, cfg.music) == (False, False)
    assert (cfg.sfx_level_db, cfg.music_level_db, cfg.music_duck_db) == (
        -12, -30, -9)
    assert cfg.music_tags == ["chill"]
    monkeypatch.setenv("CLIPFORGE_SFX", "1")
    monkeypatch.setenv("CLIPFORGE_MUSIC_LEVEL_DB", "-25")
    monkeypatch.setenv("CLIPFORGE_MUSIC_TAGS", "epic")
    cfg = r.audio_config()
    assert cfg.sfx is True and cfg.music is False
    assert cfg.music_level_db == -25 and cfg.sfx_level_db == -12
    assert cfg.music_tags == ["epic"]
    # the env var is still clamped by the F48 caps
    monkeypatch.setenv("CLIPFORGE_MUSIC_LEVEL_DB", "0")
    assert r.audio_config().music_level_db == MUSIC_MAX_DB


@pytest.mark.parametrize("env, junk", [
    ("CLIPFORGE_MUSIC_LEVEL_DB", ""), ("CLIPFORGE_MUSIC_LEVEL_DB", "loud"),
    ("CLIPFORGE_SFX_LEVEL_DB", "nan"), ("CLIPFORGE_MUSIC_DUCK_DB", " "),
])
def test_empty_or_junk_env_keeps_the_recipe_value(monkeypatch, env, junk):
    monkeypatch.setenv(env, junk)
    cfg = load_recipe("gaming").audio_config()
    assert (cfg.sfx_level_db, cfg.music_level_db, cfg.music_duck_db) == (
        -6.0, -20.0, -6.0)


def test_junk_flag_and_empty_tags_keep_the_recipe_value(tmp_path, monkeypatch):
    r = load_recipe_file(_write(tmp_path, _base(
        audio={"sfx": False, "music_tags": ["chill"]})))
    monkeypatch.setenv("CLIPFORGE_SFX", "maybe")
    monkeypatch.setenv("CLIPFORGE_MUSIC_TAGS", " , ")
    cfg = r.audio_config()
    assert cfg.sfx is False and cfg.music_tags == ["chill"]


def test_recipe_without_audio_leaves_render_on_env_settings():
    assert load_recipe("talk").audio_config() is None


# ---------------------------------------------------------------- pipeline

def _make_job(ws: Path, job_id: str) -> Path:
    src = ws / f"{job_id}.mp4"
    if not src.exists():
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc=duration=40:size=640x360:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=300:duration=40",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(src)],
            check=True, capture_output=True)
    ingest(src, ws, job_id=job_id)
    return ws / job_id


@pytest.fixture(scope="module")
def jobs(tmp_path_factory):
    """Same video run three ways: default recipe, the old hard-coded values,
    and the gaming recipe. Render calls are captured for the first two."""
    ws = tmp_path_factory.mktemp("recipes")
    mp = pytest.MonkeyPatch()
    seen: dict = {}
    real = render_run.render_job

    def spy(job_dir, *a, **kw):
        seen[Path(job_dir).name] = kw
        return real(job_dir, *a, **kw)

    mp.setattr(render_run, "render_job", spy)
    out = {}
    for jid, cfg in (
            ("t", PipelineConfig(**MOCK)),
            ("old", PipelineConfig(recipe="gaming", min_score=55,
                                   max_candidates=12, captions=True, **MOCK)),
            ("g", PipelineConfig(recipe="gaming", **MOCK))):
        job = _make_job(ws, jid)
        run_pipeline(job, cfg, progress=lambda s: None)
        out[jid] = job
    mp.undo()
    out["seen"] = seen
    return out


def test_job_records_recipe_name_and_version(jobs):
    assert read_record(jobs["t"]) == {"name": "talk",
                                      "version": load_recipe("talk").version}
    assert read_record(jobs["g"]) == {"name": "gaming",
                                      "version": load_recipe("gaming").version}


def test_recorded_recipe_sticks_on_rerun(tmp_path):
    from clipforge.pipeline import _job_recipe
    job = _make_job(tmp_path, "k")
    out = []
    m = Manifest.load(job)
    _job_recipe(job, m, PipelineConfig(**MOCK), out.append)
    r = _job_recipe(job, m, PipelineConfig(recipe="gaming", **MOCK), out.append)
    assert r.name == "talk"
    assert read_record(job)["name"] == "talk"
    assert any("already uses recipe talk" in s for s in out)


def test_unknown_recipe_marks_the_job_failed_with_a_fixed_sentence(tmp_path):
    # Corrected: the first version of this test expected the job to stay
    # untouched, which left it looking stuck at "ingested" with no reason.
    job = _make_job(tmp_path, "job9")
    with pytest.raises(RecipeError):
        run_pipeline(job, PipelineConfig(recipe="nope", **MOCK),
                     progress=lambda s: None)
    assert read_record(job) is None
    m = Manifest.load(job)
    assert m.is_failed and m.stage.value == "ingested"
    assert m.error == RECIPE_FAILED
    assert "nope" not in m.error and "/" not in m.error and "\\" not in m.error
    assert m.job_id not in m.error and str(tmp_path) not in m.error
    assert m.history[-1].error == RECIPE_FAILED and not m.history[-1].ok


def test_server_shows_a_recipe_failure_without_paths(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from clipforge.server.app import create_app
    job = _make_job(tmp_path, "s1")
    monkeypatch.setenv("CLIPFORGE_RECIPE", "nope")
    app = create_app(tmp_path, config=PipelineConfig(**MOCK),
                     sync_pipeline=True)
    c = TestClient(app, raise_server_exceptions=False)
    assert c.post("/api/jobs/s1/rerun").status_code == 200
    d = c.get("/api/jobs/s1").json()
    assert d["failed"] and d["error"] == RECIPE_FAILED
    for bad in ("/", "\\", str(tmp_path), "s1", "nope", ".json"):
        assert bad not in d["error"]
    listed = [j for j in c.get("/api/jobs").json() if j["job_id"] == "s1"][0]
    assert listed["failed"] and listed["error"] == RECIPE_FAILED
    assert job.is_dir()


def test_recipe_problem_checks_the_env_var(monkeypatch):
    assert recipe_problem() is None
    monkeypatch.setenv("CLIPFORGE_RECIPE", "gaming")
    assert recipe_problem() is None
    monkeypatch.setenv("CLIPFORGE_RECIPE", "nope")
    assert "Unknown recipe 'nope'" in recipe_problem()


def test_cli_pipeline_exits_with_a_short_message(tmp_path, monkeypatch, capsys):
    from clipforge import cli
    monkeypatch.setenv("CLIPFORGE_RECIPE", "nope")
    monkeypatch.setattr(cli, "ffmpeg_problem", lambda: None)
    monkeypatch.setattr(cli, "report_ffmpeg", lambda: None)
    monkeypatch.setattr("sys.argv", ["clipforge", "pipeline", "j1",
                                     "--workspace", str(tmp_path)])
    rc = cli.main()
    assert rc == 2
    err = capsys.readouterr().err
    assert "PIPELINE FAILED" in err and "Traceback" not in err


def _recipes_copy(tmp_path, monkeypatch, version):
    import shutil
    import clipforge.recipes as R
    d = tmp_path / "recipes"
    shutil.copytree(RECIPES_DIR, d)
    p = d / "talk.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data["version"] = version
    p.write_text(json.dumps(data), encoding="utf-8", newline="\n")
    monkeypatch.setattr(R, "RECIPES_DIR", d)


def test_recorded_version_is_updated_when_the_recipe_changes(tmp_path, monkeypatch):
    from clipforge.pipeline import _job_recipe
    job = _make_job(tmp_path, "v")
    msgs = []
    _recipes_copy(tmp_path, monkeypatch, 1)
    m = Manifest.load(job)
    _job_recipe(job, m, PipelineConfig(**MOCK), msgs.append)
    assert read_record(job) == {"name": "talk", "version": 1}
    _recipes_copy_over = tmp_path / "recipes" / "talk.json"
    data = json.loads(_recipes_copy_over.read_text(encoding="utf-8"))
    data["version"] = 2
    _recipes_copy_over.write_text(json.dumps(data), encoding="utf-8")
    r = _job_recipe(job, m, PipelineConfig(**MOCK), msgs.append)
    assert r.version == 2 and read_record(job)["version"] == 2
    assert any("version 1 to 2" in s for s in msgs)


def test_finished_job_without_a_record_is_not_given_one(jobs, tmp_path, monkeypatch):
    import shutil
    legacy = tmp_path / "legacy"
    shutil.copytree(jobs["t"], legacy)
    (legacy / "recipe.json").unlink()
    monkeypatch.setenv("CLIPFORGE_RECIPE", "nope")   # would fail a real run
    m = run_pipeline(legacy, PipelineConfig(**MOCK), progress=lambda s: None)
    assert m.stage.value == "rendered" and not m.is_failed
    assert read_record(legacy) is None


def test_finished_job_survives_a_broken_recipe_file(jobs, tmp_path, monkeypatch):
    import shutil
    import clipforge.recipes as R
    done = tmp_path / "done"
    shutil.copytree(jobs["t"], done)
    bad = tmp_path / "badrecipes"
    bad.mkdir()
    (bad / "talk.json").write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(R, "RECIPES_DIR", bad)
    m = run_pipeline(done, PipelineConfig(**MOCK), progress=lambda s: None)
    assert not m.is_failed


def test_write_record_uses_unique_temp_and_retrying_replace(tmp_path, monkeypatch):
    import clipforge.recipes as R
    seen = []
    real = R._replace_with_retry
    monkeypatch.setattr(R, "_replace_with_retry",
                        lambda t, d: (seen.append(t.name), real(t, d)))
    R.write_record(tmp_path, load_recipe("talk"))
    R.write_record(tmp_path, load_recipe("talk"))
    assert len(seen) == 2 and seen[0] != seen[1]
    assert read_record(tmp_path)["name"] == "talk"
    assert not list(tmp_path.glob("*.tmp"))


def test_talk_reproduces_the_old_candidates(jobs):
    """Explicit 55/12/captions equals the old PipelineConfig defaults; talk
    must produce the same candidates, refined clips and render arguments."""
    for rel in ("candidates.json", "refined.json"):
        a = json.loads((jobs["t"] / rel).read_text(encoding="utf-8"))
        b = json.loads((jobs["old"] / rel).read_text(encoding="utf-8"))
        a.pop("job_id", None), b.pop("job_id", None)
        assert a == b, rel


def test_talk_render_parameters_are_the_old_ones(jobs):
    kw = jobs["seen"]["t"]
    assert kw["captions"] is True
    assert kw["audio"] is None       # render reads the env exactly as before
    r = load_recipe("talk")
    assert (r.min_score, r.max_candidates, r.captions) == (55, 12, True)


def test_pipeline_config_defaults_come_from_the_recipe_not_code():
    cfg = PipelineConfig(**MOCK)
    assert cfg.min_score is None and cfg.max_candidates is None
    assert cfg.captions is None


def test_gaming_job_keeps_the_original_audio_under_the_layer(jobs):
    kw = jobs["seen"]["g"]
    assert kw["audio"].sfx_level_db == -6 and kw["audio"].music_level_db == -20
    assert kw["audio"].music_level_db <= MUSIC_MAX_DB
    clips = sorted((jobs["g"] / "renders").glob("clip_???.mp4"))
    assert clips
    for clip in clips:
        rep = load_report(jobs["g"], int(clip.stem.split("_")[1]))
        al = rep.audio_layer
        assert al["applied"]
        assert al["sound_effects"]["level_db"] == -6
        assert al["music"]["level_db"] == -20 and al["music"]["duck_db"] == -6
        x = decode_audio(clip, 0.0, 6.0)
        mono = x.mean(axis=1).astype(np.float64)
        spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2
        freqs = np.fft.rfftfreq(len(mono), 1 / SR)
        band = spec[(freqs > 285) & (freqs < 315)].sum()
        # the 300 Hz source tone is still the dominant energy: not replaced
        assert band > 0.5 * spec.sum()
