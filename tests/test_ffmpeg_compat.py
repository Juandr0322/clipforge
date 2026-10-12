"""
ffmpeg version compatibility. The suite runs in CI against the distro ffmpeg
(6.1) and a current build (8.x); these tests pin the things that differ
between releases:

  - options removed from newer ffmpeg (-vsync is gone in 8.x) must not come
    back; every ffmpeg command line we build is checked for known-removed
    options
  - the output we parse (ffprobe JSON, loudnorm JSON in stderr) is parsed
    correctly from the formats both versions print, and from real runs of
    whichever ffmpeg is first on PATH
"""
import ast
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import clipforge
from clipforge.ingest import ingest as ingest_mod
from clipforge.ingest.ingest import parse_probe, probe
from clipforge.render import render as render_mod

ROOT = Path(clipforge.__file__).parent
FIXTURE = Path(__file__).parent / "fixtures" / "speech_30s.mp4"
needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg/ffprobe not installed")

# options current ffmpeg rejects ("Unrecognized option") or deprecates
REMOVED_OPTIONS = {"-vsync", "-async", "-vol", "-psnr", "-top", "-dc",
                   "-vc", "-sameq", "-same_quant"}


def test_no_removed_ffmpeg_options_in_source():
    """No string literal anywhere in clipforge is a removed ffmpeg option."""
    found = []
    for py in ROOT.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and node.value in REMOVED_OPTIONS:
                found.append(f"{py.name}:{node.lineno}: {node.value}")
    assert not found, found


def test_ingest_normalise_uses_fps_mode(tmp_path, monkeypatch):
    cmds = []

    def fake_run(cmd, timeout=1800):
        cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ingest_mod, "_run", fake_run)
    pr = parse_probe({"streams": [
        {"codec_type": "video", "width": 1280, "height": 720,
         "avg_frame_rate": "30/1", "codec_name": "h264"},
        {"codec_type": "audio", "codec_name": "aac", "sample_rate": "44100",
         "channels": 2}], "format": {"duration": "30"}})
    ingest_mod.normalize(Path("in.mp4"), tmp_path / "out.mp4", pr)
    (cmd,) = cmds
    assert "-vsync" not in cmd
    assert cmd[cmd.index("-fps_mode") + 1] == "cfr"
    assert not REMOVED_OPTIONS & set(cmd)


# ---- parsing: canned output from both versions ---------------------------

LOUDNORM_JSON = '''{
\t"input_i" : "-21.85",
\t"input_tp" : "-17.80",
\t"input_lra" : "0.00",
\t"input_thresh" : "-31.85",
\t"output_i" : "-14.05",
\t"output_tp" : "-9.92",
\t"output_lra" : "0.00",
\t"output_thresh" : "-24.05",
\t"normalization_type" : "dynamic",
\t"target_offset" : "0.05"
}'''

# stderr of the loudnorm analysis pass, abridged from real runs of 6.1.1 and
# N-127388. They differ around the JSON (stats-line layout, where the
# "[Parsed_loudnorm_0 @ ...]" prefix lands, kB vs KiB in the summary); the JSON
# block itself is identical.
STDERR = {
    "6.1": ("frame=   82 fps=0.0 q=-0.0 size=N/A time=00:00:02.90 bitrate=N/A "
            "speed=45.1x    [out#0/null @ 0x561827a7df80] video:422kB "
            "audio:11250kB\n[Parsed_loudnorm_0 @ 0x561827b27340] \n"
            + LOUDNORM_JSON + "\n"),
    "8.x": ("frame=  894 fps=0.0 q=-0.0 size=N/A time=00:00:29.80 bitrate=N/A "
            "speed=59.6x elapsed=0:00:00.50    [Parsed_loudnorm_0 @ 0x7fc3] \n"
            + LOUDNORM_JSON + "\n[out#0/null @ 0x557c97] video:373KiB "
            "audio:11250KiB subtitle:0KiB\nframe=  900 fps=0.0 q=-0.0 "
            "Lsize=N/A time=00:00:30.00 speed=55.3x elapsed=0:00:00.54    \n"),
}


@pytest.mark.parametrize("version", STDERR)
def test_measure_loudness_parses_each_ffmpeg_stderr(monkeypatch, version):
    monkeypatch.setattr(
        render_mod.subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(
            cmd, 0, "", STDERR[version]))
    assert render_mod.measure_loudness(Path("x.mp4")) == -21.85


# `ffprobe -v error -show_streams -show_format -of json tests/fixtures/speech_30s.mp4`
# captured from ffprobe 6.1.1-3ubuntu5 and N-127388-g615b07170e-20261012, then
# trimmed to the fields the code reads. Both builds printed the same values
# for those fields, so the two entries are the same capture; the real-ffprobe
# test below is what re-checks this on whichever build runs the suite.
CAPTURED = json.loads('''{"streams": [{"index": 0, "codec_name": "h264", "codec_type": "video", "width": 640, "height": 360, "r_frame_rate": "30/1", "avg_frame_rate": "30/1"}, {"index": 1, "codec_name": "aac", "codec_type": "audio", "r_frame_rate": "0/0", "avg_frame_rate": "0/0", "sample_rate": "44100", "channels": 1}], "format": {"duration": "30.000000", "format_name": "mov,mp4,m4a,3gp,3g2,mj2", "nb_streams": 2}}''')
PROBE = {"6.1.1": CAPTURED, "N-127388": CAPTURED}


@pytest.mark.parametrize("version", PROBE)
def test_parse_probe_reads_each_ffprobe_json(version):
    pr = parse_probe(PROBE[version])
    assert (pr.width, pr.height, pr.fps, pr.duration) == (640, 360, 30.0, 30.0)
    assert (pr.video_codec, pr.audio_codec) == ("h264", "aac")
    assert (pr.audio_sample_rate, pr.audio_channels) == (44100, 1)


# ---- parsing: the real ffmpeg on PATH ------------------------------------

@needs_ffmpeg
def test_real_ffprobe_output_parses():
    pr = parse_probe(probe(FIXTURE))
    assert (pr.width, pr.height) == (640, 360)
    assert pr.fps == pytest.approx(30.0)
    assert pr.duration == pytest.approx(30.0, abs=0.2)
    assert pr.audio_codec and pr.audio_channels == 1


@needs_ffmpeg
def test_real_loudnorm_output_parses():
    lufs = render_mod.measure_loudness(FIXTURE)
    assert -30 < lufs < -10


@needs_ffmpeg
def test_real_ffmpeg_accepts_fps_mode_cfr(tmp_path):
    out = tmp_path / "o.mp4"
    p = subprocess.run(
        ["ffmpeg", "-y", "-i", str(FIXTURE), "-t", "1", "-an",
         "-r", "24", "-fps_mode", "cfr", str(out)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert p.returncode == 0, p.stderr[-300:]
    assert parse_probe(probe(out)).fps == pytest.approx(24.0)


@needs_ffmpeg
def test_real_ffmpeg_error_text_is_surfaced(tmp_path):
    """A failing ffmpeg run raises with its stderr tail, on every version."""
    bad = tmp_path / "junk.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ingest_mod.IngestError) as e:
        ingest_mod._run(["ffmpeg", "-y", "-i", str(bad), str(tmp_path / "x.mp4")])
    assert "Command failed (ffmpeg)" in str(e.value) and len(str(e.value)) > 40
