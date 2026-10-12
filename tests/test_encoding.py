"""
F21 - Windows compatibility: every text file ClipForge touches is UTF-8.

On Windows, Python's default text encoding is the ANSI code page (cp1252), so
`Path.read_text()` / `open()` with no `encoding=` decode UTF-8 files wrongly
(index.html holds byte 0x9D, which cp1252 cannot decode -> GET / returned 500).
Linux CI defaults to UTF-8 and hides this, so these tests:

  * scan the source with `ast` and fail on any text-mode call without an
    explicit UTF-8 encoding (the guard),
  * simulate Windows by making every default-encoding text open use cp1252
    (patched in-process) and, separately, by running in a child process with
    a real non-UTF-8 locale,
  * check that non-ASCII data round-trips and that .ass captions are UTF-8.
"""
import ast
import builtins
import io
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import clipforge
from clipforge.common.manifest import (
    IngestOutputs, Manifest, SourceInfo, Stage,
)
from clipforge.highlight.backends import load_examples
from clipforge.highlight.schema import CandidateSet, ClipCandidate
from clipforge.pipeline import PipelineConfig
from clipforge.refine.refine import RefinedClip, RefinedSet
from clipforge.reframe.detect import CropKeyframe, CropPath
from clipforge.render import run as render_run
from clipforge.render.run import render_job
from clipforge.review.store import (
    Review, ReviewSet, export_examples, load_reviews, save_reviews,
)
from clipforge.server.app import create_app
from clipforge.transcribe.schema import Segment, Transcript, Word

PKG = Path(clipforge.__file__).resolve().parent
ROOT = PKG.parent

# emoji, accented letters, curly quotes, an ellipsis and a dash
TEXT = "¡Qué día, señor Müller! “naïve” café … 😀 — ok"


# ------------------------------------------------------------ AST guard

UTF8_NAMES = {"utf-8", "utf8", "utf-8-sig"}
# receivers whose .open() is binary-only (wave.open(path, "rb") has no
# encoding parameter at all)
BINARY_ONLY_OPEN = {"wave"}


def _const(node):
    return node.value if isinstance(node, ast.Constant) else None


def _arg(call, pos, name):
    """Return the AST node for an argument given by position or keyword."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return call.args[pos] if len(call.args) > pos else None


def find_unsafe_text_io(source: str, filename: str = "<src>") -> list[str]:
    """Return 'file:line: what' for every text-mode read_text / write_text /
    open / io.open / Path.open call that lacks an explicit UTF-8 encoding."""
    problems = []
    for node in ast.walk(ast.parse(source, filename)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        # **kwargs could hide anything: treat as not explicit
        if isinstance(f, ast.Attribute):
            kind = f.attr
            recv = f.value.id if isinstance(f.value, ast.Name) else None
        elif isinstance(f, ast.Name):
            kind, recv = f.id, None
        else:
            continue

        if kind == "read_text":
            enc = _arg(node, 0, "encoding")
        elif kind == "write_text":
            enc = _arg(node, 1, "encoding")
        elif kind == "open" and isinstance(f, ast.Name):         # open(...)
            mode, enc = _arg(node, 1, "mode"), _arg(node, 3, "encoding")
        elif kind == "open" and recv == "io":                    # io.open(...)
            mode, enc = _arg(node, 1, "mode"), _arg(node, 3, "encoding")
        elif kind == "open" and recv in BINARY_ONLY_OPEN:
            continue
        elif kind == "open":                                     # Path.open
            mode, enc = _arg(node, 0, "mode"), _arg(node, 2, "encoding")
        else:
            continue

        if kind == "open":
            if mode is not None:
                m = _const(mode)
                if isinstance(m, str) and "b" in m:
                    continue                          # binary: no encoding
        if _const(enc) is None or str(_const(enc)).lower() not in UTF8_NAMES:
            problems.append(f"{filename}:{node.lineno}: {kind}() without "
                            f"explicit encoding='utf-8'")
    return problems


def test_guard_catches_what_it_should():
    bad = [
        "p.read_text()", "p.write_text(s)", "open(p)", "open(p, 'w')",
        "io.open(p, 'r')", "p.open()", "p.open('w')", "p.open('r', -1)",
        "p.read_text(encoding=enc)", "p.read_text(**kw)",
        "p.write_text(s, encoding='cp1252')", "open(p, 'rt')",
        "p.read_text(errors='ignore')",
    ]
    good = [
        "p.read_text(encoding='utf-8')", "p.write_text(s, encoding='utf-8')",
        "p.write_text(s, 'utf-8')", "p.read_text('utf-8')",
        "open(p, encoding='utf-8')", "open(p, 'rb')", "open(p, 'w', -1, 'utf-8')",
        "io.open(p, 'wb')", "p.open('rb')", "p.open('wb')", "p.open(mode='ab')",
        "p.open('w', encoding='utf-8')", "wave.open(str(p), 'rb')",
        "os.path.exists(p)", "x.read()",
    ]
    for src in bad:
        assert find_unsafe_text_io(src), f"guard missed: {src}"
    for src in good:
        assert not find_unsafe_text_io(src), f"guard false positive: {src}"


def test_every_text_open_in_clipforge_is_explicit_utf8():
    files = sorted(PKG.rglob("*.py"))
    assert len(files) > 20, "guard is scanning the wrong directory"
    problems = []
    for p in files:
        problems += find_unsafe_text_io(
            p.read_text(encoding="utf-8"), str(p.relative_to(ROOT)))
    assert not problems, "\n".join(problems)


# ------------------------------------------------ simulated Windows default

@pytest.fixture
def windows_defaults(monkeypatch):
    """Make every text open with no explicit encoding use cp1252, the way it
    does on a Windows PC. Covers Path.read_text/write_text/open (they go
    through io.open) and builtins.open."""
    real_open = io.open

    def cp1252_open(file, mode="r", buffering=-1, encoding=None, *a, **k):
        if "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, *a, **k)

    def text_encoding(encoding, stacklevel=2):
        # UTF-8 mode (always on under the C locale) would answer "utf-8"
        return "locale" if encoding is None else encoding

    monkeypatch.setattr(io, "text_encoding", text_encoding)
    monkeypatch.setattr(io, "open", cp1252_open)
    monkeypatch.setattr(builtins, "open", cp1252_open)


def test_simulation_really_uses_cp1252(windows_defaults, tmp_path):
    p = tmp_path / "x.txt"
    p.write_bytes("é".encode("utf-8"))
    assert p.read_text() == "Ã©"                    # mojibake, as on Windows
    with pytest.raises(UnicodeEncodeError):
        (tmp_path / "y.txt").write_text("😀")
    assert p.read_text(encoding="utf-8") == "é"


MOCK_CFG = PipelineConfig(asr_engine="mock", highlight_engine="mock",
                          detector="haar")


def test_index_served_under_cp1252_defaults(windows_defaults, tmp_path):
    page = (PKG / "server" / "static" / "index.html").read_bytes()
    assert b"\x9d" in page, "test premise: index.html holds byte 0x9D"
    expected = page.decode("utf-8")
    client = TestClient(create_app(tmp_path, config=MOCK_CFG,
                                   sync_pipeline=True),
                        raise_server_exceptions=False)
    r = client.get("/")
    assert r.status_code == 200
    assert r.content.decode("utf-8") == expected    # characters intact
    assert "·" in r.content.decode("utf-8")


CHILD = textwrap.dedent("""
    import locale, sys
    from pathlib import Path
    enc = locale.getpreferredencoding(False)
    if enc.lower().replace("-", "") in ("utf8", "utf_8"):
        print("SKIP default encoding is " + enc); sys.exit(3)
    from fastapi.testclient import TestClient
    from clipforge.pipeline import PipelineConfig
    from clipforge.server.app import create_app
    app = create_app(Path(sys.argv[1]), sync_pipeline=True,
                     config=PipelineConfig(asr_engine="mock",
                                           highlight_engine="mock",
                                           detector="haar"))
    r = TestClient(app, raise_server_exceptions=False).get("/")
    page = Path(sys.argv[2]).read_bytes().decode("utf-8")
    assert r.status_code == 200, r.status_code
    assert r.content.decode("utf-8") == page
    print("OK " + enc)
""")


def test_index_served_in_subprocess_with_non_utf8_locale(tmp_path):
    """No patching: a child with UTF-8 mode off and the C locale (Python's
    default text encoding is then ASCII, which also rejects 0x9D)."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONUTF8", "PYTHONIOENCODING", "LC_ALL", "LANG",
                        "LC_CTYPE", "PYTHONCOERCECLOCALE")}
    env.update(PYTHONUTF8="0", PYTHONCOERCECLOCALE="0", LC_ALL="C",
               PYTHONPATH=str(ROOT))
    page = PKG / "server" / "static" / "index.html"
    r = subprocess.run([sys.executable, "-c", CHILD, str(tmp_path), str(page)],
                       cwd=ROOT, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=120)
    if r.returncode == 3:
        pytest.skip("cannot force a non-UTF-8 default encoding here")
    assert r.returncode == 0, (r.stdout + r.stderr)[-1500:]
    assert r.stdout.startswith("OK")


# ------------------------------------------------------- round trips

def make_job(job_dir: Path, text: str = TEXT) -> Manifest:
    """A hand-built job at REFRAMED with non-ASCII transcript, hook, note."""
    words = [Word(w=w, start=0.5 + i * 0.4, end=0.8 + i * 0.4)
             for i, w in enumerate(["Café", "señor", "“hola”", "naïve",
                                    "😀", "über"])]
    t = Transcript(engine="mock", audio_duration_sec=10.0, words=words,
                   segments=[Segment(id=0, text=text, start=0.5, end=3.0)])
    t.save(job_dir / "transcript.json")
    cand = ClipCandidate(id=0, start=0.0, end=4.0, score=80, hook=text,
                         reason="raison d’être — 日本語")
    CandidateSet(engine="mock", source_duration_sec=10.0,
                 candidates=[cand]).save(job_dir / "candidates.json")
    clip = RefinedClip(id=0, candidate_id=0, start=0.0, end=4.0,
                       speech_start=0.5, speech_end=3.0, score=80, hook=text,
                       type="other", n_words=len(words), segment_ids=[0])
    RefinedSet(source_duration_sec=10.0, clips=[clip]).save(
        job_dir / "refined.json")
    (job_dir / "crops").mkdir(exist_ok=True)
    CropPath(clip_id=0, mode="center",
             keyframes=[CropKeyframe(t=0.0, cx=0.5)]).save(
        job_dir / "crops" / "clip_000.json")
    m = Manifest(
        job_id=job_dir.name, stage=Stage.REFRAMED,
        source=SourceInfo(original_filename="entrevista_señor_😀.mp4",
                          duration_sec=10.0, width=640, height=360, fps=30.0,
                          video_codec="h264", size_bytes=1, sha256="0"),
        ingest=IngestOutputs(normalized_video="n.mp4", audio_wav="a.wav",
                             proxy_video="p.mp4", normalized_fps=30.0,
                             normalized_width=640, normalized_height=360),
        transcript_path="transcript.json", candidates_path="candidates.json",
        refined_path="refined.json", crop_paths_dir="crops")
    m.save(job_dir)
    return m


def test_non_ascii_json_stores_round_trip(windows_defaults, tmp_path):
    job = tmp_path / "enc"
    job.mkdir()
    m = make_job(job)

    assert Manifest.load(job) == m
    t = Transcript.load(job / "transcript.json")
    assert t.segments[0].text == TEXT
    assert [w.w for w in t.words][:2] == ["Café", "señor"]
    assert CandidateSet.load(job / "candidates.json").candidates[0].hook == TEXT
    assert RefinedSet.load(job / "refined.json").clips[0].hook == TEXT
    # stored as real UTF-8, not mojibake
    assert TEXT.encode("utf-8") in (job / "transcript.json").read_bytes()

    note = "love the “cold open” — très bien 😀"
    save_reviews(job, ReviewSet(reviews=[
        Review(clip_id=0, decision="approved", note=note)]))
    assert load_reviews(job).get(0).note == note
    assert note.encode("utf-8") in (job / "reviews.json").read_bytes()


def test_export_examples_round_trips_non_ascii(windows_defaults, tmp_path):
    job = tmp_path / "enc"
    job.mkdir()
    make_job(job)
    note = "love the “cold open” — très bien 😀"
    save_reviews(job, ReviewSet(reviews=[
        Review(clip_id=0, decision="approved", note=note)]))

    written = export_examples(job, tmp_path / "examples")
    assert len(written) == 1
    raw = written[0].read_bytes().decode("utf-8")
    assert f"hook='{TEXT}'" in raw
    assert f"reviewer note: {note}" in raw
    assert "Café señor “hola” naïve 😀 über" in raw      # clip transcript
    assert "entrevista_señor_😀.mp4" in raw
    # and the few-shot loader reads it back unchanged
    loaded = load_examples(tmp_path / "examples")
    assert loaded.endswith(raw.strip())
    assert note in loaded and TEXT in loaded


def test_ass_subtitles_written_as_utf8(windows_defaults, tmp_path, monkeypatch):
    job = tmp_path / "enc"
    job.mkdir()
    make_job(job)
    # no ffmpeg needed: only the caption file matters here
    monkeypatch.setattr(render_run, "render_clip", lambda *a, **k: None)
    monkeypatch.setattr(render_run, "verify_render", lambda *a, **k: None)
    render_job(job, contact_sheets=False)

    raw = (job / "renders" / "clip_000.ass").read_bytes()
    text = raw.decode("utf-8")                       # must be valid UTF-8
    for word in ("CAFÉ", "SEÑOR", "“HOLA”", "NAÏVE", "ÜBER"):  # .ass is upper-cased
        assert word.encode("utf-8") in raw, word
        assert word in text
    assert "😀".encode("utf-8") in raw
