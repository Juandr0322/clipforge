"""Module 3 tests: chunking, dedup, defensive parsing, backend, stage runner."""
import subprocess
from pathlib import Path

import pytest

from clipforge.common.manifest import Manifest, Stage
from clipforge.highlight.backends import (
    MockHighlighter, build_system_prompt, get_highlighter, parse_llm_response,
)
from clipforge.highlight.chunker import Chunk, chunk_transcript
from clipforge.highlight.run import HighlightError, highlight_job
from clipforge.highlight.schema import CandidateSet, ClipCandidate, dedup
from clipforge.ingest.ingest import ingest
from clipforge.transcribe.backends import MockTranscriber
from clipforge.transcribe.run import transcribe_job
from clipforge.transcribe.schema import Segment, Transcript, Word

FIX = Path(__file__).parent / "fixtures"


def make_transcript(duration=3600.0, seg_len=8.0, gap=1.0) -> Transcript:
    """Synthetic long transcript: uniform segments across `duration`."""
    segs, words, t, i = [], [], 0.0, 0
    texts = ["Nobody tells you this part.", "Here is a concrete example.",
             "Most people get it wrong.", "That honestly surprised me.",
             "Some filler talk goes here."]
    while t + seg_len < duration:
        txt = texts[i % len(texts)]
        segs.append(Segment(id=i, text=txt, start=round(t, 3),
                            end=round(t + seg_len, 3)))
        wdur = seg_len / len(txt.split())
        wt = t
        for tok in txt.split():
            words.append(Word(w=tok, start=round(wt, 3),
                              end=round(wt + wdur * 0.85, 3)))
            wt += wdur
        t += seg_len + gap
        i += 1
    return Transcript(engine="synthetic", audio_duration_sec=duration,
                      words=words, segments=segs)


# ------------------------------------------------------------------ chunker

def test_chunk_coverage_and_overlap():
    t = make_transcript(duration=3600)  # 1 hour
    chunks = chunk_transcript(t, chunk_sec=1080, overlap_sec=120)
    # every segment appears in at least one chunk
    covered = {s.id for c in chunks for s in c.segments}
    assert covered == {s.id for s in t.segments}
    # consecutive chunks overlap in time
    for a, b in zip(chunks, chunks[1:]):
        assert b.start < a.end
    # indices sequential
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_short_video_single_chunk():
    t = make_transcript(duration=300)  # 5 min
    chunks = chunk_transcript(t)
    assert len(chunks) == 1
    assert chunks[0].start == 0


def test_chunk_render_format():
    t = make_transcript(duration=60)
    line = chunk_transcript(t)[0].render().splitlines()[0]
    assert line.startswith("[0.0-8.0] S1:")


def test_bad_overlap_rejected():
    t = make_transcript(duration=600)
    with pytest.raises(ValueError):
        chunk_transcript(t, chunk_sec=100, overlap_sec=100)


# -------------------------------------------------------------------- dedup

def C(i, s, e, score):
    return ClipCandidate(id=i, start=s, end=e, score=score, hook="h", reason="r")


def test_dedup_keeps_higher_score():
    out = dedup([C(0, 10, 40, 70), C(1, 15, 45, 90)])
    assert len(out) == 1 and out[0].score == 90


def test_dedup_keeps_non_overlapping():
    out = dedup([C(0, 10, 40, 70), C(1, 100, 130, 60)])
    assert len(out) == 2


def test_dedup_partial_overlap_below_threshold_kept():
    # 10s overlap on 30s clips = 0.33 < 0.5 -> both kept
    out = dedup([C(0, 0, 30, 70), C(1, 20, 50, 80)])
    assert len(out) == 2


def test_dedup_reids_and_sorts():
    out = dedup([C(5, 0, 30, 60), C(9, 100, 130, 95)])
    assert [c.id for c in out] == [0, 1]
    assert out[0].score == 95  # sorted by score desc


# --------------------------------------------------------- defensive parsing

CHUNK = Chunk(index=0, start=0, end=600, segments=[])


def test_parse_clean_json():
    text = '[{"start": 10, "end": 50, "score": 85, "hook": "h", "reason": "r", "type": "story"}]'
    out = parse_llm_response(text, CHUNK, 600)
    assert len(out) == 1 and out[0]["type"] == "story"


def test_parse_with_markdown_fences_and_chatter():
    text = 'Sure! Here you go:\n```json\n[{"start": 10, "end": 50, "score": 85, "hook": "h", "reason": "r"}]\n```\nLet me know!'
    assert len(parse_llm_response(text, CHUNK, 600)) == 1


def test_parse_garbage_returns_empty():
    assert parse_llm_response("I could not find any clips, sorry!", CHUNK, 600) == []
    assert parse_llm_response("[not json at all", CHUNK, 600) == []


def test_parse_drops_invalid_entries_keeps_valid():
    text = ('[{"start": 10, "end": 50, "score": 85, "hook": "h", "reason": "r"},'
            ' {"start": "bad"},'
            ' {"start": 100, "end": 90, "score": 70, "hook": "h", "reason": "r"},'
            ' {"start": 0, "end": 5, "score": 99, "hook": "too short", "reason": "r"}]')
    out = parse_llm_response(text, CHUNK, 600)
    assert len(out) == 1 and out[0]["start"] == 10


def test_parse_clamps_to_source_duration():
    text = '[{"start": 580, "end": 900, "score": 85, "hook": "h", "reason": "r"}]'
    out = parse_llm_response(text, CHUNK, 600)
    assert out == [] or all(o["end"] <= 600 for o in out)


def test_parse_clamps_score_and_type():
    text = '[{"start": 10, "end": 50, "score": 150, "hook": "h", "reason": "r", "type": "viral"}]'
    out = parse_llm_response(text, CHUNK, 600)
    assert out[0]["score"] == 100 and out[0]["type"] == "other"


def test_system_prompt_includes_examples():
    sp = build_system_prompt("--- example: ep1 ---\nhook: xyz")
    assert "xyz" in sp and "calibration" in sp


# ------------------------------------------------------------- mock backend

def test_mock_backend_end_to_end():
    t = make_transcript(duration=1800)
    cset = MockHighlighter().find_highlights(t)
    assert cset.engine == "mock"
    assert 0 < len(cset.candidates) <= 12
    for c in cset.candidates:
        assert c.score >= 55
        assert c.end <= t.audio_duration_sec + 0.75
        assert c.duration >= 15 * 0.6
    # deduped: no pair overlaps > 50%
    for i, a in enumerate(cset.candidates):
        for b in cset.candidates[i + 1:]:
            assert a.overlap_fraction(b) <= 0.5


def test_mock_deterministic():
    t = make_transcript(duration=900)
    assert MockHighlighter().find_highlights(t) == \
        MockHighlighter().find_highlights(t)


# ------------------------------------------------------------- stage runner

@pytest.fixture(scope="module")
def transcribed_job(tmp_path_factory):
    src = FIX / "speech_30s.mp4"
    if not src.exists():
        FIX.mkdir(exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=30:size=640x360:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=300:duration=30",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(src)],
            check=True, capture_output=True)
    ws = tmp_path_factory.mktemp("ws")
    m = ingest(src, ws, job_id="t3job")
    transcribe_job(ws / m.job_id, MockTranscriber())
    return ws / m.job_id


def test_stage_runner_happy_path(transcribed_job):
    m = highlight_job(transcribed_job, MockHighlighter())
    assert m.stage == Stage.CANDIDATES_READY
    cset = CandidateSet.load(transcribed_job / m.candidates_path)
    assert isinstance(cset.candidates, list)


def test_stage_runner_requires_transcript(tmp_path):
    m = Manifest(job_id="fresh")
    m.save(tmp_path)
    with pytest.raises(HighlightError, match="transcribe first"):
        highlight_job(tmp_path, MockHighlighter())


def test_stage_runner_idempotent(transcribed_job):
    m1 = highlight_job(transcribed_job, MockHighlighter())
    m2 = highlight_job(transcribed_job, MockHighlighter())
    assert len(m2.history) == len(m1.history)


# ------------------------------------------------------------------ factory

def test_factory_mock():
    assert isinstance(get_highlighter("mock"), MockHighlighter)


def test_factory_anthropic_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        get_highlighter("anthropic")


# ------------------------------------- client, temperature, raw-reply debug

import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

from clipforge.highlight.backends import (  # noqa: E402
    RAW_REPLY_GLOB, AnthropicHighlighter,
)


class FakeClient:
    """Records messages.create kwargs; replies come from a list (last repeats).
    `usage`, when given, is attached to every reply like the SDK's Usage."""
    def __init__(self, replies, usage=None):
        self.replies, self.calls, self.usage = list(replies), [], usage
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=r)],
            usage=self.usage)


GOOD = json.dumps([{"start": 0, "end": 30, "score": 90, "hook": "h",
                    "reason": "r", "type": "story"}])


def _chunk():
    return chunk_transcript(make_transcript(duration=100.0))[0]


def _calls_for(monkeypatch, ws):
    if ws:
        monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", ws)
    else:
        monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    c = FakeClient([GOOD])
    AnthropicHighlighter(client=c).analyze_chunk(_chunk(), 100.0)
    return c.calls


@pytest.mark.parametrize("ws", ["ws-123", None])
def test_workspace_id_kwarg(monkeypatch, ws):
    (kw,) = _calls_for(monkeypatch, ws)
    if ws:
        assert kw["workspace_id"] == ws
    else:
        assert "workspace_id" not in kw


def test_client_built_without_workspace_header(monkeypatch):
    seen = {}
    mod = types.ModuleType("anthropic")
    mod.Anthropic = lambda **kw: seen.update(kw) or object()
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "ws-123")
    AnthropicHighlighter()
    assert seen == {}


def test_workspace_id_never_printed(monkeypatch, capsys):
    _calls_for(monkeypatch, "secret-ws-id")
    out = capsys.readouterr()
    assert "secret-ws-id" not in out.out + out.err


def test_no_temperature_kwarg(monkeypatch):
    # Replaces F23's temperature == 0 test: SDK 1.x removed the parameter.
    calls = _calls_for(monkeypatch, None) + _calls_for(monkeypatch, "ws-1")
    assert len(calls) == 2 and all("temperature" not in k for k in calls)


@pytest.mark.parametrize("ws", ["ws-123", None])
def test_kwargs_exist_in_real_sdk_signature(monkeypatch, ws):
    import inspect
    import anthropic  # real package: no skip, fails loudly if missing
    from anthropic.resources.messages import Messages
    params = inspect.signature(Messages.create).parameters
    (kw,) = _calls_for(monkeypatch, ws)
    missing = set(kw) - set(params)
    assert not missing, f"anthropic {anthropic.__version__} rejects: {missing}"


def test_requirements_pin_anthropic_1x():
    req = (Path(__file__).resolve().parent.parent / "requirements.txt"
           ).read_text(encoding="utf-8")
    lines = [ln.split("#")[0].strip() for ln in req.splitlines()]
    assert "anthropic>=1.13,<2" in lines


def test_empty_reply_saved_with_debug_dir(tmp_path, capsys):
    reply = "Désolé, pas de clips — 日本語 \U0001F600"
    h = AnthropicHighlighter(client=FakeClient([reply]))
    h.debug_dir = tmp_path
    ch = _chunk()
    assert h.analyze_chunk(ch, 100.0) == []
    f = tmp_path / f"highlight_raw_chunk_{ch.index:03d}.txt"
    assert f.read_bytes() == reply.encode("utf-8")
    err = capsys.readouterr().err
    assert "WARNING" in err and f.name in err and f"chunk {ch.index}" in err
    assert f"{ch.start:.0f}s-{ch.end:.0f}s" in err


def test_empty_reply_without_debug_dir_only_warns(tmp_path, capsys):
    h = AnthropicHighlighter(client=FakeClient(["nope"]))
    assert h.analyze_chunk(_chunk(), 100.0) == []
    assert "raw reply not saved" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_good_reply_saves_nothing(tmp_path, capsys):
    h = AnthropicHighlighter(client=FakeClient([GOOD]))
    h.debug_dir = tmp_path
    assert len(h.analyze_chunk(_chunk(), 100.0)) == 1
    assert not list(tmp_path.iterdir())
    assert capsys.readouterr().err == ""


def test_stage_keeps_debug_files_and_clears_stale(transcribed_job, tmp_path):
    job = tmp_path / "job"
    shutil.copytree(transcribed_job, job)
    m1 = highlight_job(job, AnthropicHighlighter(client=FakeClient(["garbage"])),
                       force=True)
    raws = sorted(job.glob(RAW_REPLY_GLOB))
    assert raws and m1.stage == Stage.CANDIDATES_READY
    assert raws[0].read_text(encoding="utf-8") == "garbage"
    assert Manifest.load(job).stage == Stage.CANDIDATES_READY
    # no-op rerun leaves them in place
    highlight_job(job, AnthropicHighlighter(client=FakeClient([GOOD])))
    assert sorted(job.glob(RAW_REPLY_GLOB)) == raws
    # forced run with a good reply removes stale debug files
    highlight_job(job, AnthropicHighlighter(client=FakeClient([GOOD])),
                  force=True)
    assert not list(job.glob(RAW_REPLY_GLOB))


def _stage_job(transcribed_job, tmp_path):
    job = tmp_path / "job"
    shutil.copytree(transcribed_job, job)
    return job


def test_debug_write_failure_does_not_fail_stage(
        transcribed_job, tmp_path, monkeypatch, capsys):
    job = _stage_job(transcribed_job, tmp_path)
    secret = str(tmp_path)

    real = Path.write_text

    def boom(self, *a, **k):
        if self.name.startswith("highlight_raw_chunk_"):
            raise PermissionError(13, "denied", str(self))
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "write_text", boom)
    h =AnthropicHighlighter(client=FakeClient(["garbage"]))
    m = highlight_job(job, h, force=True)
    monkeypatch.undo()
    assert m.stage == Stage.CANDIDATES_READY
    assert (job / "candidates.json").exists()
    assert not Manifest.load(job).history[-1].error
    err = capsys.readouterr().err
    assert "could not save" in err and secret not in err


def test_stale_unlink_failure_does_not_fail_stage(
        transcribed_job, tmp_path, monkeypatch, capsys):
    job = _stage_job(transcribed_job, tmp_path)
    (job / "highlight_raw_chunk_000.txt").write_text("old", encoding="utf-8")

    real = Path.unlink

    def boom(self, *a, **k):
        if self.name.startswith("highlight_raw_chunk_"):
            raise PermissionError(13, "denied", str(self))
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "unlink", boom)
    m = highlight_job(job, MockHighlighter(), force=True)
    monkeypatch.undo()
    assert m.stage == Stage.CANDIDATES_READY
    assert (job / "candidates.json").exists()
    assert not Manifest.load(job).history[-1].error
    err = capsys.readouterr().err
    assert "could not remove highlight_raw_chunk_000.txt" in err
    assert str(tmp_path) not in err


# ------------------------------------------------- token usage log (F25)

from clipforge.highlight.backends import USAGE_FIELDS, USAGE_NAME  # noqa: E402


def _usage(**kw):
    """A reply's usage like the SDK's: every field present, some None."""
    return types.SimpleNamespace(**{f: kw.get(f) for f in USAGE_FIELDS})


def test_usage_fields_match_real_sdk():
    from anthropic.types import Usage
    assert set(USAGE_FIELDS) <= set(Usage.model_fields)


def test_usage_summed_over_every_chunk():
    t = make_transcript(duration=3600.0)
    n = len(chunk_transcript(t))
    assert n > 1
    h = AnthropicHighlighter(client=FakeClient(
        [GOOD], usage=_usage(input_tokens=100, output_tokens=20,
                             cache_read_input_tokens=5)))
    h.find_highlights(t)
    rec = h.usage_record()
    assert rec["model"] == "claude-sonnet-4-6"
    assert rec["engine"] == "anthropic/claude-sonnet-4-6"
    assert rec["calls"] == n
    assert rec["input_tokens"] == 100 * n
    assert rec["output_tokens"] == 20 * n
    assert rec["cache_read_input_tokens"] == 5 * n
    assert rec["cache_creation_input_tokens"] == 0      # None counts as 0


def test_usage_resets_between_runs():
    t = make_transcript(duration=900.0)
    h = AnthropicHighlighter(client=FakeClient(
        [GOOD], usage=_usage(input_tokens=10)))
    h.find_highlights(t)
    first = h.usage_record()
    h.find_highlights(t)
    assert h.usage_record() == first


def test_reply_without_usage_counts_zero():
    h = AnthropicHighlighter(client=FakeClient([GOOD]))   # usage=None
    h.analyze_chunk(_chunk(), 100.0)
    rec = h.usage_record()
    assert all(rec[f] == 0 for f in USAGE_FIELDS)


def test_stage_writes_usage_file_for_mock(transcribed_job, tmp_path):
    job = _stage_job(transcribed_job, tmp_path)
    highlight_job(job, MockHighlighter(), force=True)
    rec = json.loads((job / USAGE_NAME).read_text(encoding="utf-8"))
    assert rec["engine"] == "mock" and rec["model"] is None
    assert rec["calls"] >= 1
    assert all(rec[f] == 0 for f in USAGE_FIELDS)


def test_forced_rerun_replaces_stale_usage(transcribed_job, tmp_path):
    job = _stage_job(transcribed_job, tmp_path)
    (job / USAGE_NAME).write_text('{"engine": "old"}', encoding="utf-8")
    highlight_job(job, MockHighlighter(), force=True)
    rec = json.loads((job / USAGE_NAME).read_text(encoding="utf-8"))
    assert rec["engine"] == "mock"


def test_usage_write_failure_does_not_fail_stage(
        transcribed_job, tmp_path, capsys):
    job = _stage_job(transcribed_job, tmp_path)
    (job / USAGE_NAME).unlink(missing_ok=True)
    (job / USAGE_NAME).mkdir()      # a directory: neither removable nor writable
    m = highlight_job(job, MockHighlighter(), force=True)
    assert m.stage == Stage.CANDIDATES_READY
    assert not Manifest.load(job).history[-1].error
    assert (job / "candidates.json").exists()
    assert not list(job.glob(f".{USAGE_NAME}.*.tmp"))
    err = capsys.readouterr().err
    assert f"could not write {USAGE_NAME}" in err
    assert str(tmp_path) not in err


def test_usage_kept_when_stage_fails(transcribed_job, tmp_path):
    class SpendsThenFails(MockHighlighter):
        def analyze_chunk(self, chunk, source_duration):
            self.add_usage(_usage(input_tokens=7, output_tokens=3))
            raise RuntimeError("model fell over")

    job = _stage_job(transcribed_job, tmp_path)
    with pytest.raises(RuntimeError):
        highlight_job(job, SpendsThenFails(), force=True)
    rec = json.loads((job / USAGE_NAME).read_text(encoding="utf-8"))
    assert rec["input_tokens"] == 7 and rec["output_tokens"] == 3
    assert Manifest.load(job).failed_stage == Stage.CANDIDATES_READY


def test_usage_records_one_hour_writes_and_geo():
    split = types.SimpleNamespace(ephemeral_1h_input_tokens=40,
                                  ephemeral_5m_input_tokens=60)
    usage = types.SimpleNamespace(
        input_tokens=1, output_tokens=1, cache_creation_input_tokens=100,
        cache_read_input_tokens=0, cache_creation=split, inference_geo="us")
    h = AnthropicHighlighter(client=FakeClient([GOOD], usage=usage))
    h.analyze_chunk(_chunk(), 100.0)
    h.analyze_chunk(_chunk(), 100.0)
    rec = h.usage_record()
    assert rec["cache_creation_input_tokens"] == 200
    assert rec["cache_write_1h_tokens"] == 80
    assert rec["inference_geo"] == ["us"]


def test_split_and_geo_fields_match_real_sdk():
    from anthropic.types import CacheCreation, Usage
    assert {"cache_creation", "inference_geo"} <= set(Usage.model_fields)
    assert "ephemeral_1h_input_tokens" in CacheCreation.model_fields


def test_failed_attempt_tokens_carried_into_retry(transcribed_job, tmp_path):
    class SpendsThenFails(MockHighlighter):
        def analyze_chunk(self, chunk, source_duration):
            self.add_usage(_usage(input_tokens=7, output_tokens=3))
            raise RuntimeError("model fell over")

    class Spends(MockHighlighter):
        def analyze_chunk(self, chunk, source_duration):
            self.add_usage(_usage(input_tokens=10, output_tokens=1))
            return super().analyze_chunk(chunk, source_duration)

    job = _stage_job(transcribed_job, tmp_path)
    with pytest.raises(RuntimeError):
        highlight_job(job, SpendsThenFails(), force=True)
    m = highlight_job(job, Spends(), force=True)       # the retry succeeds
    assert m.stage == Stage.CANDIDATES_READY and m.failed_stage is None
    rec = json.loads((job / USAGE_NAME).read_text(encoding="utf-8"))
    assert rec["input_tokens"] == 7 + 10 * rec["calls"] - 10   # 1 failed call
    assert rec["attempts"] == 2

    highlight_job(job, Spends(), force=True)   # forced redo after a success
    rec = json.loads((job / USAGE_NAME).read_text(encoding="utf-8"))
    assert rec["attempts"] == 1 and rec["input_tokens"] == 10 * rec["calls"]


@pytest.mark.parametrize("damage", [
    {"attempts": "x"}, {"inference_geo": 5}, {"input_tokens": "lots"},
    {"input_tokens": 1e400}, {"attempts": float("inf")},
])
def test_malformed_failed_attempt_usage_never_fails_retry(
        transcribed_job, tmp_path, capsys, damage):
    class SpendsThenFails(MockHighlighter):
        def analyze_chunk(self, chunk, source_duration):
            self.add_usage(_usage(input_tokens=7))
            raise RuntimeError("boom")

    job = _stage_job(transcribed_job, tmp_path)
    with pytest.raises(RuntimeError):
        highlight_job(job, SpendsThenFails(), force=True)
    path = job / USAGE_NAME
    rec = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**rec, **damage}), encoding="utf-8")

    with pytest.raises(RuntimeError):         # a second failure is recorded
        highlight_job(job, SpendsThenFails(), force=True)
    m = Manifest.load(job)
    assert len([h for h in m.history if not h.ok]) == 2
    path.write_text(json.dumps({**rec, **damage}), encoding="utf-8")

    m = highlight_job(job, MockHighlighter(), force=True)   # retry succeeds
    assert m.stage == Stage.CANDIDATES_READY and m.failed_stage is None
    assert json.loads(path.read_text(encoding="utf-8"))["attempts"] == 1
    assert "ignored a malformed" in capsys.readouterr().err


def test_unreadable_failed_attempt_usage_never_fails_retry(
        transcribed_job, tmp_path, capsys):
    class SpendsThenFails(MockHighlighter):
        def analyze_chunk(self, chunk, source_duration):
            raise RuntimeError("boom")

    job = _stage_job(transcribed_job, tmp_path)
    with pytest.raises(RuntimeError):
        highlight_job(job, SpendsThenFails(), force=True)
    # nested deeply enough to make the JSON parser give up
    (job / USAGE_NAME).write_text("[" * 200_000 + "]" * 200_000,
                                  encoding="utf-8")
    m = highlight_job(job, MockHighlighter(), force=True)
    assert m.stage == Stage.CANDIDATES_READY and m.failed_stage is None
    assert json.loads((job / USAGE_NAME).read_text(encoding="utf-8")
                      )["attempts"] == 1
    assert "ignored an unreadable" in capsys.readouterr().err
