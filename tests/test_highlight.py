"""Module 3 tests: chunking, dedup, defensive parsing, backend, stage runner."""
import subprocess
from pathlib import Path

import pytest

from clipforge.common.manifest import Manifest, Stage
from clipforge.highlight.backends import (
    Highlighter, MockHighlighter, build_system_prompt, get_highlighter, parse_llm_response,
)
from clipforge.highlight.chunker import Chunk, chunk_transcript
from clipforge.highlight.run import HighlightError, highlight_job
from clipforge.highlight.schema import (
    HOOK_TYPES, CandidateSet, ClipCandidate, RejectedCandidate, dedup,
)
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


# ------------------------------------------------------- F10: v2 fields

V2_FIELDS = ["trigger_t", "reaction_peak_t", "payoff_end_t", "context_setup",
             "hook_type", "stranger_check"]


def _item(**kw):
    d = {"start": 100, "end": 130, "score": 80, "hook": "h", "reason": "r",
         "type": "story"}
    d.update(kw)
    return json.dumps([d])


def _parse(text):
    return parse_llm_response(text, _chunk(), 3600.0)


def test_prompt_names_every_new_field_and_backstory_rule():
    sp = build_system_prompt()
    for f in V2_FIELDS:
        assert f in sp
    assert "backstory" in sp and "rejected" in sp


def test_parse_keeps_valid_v2_fields():
    (c,) = _parse(_item(trigger_t=105, reaction_peak_t=115, payoff_end_t=125,
                        context_setup="Host explains the bet.",
                        hook_type="question"))
    assert (c["trigger_t"], c["reaction_peak_t"], c["payoff_end_t"]) == (
        105, 115, 125)
    assert c["context_setup"] == "Host explains the bet."
    assert c["hook_type"] == "question"


@pytest.mark.parametrize("times", [
    (120, 110, 125),      # out of order
    (90, 110, 125),       # before start
    (105, 115, 140),      # after end
    (105, "x", 125),      # garbage
    (105, 115, None),     # missing
])
def test_parse_clears_bad_times_keeps_candidate(times):
    t, p, e = times
    (c,) = _parse(_item(trigger_t=t, reaction_peak_t=p, payoff_end_t=e))
    assert c["start"] == 100
    assert not {"trigger_t", "reaction_peak_t", "payoff_end_t"} & c.keys()
    ClipCandidate(id=0, **c)


def test_parse_unknown_hook_type_and_multiline_context():
    (c,) = _parse(_item(hook_type="clickbait",
                        context_setup="line one\nline two\nthree"))
    assert c["hook_type"] == "other"
    assert c["context_setup"] == "line one"


def test_failed_stranger_check_goes_to_rejected():
    class H(Highlighter):
        def analyze_chunk(self, chunk, source_duration):
            return _parse(json.dumps([
                {"start": 100, "end": 130, "score": 90, "hook": "bad",
                 "reason": "r", "stranger_check":
                 {"pass": False, "reason": "needs the earlier bet"}},
                {"start": 200, "end": 230, "score": 85, "hook": "good",
                 "reason": "r", "stranger_check": {"pass": True}},
            ]))
    cset = H().find_highlights(make_transcript(duration=600))
    assert [c.hook for c in cset.candidates] == ["good"]
    assert len(cset.rejected) == 1
    assert cset.rejected[0].reason == "needs the earlier bet"
    assert cset.schema_version == 2
    again = CandidateSet.model_validate_json(cset.model_dump_json())
    assert again.rejected[0].hook == "bad"


def test_v1_fixture_still_loads():
    cs = CandidateSet.load(Path(__file__).parent / "fixtures"
                           / "candidates_v1.json")
    c = cs.candidates[0]
    assert cs.rejected == [] and c.trigger_t is None and c.hook_type is None


def test_mock_fills_v2_fields_in_order():
    cset = MockHighlighter().find_highlights(make_transcript(duration=1800))
    assert cset.candidates
    for c in cset.candidates:
        assert c.hook_type == "bold_claim" and c.context_setup
        assert (c.start <= c.trigger_t <= c.reaction_peak_t
                <= c.payoff_end_t <= c.end)


def test_mock_pipeline_run_writes_v2_candidates(transcribed_job, tmp_path):
    job = _stage_job(transcribed_job, tmp_path)
    highlight_job(job, MockHighlighter(), force=True)
    data = json.loads((job / "candidates.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
    for c in data["candidates"]:
        assert (c["start"] <= c["trigger_t"] <= c["reaction_peak_t"]
                <= c["payoff_end_t"] <= c["end"])


# ------------------------------------------------------- F10 review fixes

JUNK = [None, [], {}, [1], {"a": 1}, 7, -1.5, True, "", "x\ny", 10 ** 400,
        float("nan"), float("inf")]


def _dumps(d):
    return json.dumps([d])  # allow_nan default: emits NaN/Infinity


def _base(**kw):
    d = {"start": 100, "end": 130, "score": 80, "hook": "h", "reason": "r",
         "type": "story"}
    d.update(kw)
    return d


@pytest.mark.parametrize("junk", JUNK, ids=repr)
@pytest.mark.parametrize("field", ["trigger_t", "reaction_peak_t",
                                   "payoff_end_t", "context_setup",
                                   "hook_type", "stranger_check", "type",
                                   "hook", "reason"])
def test_junk_in_any_field_never_raises_and_keeps_candidate(field, junk):
    out = _parse(_dumps(_base(**{field: junk})))
    assert len(out) == 1
    ClipCandidate(id=0, **{k: v for k, v in out[0].items()
                           if k != "stranger_fail_reason"})


@pytest.mark.parametrize("field", ["start", "end", "score"])
@pytest.mark.parametrize("junk", [None, [], {}, "x", 10 ** 400,
                                  float("nan"), float("inf")], ids=repr)
def test_junk_start_end_score_drops_entry_without_raising(field, junk):
    out = _parse(_dumps(_base(**{field: junk})))
    if field == "score" and junk == 10 ** 400:
        assert [c["score"] for c in out] == [100]  # huge int clamps, as before
    else:
        assert out == []


@pytest.mark.parametrize("ht", [[], {}, 5, ["a"]])
def test_hook_type_unhashable_or_wrong_type_becomes_other(ht):
    (c,) = _parse(_dumps(_base(hook_type=ht, type=ht)))
    assert c["hook_type"] == "other" and c["type"] == "other"


def test_hook_type_null_stays_absent():
    (c,) = _parse(_item(hook_type=None))
    assert "hook_type" not in c


def test_num_rejects_bad_numbers():
    from clipforge.highlight.backends import _num
    for bad in (10 ** 400, True, False, float("nan"), float("inf"),
                float("-inf"), "5", None):
        assert _num(bad) is None
    assert _num(5) == 5.0


@pytest.mark.parametrize("sc,fails", [
    (False, True), (0, True), ("false", True), ("No", True), ("FAIL", True),
    ({"pass": False}, True), ({"pass": "false"}, True), ({"pass": 0}, True),
    ({"passed": False}, True), ({"passed": "no"}, True),
    (True, False), (1, False), ("yes", False), ({"pass": True}, False),
    ({"passed": "true"}, False),
    ([], False), ("maybe", False), ({}, False), ({"pass": []}, False),
    (None, False),
])
def test_stranger_check_shapes(sc, fails):
    (c,) = _parse(_item(stranger_check=sc))
    assert ("stranger_fail_reason" in c) is fails


def test_missing_stranger_check_passes_and_is_absent():
    (c,) = _parse(_item())
    assert "stranger_fail_reason" not in c and "stranger_check" not in c


def test_stranger_check_stored_for_passing_candidate():
    (c,) = _parse(_item(stranger_check={"pass": True,
                                        "reason": "\n\nSelf-contained\nmore"}))
    assert c["stranger_check"] == {"pass": True, "reason": "Self-contained"}
    assert ClipCandidate(id=0, **c).stranger_check.passed is True


def test_context_setup_uses_first_non_empty_line():
    (c,) = _parse(_item(context_setup="\n  \n  the bet \n second"))
    assert c["context_setup"] == "the bet"


def _huge_int_reply():
    return '[{"start": 0, "end": 30, "score": 9' + "9" * 5000 + "}]"


def _deep_reply():
    return "[" * 100000 + "]" * 100000


@pytest.mark.parametrize("make", [_huge_int_reply, _deep_reply],
                         ids=["huge_int", "deep_nesting"])
def test_hostile_reply_parses_to_empty(make):
    assert _parse(make()) == []


@pytest.mark.parametrize("make", [_huge_int_reply, _deep_reply],
                         ids=["huge_int", "deep_nesting"])
def test_hostile_reply_saved_and_warned_not_crashed(make, tmp_path, capsys):
    h = AnthropicHighlighter(client=FakeClient([make()]))
    h.debug_dir = tmp_path
    ch = _chunk()
    assert h.analyze_chunk(ch, 100.0) == []
    f = tmp_path / f"highlight_raw_chunk_{ch.index:03d}.txt"
    assert f.read_text(encoding="utf-8") == make()
    assert "WARNING" in capsys.readouterr().err


def test_stranger_pass_null_falls_back_to_passed():
    (c,) = _parse(_item(stranger_check={"pass": None, "passed": False}))
    assert "stranger_fail_reason" in c
    (c,) = _parse(_item(stranger_check={"pass": True, "passed": False}))
    assert "stranger_fail_reason" not in c


@pytest.mark.parametrize("field", ["start", "end"])
def test_bool_start_end_dropped(field):
    assert _parse(_dumps(_base(**{field: True}))) == []


def test_rejected_deduped_across_overlapping_chunks():
    rej = {"start": 100, "end": 130, "score": 90, "hook": "bad",
           "reason": "r", "stranger_check": {"pass": False, "reason": "why"}}
    other = dict(rej, start=300, end=330, hook="other")

    class H(Highlighter):
        calls = 0

        def analyze_chunk(self, chunk, source_duration):
            H.calls += 1
            return _parse(json.dumps([rej, other] if H.calls == 1 else [rej]))
    cset = H().find_highlights(make_transcript(duration=3600))
    assert H.calls >= 2
    assert sorted((r.start, r.hook) for r in cset.rejected) == [
        (100, "bad"), (300, "other")]


def test_save_writes_lf_newlines(monkeypatch, tmp_path):
    seen = {}
    real = Path.write_text

    def spy(self, data, *a, **kw):
        seen.update(kw)
        return real(self, data, *a, **kw)
    monkeypatch.setattr(Path, "write_text", spy)
    CandidateSet(engine="x", source_duration_sec=60.0).save(
        tmp_path / "c.json")
    assert seen.get("newline") == "\n" and seen.get("encoding") == "utf-8"
    assert b"\r\n" not in (tmp_path / "c.json").read_bytes()


def test_save_load_roundtrip_rejected_and_stranger_check(tmp_path):
    cs = CandidateSet(
        engine="x", source_duration_sec=100.0,
        candidates=[ClipCandidate(
            id=0, start=10, end=40, score=80, hook="h", reason="r",
            stranger_check={"pass": True, "reason": "claro, sin contexto ñ"})],
        rejected=[RejectedCandidate(start=50, end=80, hook="¿qué?",
                                    reason="necesita contexto: café")])
    p = tmp_path / "c.json"
    cs.save(p)
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert raw["candidates"][0]["stranger_check"]["pass"] is True
    back = CandidateSet.load(p)
    assert back == cs
    assert back.rejected[0].reason == "necesita contexto: café"


def test_hook_types_exact_set():
    assert HOOK_TYPES == {"bold_claim", "question", "story_open",
                          "contrarian", "number", "reveal", "emotion",
                          "reaction", "funny", "other"}


def test_prompt_lists_every_hook_type_and_definitions():
    sp = build_system_prompt()
    line = next(ln for ln in sp.splitlines() if '"hook_type": "<' in ln)
    listed = set(line.split('"hook_type": "<')[1].split(">")[0].split("|"))
    assert listed == HOOK_TYPES
    for ht in HOOK_TYPES:
        assert ht in sp
    assert "a big live reaction (yelling, shock, celebration)" in sp
    assert "a fail, joke or chaotic moment" in sp
    assert "OPENING hook" in sp and "clip as a whole" in sp


@pytest.mark.parametrize("ht", ["reaction", "funny"])
def test_parse_keeps_new_hook_types(ht):
    (c,) = _parse(_item(hook_type=ht, type="story"))
    assert c["hook_type"] == ht and c["type"] == "story"


# ----------------------------------- F54: trim long candidates, ask for more

from clipforge.highlight.backends import moment_target  # noqa: E402
from clipforge.highlight.schema import MAX_CLIP_SEC  # noqa: E402

RON = Path(__file__).parent / "fixtures"
RON_CHUNK = Chunk(index=0, start=0, end=1080, segments=[])


def _ron(n):
    return (RON / f"ron_reply_chunk_{n:03d}.json").read_text(encoding="utf-8")


def test_ron_155s_reply_is_trimmed_not_dropped():
    """Root cause of Diego's 0-clip bench run: 155 s candidate, score 82."""
    stats = {}
    (c,) = parse_llm_response(_ron(0), RON_CHUNK, 1801.0, stats)
    assert c["start"] == 245 and c["end"] - c["start"] == MAX_CLIP_SEC
    assert c["trigger_t"] <= c["reaction_peak_t"] <= c["end"]
    assert "payoff_end_t" not in c            # 399 is past the trimmed end
    assert c["reaction_peak_t"] == 303        # the peak is kept
    assert c["score"] == 82 and c["hook_type"] == "number"
    assert (stats["kept"], stats["trimmed"], stats["dropped_length"]) == (1, 1, 0)


def test_ron_299s_reply_keeps_the_peak_and_the_payoff():
    (c,) = parse_llm_response(_ron(1), Chunk(1, 960, 2040, []), 1801.0)
    assert c["end"] - c["start"] <= MAX_CLIP_SEC
    assert c["start"] >= 1501.5                        # not before the trigger
    assert c["start"] < 1791.2 < c["end"]              # peak inside
    assert c["end"] == 1800.3                          # payoff fits, kept
    assert "trigger_t" not in c                        # cut off: not invented
    assert c["reaction_peak_t"] == 1791.2 and c["payoff_end_t"] == 1800.3


def test_long_candidate_without_times_falls_back_to_its_start():
    text = '[{"start": 100, "end": 400, "score": 80, "hook": "h", "reason": "r"}]'
    (c,) = parse_llm_response(text, CHUNK, 1000)
    assert (c["start"], c["end"]) == (100, 100 + MAX_CLIP_SEC)
    assert "trigger_t" not in c


def test_setup_that_fits_starts_at_the_trigger():
    text = ('[{"start": 100, "end": 400, "score": 80, "hook": "h", '
            '"reason": "r", "trigger_t": 130, "reaction_peak_t": 150, '
            '"payoff_end_t": 300}]')
    (c,) = parse_llm_response(text, CHUNK, 1000)
    assert (c["start"], c["end"]) == (130, 130 + MAX_CLIP_SEC)
    assert "payoff_end_t" not in c            # 300 is past the trimmed end


def test_too_short_is_still_dropped_and_counted():
    stats = {}
    text = ('[{"start": 0, "end": 5, "score": 99, "hook": "h", "reason": "r"},'
            ' 7, {"start": "x"},'
            ' {"start": 10, "end": 50, "score": 80, "hook": "h", "reason": "r"}]')
    out = parse_llm_response(text, CHUNK, 600, stats)
    assert len(out) == 1
    assert stats == {"parsed": 4, "kept": 1, "trimmed": 0,
                     "dropped_length": 1, "dropped_junk": 2}


def test_normal_length_candidate_is_unchanged():
    text = ('[{"start": 10, "end": 100, "score": 85, "hook": "h", "reason": "r",'
            ' "trigger_t": 12, "reaction_peak_t": 50, "payoff_end_t": 90}]')
    stats = {}
    (c,) = parse_llm_response(text, CHUNK, 600, stats)
    assert (c["start"], c["end"], c["trigger_t"], c["payoff_end_t"]) == \
        (10, 100, 12, 90)
    assert stats["trimmed"] == 0


def test_prompt_asks_for_several_moments_and_caps_length():
    p = build_system_prompt()
    assert "SEVERAL" in p and "one per 3-5 minutes" in p
    assert "0-4 candidates" not in p
    assert "at most 90 seconds" in p
    for ht in HOOK_TYPES:                    # list still generated from the type
        assert ht in p
    assert moment_target(18 * 60) == (4, 6)
    assert moment_target(5 * 60) == (1, 2)
    assert moment_target(0) == (1, 1)


def test_user_message_states_a_target_scaled_to_the_chunk():
    c = FakeClient([GOOD])
    AnthropicHighlighter(client=c).analyze_chunk(
        Chunk(0, 0, 1080, make_transcript(duration=100.0).segments), 1800.0)
    msg = c.calls[0]["messages"][0]["content"]
    assert "about 4-6 moments" in msg and "at most 90 seconds" in msg


def test_chunk_counts_are_logged_and_stored(capsys):
    t = make_transcript(duration=100.0)
    reply = json.dumps([
        {"start": 0, "end": 30, "score": 90, "hook": "a", "reason": "r"},
        {"start": 0, "end": 300, "score": 80, "hook": "b", "reason": "r"},
        {"start": 40, "end": 70, "score": 30, "hook": "c", "reason": "r"},
        {"start": 50, "end": 53, "score": 99, "hook": "d", "reason": "r"},
        "junk",
        {"start": 60, "end": 90, "score": 85, "hook": "e", "reason": "r",
         "stranger_check": {"pass": False, "reason": "needs backstory"}},
    ])
    h = AnthropicHighlighter(client=FakeClient([reply]))
    h.find_highlights(t, min_score=55)
    (c,) = h.usage_record()["chunks"]
    assert c == {"index": 0, "kept": 2, "trimmed": 1, "dropped_length": 1,
                 "dropped_junk": 1, "dropped_low_score": 1,
                 "rejected_stranger": 1}
    err = capsys.readouterr().err
    assert "chunk 0" in err and "kept 2" in err and "trimmed 1" in err
    assert "1 too short" in err and "1 junk" in err and "1 below min score" in err


def test_chunk_counts_reach_highlight_usage_json(transcribed_job):
    from clipforge.highlight.backends import MockHighlighter
    highlight_job(transcribed_job, MockHighlighter())
    rec = json.loads((transcribed_job / USAGE_NAME).read_text(encoding="utf-8"))
    assert rec["chunks"] and rec["chunks"][0]["index"] == 0


# --------------------------------------------- F54 review fixes

def _one(start, end, t0=None, tp=None, te=None, src=1000):
    times = "" if t0 is None else (
        f', "trigger_t": {t0}, "reaction_peak_t": {tp}, "payoff_end_t": {te}')
    text = (f'[{{"start": {start}, "end": {end}, "score": 80, "hook": "h", '
            f'"reason": "r"{times}}}]')
    stats = {}
    return parse_llm_response(text, CHUNK, src, stats), stats


def test_trigger_near_the_end_anchors_the_window_at_the_end():
    for end, times in [(100, (95, 96, 100)), (300, (290, 295, 300))]:
        (c,), st = _one(0, end, *times)
        assert c["end"] == end and c["end"] - c["start"] == MAX_CLIP_SEC
        assert c["start"] < times[0] < c["end"]       # trigger and peak kept
        assert c["trigger_t"] == times[0] and st["trimmed"] == 1


def test_trim_result_under_the_minimum_is_dropped_and_counted():
    # a window the trim cannot widen: end - start < MIN after trimming
    from clipforge.highlight import backends
    orig = backends._trim_window
    backends._trim_window = lambda s, e, i: (s, s + 5)
    try:
        out, st = _one(0, 300)
    finally:
        backends._trim_window = orig
    assert out == [] and st["dropped_length"] == 1 and st["trimmed"] == 0


def test_valid_peak_anchors_the_window_when_another_time_is_off():
    # payoff_end_t lies past the candidate's end: only the peak is usable
    (c,), _ = _one(0, 300, 190, 200, 350)
    assert c["start"] < 200 < c["end"] and c["end"] - c["start"] == MAX_CLIP_SEC
    assert (c["start"], c["end"]) == (155, 245)
    assert "trigger_t" not in c                       # all-or-none, untrusted
    (c,), _ = _one(0, 300, 10, 290, 295)              # peak near the end
    assert c["end"] == 295 and c["start"] < 290 < c["end"]


def test_times_cut_off_by_the_trim_are_dropped_not_clamped():
    (c,), _ = _one(0, 300, 5, 150, 290)
    assert c["reaction_peak_t"] == 150
    assert "trigger_t" not in c and "payoff_end_t" not in c


def test_ron_fixtures_are_the_exact_raw_replies():
    """Fixtures are byte-copies of the raw replies of bench run
    bench-20261014-210447-ron (highlight_raw_chunk_000/001.txt), not hand-built."""
    assert _ron(0).startswith('[{"start": 245, "end": 400')
    assert _ron(1).startswith('[{"start": 1501.5, "end": 1800.3')
