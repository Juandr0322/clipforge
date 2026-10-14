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
    CandidateSet, ClipCandidate, RejectedCandidate, dedup,
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
    """Records messages.create kwargs; replies come from a list (last repeats)."""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=r)])


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
