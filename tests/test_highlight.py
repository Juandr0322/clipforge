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
