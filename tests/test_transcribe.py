"""Module 2 tests: schema invariants, mock backend, stage-runner behavior."""
import subprocess
from pathlib import Path

import pytest

from clipforge.common.manifest import Manifest, Stage
from clipforge.ingest.ingest import ingest
from clipforge.transcribe.backends import MockTranscriber, get_transcriber
from clipforge.transcribe.run import TranscribeError, transcribe_job, verify_transcript
from clipforge.transcribe.schema import Segment, Transcript, Word

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def ingested_job(tmp_path_factory):
    """One real ingested job (30s video) shared by the stage-runner tests."""
    src = FIX / "speech_30s.mp4"
    if not src.exists():
        FIX.mkdir(exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=30:size=640x360:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=300:duration=30",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(src),
        ], check=True, capture_output=True)
    ws = tmp_path_factory.mktemp("ws")
    m = ingest(src, ws, job_id="t2job")
    return ws / m.job_id


# ------------------------------------------------------------------ schema

def W(w, s, e):
    return Word(w=w, start=s, end=e)


def test_schema_rejects_unordered_words():
    with pytest.raises(ValueError, match="out of order"):
        Transcript(engine="x", audio_duration_sec=10,
                   words=[W("b", 5, 6), W("a", 1, 2)],
                   segments=[])


def test_schema_rejects_word_beyond_duration():
    with pytest.raises(ValueError, match="beyond audio"):
        Transcript(engine="x", audio_duration_sec=10,
                   words=[W("a", 9, 14)], segments=[])


def test_schema_rejects_negative_span():
    with pytest.raises(ValueError):
        Word(w="a", start=5.0, end=4.0)


def test_schema_rejects_duplicate_segment_ids():
    segs = [Segment(id=1, text="a", start=0, end=1),
            Segment(id=1, text="b", start=1, end=2)]
    with pytest.raises(ValueError, match="duplicate"):
        Transcript(engine="x", audio_duration_sec=10, words=[], segments=segs)


def test_words_between():
    t = Transcript(engine="x", audio_duration_sec=10,
                   words=[W("a", 0, 1), W("b", 2, 3), W("c", 8, 9)],
                   segments=[])
    assert [w.w for w in t.words_between(1.5, 4.0)] == ["b"]


def test_roundtrip(tmp_path):
    t = Transcript(engine="x", audio_duration_sec=5,
                   words=[W("hi", 0, 0.4)],
                   segments=[Segment(id=0, text="hi", start=0, end=0.4)])
    p = tmp_path / "t.json"
    t.save(p)
    assert Transcript.load(p) == t


# ------------------------------------------------------------- mock backend

def test_mock_spans_audio_and_validates(ingested_job):
    tr = MockTranscriber().transcribe(ingested_job / "audio.wav")
    assert tr.audio_duration_sec == pytest.approx(30, abs=0.5)
    assert tr.words and tr.segments
    # coverage: transcript should reach near the end of the audio
    assert tr.words[-1].end > tr.audio_duration_sec * 0.7
    # words fall inside their segments
    for seg in tr.segments:
        inside = tr.words_between(seg.start, seg.end + 0.01)
        assert inside, f"segment {seg.id} has no words"


def test_mock_is_deterministic(ingested_job):
    wav = ingested_job / "audio.wav"
    assert MockTranscriber().transcribe(wav) == MockTranscriber().transcribe(wav)


# ----------------------------------------------------------- verifications

def test_verify_rejects_empty():
    t = Transcript(engine="x", audio_duration_sec=60, words=[], segments=[])
    with pytest.raises(TranscribeError, match="zero words"):
        verify_transcript(t)


def test_verify_rejects_sparse():
    t = Transcript(engine="x", audio_duration_sec=600,
                   words=[W("a", 0, 1)], segments=[Segment(id=0, text="a", start=0, end=1)])
    with pytest.raises(TranscribeError, match="sparse"):
        verify_transcript(t)


# ------------------------------------------------------------ stage runner

def test_stage_runner_happy_path(ingested_job):
    m = transcribe_job(ingested_job, MockTranscriber())
    assert m.stage == Stage.TRANSCRIBED
    assert (ingested_job / m.transcript_path).exists()
    t = Transcript.load(ingested_job / m.transcript_path)
    assert t.engine == "mock"


def test_stage_runner_idempotent(ingested_job):
    m1 = transcribe_job(ingested_job, MockTranscriber())
    n_hist = len(m1.history)
    m2 = transcribe_job(ingested_job, MockTranscriber())
    assert len(m2.history) == n_hist  # no extra run recorded


def test_stage_runner_requires_ingest(tmp_path):
    m = Manifest(job_id="fresh")
    m.save(tmp_path)
    with pytest.raises(TranscribeError, match="run ingest first"):
        transcribe_job(tmp_path, MockTranscriber())


def test_failure_recorded_in_manifest(ingested_job):
    class Broken(MockTranscriber):
        def transcribe(self, wav_path):
            raise RuntimeError("engine exploded")
    stage_before = Manifest.load(ingested_job).stage
    with pytest.raises(RuntimeError):
        transcribe_job(ingested_job, Broken(), force=True)
    m = Manifest.load(ingested_job)
    assert m.is_failed and m.failed_stage == Stage.TRANSCRIBED
    assert m.stage == stage_before  # progress preserved, job resumable
    assert "exploded" in m.error
    # and it can recover:
    m2 = transcribe_job(ingested_job, MockTranscriber(), force=True)
    assert m2.stage == Stage.TRANSCRIBED


# ------------------------------------------------------------------ factory

def test_factory():
    assert isinstance(get_transcriber("mock"), MockTranscriber)
    with pytest.raises(ValueError):
        get_transcriber("nonsense")
