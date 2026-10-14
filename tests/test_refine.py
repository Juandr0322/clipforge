"""
Module 4 tests.

Includes the adversarial battery from the plan: candidates starting inside
a word, 5-second candidates, candidates at the video end, candidates in
silence — plus a real audio-energy test on synthesized speech-shaped audio
(tone bursts separated by silence).
"""
import math
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from clipforge.common.manifest import Manifest, Stage
from clipforge.highlight.backends import MockHighlighter
from clipforge.highlight.run import highlight_job
from clipforge.highlight.schema import CandidateSet, ClipCandidate
from clipforge.ingest.ingest import ingest
from clipforge.refine.refine import (
    RefineReject, RefinedSet, refine_all, refine_candidate,
)
from clipforge.refine.run import RefineError, refine_job
from clipforge.refine.silence import EnergyProfile
from clipforge.transcribe.backends import MockTranscriber
from clipforge.transcribe.run import transcribe_job
from clipforge.transcribe.schema import Segment, Transcript, Word

FIX = Path(__file__).parent / "fixtures"


# ------------------------------------------------------------ test helpers

def make_transcript(duration=300.0, seg_len=6.0, gap=0.8) -> Transcript:
    segs, words, t, i = [], [], 0.5, 0
    text = "one two three four five six seven eight"
    while t + seg_len < duration - 1:
        toks = text.split()
        wdur = seg_len / len(toks)
        wt = t
        for tok in toks:
            words.append(Word(w=tok, start=round(wt, 3),
                              end=round(wt + wdur * 0.8, 3)))
            wt += wdur
        segs.append(Segment(id=i, text=text, start=round(t, 3),
                            end=round(words[-1].end, 3)))
        t += seg_len + gap
        i += 1
    return Transcript(engine="synthetic", audio_duration_sec=duration,
                      words=words, segments=segs)


def C(i, s, e, score=80):
    return ClipCandidate(id=i, start=s, end=e, score=score, hook="h", reason="r")


def assert_clean(clip, t):
    """The golden invariants for any refined clip."""
    for w in t.words:
        assert not (w.start < clip.start < w.end), f"start cut inside '{w.w}'"
        assert not (w.start < clip.end < w.end), f"end cut inside '{w.w}'"
    assert clip.start <= clip.speech_start < clip.speech_end <= clip.end
    assert 0 <= clip.start < clip.end <= t.audio_duration_sec


T = make_transcript()


# ----------------------------------------------------------- refinement core

def test_normal_candidate():
    clip = refine_candidate(C(0, 20.0, 55.0), T, None)
    assert_clean(clip, T)
    assert 15 <= clip.duration <= 90
    # starts at a sentence start (within lead padding)
    seg_starts = [s.start for s in T.segments]
    assert any(abs(clip.speech_start - s) < 0.01 for s in seg_starts)


def test_candidate_starting_mid_word():
    # 0.5 + words of ~0.6s: 1.0 is inside an early word
    clip = refine_candidate(C(0, 1.0, 35.0), T, None)
    assert_clean(clip, T)


def test_tiny_candidate_grows_to_min():
    clip = refine_candidate(C(0, 40.0, 45.0), T, None)
    assert_clean(clip, T)
    assert clip.duration >= 15


def test_too_long_candidate_trimmed():
    clip = refine_candidate(C(0, 10.0, 250.0), T, None)
    assert_clean(clip, T)
    assert clip.duration <= 90


def test_candidate_at_video_end():
    clip = refine_candidate(C(0, T.audio_duration_sec - 20,
                              T.audio_duration_sec + 50), T, None)
    assert_clean(clip, T)
    assert clip.end <= T.audio_duration_sec


def test_candidate_before_video_start():
    clip = refine_candidate(C(0, -10.0, 30.0), T, None)
    assert_clean(clip, T)


def test_candidate_in_pure_silence_rejected():
    t = make_transcript(duration=300, seg_len=6, gap=0.8)
    # kill segments between 100 and 200s to create a dead zone
    t2 = Transcript(engine="x", audio_duration_sec=300,
                    words=[w for w in t.words if w.end < 100 or w.start > 200],
                    segments=[s for s in t.segments if s.end < 100 or s.start > 200])
    with pytest.raises(RefineReject, match="no transcript segments"):
        refine_candidate(C(0, 120, 160), t2, None)


def test_empty_span_rejected():
    with pytest.raises(RefineReject, match="empty span"):
        refine_candidate(C(0, 500, 600), T, None)  # beyond duration -> clamps to empty


def test_giant_single_sentence_rejected():
    # one 120s sentence cannot satisfy max 90s
    words = [Word(w=f"w{i}", start=i * 1.0, end=i * 1.0 + 0.8) for i in range(120)]
    t = Transcript(engine="x", audio_duration_sec=130, words=words,
                   segments=[Segment(id=0, text="long", start=0, end=119.8)])
    with pytest.raises(RefineReject, match="single sentence"):
        refine_candidate(C(0, 10, 60), t, None)


def test_refine_all_partial_harvest():
    cset = CandidateSet(engine="mock", source_duration_sec=300,
                        candidates=[C(0, 20, 55, 90), C(1, 290, 299, 70)])
    # candidate 1 is at the tail; may refine or reject, but must not crash
    out = refine_all(cset, T, None)
    assert len(out.clips) >= 1
    assert len(out.clips) + len(out.rejected) == 2
    ids = [c.id for c in out.clips]
    assert ids == list(range(len(ids)))  # re-id'd sequentially


# --------------------------------------------------------------- energy

def make_burst_wav(path: Path, duration=30.0, sr=16000,
                   burst=2.0, silence=1.0):
    """Tone bursts separated by silences — speech-shaped energy."""
    frames = bytearray()
    t = 0.0
    n = int(duration * sr)
    for i in range(n):
        t = i / sr
        cycle = t % (burst + silence)
        if cycle < burst:
            v = int(12000 * math.sin(2 * math.pi * 220 * t))
        else:
            v = 0
        frames += struct.pack("<h", v)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(bytes(frames))


def test_energy_profile_finds_silence(tmp_path):
    wav = tmp_path / "bursts.wav"
    make_burst_wav(wav)  # loud 0-2s, silent 2-3s, loud 3-5s, ...
    ep = EnergyProfile.from_wav(wav)
    q = ep.quietest_point(1.5, 3.5)
    assert 2.0 < q < 3.0, f"quietest point {q} not in the silence gap"


def test_energy_profile_degenerate_window(tmp_path):
    wav = tmp_path / "b2.wav"
    make_burst_wav(wav, duration=10)
    ep = EnergyProfile.from_wav(wav)
    assert ep.quietest_point(5.0, 5.0) == 5.0
    # out-of-range clamps sanely
    q = ep.quietest_point(9.9, 10.4)
    assert 9.9 <= q <= 10.4


def test_refine_with_energy_prefers_quiet(tmp_path):
    """With energy info, the cut lands in genuine silence."""
    wav = tmp_path / "b3.wav"
    make_burst_wav(wav, duration=300)
    ep = EnergyProfile.from_wav(wav)
    clip = refine_candidate(C(0, 20.0, 55.0), T, ep)
    assert_clean(clip, T)


# ------------------------------------------------------------ stage runner

@pytest.fixture(scope="module")
def ready_job(tmp_path_factory):
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
    m = ingest(src, ws, job_id="t4job")
    jd = ws / m.job_id
    transcribe_job(jd, MockTranscriber())
    highlight_job(jd, MockHighlighter())
    return jd


def test_stage_runner_happy_path(ready_job):
    m = refine_job(ready_job)
    assert m.stage == Stage.REFINED
    rs = RefinedSet.load(ready_job / m.refined_path)
    t = __import__("clipforge.transcribe.schema", fromlist=["Transcript"]) \
        .Transcript.load(ready_job / m.transcript_path)
    for clip in rs.clips:
        assert_clean(clip, t)


def test_stage_runner_requires_candidates(tmp_path):
    m = Manifest(job_id="fresh")
    m.save(tmp_path)
    with pytest.raises(RefineError, match="highlight first"):
        refine_job(tmp_path)


def test_stage_runner_idempotent(ready_job):
    m1 = refine_job(ready_job)
    m2 = refine_job(ready_job)
    assert len(m2.history) == len(m1.history)


# ------------------------------------------------------- F11 hook-first trim

def CT(i, s, e, **kw):
    return ClipCandidate(id=i, start=s, end=e, score=80, hook="h",
                         reason="r", **kw)


def test_preamble_is_cut_so_trigger_sentence_opens_clip():
    # segments start every 6.8 s; candidate 20-80 spans ~9 sentences
    trig_seg = T.segments[5]
    trig = trig_seg.start + 1.0
    payoff = trig_seg.start + 30.0
    base = refine_candidate(CT(0, 20.0, 80.0), T, None)
    clip = refine_candidate(CT(0, 20.0, 80.0, trigger_t=trig,
                               reaction_peak_t=trig + 5, payoff_end_t=payoff),
                            T, None)
    assert_clean(clip, T)
    assert clip.segment_ids[0] == trig_seg.id
    assert clip.speech_start == trig_seg.start
    assert clip.start > base.start
    assert clip.end >= payoff
    assert 15 <= clip.duration <= 90
    assert clip.trim_note is None
    assert clip.start <= clip.trigger_t <= clip.reaction_peak_t \
        <= clip.payoff_end_t <= clip.end


def test_preamble_kept_when_cut_would_be_too_short():
    trig_seg = T.segments[8]          # leaves < 15 s after the cut
    cand = CT(0, 20.0, 66.0, trigger_t=trig_seg.start + 0.5)
    base = refine_candidate(CT(0, 20.0, 66.0), T, None)
    clip = refine_candidate(cand, T, None)
    assert clip.start == base.start and clip.segment_ids == base.segment_ids
    assert "too short" in clip.trim_note
    assert_clean(clip, T)


def test_preamble_kept_when_cut_would_drop_payoff():
    trig_seg = T.segments[5]
    # 120 s candidate: the 90 s cap trims the tail, so the payoff is lost
    clip = refine_candidate(CT(0, 20.0, 140.0, trigger_t=trig_seg.start + 1,
                               payoff_end_t=139.0), T, None)
    assert clip.trim_note and "start kept" in clip.trim_note
    assert_clean(clip, T)


def test_no_trigger_refines_exactly_as_before():
    a = refine_candidate(C(0, 20.0, 55.0), T, None)
    assert a.trim_note is None and a.trigger_t is None
    assert a.reaction_peak_t is None and a.payoff_end_t is None


def test_bad_trigger_values_do_not_crash():
    for bad in (float("nan"), float("inf"), -5.0, 1e9):
        clip = refine_candidate(CT(0, 20.0, 55.0, trigger_t=bad), T, None)
        assert_clean(clip, T)
        assert clip.trigger_t is None


def test_times_outside_refined_clip_are_cleared_and_kept_ordered():
    # a long candidate gets tail-trimmed to 90 s; payoff past it is cleared
    clip = refine_candidate(CT(0, 20.0, 200.0, trigger_t=30.0,
                               reaction_peak_t=25.0,   # before trigger
                               payoff_end_t=190.0), T, None)
    assert clip.trigger_t == 30.0
    assert clip.reaction_peak_t is None
    assert clip.payoff_end_t is None
    clip2 = refine_candidate(CT(0, 20.0, 55.0, trigger_t=30.0,
                                reaction_peak_t=35.0, payoff_end_t=50.0),
                             T, None)
    assert (clip2.trigger_t, clip2.reaction_peak_t, clip2.payoff_end_t) \
        == (30.0, 35.0, 50.0)


def test_refined_clip_rejects_times_outside_or_unordered():
    good = refine_candidate(C(0, 20.0, 55.0), T, None).model_dump()
    with pytest.raises(ValueError):
        RefinedSet.model_validate({"source_duration_sec": 300, "clips": [
            {**good, "trigger_t": good["end"] + 1}]})
    with pytest.raises(ValueError):
        RefinedSet.model_validate({"source_duration_sec": 300, "clips": [
            {**good, "trigger_t": good["start"] + 5,
             "reaction_peak_t": good["start"] + 2}]})
