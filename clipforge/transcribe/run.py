"""
Module 2 — Transcription stage runner.

Reads the job manifest, requires stage >= INGESTED, runs the chosen
Transcriber backend on audio.wav, verifies post-conditions, writes
transcript.json, and advances the manifest to TRANSCRIBED.

Idempotent + resumable, same rules as Module 1.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from clipforge.common.manifest import Manifest, Stage
from clipforge.transcribe.backends import Transcriber
from clipforge.transcribe.schema import Transcript


class TranscribeError(Exception):
    pass


MIN_WORDS_PER_MIN = 20   # below this something is wrong (music? wrong file?)


def verify_transcript(t: Transcript) -> None:
    """Post-conditions beyond schema validation."""
    if not t.words:
        raise TranscribeError("Transcript contains zero words.")
    minutes = max(t.audio_duration_sec / 60.0, 1e-6)
    wpm = len(t.words) / minutes
    if wpm < MIN_WORDS_PER_MIN:
        raise TranscribeError(
            f"Suspiciously sparse transcript: {wpm:.0f} words/min. "
            "Is this actually speech content?")
    # every word must belong to some segment's time range (loose check)
    if t.segments:
        last_seg_end = max(s.end for s in t.segments)
        if t.words[-1].end > last_seg_end + 2.0:
            raise TranscribeError("Words extend beyond final segment.")


def transcribe_job(job_dir: str | Path, transcriber: Transcriber,
                   force: bool = False) -> Manifest:
    job_dir = Path(job_dir)
    m = Manifest.load(job_dir)

    if not m.is_at_least(Stage.INGESTED):
        raise TranscribeError(
            f"Job is at stage '{m.stage.value}' — run ingest first.")
    if m.is_at_least(Stage.TRANSCRIBED) and not force:
        return m  # no-op

    started = datetime.now(timezone.utc)
    try:
        wav = job_dir / m.ingest.audio_wav
        if not wav.exists():
            raise TranscribeError(f"audio.wav missing from job dir: {wav}")

        transcript = transcriber.transcribe(wav)
        verify_transcript(transcript)

        out = job_dir / "transcript.json"
        transcript.save(out)
        m.transcript_path = out.name
        m.record(Stage.TRANSCRIBED, started, ok=True)
    except Exception as e:  # noqa: BLE001
        m.record(Stage.TRANSCRIBED, started, ok=False, error=str(e))
        m.save(job_dir)
        raise
    m.save(job_dir)
    return m
