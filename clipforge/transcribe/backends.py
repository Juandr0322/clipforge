"""
Transcriber backends.

The stage runner (run.py) only knows the `Transcriber` interface, so the
engine is swappable: local faster-whisper today, Deepgram/AssemblyAI later,
mock in tests — with zero changes anywhere else in the pipeline.
"""
from __future__ import annotations

import abc
import wave
from pathlib import Path

from clipforge.transcribe.schema import Segment, Transcript, Word


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()


class Transcriber(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def transcribe(self, wav_path: Path) -> Transcript:
        """16 kHz mono WAV in -> validated Transcript out."""


# ---------------------------------------------------------------------------
# Real backend: faster-whisper (local, free, word-level timestamps built in)
# ---------------------------------------------------------------------------

class FasterWhisperTranscriber(Transcriber):
    """
    Local Whisper via CTranslate2.

    model_size guide (English content, laptop CPU):
      tiny/base  — fast, rough; fine for pipeline dev
      small      — good default: solid accuracy, ~1-2x realtime on CPU
      medium/large-v3 — best accuracy; use if you have a GPU or patience
    First run downloads the model from Hugging Face (~500MB for small).
    """

    def __init__(self, model_size: str = "small", device: str = "auto",
                 compute_type: str = "auto", language: str | None = None):
        self.model_size = model_size
        self.language = language
        self.name = f"faster-whisper/{model_size}"
        # Lazy import so the rest of the pipeline doesn't require the package
        from faster_whisper import WhisperModel
        self._model = WhisperModel(model_size, device=device,
                                   compute_type=compute_type)

    def transcribe(self, wav_path: Path) -> Transcript:
        duration = wav_duration(wav_path)
        seg_iter, info = self._model.transcribe(
            str(wav_path),
            word_timestamps=True,          # non-negotiable for this pipeline
            vad_filter=True,               # skip silence; big speedup + fewer hallucinations
            language=self.language,
            beam_size=5,
        )
        words: list[Word] = []
        segments: list[Segment] = []
        for i, seg in enumerate(seg_iter):
            segments.append(Segment(
                id=i, text=seg.text.strip(),
                start=round(seg.start, 3), end=round(seg.end, 3),
            ))
            for w in (seg.words or []):
                words.append(Word(
                    w=w.word.strip(),
                    start=round(w.start, 3), end=round(w.end, 3),
                    confidence=round(w.probability, 3),
                ))
        return Transcript(
            language=info.language or "en",
            engine=self.name,
            audio_duration_sec=round(duration, 3),
            words=words,
            segments=segments,
        )


# ---------------------------------------------------------------------------
# Mock backend: deterministic, offline — used by tests and pipeline dev
# ---------------------------------------------------------------------------

MOCK_SENTENCES = [
    "Welcome back to the show everyone.",
    "Today we are talking about something nobody tells you.",
    "Here is the thing that changed my mind completely.",
    "Most people get this totally wrong and it costs them.",
    "Let me give you a concrete example from last year.",
    "The results honestly surprised even me.",
    "So the real lesson is simpler than you think.",
    "Thanks for watching and see you next time.",
]


class MockTranscriber(Transcriber):
    """
    Produces a plausible, schema-valid transcript that spans the real audio
    duration: ~2.5 words/sec, sentences looped from MOCK_SENTENCES, small
    silence gap between sentences. Deterministic for stable tests.
    """
    name = "mock"

    def __init__(self, words_per_sec: float = 2.5, gap_sec: float = 0.6):
        self.wps = words_per_sec
        self.gap = gap_sec

    def transcribe(self, wav_path: Path) -> Transcript:
        duration = wav_duration(wav_path)
        words: list[Word] = []
        segments: list[Segment] = []
        t, seg_id, si = 0.25, 0, 0
        wdur = 1.0 / self.wps
        margin = 0.5
        while t < duration - margin:
            sentence = MOCK_SENTENCES[si % len(MOCK_SENTENCES)]
            si += 1
            toks = sentence.split()
            if t + len(toks) * wdur > duration - margin:
                break
            seg_start = t
            for tok in toks:
                words.append(Word(w=tok, start=round(t, 3),
                                  end=round(t + wdur * 0.85, 3)))
                t += wdur
            segments.append(Segment(id=seg_id, text=sentence,
                                    start=round(seg_start, 3),
                                    end=round(t - wdur * 0.15, 3)))
            seg_id += 1
            t += self.gap
        return Transcript(
            language="en", engine=self.name,
            audio_duration_sec=round(duration, 3),
            words=words, segments=segments,
        )


def get_transcriber(engine: str, **kwargs) -> Transcriber:
    """Factory used by the CLI: --engine mock|whisper[:model_size]"""
    if engine == "mock":
        return MockTranscriber()
    if engine.startswith("whisper"):
        _, _, size = engine.partition(":")
        return FasterWhisperTranscriber(model_size=size or "small", **kwargs)
    raise ValueError(f"Unknown transcription engine: {engine}")
