"""
Module 1 — Ingestion & Preprocessing.

Responsibilities:
  1. Probe the source file (ffprobe) and validate it's usable.
  2. Normalize to a consistent working format: H.264, constant frame rate,
     max 1080p, AAC audio. Every downstream module sees identical input shape.
  3. Extract 16 kHz mono WAV (what Whisper/ASR expects).
  4. Build a 480p proxy for cheap vision analysis (Module 5).
  5. Write everything into a job directory with a manifest.json.

Design rules:
  - Idempotent: re-running on a completed job is a no-op unless force=True.
  - Fail loudly and cleanly: a bad file marks the job FAILED with a clear
    error in the manifest — never silent corruption.
  - All heavy lifting is delegated to ffmpeg via subprocess; we never load
    video frames into Python here.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Optional

from clipforge.common.manifest import (
    IngestOutputs, Manifest, SourceInfo, Stage, new_job_id,
)

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}
MIN_DURATION_SEC = 5.0          # shorter than this can't produce a clip
MAX_DURATION_SEC = 4 * 3600     # sanity cap for a prototype
TARGET_MAX_HEIGHT = 1080
PROXY_HEIGHT = 480
AUDIO_SR = 16_000               # Whisper's expected sample rate


class IngestError(Exception):
    """Raised for any unrecoverable problem with the source file."""


# --------------------------------------------------------------------------
# ffprobe
# --------------------------------------------------------------------------

def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    # ffprobe/ffmpeg emit UTF-8 (metadata tags can hold curly quotes, CJK...);
    # without this Windows decodes with cp1252 and can raise UnicodeDecodeError
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
    if proc.returncode != 0:
        raise IngestError(
            f"Command failed ({cmd[0]}): {proc.stderr.strip()[-500:]}"
        )
    return proc


def probe(path: Path) -> dict:
    """Return the parsed ffprobe JSON for a media file."""
    proc = _run([
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ], timeout=120)
    return json.loads(proc.stdout)


@dataclass
class ProbeResult:
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: Optional[str]
    audio_sample_rate: Optional[int]
    audio_channels: Optional[int]


def parse_probe(raw: dict) -> ProbeResult:
    vstreams = [s for s in raw.get("streams", []) if s.get("codec_type") == "video"]
    astreams = [s for s in raw.get("streams", []) if s.get("codec_type") == "audio"]
    if not vstreams:
        raise IngestError("File has no video stream.")
    v = vstreams[0]

    # fps: prefer avg_frame_rate; r_frame_rate can be inflated for VFR files
    fps_str = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1"
    try:
        fps = float(Fraction(fps_str)) if fps_str != "0/0" else 0.0
    except (ZeroDivisionError, ValueError):
        fps = 0.0

    duration = float(raw.get("format", {}).get("duration", 0) or 0)
    if duration <= 0:  # some containers only carry duration on the stream
        duration = float(v.get("duration", 0) or 0)

    a = astreams[0] if astreams else None
    return ProbeResult(
        duration=duration,
        width=int(v.get("width", 0)),
        height=int(v.get("height", 0)),
        fps=fps,
        video_codec=v.get("codec_name", "unknown"),
        audio_codec=a.get("codec_name") if a else None,
        audio_sample_rate=int(a["sample_rate"]) if a and a.get("sample_rate") else None,
        audio_channels=int(a["channels"]) if a and a.get("channels") else None,
    )


def validate(pr: ProbeResult) -> None:
    """Reject files the pipeline cannot meaningfully process."""
    if pr.duration < MIN_DURATION_SEC:
        raise IngestError(
            f"Video too short ({pr.duration:.1f}s); minimum is {MIN_DURATION_SEC}s.")
    if pr.duration > MAX_DURATION_SEC:
        raise IngestError(
            f"Video too long ({pr.duration/3600:.1f}h); maximum is "
            f"{MAX_DURATION_SEC/3600:.0f}h for the prototype.")
    if pr.audio_codec is None:
        raise IngestError(
            "File has no audio track — transcription-driven clipping "
            "requires audio.")
    if pr.width <= 0 or pr.height <= 0:
        raise IngestError("Could not determine video dimensions.")
    if pr.fps <= 0 or pr.fps > 240:
        raise IngestError(f"Unusable frame rate: {pr.fps}")


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def _even(n: int) -> int:
    return n - (n % 2)


def normalize(src: Path, dst: Path, pr: ProbeResult) -> tuple[float, int, int]:
    """
    Transcode to the standard working format:
      H.264 CFR (30fps if source is VFR/odd, else min(source, 60)),
      max 1080p (preserve aspect), yuv420p, AAC 48k stereo.
    Returns (fps, width, height) of the normalized file.
    """
    # Choose output fps: keep common rates, force weird/VFR ones to 30
    fps = pr.fps
    if fps < 10 or fps > 60:
        out_fps = 30.0
    else:
        out_fps = round(fps, 3)

    # Scale down to <=1080 height, keep aspect, force even dims (H.264 req.)
    if pr.height > TARGET_MAX_HEIGHT:
        out_h = TARGET_MAX_HEIGHT
        out_w = _even(round(pr.width * TARGET_MAX_HEIGHT / pr.height))
    else:
        out_h = _even(pr.height)
        out_w = _even(pr.width)

    _run([
        "ffmpeg", "-y", "-i", str(src),
        "-map", "0:v:0", "-map", "0:a:0",
        "-vf", f"scale={out_w}:{out_h}",
        "-r", str(out_fps), "-fps_mode", "cfr",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        str(dst),
    ])
    return out_fps, out_w, out_h


def extract_audio(src: Path, dst: Path) -> None:
    """16 kHz mono PCM WAV — the canonical ASR input."""
    _run([
        "ffmpeg", "-y", "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(AUDIO_SR),
        "-c:a", "pcm_s16le",
        str(dst),
    ])


def make_proxy(src: Path, dst: Path) -> None:
    """Small 480p proxy for per-frame vision work in Module 5."""
    _run([
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"scale=-2:{PROXY_HEIGHT}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-an",
        str(dst),
    ])


# --------------------------------------------------------------------------
# post-conditions: verify our own outputs before declaring success
# --------------------------------------------------------------------------

def verify_outputs(job_dir: Path, out: IngestOutputs,
                   source_duration: float) -> None:
    """Invariant checks. If any fail, the stage fails — no silent corruption."""
    norm = parse_probe(probe(job_dir / out.normalized_video))
    wav = parse_probe_audio_only(job_dir / out.audio_wav)

    tol = max(0.5, source_duration * 0.01)  # 1% or 0.5s
    if abs(norm.duration - source_duration) > tol:
        raise IngestError(
            f"Normalized duration drifted: {norm.duration:.2f}s vs "
            f"source {source_duration:.2f}s")
    if abs(wav["duration"] - source_duration) > tol:
        raise IngestError(
            f"Extracted audio duration drifted: {wav['duration']:.2f}s vs "
            f"source {source_duration:.2f}s")
    if wav["sample_rate"] != AUDIO_SR or wav["channels"] != 1:
        raise IngestError(
            f"Audio not {AUDIO_SR}Hz mono: got {wav['sample_rate']}Hz "
            f"{wav['channels']}ch")
    if norm.width % 2 or norm.height % 2:
        raise IngestError("Normalized dimensions not even.")


def parse_probe_audio_only(path: Path) -> dict:
    raw = probe(path)
    a = [s for s in raw["streams"] if s["codec_type"] == "audio"][0]
    return {
        "duration": float(raw["format"]["duration"]),
        "sample_rate": int(a["sample_rate"]),
        "channels": int(a["channels"]),
    }


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------

def ingest(source_path: str | Path, workspace: str | Path,
           job_id: Optional[str] = None, force: bool = False) -> Manifest:
    """
    Run Module 1 on a source video. Creates workspace/<job_id>/ containing:
        manifest.json, source.<ext>, normalized.mp4, audio.wav, proxy.mp4

    Idempotent: if the job is already INGESTED and force=False, returns
    the existing manifest untouched.
    """
    source_path = Path(source_path)
    workspace = Path(workspace)
    if not source_path.exists():
        raise IngestError(f"Source file not found: {source_path}")
    if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise IngestError(
            f"Unsupported extension '{source_path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    job_id = job_id or new_job_id()
    job_dir = workspace / job_id

    # Resume support
    if Manifest.path_for(job_dir).exists() and not force:
        m = Manifest.load(job_dir)
        if m.is_at_least(Stage.INGESTED):
            return m  # nothing to do

    manifest = Manifest(job_id=job_id)
    started = datetime.now(timezone.utc)
    try:
        job_dir.mkdir(parents=True, exist_ok=True)

        # 1. copy source in (jobs must be self-contained/reproducible)
        local_src = job_dir / f"source{source_path.suffix.lower()}"
        if not local_src.exists() or force:
            shutil.copy2(source_path, local_src)

        # 2. probe + validate
        pr = parse_probe(probe(local_src))
        validate(pr)
        manifest.source = SourceInfo(
            original_filename=source_path.name,
            duration_sec=round(pr.duration, 3),
            width=pr.width, height=pr.height, fps=round(pr.fps, 3),
            video_codec=pr.video_codec, audio_codec=pr.audio_codec,
            audio_sample_rate=pr.audio_sample_rate,
            audio_channels=pr.audio_channels,
            size_bytes=local_src.stat().st_size,
            sha256=_sha256(local_src),
        )

        # 3. transforms
        norm_path = job_dir / "normalized.mp4"
        wav_path = job_dir / "audio.wav"
        proxy_path = job_dir / "proxy.mp4"
        out_fps, out_w, out_h = normalize(local_src, norm_path, pr)
        extract_audio(norm_path, wav_path)
        make_proxy(norm_path, proxy_path)

        outputs = IngestOutputs(
            normalized_video=norm_path.name,
            audio_wav=wav_path.name,
            proxy_video=proxy_path.name,
            normalized_fps=out_fps,
            normalized_width=out_w,
            normalized_height=out_h,
        )

        # 4. verify our own work
        verify_outputs(job_dir, outputs, pr.duration)

        manifest.ingest = outputs
        manifest.record(Stage.INGESTED, started, ok=True)
    except Exception as e:  # noqa: BLE001 — we want ANY failure in the manifest
        manifest.record(Stage.INGESTED, started, ok=False, error=str(e))
        manifest.save(job_dir)
        raise
    manifest.save(job_dir)
    return manifest
