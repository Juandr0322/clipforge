# ClipForge

[![tests](https://github.com/Juandr0322/clipforge/actions/workflows/tests.yml/badge.svg)](https://github.com/Juandr0322/clipforge/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

AI agent that turns long-form video into short-form vertical clips —
transcription, LLM highlight detection, clean cut refinement, auto-reframe
to 9:16, word-synced captions, and a review UI with a feedback loop.

Built by **Juan Diego Ramirez** (Computer Engineering, Boston University).

```mermaid
flowchart LR
    A[Long video] --> B[1 Ingest]
    B --> C[2 Transcribe<br/>faster-whisper]
    C --> D[3 Highlight<br/>Claude API]
    D --> E[4 Refine cuts]
    E --> F[5 Reframe 9:16<br/>face tracking]
    F --> G[6 Render<br/>ffmpeg + captions]
    G --> H[7 Review UI<br/>FastAPI]
    H -. approved clips become<br/>few-shot examples .-> D
```

**Tech stack:** Python · faster-whisper · Anthropic Claude API · MediaPipe / OpenCV ·
ffmpeg + libass · FastAPI · Pydantic · pytest · GitHub Actions

## Status: all 7 modules complete — 117 tests passing

| Module | What it does |
|---|---|
| 1. Ingestion | probe/validate/normalize, extract audio, 480p proxy |
| 2. Transcription | faster-whisper, word-level timestamps |
| 3. Highlight detection | Claude ranks clip candidates; few-shot library |
| 4. Boundary refinement | sentence snap + silence-aware cuts |
| 5. Visual reframing | face tracking -> smoothed 9:16 camera path |
| 6. Render | dynamic crop, burned ASS captions, -14 LUFS, 1080x1920 |
| 7. Review UI | web app: approve/reject/trim, metrics, feedback export |

## Setup

Python 3.11+ and ffmpeg on PATH.

```bash
pip install -r requirements.txt
pip install mediapipe          # optional, better face tracking than haar
export ANTHROPIC_API_KEY=...   # for real highlight detection
```

On Windows PowerShell, set the key with `$env:ANTHROPIC_API_KEY="..."`.
See `.env.example` for every setting; never commit a real `.env`.

## Run the web app (recommended)

```bash
python -m clipforge.cli serve
# open http://localhost:8000 — upload a video, wait, review clips
```

Engine overrides via env: `CLIPFORGE_ASR` (whisper:small|whisper:medium|mock),
`CLIPFORGE_HL` (anthropic|mock), `CLIPFORGE_DETECTOR` (mediapipe|haar),
`CLIPFORGE_EXAMPLES` (few-shot dir; defaults to ./examples via export).

## Or run stage by stage (CLI)

```bash
python -m clipforge.cli ingest video.mp4
python -m clipforge.cli transcribe <job> --engine whisper:small
python -m clipforge.cli highlight  <job> --engine anthropic --examples-dir examples
python -m clipforge.cli refine     <job>
python -m clipforge.cli reframe    <job>            # mediapipe or haar
python -m clipforge.cli render     <job>
python -m clipforge.cli pipeline   <job>            # or: all remaining stages
python -m clipforge.cli status     <job>
```

Outputs land in `workspace/<job>/renders/` (clips + QA contact sheets).

## The feedback loop

Review clips in the UI (approve / reject / trim / note). Then click
"export approved -> few-shot examples". Approved clips become example files
that calibrate Module 3's prompt on the next run — the system learns your
taste from your own decisions. Metrics to watch over time (shown in the
header): **approval rate** should rise, **avg trim adjustment** should fall.

## Architecture

- Modules communicate ONLY through JSON files in the job dir
  (`manifest.json` is the state machine + audit trail).
- Every stage is idempotent and resumable; failures record the error and
  preserve progress.
- Every stage verifies its own outputs before marking itself done.
- Engines (ASR / LLM / face detector) are swappable interfaces with mock
  implementations, so the entire pipeline is testable offline.

## Testing

```bash
python -m pytest tests/ -q     # 117 tests, fully offline (mock engines)
```

## Roadmap

Next upgrades, informed by 2026 short-form platform research:

- [ ] Platform profiles (length limits, caption safe zones per app)
- [ ] Smarter highlight picking: moments that make sense to a stranger
- [ ] Hook-first trimming and a first-3-seconds hook score
- [ ] Originality layer: context overlay, punch-in zooms, end cards
- [ ] Word-chunked captions and a pacing check
- [ ] Hook variants and loop-point detection
- [ ] Per-platform exports
- [ ] Review UI as a control panel
- [ ] Learning loop: clip database + real performance data fed back into highlight selection

## Scaling later (deliberately not built yet)

Single-user prototype by design. To commercialize: real job queue
(Celery/Redis) instead of threads, auth, object storage for videos,
GPU workers for Whisper/rendering, and per-user workspaces. The
file-contract architecture means none of that touches module internals.
