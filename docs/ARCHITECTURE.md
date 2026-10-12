# ClipForge architecture

ClipForge turns a long video into captioned 9:16 clips. Seven modules under `clipforge/` each own one stage. Stages never import each other's results: they talk only through files in the job directory (`<workspace>/<job_id>/`), and `manifest.json` (`clipforge/common/manifest.py`, class `Manifest`) is the state machine. `Manifest.stage` is the last successfully completed `Stage`; a failure sets `failed_stage` and `error` but keeps `stage`, so a job resumes where it stopped. Each stage 1-6 entry point in the table below is idempotent (a finished stage returns early unless `force=True`) and checks its own outputs before recording success.

`clipforge/pipeline.py` (`run_pipeline`) calls the stages in order and skips any that are done. `PipelineConfig` reads `CLIPFORGE_ASR` (default `whisper:small`), `CLIPFORGE_HL` (default `anthropic`), `CLIPFORGE_DETECTOR` (default `mediapipe`) and `CLIPFORGE_EXAMPLES`. `PipelineConfig.resolve` falls back to `mock` highlights when `ANTHROPIC_API_KEY` is unset, and to `haar` when `mediapipe` is not installed. Tests and the offline dev server use `CLIPFORGE_ASR=mock CLIPFORGE_HL=mock CLIPFORGE_DETECTOR=haar`.

## Stages

| # | Module | Entry point | Reads | Writes (job dir) | Manifest stage set | Engines |
|---|--------|-------------|-------|------------------|--------------------|---------|
| 1 | `clipforge/ingest/` | `ingest` | the uploaded video | `source.<ext>`, `normalized.mp4`, `audio.wav`, `proxy.mp4`, `manifest.json` | `Stage.INGESTED` | none (ffmpeg/ffprobe) |
| 2 | `clipforge/transcribe/` | `transcribe_job` | `audio.wav` | `transcript.json` | `Stage.TRANSCRIBED` | `whisper[:size]` (`FasterWhisperTranscriber`), `mock` (`MockTranscriber`) via `get_transcriber` |
| 3 | `clipforge/highlight/` | `highlight_job` | `transcript.json`, optional few-shot `*.txt` in `CLIPFORGE_EXAMPLES` | `candidates.json` | `Stage.CANDIDATES_READY` | `anthropic[:model]` (`AnthropicHighlighter`, needs `ANTHROPIC_API_KEY`), `mock` (`MockHighlighter`) via `get_highlighter` |
| 4 | `clipforge/refine/` | `refine_job` | `transcript.json`, `candidates.json`, `audio.wav` (energy profile) | `refined.json` | `Stage.REFINED` | none (rule-based; `use_energy=False` skips the audio profile) |
| 5 | `clipforge/reframe/` | `reframe_job` | `refined.json`, `proxy.mp4` | `crops/clip_<id>.json` | `Stage.REFRAMED` | `haar` (`HaarDetector`), `mediapipe` (`MediaPipeDetector`) via `get_detector`; `MockDetector` is built directly in tests and is not selectable by name |
| 6 | `clipforge/render/` | `render_job` | `transcript.json`, `refined.json`, `crops/`, `normalized.mp4` | `renders/clip_<id>.mp4`, `renders/clip_<id>.ass` (captions on), `renders/clip_<id>_qa.png` | `Stage.RENDERED` | none (ffmpeg with libass) |
| 7 | `clipforge/review/` | `save_reviews`, `export_examples` | `store.py` reads `manifest.json`, `transcript.json`, `refined.json`, `reviews.json` | `reviews.json`; `export_examples` writes `<job_id>_clip<NNN>.txt` to an examples dir | none (see below) | none |

Notes on stage 7: only `clipforge/server/app.py` reads `candidates.json` (to show candidate details next to each clip). `export_examples` is not idempotent in the stage sense: it rewrites its files on every call. `Stage.REVIEWED` exists in `Stage` and `STAGE_ORDER` but no code sets it today. Review decisions live only in `reviews.json`, and `export_examples` turns approved clips into the few-shot files that stage 3 loads, closing the feedback loop.

## Review website

`clipforge/server/app.py` (`create_app`) is the FastAPI API and serves `clipforge/server/static/index.html` (vanilla JS, no build step). It runs `run_pipeline` for uploaded jobs and reads and writes `reviews.json` through `clipforge/review/store.py`. The CLI (`clipforge/cli.py`, subcommands `ingest`, `transcribe`, `highlight`, `refine`, `reframe`, `render`, `pipeline`, `status`, `serve`) calls `load_env` from `clipforge/settings.py` once at start-up. That loads a repo-root `.env` without overriding real environment variables. Library imports never read `.env`, so tests cannot pick up a real API key.

## Rules

- Engines sit behind small abstract classes (`Transcriber`, `Highlighter`, `FaceDetector`) so mocks drop in with no other change. Tests must not call the Claude API or download models.
- Lint is `python -m pyflakes clipforge tests`; CI runs it before the tests (`.github/workflows/tests.yml`).
