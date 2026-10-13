# Working on ClipForge

## Layout
- `clipforge/` — the 7 pipeline modules (ingest, transcribe, highlight, refine, reframe, render, review/server).
- `tests/` — pytest suite; runs fully offline with mock engines.
- `examples/` — few-shot example library for the highlight step.
- Modules talk only through JSON files in the job dir; `manifest.json` is the state machine. Keep it that way.

## Rules for every change
1. One upgrade step per branch (`step-N-short-name`), opened as a pull request against `main`.
2. Run `python -m pytest tests/ -q` before and after. All tests must pass; new behavior gets new tests.
3. Before the PR, the `reviewer` agent checks the diff (see `.claude/agents/reviewer.md`).
4. The PR description explains in plain words what changed, what it does, and anything Diego must set up or test on real footage.
5. Never commit secrets (`.env`, `client_secret*.json`, tokens) or video files. `.gitignore` covers them; double-check with `git status`.
6. Nothing posts to social platforms automatically until Phase 4 of the upgrade plan.

## Pins and quirks
- OpenCV must stay below 5: 5.x removed `CascadeClassifier`, which the haar face-detector fallback uses.
- `tests/test_render.py::test_tracked_crop_actually_moves` used to fail about 1 run in 3: its fixture used ffmpeg's randomly seeded `gradients` source. The fixture is now two flat colors, so a failure there is real.
