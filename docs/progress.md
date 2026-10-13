# Progress log
<!-- Newest entry at the top. The lead adds one entry per feature attempt.
     Every agent reads the top of this file first, so keep entries to 1–2 lines. -->

- 2026-10-13 · F23 · merged · PR #2 re-applied by hand: av<19 pin, Whisper CPU fallback on missing CUDA libs (GPU still used when it works), optional ANTHROPIC_WORKSPACE_ID header, temperature=0, raw reply saved + warning on 0-clip parse. Review FAIL fixed (debug I/O can't fail the stage, tighter CUDA match). 291 tests + mock pipeline e2e OK. Sandbox masked .env.* so Diego committed .env.example. Next: F08.
- 2026-10-12 · F22 · merged · -vsync→-fps_mode (ffmpeg 8 removed it), ffmpeg version preflight (refuse <5.1), CI matrix distro+current ffmpeg, .gitattributes LF, LF example files. All 3 CI jobs green incl. windows-latest. 271 tests. F21 now passes too. Next: Diego tests on Windows, then F08.
- 2026-10-12 · F21 · merged · UTF-8 on all text I/O + subprocess output, ffmpeg filter-path escaping (drive-letter colon), serve/ingest preflight (missing ffmpeg, busy port), Windows CI job. Review FAIL fixed. Windows CI exposed ffmpeg 8 incompatibility → F22.
- 2026-10-12 · F07 · merged + tested · drag-drop multi-upload with progress, phone layout, non-blocking upload route, locked + Windows-safe review saves, trim validation, no path leaks in errors. Review FAIL fixed (path leak, Windows replace race). Next: F08 (group C) after Diego's review + /retro.
- 2026-10-12 · F06 · merged + tested · keyboard review (A/R/Space/J/K/[ ]), filter counts, persisted sort, trim from playhead; selection-jump fix. 172 tests. Tester found concurrent review saves lose data → added to F07. Next: F07.
- 2026-10-12 · F05 · merged + tested · DOM-built page (no innerHTML), per-card job ids (review FAIL fixed: approvals could land on another job), polling only while running, toasts. 169 tests. Tester found upload blocks event loop + no phone layout → added to F07. Next: F06.
- 2026-10-12 · F04 · merged · ids validated everywhere (closed %2e%2e read of workspace.parent manifest), upload/ingest failures 400/500 without leaking paths, running flag + 409 rerun lock. 163 tests. Next: F05.
- 2026-10-12 · F03 · merged · pyflakes clean + CI lint step; docs/ARCHITECTURE.md; mediapipe fallback now also covers broken installs. 129 tests. Next: F04.
- 2026-10-12 · F02 · merged · flaky test root cause: ffmpeg gradients x1=1280 out of range → random crossover; fixture now a deterministic geq ramp + frozen-camera negative control. 127 tests pass.
- 2026-10-12 · F01 · merged · .env loaded from repo root at CLI start, real env wins, library imports never read .env. Next: F03.

<!-- Format:
- YYYY-MM-DD · F00 · built / review FAIL / test FAIL / merged · what happened, what's next
-->
