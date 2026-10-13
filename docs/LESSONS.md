<!-- Short, specific rules learned from real mistakes in THIS project.
     /retro proposes additions and deletions; you approve them. Max 40 lines,
     because every agent loads this file on every run. -->
## Errors and security
- Error text shown on the page (HTTPException detail, toasts) is a fixed user-facing sentence. Never send str(exc), file paths, job ids or ffmpeg/ffprobe output; log those server-side. Tests assert detail has no "/", "\\", workspace path or job id.
- Every job/clip id from a URL or the CLI goes through is_safe_job_id (clipforge/common/manifest.py) before touching disk. Traversal tests use %2e%2e with a real manifest in the workspace's parent.

## Concurrency and Windows
- Diego runs Windows; CI and agents run Linux. Any file written by more than one request (reviews.json, the future clip DB/exports) is read AND written under the per-job lock in clipforge/review/store.py: unique temp file, then os.replace with PermissionError retry.
- Routes that run ffmpeg/ffprobe, ingest or other blocking work are plain `def`, never `async def`. To test this, run real uvicorn in a thread; TestClient gives each request its own event loop.

## Tests
- A regression test must fail against main's version of the file (git show main:<path>) before the fix, or it doesn't count.
- Tests never touch the real ~/clipforge-workspace or .env: pass --workspace tmp_path and patch env loading.
- Compare names as sets/regex, not substrings (CLIPFORGE_A "matches" CLIPFORGE_ASR).
- ffmpeg lavfi silently randomizes out-of-range args (gradients x1 >= width). Fixtures use deterministic sources (geq, testsrc).
- Fake clients must be checked against the real SDK signature with an inspect.signature test: a fake that accepted any kwarg hid a parameter (temperature) the SDK had removed.

## Review page (index.html)
- Build DOM with h()/textContent only; static tests forbid innerHTML, inline on*, alert/confirm/prompt, setInterval.
- Every action reads the job id stored on its element (data-job), never the global currentJob. openJob clears the pane in the same tick.
- Never recreate a card's <video> or overwrite a field the user edited (dirty flag); update cards in place.
- Poll only while /api/jobs reports a running job; always schedule one retry after a failed refresh.

## Browser tests (Playwright)
- Bundled Chromium can't play H.264: route clip URLs to a VP8 webm and honor Range requests (needed for currentTime/seeking).
- Waits on ingest of a long video use timeout=120000 (a 90 s video takes 27–40 s on a loaded 2-CPU box).
- Kill servers you start by PID; `pkill -f` patterns can match your own shell.
