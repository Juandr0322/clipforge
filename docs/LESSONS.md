<!-- Short, specific rules learned from real mistakes in THIS project.
     /retro proposes additions and deletions; you approve them. Max 40 lines,
     because every agent loads this file on every run. -->
## Errors and security
- Error text shown on the page (HTTPException detail, toasts) is a fixed user-facing sentence. Never send str(exc), file paths, job ids or ffmpeg/ffprobe output; log those server-side. Tests assert detail has no "/", "\\", workspace path or job id.
- Every job/clip id from a URL or the CLI goes through is_safe_job_id (clipforge/common/manifest.py) before touching disk. Traversal tests use %2e%2e with a real manifest in the workspace's parent.
- Parsers of LLM output never raise: wrap json.loads in except (ValueError, RecursionError) (a >4300-digit int or deep nesting raises), isinstance-check every field before `in`/float(), and fuzz-test junk types. A bad reply is dropped or logged with its raw text (F23), never a failed stage.
- Anything that stops a job records a failed stage with a fixed sentence in the manifest BEFORE raising (F31: a bad recipe left the job stuck at "ingested"); server run_bg and the CLI assume it did. Test it through /api/jobs, not just the exception.
- An env var that is empty, whitespace or junk counts as unset, in one shared parser (AudioLayerConfig.from_env, recipes.audio_config). Never "key in os.environ" (F23, F31).

## Concurrency and Windows
- Diego runs Windows; CI and agents run Linux. Any file written by more than one request (reviews.json, the future clip DB/exports) is read AND written under the per-job lock in clipforge/review/store.py: unique temp file, then os.replace with PermissionError retry.
- Routes that run ffmpeg/ffprobe, ingest or other blocking work are plain `def`, never `async def`. To test this, run real uvicorn in a thread; TestClient gives each request its own event loop.
- Text I/O: encoding="utf-8" on every read_text/write_text/open/subprocess text call, and newline="\n" on every write. Files Diego edits by hand (profiles, examples) are read with utf-8-sig, since PowerShell 5 adds a BOM.
- Best-effort side files (debug dumps, reports) are wrapped in try/except OSError and never fail the stage; log the file name only, never the path or str(exc) (manifest errors reach the page).
- A test for Windows-only behavior (CRLF, cp1252) must fail on Linux with the bug put back: stub the call and assert the newline/encoding argument, not the file bytes.

## Tests
- A regression test must fail against main's version of the file (git show main:<path>) before the fix, or it doesn't count.
- Tests never touch the real ~/clipforge-workspace or .env: pass --workspace tmp_path and patch env loading.
- ffmpeg lavfi silently randomizes out-of-range args (gradients x1 >= width). Fixtures use deterministic sources (geq, testsrc).
- Fake clients must be checked against the real SDK signature with an inspect.signature test: a fake that accepted any kwarg hid a parameter (temperature) the SDK had removed.
- Hand-edited data files (profiles, configs): the loader rejects NaN/Infinity, bools, fractions where ints are required and a name that differs from the filename, and every error names file + field. Never write a "checked"/"verified" marker for something you didn't check; use "unverified".
- A "same as before" test compares against values captured from main (hard-coded, or run via git show main:<file>), not the new code with old arguments (F12, F31).
- When a feature reads a field another open branch can clear or change (refined times, report rows), run the merged result before review and add a test for the None/missing case (F12 round 1).

## Review page (index.html)
- Build DOM with h()/textContent only; static tests forbid innerHTML, inline on*, alert/confirm/prompt, setInterval.
- Every action reads the job id stored on its element (data-job), never the global currentJob. openJob clears the pane in the same tick.
- Never recreate a card's <video> or overwrite a field the user edited (dirty flag); update cards in place.
- Poll only while /api/jobs reports a running job; always schedule one retry after a failed refresh.

## Browser tests (Playwright)
- Bundled Chromium can't play H.264: route clip URLs to a VP8 webm and honor Range requests (needed for currentTime/seeking).
- Waits on ingest of a long video use timeout=120000 (a 90 s video takes 27–40 s on a loaded 2-CPU box). Kill servers by PID, not `pkill -f` (it can match your own shell).

## Agents and sandbox
- Agent sandboxes have no bare `python`: use .venv/bin/python -m pytest / pyflakes.
- Stage files by explicit path, never `git add -A` or `git add .`. The sandbox mounts empty device files (.bashrc, .gitconfig, .mcp.json, .claude/commands, ...) that show up as untracked; they must never be committed. A ` M` on a file you can't read is the sandbox mask, not your edit.
