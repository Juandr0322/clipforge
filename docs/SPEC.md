# ClipForge: next stage spec

ClipForge already works end to end. It has 7 pipeline stages (ingest, transcribe, highlight, refine, reframe, render, review), a FastAPI review website, and 117 passing offline tests. This spec covers the **next stage**, which builds on that code. It is not a rewrite. The stage-to-stage JSON contracts, idempotent stages, `manifest.json` state machine and swappable mock engines all stay as they are (see CLAUDE.md, "Architecture").

## 1. Overview and why

Diego makes short vertical clips for small streamers and podcasters on paid monthly retainers. The clips go to TikTok, Instagram Reels and YouTube Shorts (main platforms), and to X and Facebook (side platforms).

In 2026 every platform cuts reach for reposts with little editing, and the first ~3 seconds decide whether a clip gets pushed. A clip that is just "a cut of the stream plus captions" now underperforms. To keep retainer clients, ClipForge has to:

1. **Be trustworthy and quick to run day to day.** Settings come from a `.env` file, tests aren't flaky, and the code is lint-clean and documented. It is also a resume project, so quality shows.
2. **Make reviewing fast and safe.** Diego reviews dozens of clips per VOD in the local website. Today that page restarts videos, wipes typed notes, has no shortcuts and inserts text without escaping it.
3. **Produce clips that platforms treat as original.** That means hook-first openings, context for strangers, visible editing (punch-ins, overlays), platform-safe captions and per-platform exports.
4. **Learn from real results.** Record every exported clip, import platform analytics, and feed the winners (and flops) back into highlight picking, with Diego approving each change.

## 2. Target user

There is one user: Diego, a CompE student working part-time on a clipping retainer business. He runs ClipForge locally on his own PC, and some runs can't use a GPU. He has the rights to every source video he processes. He posts by hand; ClipForge never posts anything.

The things he cares about are review throughput (clips per minute), clip quality per platform, a clear record of what was posted where, and weekly proof to clients that clips perform.

## 3. Core user flows

**F1: First-time setup.** Diego copies `.env.example` to `.env`, fills in `ANTHROPIC_API_KEY` and any `CLIPFORGE_*` overrides, and runs `python -m clipforge.cli serve`. Nothing needs exporting in the shell. If a real environment variable is also set, it wins over `.env`.

**F2: Process a VOD.** Diego drags a stream or podcast file onto the website. He sees upload progress, then live stage progress. Polling stops once nothing is running. Files that aren't video are refused at once with a clear inline message.

**F3: Review fast.** For each clip he sees the video, its score, hook, reason and (after the Group C work) the hook score, originality checklist, pacing flags and hook variants side by side. He works by keyboard: play/pause, set trim start/end from the playhead, approve or reject, then jump to the next unreviewed clip. He can filter by unreviewed / approved / rejected and sort by score. Background refreshes never interrupt a playing video or text he is typing. A saved review shows a small confirmation. Errors appear inline and never as blocking pop-ups.

**F4: Approve per platform and export.** A clip can be approved for some platforms and not others. Export creates one file per approved (clip, platform) pair from that clip's single master render. Each export comes with a text file holding a suggested title, caption, hashtags and a credit line for the creator. Diego posts by hand and pastes the live post URL back into the website.

**F5: Weekly learning loop.** Diego downloads analytics CSVs from YouTube Studio, TikTok and Instagram, then imports them. Rows are matched to clips through the post URLs. A weekly ranking shows the top performers and the flops. He ticks which ones become positive few-shot examples and which become negative "avoid this" examples. The next highlight runs use them.

**F6: Log trend references.** When he sees a trending clip in a client's niche, he logs it in a website form: URL, platform, niche, hook type, why it works. These become reference material for later prompt work.

## 4. What changes, by group

### Group A: code polish and foundation (build first)
- **Settings from `.env`.** The CLI entry point loads `ANTHROPIC_API_KEY` and all `CLIPFORGE_*` settings from `.env` in the project root, using python-dotenv as the user asked. Real environment variables take precedence. Importing ClipForge modules must **not** load `.env`, so the test suite never picks up Diego's real settings or key. Nothing ever prints secret values.
- **Deterministic flaky test.** `tests/test_render.py::test_tracked_crop_actually_moves` must pass every time, under load too, and must still prove that the camera moves.
- **Lint and architecture map.** `pyflakes` reports nothing, and CI enforces that from then on. `docs/ARCHITECTURE.md` gives a one-page map of the 7 stages: their JSON inputs and outputs, the manifest stage each one sets, and the swappable engines.

### Group B: review website (`clipforge/server/`)
- **Server hardening.** Job ids can't escape the workspace (`..`, `.uploads`, dot-files, odd characters). Clip ids are validated. Uploads that aren't video, are empty, have no name, or have path-like names are rejected with a readable 400 and leave no files behind. The API reports whether a job's pipeline is currently running, so the page knows when to stop polling. It also refuses to start a second concurrent run of the same job.
- **Safe, non-destructive page.** All user, transcript and model text is displayed literally, with no HTML injection and no ids inside inline handlers. Refreshes update in place and don't rebuild the grid, so playing videos and typed inputs survive. Polling runs only while something is in flight. Inline toasts replace `alert()`.
- **Review speed.** Keyboard shortcuts, filter, sort by score, and "set trim from playhead". The trim fields stay in **absolute source seconds**, because the API and the metrics use that unit.
- **Upload UX.** Drag-and-drop, a progress bar, and client-side file-type feedback.
- The frontend is a single file (`static/index.html`), so all frontend features run **one after another**.

### Group C: clip-quality upgrade (later)
1. **Platform profiles.** Each platform has its own editable settings file: max length, target length range, 1080x1920 output, caption safe zone and main/side tier. Later steps read these files and hard-code none of these values.
2. **Clip quality report.** Each rendered clip gets a per-clip report file that later features fill in: hook score, originality checklist, pacing flags, variants, loop point and exports. A post-render quality pass writes it. It is a shared foundation and gets a feature of its own.
3. **Smarter highlight picking.** Each candidate also carries:
   - a trigger time, reaction-peak time and payoff-end time
   - a one-line context setup a stranger needs
   - a hook type
   - a "would a stranger get this?" check

   Candidates that need backstory are rejected and the rejection is recorded. Old `candidates.json` files still load.
4. **Hook-first trimming and hook score.** Clips open on the hook or trigger, not the preamble. A 0–100 first-3-seconds hook score is made of sub-scores: speech in second one, motion in second one, a face visible, and on-screen text present. Each sub-score has a reason.
5. **Originality layer.** Adds a context text overlay in the opening seconds, a punch-in zoom on the reaction peak (following the existing face-tracking crop path), and an optional end card. A per-clip checklist records each value-add as applied or skipped, with the reason.
6. **Captions and pacing.** Captions are word-chunked and placed inside the safe zone shared by all main platforms. A pacing check flags any stretch over 3 s with no visual change.
7. **Hook variants and loop point.** Each clip gets 2–3 alternative openings, each with its own hook score. The pipeline also suggests a loop-friendly end point.
8. **Per-platform exports.** Each platform's file is derived from the single master render, so the full render doesn't run again. Each export has a text file with title, caption, hashtags and a credit line. Text generation is a swappable engine with a mock.
9. **Review control panel.** The website shows everything above (hook score, checklist, pacing flags, variants side by side). It adds per-platform approve, an export button and a trend-reference log form.
10. **Clip database.** SQLite in the workspace. It records every exported clip and the choices behind it (hook type, variant, value-adds, platform), plus a pasted post URL.
11. **Analytics CSV import.** Imports YouTube, TikTok and Instagram analytics exports into the database, matched to clips through post URLs. Connecting to the platform APIs is out of scope because it needs Diego's credentials.
12. **Weekly ranking and learning.** Ranks clips fairly across platforms and creators. Top performers become positive few-shot examples and flops become negative examples, **but only the ones Diego approves**.

### Group D: v2 roadmap (confirmed by Diego, 2026-10-13)
The full roadmap, with reasons and sources, is the "ClipForge v2 Roadmap" doc (https://claude.ai/code/artifact/a3f102b2-a1a6-44ad-b68f-502685badf6b). ClipForge grows from "podcast and stream clipper" into an editor for several content types: talk/podcast, gaming, storytime, talking head, cinematic and day-in-the-life. Each phase ends at a gate measured on the benchmark set (F25) before the next starts. Group C features keep their ids and move into these phases.

- **Phase 0, measure first.** F25: a benchmark command that runs a fixed set of Diego's real videos and records clips found, approval rate, minutes, tokens and $ per video. Every later feature must not get worse on the talk video and must get better on its target type.
- **Phase 1, editor-style workspace (F26–F30).** The review page becomes a three-pane editor (media bin, 9:16 player, inspector) with a waveform timeline. Creators are folders with a brand kit. Jobs can be deleted (to a trash with undo), failed stages retried, captions corrected and one clip re-rendered. Still vanilla JS with no build step; wavesurfer.js (BSD-3) is loaded from a CDN.
- **Phase 2, recipe engine (F08, F09, F31, F32, F10, F33, F34).** A recipe is one versioned data file per content type that tells every stage how to edit. The new Signals stage (F32) measures loudness, scene cuts (PySceneDetect, BSD-3), motion and faces locally, so highlight picking can find moments with little speech. F34 cuts AI cost: prompt caching, a compact transcript, a cheap first pass, optional batch mode and a usage log.
- **Phase 3, recipes one at a time (F11–F13, F41–F45).** Talk first, then gaming, storytime, talking head and cinematic. Each recipe ships alone and is checked on the benchmark plus 5–10 real test posts.
- **Phase 4, multi-file edits (F35–F38, F46).** A job can hold several files; Claude writes an edit plan (`edit_plan.json`) and ffmpeg assembles it. Plans can be reordered on the timeline and exported to editors through OpenTimelineIO (Apache-2.0).
- **Phase 5, learning loop (F14–F20, F39, F40).** Exports, the clip database and analytics import, plus recipe A/B tracking and a trend log that turns into recipe proposals Diego approves.

Decisions: gaming is the first recipe after talk; the UI stays vanilla JS; Signals is an 8th stage; multi-file editing comes after the gaming recipe; the repo gets an MIT license. Code copied (not just ideas) from MIT/BSD projects keeps its notice in `THIRD_PARTY_NOTICES.md`; GPL/AGPL code is never copied in.

## 5. Data model sketch

These are additions. Every existing file keeps its shape or gains only optional fields.

**Per job (`workspace/<job>/`)**
- `candidates.json` gains schema v2, with these optional per-candidate fields:
  - `trigger_t`, `reaction_peak_t`, `payoff_end_t` (absolute seconds, ordered, inside the clip)
  - `context_setup` (one line)
  - `hook_type` (fixed vocabulary)
  - `stranger_check` (pass/fail plus why)

  It also gets a `rejected` list holding the candidates dropped for needing backstory. v1 files still load.
- `refined.json` keeps its shape. Hook-first trimming changes the cut points, not the contract.
- `renders/clip_NNN.mp4` is the **master** render. Variants sit next to it.
- `renders/clip_NNN.report.json` (new) is the per-clip quality report. Its sections are `hook_score` (total plus sub-scores and reasons), `value_adds` (list of {name, applied, reason}), `pacing_flags` (list of [start, end]), `variants` (list of {label, file, hook_score}), `loop` (score and suggested end) and `exports` (per platform: file, status, reason). Every section is optional.
- `reviews.json` (backward compatible). A review can carry per-platform decisions. Old reviews are read as "same decision for all platforms".
- `exports/<platform>/clip_NNN.mp4` and `clip_NNN.txt`.

**Per workspace**
- `clipforge.db` (SQLite, git-ignored). It holds:
  - exported clips: job, source, creator, clip timing, hook text/type, hook score, variant, value-adds, platform, file, post URL
  - analytics snapshots per clip per day: views, likes, comments, shares, average view duration or retention when present
  - example decisions: positive or negative, approved when, by Diego
- `references.json` holds trend references (URL, platform, niche, hook type, notes, date).

**Per install**
- Platform profiles: one data file per platform inside the package, edited by hand.
- Few-shot library: the existing `examples/` directory holds positive examples. Negative examples are kept separately, so the prompt can label them "avoid".

**Settings:** `.env` in the project root, with `CLIPFORGE_*` keys. New knobs from Group C, such as value-adds on/off, variant count and end-card text, are `CLIPFORGE_*` settings too, so they work from `.env`.

## 6. Visual direction (review website)

Keep the current look: dark background, monospace, teal accent, the green/red approve/reject states, and the existing CSS variables. It should feel like a dense, keyboard-first tool, not a marketing page. Avoid modal dialogs; use inline toasts and badges. Small badges carry status (hook score, pacing flags, per-platform approvals) so a whole grid can be scanned at a glance. Variants sit side by side in the clip card. There is still no build step: vanilla JS in one static file.

## 7. Definition of done for every feature
- Its acceptance criteria pass, checked by a test or on the running website with mock engines (`CLIPFORGE_ASR=mock CLIPFORGE_HL=mock CLIPFORGE_DETECTOR=haar`).
- The full suite still passes, offline. Tests never call Claude, never download models and never read the project's real `.env`.
- `python -m pyflakes clipforge tests` stays clean (from F03 onward).
- New test videos are short synthetic ones (≤ 12 s, ffmpeg `testsrc`/`sine`/`gradients`). Keep the full-suite runtime reasonable; render-heavy features should reuse module-scoped fixtures.

## 8. Out of scope
- Posting to any platform automatically, or scheduling posts.
- YouTube, Instagram, TikTok, X or Facebook API connections. Analytics arrive only as CSV exports, and auth or OAuth flows are not planned.
- Multi-user use, auth, cloud hosting, job queues, or exposing the server beyond localhost.
- Trending or licensed music baked into exported files, AI voiceover, stock footage, or face/voice swaps. Trending sounds are added in each platform's app at posting time. A track the creator supplies may be used for beat-matched cuts (Group D).
- Processing footage Diego doesn't have rights to.
- Rewriting the stage architecture or the manifest state machine. Adding a stage built the same way (JSON in and out, idempotent, verifies its own outputs) is allowed: F32 adds Signals as the 8th stage.
