"""
ClipForge CLI.

Usage:
    python -m clipforge.cli ingest <video-file> [--workspace DIR] [--force]
    python -m clipforge.cli status <job-id>    [--workspace DIR]
    python -m clipforge.cli audio check|add ...
"""
import argparse
import datetime
import errno
import functools
import json
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from clipforge.common.manifest import Manifest, is_safe_job_id
from clipforge.ingest.ingest import IngestError, ingest
from clipforge.settings import load_env

DEFAULT_WS = Path.home() / "clipforge-workspace"

FFMPEG_FIX = ("Windows: winget install Gyan.FFmpeg, then open a new terminal; "
              "macOS: brew install ffmpeg; Linux: sudo apt install ffmpeg")
ADDR_IN_USE = {errno.EADDRINUSE, 10048}      # 10048 = WSAEADDRINUSE


def missing_ffmpeg_tools() -> list[str]:
    """Names of ffmpeg/ffprobe that are not on PATH."""
    return [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]


MIN_FFMPEG = (5, 1)    # first release with -fps_mode (replaces -vsync)
MIN_GIT_BUILD_DATE = datetime.date(2022, 7, 1)   # 5.1 branched around here


def parse_ffmpeg_version(banner: str) -> tuple[str, tuple[int, int] | None]:
    """(version text, (major, minor)) from the first line of `ffmpeg -version`.

    Release builds read "ffmpeg version 6.1.1-3ubuntu5" or "n7.0". Git builds
    have no release number, so the tuple is None; see ffmpeg_build_date and
    ffmpeg_problem for how those are judged. Anything unrecognised is
    ("unknown", None).
    """
    m = re.search(r"ffmpeg version\s+(\S+)", banner)
    if not m:
        return "unknown", None
    text = m.group(1)
    r = re.match(r"[nN]?(\d+)\.(\d+)", text)
    return text, (int(r.group(1)), int(r.group(2))) if r else None


def ffmpeg_build_date(text: str) -> datetime.date | None:
    """Build date embedded in a git-build version string, if there is one:
    Gyan "2024-05-13-git-37db0454e4-full_build-..." (leading date) or BtbN
    "N-127388-g615b07170e-20261012" (trailing date). None when undated."""
    m = (re.match(r"(\d{4})-(\d{2})-(\d{2})-git", text)
         or re.search(r"-(\d{4})(\d{2})(\d{2})$", text))
    if not m:
        return None
    try:
        return datetime.date(*map(int, m.groups()))
    except ValueError:
        return None


@functools.lru_cache(maxsize=1)
def ffmpeg_version() -> tuple[str, tuple[int, int] | None]:
    """Detected ffmpeg version; `ffmpeg -version` runs once per process."""
    try:
        p = subprocess.run(["ffmpeg", "-version"], capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "unknown", None
    return parse_ffmpeg_version(p.stdout.splitlines()[0] if p.stdout else "")


def ffmpeg_problem() -> str | None:
    """Why the ffmpeg install is unusable (missing or too old), else None.

    Releases older than 5.1 are refused. Git builds are judged by their build
    date: dated before 2022-07-01 (pre-5.1, no -fps_mode) is refused; undated
    git builds and unrecognised banners are given the benefit of the doubt.
    """
    missing = missing_ffmpeg_tools()
    if missing:
        return (f"{' and '.join(missing)} not found on PATH. Install it: "
                f"{FFMPEG_FIX}")
    text, ver = ffmpeg_version()
    built = ffmpeg_build_date(text) if ver is None else None
    if (ver is not None and ver < MIN_FFMPEG) or (
            built is not None and built < MIN_GIT_BUILD_DATE):
        return (f"ffmpeg {text} is too old: ClipForge needs ffmpeg "
                f"{MIN_FFMPEG[0]}.{MIN_FFMPEG[1]} or newer. Upgrade it: "
                f"{FFMPEG_FIX}")
    return None


def report_ffmpeg() -> None:
    """Print which ffmpeg is in use (stderr, so job output stays parseable)."""
    print(f"ffmpeg: {ffmpeg_version()[0]}", file=sys.stderr)


def port_problem(host: str, port: int) -> str | None:
    """Try to bind host:port and close again; return a message if we can't."""
    try:
        # every address the host name resolves to (localhost -> ::1 and
        # 127.0.0.1), so a server on either one is noticed
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        return f"Cannot listen on {host}:{port}: {e.strerror or e}"
    usable, failure = 0, None
    for family, socktype, proto, _canon, addr in infos:
        try:
            s = socket.socket(family, socktype, proto)
        except OSError as e:
            failure = failure or e
            continue
        try:
            if sys.platform != "win32":
                # like uvicorn: a lingering TIME_WAIT must not count as busy.
                # Never on Windows, where it would let us steal a live port.
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(addr)
            usable += 1
        except OSError as e:
            if e.errno in ADDR_IN_USE:
                return (f"Port {port} is already in use — another ClipForge "
                        "server may still be running. Close it, or start on "
                        "another port: python -m clipforge.cli serve "
                        f"--port {port + 1}")
            failure = failure or e
        finally:
            s.close()
    # an unusable secondary address (e.g. no IPv6) is fine; none usable is not
    if not usable and failure is not None:
        return f"Cannot listen on {host}:{port}: {failure.strerror or failure}"
    return None


def main() -> int:
    load_env()
    p = argparse.ArgumentParser(prog="clipforge")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="Run Module 1 on a video file")
    pi.add_argument("video")
    pi.add_argument("--workspace", default=str(DEFAULT_WS))
    pi.add_argument("--job-id", default=None)
    pi.add_argument("--force", action="store_true")

    pt = sub.add_parser("transcribe", help="Run Module 2 on an ingested job")
    pt.add_argument("job_id")
    pt.add_argument("--workspace", default=str(DEFAULT_WS))
    pt.add_argument("--engine", default="whisper:small",
                    help="whisper[:tiny|base|small|medium|large-v3] or mock")
    pt.add_argument("--language", default=None,
                    help="Force language code, e.g. 'en' (default: auto)")
    pt.add_argument("--force", action="store_true")

    ph = sub.add_parser("highlight", help="Run Module 3 on a transcribed job")
    ph.add_argument("job_id")
    ph.add_argument("--workspace", default=str(DEFAULT_WS))
    ph.add_argument("--engine", default="anthropic",
                    help="anthropic[:model] or mock")
    ph.add_argument("--examples-dir", default=None,
                    help="Directory of .txt few-shot examples")
    ph.add_argument("--min-score", type=int, default=55)
    ph.add_argument("--max-candidates", type=int, default=12)
    ph.add_argument("--force", action="store_true")

    pr = sub.add_parser("refine", help="Run Module 4 on candidate clips")
    pr.add_argument("job_id")
    pr.add_argument("--workspace", default=str(DEFAULT_WS))
    pr.add_argument("--no-energy", action="store_true",
                    help="Skip audio-energy analysis (midpoint cuts only)")
    pr.add_argument("--force", action="store_true")

    pf = sub.add_parser("reframe", help="Run Module 5: 9:16 crop paths")
    pf.add_argument("job_id")
    pf.add_argument("--workspace", default=str(DEFAULT_WS))
    pf.add_argument("--detector", default="mediapipe",
                    help="mediapipe (best, pip install mediapipe) or haar")
    pf.add_argument("--force", action="store_true")

    pn = sub.add_parser("render", help="Run Module 6: final vertical clips")
    pn.add_argument("job_id")
    pn.add_argument("--workspace", default=str(DEFAULT_WS))
    pn.add_argument("--no-captions", action="store_true")
    pn.add_argument("--font", default="Poppins")
    pn.add_argument("--force", action="store_true")

    pp = sub.add_parser("pipeline", help="Run ALL remaining stages on a job")
    pp.add_argument("job_id")
    pp.add_argument("--workspace", default=str(DEFAULT_WS))

    pv = sub.add_parser("serve", help="Start the ClipForge web app")
    pv.add_argument("--workspace", default=str(DEFAULT_WS))
    pv.add_argument("--host", default="127.0.0.1")
    pv.add_argument("--port", type=int, default=8000)

    ps = sub.add_parser("status", help="Show a job's manifest")
    ps.add_argument("job_id")
    ps.add_argument("--workspace", default=str(DEFAULT_WS))

    pa = sub.add_parser("audio", help="Manage the sound-effect and music "
                        "library (see docs/AUDIO_SOURCES.md)")
    asub = pa.add_subparsers(dest="audio_cmd", required=True)
    ac = asub.add_parser("check", help="Validate library.json and list "
                         "every problem")
    ac.add_argument("--library", default=None,
                    help="Library folder (default: CLIPFORGE_AUDIO_LIBRARY "
                         "or ~/clipforge-audio)")
    aa = asub.add_parser("add", help="Copy an audio file into the library "
                         "with its licence and source")
    aa.add_argument("file")
    aa.add_argument("--kind", required=True, choices=["sfx", "music"])
    aa.add_argument("--tags", required=True,
                    help="Comma-separated, e.g. impact or gaming,energetic")
    aa.add_argument("--license", required=True)
    aa.add_argument("--source", required=True,
                    help="Where the file came from (URL)")
    aa.add_argument("--attribution", default="",
                    help="Credit line, when the licence needs one")
    aa.add_argument("--platforms", default="all",
                    help="Comma-separated: all, tiktok, reels, shorts, x, "
                         "facebook (default: all)")
    aa.add_argument("--library", default=None)

    args = p.parse_args()

    if args.cmd == "ingest":
        if args.job_id is not None and not is_safe_job_id(args.job_id):
            print(f"INGEST FAILED: invalid --job-id {args.job_id!r}: use "
                  "letters, digits, '-' and '_', starting with a letter or "
                  "digit (other ids are invisible to the review website)",
                  file=sys.stderr)
            return 2
        problem = ffmpeg_problem()
        if problem:
            print(f"INGEST FAILED: {problem}", file=sys.stderr)
            return 2
        report_ffmpeg()
        try:
            m = ingest(args.video, args.workspace,
                       job_id=args.job_id, force=args.force)
        except IngestError as e:
            print(f"INGEST FAILED: {e}", file=sys.stderr)
            return 1
        print(f"job_id: {m.job_id}")
        print(f"stage:  {m.stage.value}")
        print(f"source: {m.source.original_filename} "
              f"({m.source.duration_sec:.1f}s, "
              f"{m.source.width}x{m.source.height} @ {m.source.fps:.3g}fps)")
        print(f"normalized: {m.ingest.normalized_width}x"
              f"{m.ingest.normalized_height} @ {m.ingest.normalized_fps}fps")
        print(f"job dir: {Path(args.workspace) / m.job_id}")
        return 0

    if args.cmd == "transcribe":
        from clipforge.transcribe.backends import get_transcriber
        from clipforge.transcribe.run import TranscribeError, transcribe_job
        job_dir = Path(args.workspace) / args.job_id
        try:
            kw = {"language": args.language} if args.engine.startswith("whisper") else {}
            m = transcribe_job(job_dir, get_transcriber(args.engine, **kw),
                               force=args.force)
        except TranscribeError as e:
            print(f"TRANSCRIBE FAILED: {e}", file=sys.stderr)
            return 1
        from clipforge.transcribe.schema import Transcript
        t = Transcript.load(job_dir / m.transcript_path)
        print(f"job_id: {m.job_id}")
        print(f"stage:  {m.stage.value}")
        print(f"engine: {t.engine}  language: {t.language}")
        print(f"words: {len(t.words)}  segments: {len(t.segments)}")
        preview = t.full_text[:200]
        print(f"preview: {preview}{'...' if len(t.full_text) > 200 else ''}")
        return 0

    if args.cmd == "highlight":
        from clipforge.highlight.backends import get_highlighter
        from clipforge.highlight.run import HighlightError, highlight_job
        from clipforge.highlight.schema import CandidateSet
        job_dir = Path(args.workspace) / args.job_id
        try:
            hl = get_highlighter(
                args.engine,
                examples_dir=Path(args.examples_dir) if args.examples_dir else None)
            m = highlight_job(job_dir, hl, min_score=args.min_score,
                              max_candidates=args.max_candidates,
                              force=args.force)
        except (HighlightError, ValueError) as e:
            print(f"HIGHLIGHT FAILED: {e}", file=sys.stderr)
            return 1
        cset = CandidateSet.load(job_dir / m.candidates_path)
        print(f"job_id: {m.job_id}\nstage:  {m.stage.value}")
        print(f"engine: {cset.engine}\ncandidates: {len(cset.candidates)}")
        for c in cset.candidates:
            print(f"  #{c.id} [{c.start:7.1f}s - {c.end:7.1f}s] "
                  f"score={c.score:3d} {c.type:8s} hook: {c.hook[:60]}")
        return 0

    if args.cmd == "refine":
        from clipforge.refine.refine import RefinedSet
        from clipforge.refine.run import RefineError, refine_job
        job_dir = Path(args.workspace) / args.job_id
        try:
            m = refine_job(job_dir, force=args.force,
                           use_energy=not args.no_energy)
        except RefineError as e:
            print(f"REFINE FAILED: {e}", file=sys.stderr)
            return 1
        rs = RefinedSet.load(job_dir / m.refined_path)
        print(f"job_id: {m.job_id}\nstage:  {m.stage.value}")
        print(f"clips: {len(rs.clips)}  rejected: {len(rs.rejected)}")
        for c in rs.clips:
            print(f"  #{c.id} cut[{c.start:7.2f}s - {c.end:7.2f}s] "
                  f"({c.duration:5.1f}s, {c.n_words} words) "
                  f"score={c.score} hook: {c.hook[:50]}")
        for r in rs.rejected:
            print(f"  rejected cand {r['candidate_id']}: {r['reason']}")
        return 0

    if args.cmd == "reframe":
        from clipforge.reframe.core import path_jitter
        from clipforge.reframe.detect import CropPath, get_detector
        from clipforge.reframe.run import ReframeError, reframe_job
        job_dir = Path(args.workspace) / args.job_id
        try:
            det = get_detector(args.detector)
        except ImportError:
            print("mediapipe not installed — falling back to haar. "
                  "For better tracking: pip install mediapipe", file=sys.stderr)
            det = get_detector("haar")
        try:
            m = reframe_job(job_dir, det, force=args.force)
        except ReframeError as e:
            print(f"REFRAME FAILED: {e}", file=sys.stderr)
            return 1
        print(f"job_id: {m.job_id}\nstage:  {m.stage.value}")
        for p in sorted((job_dir / m.crop_paths_dir).glob("clip_*.json")):
            cp = CropPath.load(p)
            print(f"  {p.name}: mode={cp.mode:7s} keyframes={len(cp.keyframes):4d} "
                  f"cuts={sum(k.cut for k in cp.keyframes)} "
                  f"jitter={path_jitter(cp):.4f}")
        return 0

    if args.cmd == "render":
        from clipforge.render.captions import CaptionStyle
        from clipforge.render.render import RenderError
        from clipforge.render.run import render_job
        job_dir = Path(args.workspace) / args.job_id
        try:
            m = render_job(job_dir, force=args.force,
                           captions=not args.no_captions,
                           style=CaptionStyle(font=args.font))
        except RenderError as e:
            print(f"RENDER FAILED: {e}", file=sys.stderr)
            return 1
        renders = job_dir / m.renders_dir
        print(f"job_id: {m.job_id}\nstage:  {m.stage.value}")
        for c in sorted(renders.glob("clip_*.mp4")):
            mb = c.stat().st_size / 1e6
            print(f"  {c.name}  ({mb:.1f} MB)   QA sheet: {c.stem}_qa.png")
        print(f"output dir: {renders}")
        return 0

    if args.cmd == "pipeline":
        problem = ffmpeg_problem()
        if problem:
            print(f"PIPELINE FAILED: {problem}", file=sys.stderr)
            return 2
        report_ffmpeg()
        from clipforge.pipeline import PipelineConfig, run_pipeline
        from clipforge.recipes import RecipeError, recipe_problem
        problem = recipe_problem()
        if problem:
            print(f"PIPELINE FAILED: {problem}", file=sys.stderr)
            return 2
        try:
            m = run_pipeline(Path(args.workspace) / args.job_id,
                             PipelineConfig())
        except RecipeError:
            print("PIPELINE FAILED: the recipe could not be loaded "
                  "(details above)", file=sys.stderr)
            return 2
        print(f"job {m.job_id} -> {m.stage.value}")
        return 0

    if args.cmd == "serve":
        problem = port_problem(args.host, args.port)
        if problem:
            print(problem, file=sys.stderr)
            return 2
        from clipforge.recipes import recipe_problem
        problem = recipe_problem()
        if problem:
            print(f"SERVE FAILED: {problem}", file=sys.stderr)
            return 2
        warning = ffmpeg_problem()
        if warning:
            print(f"WARNING: {warning}\n"
                  "The review page will load, but uploads and rendering "
                  "will fail until that is fixed.", file=sys.stderr)
        else:
            report_ffmpeg()
        import uvicorn
        from clipforge.server.app import create_app
        app = create_app(Path(args.workspace))
        print(f"ClipForge running at http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
        return 0

    if args.cmd == "status":
        m = Manifest.load(Path(args.workspace) / args.job_id)
        print(json.dumps(json.loads(m.model_dump_json()), indent=2))
        return 0

    if args.cmd == "audio":
        return audio_command(args)
    return 2


def audio_command(args) -> int:
    from clipforge.audio.library import (
        LIBRARY_FILE, add_file, default_library_dir, load_library,
    )
    root = Path(args.library) if args.library else default_library_dir()
    if args.audio_cmd == "add":
        split = lambda v: [x.strip() for x in v.split(",") if x.strip()]
        try:
            item = add_file(Path(args.file), args.kind, split(args.tags),
                            args.license, args.source, args.attribution,
                            split(args.platforms) or ["all"], root=root)
        except ValueError as e:
            print(f"AUDIO ADD FAILED: {e}", file=sys.stderr)
            return 1
        except OSError as e:
            print(f"AUDIO ADD FAILED: could not copy the file "
                  f"({e.strerror or 'I/O error'})", file=sys.stderr)
            return 1
        print(f"added {item.file} ({item.kind}, tags: {', '.join(item.tags)})")
        return 0
    lib = load_library(root)
    if not lib.found:
        print(f"No {LIBRARY_FILE} in {root}. Add a file with "
              "`clipforge audio add` to create one; until then the built-in "
              "sound effects are used and there is no music.")
        return 0
    sfx, music = len(lib.of_kind("sfx")), len(lib.of_kind("music"))
    print(f"{root}: {sfx} sound effects, {music} music tracks usable")
    for prob in lib.problems:
        print(f"  problem: {prob}")
    return 1 if lib.problems else 0


if __name__ == "__main__":
    sys.exit(main())
