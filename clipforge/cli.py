"""
ClipForge CLI.

Usage:
    python -m clipforge.cli ingest <video-file> [--workspace DIR] [--force]
    python -m clipforge.cli status <job-id>    [--workspace DIR]
"""
import argparse
import errno
import json
import shutil
import socket
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


def ffmpeg_problem() -> str | None:
    missing = missing_ffmpeg_tools()
    if not missing:
        return None
    return (f"{' and '.join(missing)} not found on PATH. Install it: "
            f"{FFMPEG_FIX}")


def port_problem(host: str, port: int) -> str | None:
    """Try to bind host:port and close again; return a message if we can't."""
    s = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET,
                      socket.SOCK_STREAM)
    try:
        if sys.platform != "win32":
            # like uvicorn: a lingering TIME_WAIT must not count as "busy".
            # Never on Windows, where it would let us steal a live port.
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
    except OSError as e:
        if e.errno in ADDR_IN_USE:
            return (f"Port {port} is already in use — another ClipForge "
                    "server may still be running. Close it, or start on "
                    "another port: python -m clipforge.cli serve "
                    f"--port {port + 1}")
        return f"Cannot listen on {host}:{port}: {e.strerror or e}"
    finally:
        s.close()
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
        from clipforge.pipeline import PipelineConfig, run_pipeline
        m = run_pipeline(Path(args.workspace) / args.job_id, PipelineConfig())
        print(f"job {m.job_id} -> {m.stage.value}")
        return 0

    if args.cmd == "serve":
        problem = port_problem(args.host, args.port)
        if problem:
            print(problem, file=sys.stderr)
            return 2
        warning = ffmpeg_problem()
        if warning:
            print(f"WARNING: {warning}\n"
                  "The review page will load, but uploads and rendering "
                  "will fail until it is installed.", file=sys.stderr)
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
    return 2


if __name__ == "__main__":
    sys.exit(main())
