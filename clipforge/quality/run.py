"""
Post-render quality pass: fills the hook_score section of each clip's report.

Best effort and idempotent: a clip that already has a hook_score is skipped,
and a clip that fails is skipped (and retried on the next run) without
failing the others.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from clipforge.common.manifest import Manifest
from clipforge.quality.hook import hook_score
from clipforge.quality.report import load_report, update_section
from clipforge.refine.refine import RefinedSet
from clipforge.render.captions import group_words
from clipforge.transcribe.schema import Transcript


def _overlay_span(rep) -> list[tuple[float, float]]:
    """Clip-relative (start, end) of the applied context overlay, read from
    the report's value_adds. Missing, not applied or malformed -> []."""
    try:
        for v in (rep.value_adds or []) if rep is not None else []:
            if isinstance(v, dict) and v.get("name") == "context_overlay" \
                    and v.get("applied") is True:
                a, b = v["start"], v["end"]
                ok = all(isinstance(x, (int, float))
                         and not isinstance(x, bool) and math.isfinite(x)
                         for x in (a, b))
                if ok and 0 <= a < b:
                    return [(float(a), float(b))]
    except Exception:  # noqa: BLE001 - caption-only scoring still works
        pass
    return []


def hook_scores_job(job_dir: Path, detector, captions: bool = True) -> int:
    """Write hook_score for every rendered clip that lacks one. Returns the
    number of clips scored."""
    job_dir = Path(job_dir)
    m = Manifest.load(job_dir)
    rs = RefinedSet.load(job_dir / m.refined_path)
    t = Transcript.load(job_dir / m.transcript_path)
    done = failed = 0
    for clip in rs.clips:
        mp4 = job_dir / "renders" / f"clip_{clip.id:03d}.mp4"
        if not mp4.exists():
            continue
        rep = load_report(job_dir, clip.id)
        if rep is not None and rep.hook_score is not None:
            continue
        try:
            spans = _overlay_span(rep)
            if captions:
                spans += [(c.start, c.end)
                          for c in group_words(t.words, clip.start,
                                               clip.end)]
            score = hook_score(mp4, t.words, clip.start, clip.end,
                               detector, spans)
            update_section(job_dir, clip.id, "hook_score", score)
            done += 1
        except Exception:  # noqa: BLE001 - one clip must not block the rest
            failed += 1
    if failed:
        print(f"hook score failed for {failed} clip(s); "
              "re-run the pipeline to retry", file=sys.stderr)
    return done
