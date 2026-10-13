"""
Module 7 tests: the whole product exercised through the HTTP API.

Uses sync_pipeline=True (stages run inline) and mock engines, so this is a
true end-to-end test: upload bytes in -> rendered clips + reviews +
metrics + few-shot export out.
"""
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clipforge.pipeline import PipelineConfig, run_pipeline
from clipforge.review.store import (
    Review, ReviewSet, export_examples, job_metrics, load_reviews,
    save_reviews,
)
from clipforge.server.app import create_app

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def source_video():
    FIX.mkdir(exist_ok=True)
    src = FIX / "speech_60s.mp4"
    if not src.exists():
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=60:size=640x360:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=300:duration=60",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(src)],
            check=True, capture_output=True)
    return src


MOCK_CFG = PipelineConfig(asr_engine="mock", highlight_engine="mock",
                          detector="haar")


@pytest.fixture(scope="module")
def client(tmp_path_factory, source_video):
    ws = tmp_path_factory.mktemp("ws")
    app = create_app(ws, config=MOCK_CFG, sync_pipeline=True)
    return TestClient(app, raise_server_exceptions=False)


# ------------------------------------------------------------ orchestrator

def test_pipeline_config_fallbacks(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = PipelineConfig(highlight_engine="anthropic",
                         detector="mediapipe").resolve()
    assert cfg.highlight_engine == "mock"      # no key -> mock
    assert cfg.detector in ("haar", "mediapipe")  # haar if mp not installed


def test_run_pipeline_end_to_end(tmp_path, source_video):
    from clipforge.ingest.ingest import ingest
    m = ingest(source_video, tmp_path, job_id="orch")
    m = run_pipeline(tmp_path / "orch", MOCK_CFG, progress=lambda s: None)
    assert m.stage.value == "rendered"
    assert list((tmp_path / "orch" / "renders").glob("clip_*.mp4"))
    # re-run = no-op (idempotent orchestration)
    n = len(m.history)
    m2 = run_pipeline(tmp_path / "orch", MOCK_CFG, progress=lambda s: None)
    assert len(m2.history) == n


# ------------------------------------------------------------------- API

def test_upload_and_full_pipeline(client, source_video):
    with source_video.open("rb") as f:
        r = client.post("/api/jobs",
                        files={"file": ("myvideo.mp4", f, "video/mp4")})
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]

    r = client.get(f"/api/jobs/{jid}")
    assert r.status_code == 200
    d = r.json()
    assert d["stage"] == "rendered"
    assert len(d["clips"]) >= 1
    assert all(c["video_ready"] for c in d["clips"])


def test_job_list(client):
    jobs = client.get("/api/jobs").json()
    assert len(jobs) == 1
    assert jobs[0]["filename"] == "myvideo.mp4"
    assert jobs[0]["stage"] == "rendered"


def test_clip_video_served(client):
    jid = client.get("/api/jobs").json()[0]["job_id"]
    cid = client.get(f"/api/jobs/{jid}").json()["clips"][0]["id"]
    r = client.get(f"/api/jobs/{jid}/clips/{cid}/video")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    assert len(r.content) > 10_000
    r = client.get(f"/api/jobs/{jid}/clips/{cid}/qa")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"


def test_review_flow_and_metrics(client):
    jid = client.get("/api/jobs").json()[0]["job_id"]
    clips = client.get(f"/api/jobs/{jid}").json()["clips"]

    # approve first clip untouched
    r = client.post(f"/api/jobs/{jid}/clips/{clips[0]['id']}/review",
                    json={"decision": "approved"})
    assert r.status_code == 200

    if len(clips) > 1:
        # reject second with a trim adjustment
        c1 = clips[1]
        r = client.post(f"/api/jobs/{jid}/clips/{c1['id']}/review",
                        json={"decision": "rejected",
                              "adjusted_start": c1["start"] + 2.0,
                              "note": "hook starts late"})
        assert r.status_code == 200

    m = client.get(f"/api/jobs/{jid}").json()["metrics"]
    assert m["reviewed"] == min(len(clips), 2)
    assert m["approved"] == 1
    assert m["untouched_approvals"] == 1
    if len(clips) > 1:
        assert m["avg_trim_adjustment_sec"] == pytest.approx(1.0, abs=0.01)

    gm = client.get("/api/metrics").json()
    assert gm["total_reviewed"] == m["reviewed"]
    assert gm["overall_approval_rate"] is not None


def test_review_upsert_no_duplicates(client):
    jid = client.get("/api/jobs").json()[0]["job_id"]
    cid = client.get(f"/api/jobs/{jid}").json()["clips"][0]["id"]
    client.post(f"/api/jobs/{jid}/clips/{cid}/review",
                json={"decision": "rejected"})
    client.post(f"/api/jobs/{jid}/clips/{cid}/review",
                json={"decision": "approved"})
    d = client.get(f"/api/jobs/{jid}").json()
    reviews = [c["review"] for c in d["clips"] if c["review"]]
    assert sum(1 for r in reviews if r["clip_id"] == cid) == 1
    assert next(r for r in reviews if r["clip_id"] == cid)["decision"] == "approved"


def test_export_examples_endpoint(client):
    jid = client.get("/api/jobs").json()[0]["job_id"]
    r = client.post(f"/api/jobs/{jid}/export-examples")
    assert r.status_code == 200
    body = r.json()
    assert len(body["written"]) >= 1          # the approved clip
    ex = Path(body["examples_dir"]) / body["written"][0]
    text = ex.read_text(encoding="utf-8")
    assert "clip transcript:" in text and "human-approved" in text


def test_bad_upload_rejected(client, tmp_path):
    r = client.post("/api/jobs",
                    files={"file": ("junk.mp4", b"not a video", "video/mp4")})
    assert r.status_code == 400


def test_unknown_job_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


def test_invalid_decision_400(client):
    jid = client.get("/api/jobs").json()[0]["job_id"]
    r = client.post(f"/api/jobs/{jid}/clips/0/review",
                    json={"decision": "maybe"})
    assert r.status_code == 400


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "CLIPFORGE" in r.text


# ------------------------------------------------------------ store units

def test_review_store_roundtrip(tmp_path):
    rs = ReviewSet()
    rs.upsert(Review(clip_id=0, decision="approved"))
    rs.upsert(Review(clip_id=0, decision="rejected"))   # upsert replaces
    save_reviews(tmp_path, rs)
    loaded = load_reviews(tmp_path)
    assert len(loaded.reviews) == 1
    assert loaded.get(0).decision == "rejected"


def test_metrics_empty_job(tmp_path):
    m = job_metrics(tmp_path)
    assert m["reviewed"] == 0 and m["approval_rate"] is None
