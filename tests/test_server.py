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
    Review, ReviewSet, job_metrics, load_reviews, save_reviews,
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


@pytest.mark.parametrize("exc", [ImportError("no mediapipe"),
                                 RuntimeError("protobuf mismatch")])
def test_pipeline_falls_back_to_haar_without_mediapipe(monkeypatch, exc):
    """Covers both 'not installed' and 'installed but import raises'."""
    import importlib
    real = importlib.import_module

    def fake(name, *a, **k):
        if name == "mediapipe":
            raise exc
        return real(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", fake)
    cfg = PipelineConfig(detector="mediapipe").resolve()
    assert cfg.detector == "haar"


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
    text = ex.read_text()
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


# ------------------------------------------------------- F04: hardening

@pytest.fixture()
def fresh(tmp_path):
    """A private workspace (inside tmp_path) and an app that runs inline."""
    ws = tmp_path / "ws"
    app = create_app(ws, config=MOCK_CFG, sync_pipeline=True)
    return ws, TestClient(app, raise_server_exceptions=False)


def _job_dirs(ws):
    return sorted(p.name for p in ws.iterdir() if not p.name.startswith("."))


def _uploads_empty(ws):
    up = ws / ".uploads"
    return not up.exists() or not any(up.iterdir())


def test_job_id_check():
    from clipforge.common.manifest import new_job_id
    from clipforge.server.app import is_safe_job_id
    for bad in ["..", ".", ".uploads", ".hidden", "a/b", "a\\b", "", "a b",
                "../x"]:
        assert not is_safe_job_id(bad), bad
    for _ in range(5):
        assert is_safe_job_id(new_job_id())
    assert is_safe_job_id("orch")


@pytest.mark.parametrize("bad", ["..", ".uploads", ".hidden", "%2e%2e",
                                 "a%2fb", "a%5cb", "..%2foutside"])
def test_unsafe_job_ids_get_404(fresh, bad):
    ws, c = fresh
    routes = [("get", "/api/jobs/{}"), ("post", "/api/jobs/{}/rerun"),
              ("get", "/api/jobs/{}/clips/0/video"),
              ("get", "/api/jobs/{}/clips/0/qa"),
              ("post", "/api/jobs/{}/export-examples")]
    for method, tmpl in routes:
        r = getattr(c, method)(tmpl.format(bad))
        assert r.status_code in (400, 404), (tmpl, bad, r.status_code)
    r = c.post(f"/api/jobs/{bad}/clips/0/review", json={"decision": "approved"})
    assert r.status_code in (400, 404)


def test_manifest_outside_workspace_is_unreachable(tmp_path, source_video):
    from clipforge.ingest.ingest import ingest
    ws = tmp_path / "ws"
    app = create_app(ws, config=MOCK_CFG, sync_pipeline=True)
    c = TestClient(app, raise_server_exceptions=False)
    # a complete, valid job that lives NEXT TO the workspace, not in it
    ingest(source_video, tmp_path, job_id="outside")
    renders = tmp_path / "outside" / "renders"
    renders.mkdir()
    (renders / "clip_000.mp4").write_bytes(b"x" * 100)
    (renders / "clip_000_qa.png").write_bytes(b"x" * 100)
    assert (tmp_path / "outside" / "manifest.json").exists()

    for jid in ["..%2foutside", "%2e%2e%2foutside", "..\\outside",
                "../outside", "outside"]:
        for url in [f"/api/jobs/{jid}", f"/api/jobs/{jid}/clips/0/video",
                    f"/api/jobs/{jid}/clips/0/qa"]:
            assert c.get(url).status_code in (400, 404), url
        assert c.post(f"/api/jobs/{jid}/rerun").status_code in (400, 404)
        assert c.post(f"/api/jobs/{jid}/export-examples").status_code in (400, 404)
        assert c.post(f"/api/jobs/{jid}/clips/0/review",
                      json={"decision": "approved"}).status_code in (400, 404)
    assert c.get("/api/jobs").json() == []
    assert c.get("/api/metrics").json()["total_reviewed"] == 0


@pytest.mark.parametrize("name,ctype,data", [
    ("notes.txt", "text/plain", b"hello"),
    ("pic.png", "image/png", b"\x89PNG\r\n\x1a\n" + b"0" * 50),
    ("empty.mp4", "video/mp4", b""),
    ("", "video/mp4", b"abc"),
    ("..", "video/mp4", b"abc"),
    ("../../x.mp4", "video/mp4", b"not a video"),
    ("..\\..\\x.mp4", "video/mp4", b"not a video"),
])
def test_bad_uploads_rejected_cleanly(fresh, name, ctype, data):
    ws, c = fresh
    r = c.post("/api/jobs", files={"file": (name, data, ctype)})
    assert r.status_code in (400, 422), r.text
    assert _job_dirs(ws) == []
    assert _uploads_empty(ws)
    assert not (ws.parent / "x.mp4").exists()


def test_non_video_message_names_accepted_types(fresh):
    ws, c = fresh
    r = c.post("/api/jobs", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert r.status_code == 400
    assert ".mp4" in r.json()["detail"] and ".mov" in r.json()["detail"]


def test_traversal_filename_is_stored_safely(fresh, source_video, monkeypatch):
    from clipforge.server import app as appmod
    monkeypatch.setattr(appmod, "run_pipeline", lambda d, cfg, progress=None: None)
    ws, c = fresh
    with source_video.open("rb") as f:
        r = c.post("/api/jobs", files={"file": ("../../evil.mp4", f, "video/mp4")})
    assert r.status_code == 200, r.text
    assert not (ws.parent / "evil.mp4").exists()
    detail = c.get(f"/api/jobs/{r.json()['job_id']}").json()
    assert detail["source"]["original_filename"] == "evil.mp4"


def test_running_flag_false_after_sync_upload(fresh, source_video,
                                              monkeypatch):
    from clipforge.server import app as appmod
    seen = []
    monkeypatch.setattr(appmod, "run_pipeline",
                        lambda d, cfg, progress=None: seen.append(d.name))
    ws, c = fresh
    with source_video.open("rb") as f:
        jid = c.post("/api/jobs",
                     files={"file": ("a.mp4", f, "video/mp4")}).json()["job_id"]
    assert seen == [jid]                         # the pipeline did run inline
    assert c.get(f"/api/jobs/{jid}").json()["running"] is False
    jobs = c.get("/api/jobs").json()
    assert jobs[0]["running"] is False


def test_rerun_while_running_is_409(tmp_path, source_video, monkeypatch):
    import threading
    from clipforge.ingest.ingest import ingest
    from clipforge.server import app as appmod

    started, release = threading.Event(), threading.Event()
    calls = []

    def blocking_pipeline(job_dir, cfg, progress=None):
        calls.append(job_dir)
        started.set()
        assert release.wait(30)

    monkeypatch.setattr(appmod, "run_pipeline", blocking_pipeline)
    ws = tmp_path / "ws"
    c = TestClient(create_app(ws, config=MOCK_CFG, sync_pipeline=False),
                   raise_server_exceptions=False)
    ingest(source_video, ws, job_id="job1")

    assert c.post("/api/jobs/job1/rerun").status_code == 200
    assert started.wait(10)
    assert c.get("/api/jobs/job1").json()["running"] is True
    assert [j["running"] for j in c.get("/api/jobs").json()] == [True]

    assert c.post("/api/jobs/job1/rerun").status_code == 409
    assert len(calls) == 1                       # no second run started

    release.set()
    for _ in range(100):                         # wait for the flag to clear
        if not c.get("/api/jobs/job1").json()["running"]:
            break
        threading.Event().wait(0.05)
    assert c.get("/api/jobs/job1").json()["running"] is False
    started.clear()
    assert c.post("/api/jobs/job1/rerun").status_code == 200   # allowed again
    assert started.wait(10)
    assert len(calls) == 2


def test_running_flag_clears_when_pipeline_raises(tmp_path, source_video,
                                                  monkeypatch):
    from clipforge.ingest.ingest import ingest
    from clipforge.server import app as appmod

    def boom(job_dir, cfg, progress=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(appmod, "run_pipeline", boom)
    ws = tmp_path / "ws"
    c = TestClient(create_app(ws, config=MOCK_CFG, sync_pipeline=True),
                   raise_server_exceptions=False)
    ingest(source_video, ws, job_id="job1")
    assert c.post("/api/jobs/job1/rerun").status_code == 200
    assert c.get("/api/jobs/job1").json()["running"] is False
    assert c.post("/api/jobs/job1/rerun").status_code == 200


@pytest.fixture(scope="module")
def ingested_job(tmp_path_factory, source_video):
    from clipforge.ingest.ingest import ingest
    ws = tmp_path_factory.mktemp("ws_ingested")
    c = TestClient(create_app(ws, config=MOCK_CFG, sync_pipeline=True),
                   raise_server_exceptions=False)
    return c, ingest(source_video, ws, job_id="job1").job_id


@pytest.mark.parametrize("cid", ["-1", "-5", "abc", "1.5"])
def test_bad_clip_ids_are_404_or_422(ingested_job, cid):
    client, jid = ingested_job
    assert client.get(f"/api/jobs/{jid}/clips/{cid}/video").status_code in (404, 422)
    assert client.get(f"/api/jobs/{jid}/clips/{cid}/qa").status_code in (404, 422)
    r = client.post(f"/api/jobs/{jid}/clips/{cid}/review",
                    json={"decision": "approved"})
    assert r.status_code in (404, 422)
