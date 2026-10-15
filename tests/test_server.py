"""
Module 7 tests: the whole product exercised through the HTTP API.

Uses sync_pipeline=True (stages run inline) and mock engines, so this is a
true end-to-end test: upload bytes in -> rendered clips + reviews +
metrics + few-shot export out.
"""
import re
import shutil
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
    from clipforge.common.manifest import is_safe_job_id, new_job_id
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


def test_manifest_in_workspace_parent_unreachable_via_dotdot(tmp_path):
    """Regression for the original hole: /api/jobs/%2e%2e resolved to the
    workspace's PARENT, so a manifest.json there was served as a job."""
    from clipforge.common.manifest import Manifest
    ws = tmp_path / "ws"
    c = TestClient(create_app(ws, config=MOCK_CFG, sync_pipeline=True),
                   raise_server_exceptions=False)
    Manifest(job_id="parent").save(tmp_path)          # tmp_path/manifest.json
    assert (tmp_path / "manifest.json").exists()
    (tmp_path / "renders").mkdir()
    (tmp_path / "renders" / "clip_000.mp4").write_bytes(b"x" * 100)
    (tmp_path / "renders" / "clip_000_qa.png").write_bytes(b"x" * 100)

    jid = "%2e%2e"
    assert c.get(f"/api/jobs/{jid}").status_code == 404
    assert c.post(f"/api/jobs/{jid}/rerun").status_code == 404
    assert c.get(f"/api/jobs/{jid}/clips/0/video").status_code == 404
    assert c.get(f"/api/jobs/{jid}/clips/0/qa").status_code == 404
    assert c.post(f"/api/jobs/{jid}/clips/0/review",
                  json={"decision": "approved"}).status_code == 404
    assert c.post(f"/api/jobs/{jid}/export-examples").status_code == 404
    assert not (tmp_path / "reviews.json").exists()


def test_very_long_filename_is_not_a_500(fresh, source_video, monkeypatch):
    from clipforge.server import app as appmod
    monkeypatch.setattr(appmod, "run_pipeline", lambda d, cfg, progress=None: None)
    ws, c = fresh
    long_name = "a" * 300 + ".mp4"
    with source_video.open("rb") as f:
        r = c.post("/api/jobs", files={"file": (long_name, f, "video/mp4")})
    assert r.status_code in (200, 400), r.text
    if r.status_code == 200:
        fn = c.get(f"/api/jobs/{r.json()['job_id']}").json()["source"][
            "original_filename"]
        assert fn.endswith(".mp4") and len(fn) < 200
    assert str(ws) not in r.text
    # a long name with no usable extension is a clean 400
    r = c.post("/api/jobs", files={"file": ("b" * 300, b"abc", "video/mp4")})
    assert r.status_code == 400 and str(ws) not in r.text
    assert _uploads_empty(ws)


def test_unexpected_ingest_error_cleans_up(fresh, monkeypatch):
    from clipforge.server import app as appmod

    def boom(src, workspace, job_id=None, force=False):
        (workspace / job_id).mkdir(parents=True)        # half-made job dir
        (workspace / job_id / "source.mp4").write_bytes(b"x")
        raise ValueError(f"bad thing at {src}")

    monkeypatch.setattr(appmod, "ingest", boom)
    ws, c = fresh
    r = c.post("/api/jobs", files={"file": ("a.mp4", b"abc", "video/mp4")})
    assert r.status_code in (400, 500)
    assert "bad thing" not in r.text and str(ws) not in r.text
    assert _job_dirs(ws) == []
    assert _uploads_empty(ws)


def test_workspace_metrics_skips_unsafe_dirs(tmp_path):
    from clipforge.common.manifest import Manifest
    from clipforge.review.store import workspace_metrics
    for name in ("good", ".hidden", "has space"):
        d = tmp_path / name
        Manifest(job_id=name).save(d)
        rs = ReviewSet()
        rs.upsert(Review(clip_id=0, decision="approved"))
        save_reviews(d, rs)
    m = workspace_metrics(tmp_path)
    assert list(m["per_job"]) == ["good"]
    assert m["jobs_with_reviews"] == 1 and m["total_reviewed"] == 1


@pytest.mark.parametrize("bad", ["..", ".hidden", "a/b", "a b", ""])
def test_cli_ingest_rejects_unsafe_job_id(tmp_path, monkeypatch, capsys, bad):
    import sys
    from clipforge import cli
    monkeypatch.setattr(cli, "load_env", lambda *a, **k: None)
    called = []
    monkeypatch.setattr(cli, "ingest", lambda *a, **k: called.append(1))
    monkeypatch.setattr(sys, "argv", ["clipforge", "ingest", "v.mp4",
                                      "--workspace", str(tmp_path / "ws"),
                                      "--job-id", bad])
    assert cli.main() != 0
    assert not called
    assert "job-id" in capsys.readouterr().err
    assert not (tmp_path / "ws").exists()


# ------------------------------------------- F07: upload + review robustness

def _real_server(app):
    """Run the app under uvicorn in a thread, so requests really overlap
    (TestClient gives each request its own event loop)."""
    import socket
    import threading
    import uvicorn
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    import time
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started
    return server, t, f"http://127.0.0.1:{port}"


def test_slow_upload_does_not_block_other_requests(tmp_path, source_video,
                                                   monkeypatch):
    import threading
    import time
    from types import SimpleNamespace

    import httpx

    from clipforge.server import app as appmod

    def slow_ingest(path, workspace, job_id=None):
        time.sleep(3)                       # stands in for a 90 s video
        return SimpleNamespace(job_id=job_id)

    monkeypatch.setattr(appmod, "ingest", slow_ingest)
    monkeypatch.setattr(appmod, "run_pipeline", lambda d, cfg, progress=None: None)
    server, thread, base = _real_server(
        create_app(tmp_path / "ws", config=MOCK_CFG))
    try:
        result = {}

        def upload():
            with source_video.open("rb") as f:
                result["r"] = httpx.post(
                    base + "/api/jobs", timeout=30,
                    files={"file": ("slow.mp4", f, "video/mp4")})

        up = threading.Thread(target=upload)
        up.start()
        time.sleep(0.5)                     # the upload is now inside ingest
        assert up.is_alive()
        worst = 0.0
        for _ in range(5):
            t0 = time.time()
            assert httpx.get(base + "/api/jobs", timeout=10).status_code == 200
            worst = max(worst, time.time() - t0)
            time.sleep(0.1)
        assert up.is_alive(), "ingest finished too early to prove anything"
        assert worst < 1.0, f"GET /api/jobs took {worst:.2f}s during an upload"
        up.join(15)
        assert result["r"].status_code == 200
    finally:
        server.should_exit = True
        thread.join(10)


@pytest.fixture(scope="module")
def _ingested_once(tmp_path_factory, source_video):
    from clipforge.ingest.ingest import ingest
    ws = tmp_path_factory.mktemp("ws_f07")
    ingest(source_video, ws, job_id="job1")
    return ws / "job1"


@pytest.fixture()
def ingested(tmp_path, _ingested_once):
    """A private workspace holding one ingested (not yet processed) job."""
    ws = tmp_path / "ws"
    app = create_app(ws, config=MOCK_CFG, sync_pipeline=True)
    shutil.copytree(_ingested_once, ws / "job1")
    return ws, TestClient(app, raise_server_exceptions=False), None


def test_simultaneous_reviews_are_all_saved(ingested):
    import threading
    ws, c, _ = ingested
    n = 8
    barrier = threading.Barrier(n)
    codes: dict[int, int] = {}

    def post(cid):
        barrier.wait()
        r = c.post(f"/api/jobs/job1/clips/{cid}/review",
                   json={"decision": "approved", "note": f"n{cid}"})
        codes[cid] = r.status_code

    for rnd in range(3):                    # a race is not certain: try 3 rounds
        ids = range(rnd * n, rnd * n + n)
        threads = [threading.Thread(target=post, args=(i,)) for i in ids]
        barrier.reset()
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
    assert set(codes.values()) == {200}, codes
    saved = load_reviews(ws / "job1")
    assert sorted(r.clip_id for r in saved.reviews) == list(range(3 * n))
    assert not list((ws / "job1").glob("*.tmp"))     # no temp file left behind


def test_save_reviews_uses_a_fresh_temp_file_per_write(tmp_path, monkeypatch):
    seen = []
    real_replace = Path.replace

    def spy(self, target):
        seen.append(self.name)
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", spy)
    save_reviews(tmp_path, ReviewSet())
    save_reviews(tmp_path, ReviewSet())
    assert len(seen) == 2 and seen[0] != seen[1]
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_save_leaves_old_file_and_no_temp(tmp_path, monkeypatch):
    save_reviews(tmp_path, ReviewSet(reviews=[
        Review(clip_id=1, decision="approved")]))

    def boom(self, target):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(OSError):
        save_reviews(tmp_path, ReviewSet())
    monkeypatch.undo()
    assert [r.clip_id for r in load_reviews(tmp_path).reviews] == [1]
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("body", [
    {"adjusted_start": -0.5},                        # before the video
    {"adjusted_end": 60.5},                          # past the 60 s source
    {"adjusted_start": 61.0, "adjusted_end": 62.0},
    {"adjusted_start": 10.0, "adjusted_end": 10.0},  # empty
    {"adjusted_start": 20.0, "adjusted_end": 10.0},  # reversed
])
def test_out_of_range_or_reversed_trim_is_400_and_stores_nothing(ingested, body):
    _post_bad_trim(ingested, body)


@pytest.mark.parametrize("raw", ['{"decision":"approved","adjusted_start":NaN}',
                                 '{"decision":"approved","adjusted_end":Infinity}'])
def test_non_finite_trim_is_400(ingested, raw):
    ws, c, _ = ingested
    r = c.post("/api/jobs/job1/clips/0/review", content=raw,
               headers={"Content-Type": "application/json"})
    assert r.status_code in (400, 422), r.text
    assert load_reviews(ws / "job1").reviews == []


def _post_bad_trim(ingested, body):
    ws, c, _ = ingested
    r = c.post("/api/jobs/job1/clips/0/review",
               json={"decision": "approved", **body})
    assert r.status_code == 400, r.text
    assert "trim" in r.json()["detail"]
    assert load_reviews(ws / "job1").reviews == []
    assert not (ws / "job1" / "reviews.json").exists()


@pytest.mark.parametrize("body", [
    {}, {"adjusted_start": 0.0}, {"adjusted_end": 60.0},
    {"adjusted_start": 0.0, "adjusted_end": 60.0},
    {"adjusted_start": 12.5, "adjusted_end": 30.25},
])
def test_in_range_trim_is_saved(ingested, body):
    ws, c, _ = ingested
    r = c.post("/api/jobs/job1/clips/0/review",
               json={"decision": "approved", **body})
    assert r.status_code == 200, r.text
    got = load_reviews(ws / "job1").get(0)
    assert got.adjusted_start == body.get("adjusted_start")
    assert got.adjusted_end == body.get("adjusted_end")


def test_one_sided_trim_is_checked_against_the_clip(client):
    jid = client.get("/api/jobs").json()[0]["job_id"]
    clip = client.get(f"/api/jobs/{jid}").json()["clips"][0]
    for body in ({"adjusted_start": clip["end"] + 1},
                 {"adjusted_end": max(0.0, clip["start"] - 1)}):
        r = client.post(f"/api/jobs/{jid}/clips/{clip['id']}/review",
                        json={"decision": "approved", **body})
        assert r.status_code == 400, (body, r.text)


def test_junk_mp4_gives_a_safe_fixed_message_and_leaves_no_job(fresh, capfd):
    ws, c = fresh
    r = c.post("/api/jobs", files={"file": ("junk.mp4", b"just text", "video/mp4")})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert isinstance(detail, str) and detail
    for leak in ("/", "\\", str(ws), str(ws.parent), "ffprobe", "ffmpeg",
                 "moov", "source.mp4", "Command failed"):
        assert leak not in detail, leak
    jid_like = re.compile(r"\d{8}-\d{6}")
    assert not jid_like.search(detail)
    assert _job_dirs(ws) == [] and _uploads_empty(ws)
    # the full reason is still available to the operator
    assert "ingest rejected" in capfd.readouterr().err


def test_missing_source_error_does_not_leak_the_path(fresh, monkeypatch):
    from clipforge.ingest.ingest import IngestError
    from clipforge.server import app as appmod

    def boom(path, workspace, job_id=None):
        raise IngestError(f"Source file not found: {path}")

    monkeypatch.setattr(appmod, "ingest", boom)
    ws, c = fresh
    r = c.post("/api/jobs", files={"file": ("a.mp4", b"abc", "video/mp4")})
    assert r.status_code == 400
    assert str(ws) not in r.text and "not found" not in r.text and "/" not in r.json()["detail"]


def test_unexpected_errors_return_a_generic_500(ingested, monkeypatch):
    from clipforge.server import app as appmod

    def boom(*a, **k):
        raise RuntimeError("secret /abs/path detail")

    monkeypatch.setattr(appmod, "upsert_review", boom)
    _, c, _ = ingested
    r = c.post("/api/jobs/job1/clips/0/review", json={"decision": "approved"})
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal error"}


def test_review_for_a_clip_the_job_does_not_have_is_404(client):
    jid = client.get("/api/jobs").json()[0]["job_id"]
    ids = {c["id"] for c in client.get(f"/api/jobs/{jid}").json()["clips"]}
    missing = max(ids) + 100
    r = client.post(f"/api/jobs/{jid}/clips/{missing}/review",
                    json={"decision": "approved"})
    assert r.status_code == 404
    ok = client.post(f"/api/jobs/{jid}/clips/{min(ids)}/review",
                     json={"decision": "approved"})
    assert ok.status_code == 200


def _flaky_replace(monkeypatch, failures):
    """Behave like Windows: the first `failures` renames raise PermissionError."""
    real = Path.replace
    calls = {"n": 0}

    def replace(self, target):
        calls["n"] += 1
        if calls["n"] <= failures:
            raise PermissionError(5, "Access is denied (WinError 5)")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr("clipforge.review.store.REPLACE_PAUSE_SEC", 0)
    return calls


def test_save_retries_when_the_rename_is_refused(tmp_path, monkeypatch):
    calls = _flaky_replace(monkeypatch, failures=2)
    save_reviews(tmp_path, ReviewSet(reviews=[Review(clip_id=3, decision="approved")]))
    assert calls["n"] == 3
    assert [r.clip_id for r in load_reviews(tmp_path).reviews] == [3]
    assert not list(tmp_path.glob("*.tmp"))


def test_save_gives_up_after_a_few_tries_and_cleans_up(tmp_path, monkeypatch):
    _flaky_replace(monkeypatch, failures=99)
    with pytest.raises(PermissionError):
        save_reviews(tmp_path, ReviewSet())
    assert not list(tmp_path.glob("*.tmp"))


def test_review_request_survives_a_refused_rename(ingested, monkeypatch):
    ws, c, _ = ingested
    _flaky_replace(monkeypatch, failures=2)
    r = c.post("/api/jobs/job1/clips/0/review", json={"decision": "approved"})
    assert r.status_code == 200, r.text
    assert load_reviews(ws / "job1").get(0).decision == "approved"


def test_readers_wait_for_a_writer_holding_the_job_lock(tmp_path):
    """On Windows a reader would block the writer's rename; every reader must
    therefore take the same lock the writer holds."""
    import threading
    import time

    from clipforge.review import store

    save_reviews(tmp_path, ReviewSet(reviews=[Review(clip_id=1, decision="approved")]))
    done = {}
    started = threading.Event()

    def read():
        started.set()
        done["reviews"] = load_reviews(tmp_path)
        done["metrics"] = job_metrics(tmp_path)

    with store.job_lock(tmp_path):
        t = threading.Thread(target=read)
        t.start()
        started.wait(5)
        time.sleep(0.3)
        assert t.is_alive() and not done      # blocked while the writer holds it
    t.join(5)
    assert not t.is_alive() and done["metrics"]["reviewed"] == 1


# ------------------------------------------ F54: 0 clips is a warning

def _finished_job(ws, jid):
    from clipforge.common.manifest import Manifest, Stage
    from clipforge.refine.refine import RefinedSet
    d = ws / jid
    m = Manifest(job_id=jid, stage=Stage.RENDERED, refined_path="refined.json")
    m.save(d)
    RefinedSet(source_duration_sec=60.0).save(d / "refined.json")
    return d


def test_zero_clip_job_carries_the_fixed_warning(tmp_path):
    from clipforge.highlight.schema import ZERO_CLIPS_WARNING
    ws = tmp_path / "ws"
    client = TestClient(create_app(ws, config=MOCK_CFG, sync_pipeline=True))
    _finished_job(ws, "job-zero")
    (listed,) = [j for j in client.get("/api/jobs").json()
                 if j["job_id"] == "job-zero"]
    assert listed["warning"] == ZERO_CLIPS_WARNING
    assert client.get("/api/jobs/job-zero").json()["warning"] == ZERO_CLIPS_WARNING
    assert "/" not in ZERO_CLIPS_WARNING and "job-zero" not in ZERO_CLIPS_WARNING


def test_job_with_clips_or_unfinished_has_no_warning(fresh, source_video):
    ws, client = fresh
    with open(source_video, "rb") as f:
        r = client.post("/api/jobs", files={"file": ("a.mp4", f, "video/mp4")})
    jid = r.json()["job_id"]
    detail = client.get(f"/api/jobs/{jid}").json()
    assert detail["clips"] and detail["warning"] is None
    assert client.get("/api/jobs").json()[0]["warning"] is None



def test_failed_zero_clip_job_has_no_warning(tmp_path):
    from clipforge.common.manifest import Manifest, Stage
    ws = tmp_path / "ws"
    client = TestClient(create_app(ws, config=MOCK_CFG, sync_pipeline=True))
    d = _finished_job(ws, "job-failed")
    m = Manifest.load(d)
    m.failed_stage = Stage.RENDERED
    m.error = "boom"
    m.save(d)
    assert client.get("/api/jobs/job-failed").json()["warning"] is None
    (listed,) = [j for j in client.get("/api/jobs").json()
                 if j["job_id"] == "job-failed"]
    assert listed["warning"] is None


def test_running_zero_clip_job_has_no_warning_until_it_finishes(
        tmp_path, monkeypatch):
    """Through the real rerun route: `running` is a closure, so the pipeline
    is replaced by one that blocks until released."""
    import threading
    import time
    import clipforge.server.app as appmod
    from clipforge.highlight.schema import ZERO_CLIPS_WARNING
    ws = tmp_path / "ws"
    gate = threading.Event()
    monkeypatch.setattr(appmod, "run_pipeline",
                        lambda *a, **k: gate.wait(30))
    client = TestClient(create_app(ws, config=MOCK_CFG, sync_pipeline=False))
    _finished_job(ws, "job-busy")
    assert client.post("/api/jobs/job-busy/rerun").status_code == 200
    try:
        detail = client.get("/api/jobs/job-busy").json()
        assert detail["running"] is True and detail["warning"] is None
        (listed,) = client.get("/api/jobs").json()
        assert listed["running"] is True and listed["warning"] is None
    finally:
        gate.set()
    for _ in range(100):
        if not client.get("/api/jobs/job-busy").json()["running"]:
            break
        time.sleep(0.05)
    assert client.get("/api/jobs/job-busy").json()["warning"] == ZERO_CLIPS_WARNING
