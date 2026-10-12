"""
Browser-level checks of the review page (F05), driven by Playwright.

Starts the real server (mock engines, temp workspace) in a subprocess and
drives headless Chromium against it. Skipped when playwright or its
browser is not installed. These tests take about a minute because they
watch real time (video playback, idle polling) and take about 2.5 minutes,
so they only run when asked:

    CLIPFORGE_BROWSER_TESTS=1 python -m pytest tests/test_ui_browser.py -q
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

if not os.environ.get("CLIPFORGE_BROWSER_TESTS"):
    pytest.skip("slow browser tests: set CLIPFORGE_BROWSER_TESTS=1",
                allow_module_level=True)
pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Error as PlaywrightError  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

ROOT = Path(__file__).parents[1]
EVIL_NAME = "<img src=x onerror=alert(1)>.mp4"
EVIL_NOTE = '<b>hi</b> "quoted"'


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=5) as r:
        return json.load(r)


def _make_video(path: Path, seconds: int) -> Path:
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=1280x720:rate=30",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True)
    return path


class Server:
    def __init__(self, tmp: Path):
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        env = dict(os.environ, CLIPFORGE_ASR="mock", CLIPFORGE_HL="mock",
                   CLIPFORGE_DETECTOR="haar",
                   CLIPFORGE_EXAMPLES=str(tmp / "examples"))
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "clipforge.cli", "serve",
             "--workspace", str(tmp / "ws"), "--port", str(self.port)],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                _get(self.base, "/api/jobs")
                return
            except Exception:  # noqa: BLE001 — not up yet
                time.sleep(0.2)
        self.stop()
        raise RuntimeError("server did not start")

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def wait_idle(self, timeout=120):
        end = time.time() + timeout
        while time.time() < end:
            jobs = _get(self.base, "/api/jobs")
            if jobs and not any(j["running"] for j in jobs):
                return jobs
            time.sleep(0.5)
        raise TimeoutError("jobs still running")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("f05")
    server = Server(tmp)
    try:
        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch()
            except PlaywrightError as e:
                pytest.skip(f"chromium not available: {e}")
            yield server, browser, tmp
            browser.close()
    finally:
        server.stop()


@pytest.fixture(scope="module")
def seeded(env):
    """A finished job whose filename is an HTML injection attempt."""
    import httpx
    server, _, tmp = env
    vid = _make_video(tmp / "a.mp4", 30)
    r = httpx.post(server.base + "/api/jobs", timeout=60, files={
        "file": (EVIL_NAME, vid.read_bytes(), "video/mp4")})
    assert r.status_code == 200, r.text
    server.wait_idle()
    return server


@pytest.fixture
def page(env):
    _, browser, _ = env
    pg = browser.new_page(viewport={"width": 1280, "height": 900})
    pg.dialogs = []
    pg.on("dialog", lambda d: (pg.dialogs.append(d.message), d.dismiss()))
    pg.api_calls = []
    pg.on("request", lambda r: pg.api_calls.append(r.url)
          if "/api/" in r.url else None)
    yield pg
    pg.close()


def _serve_webm_for_clips(page, tmp: Path):
    """Playwright's Chromium cannot decode H.264, so answer clip-video
    requests with a VP8 webm of the same length. The page's own behaviour
    (element lifetime, playback not being reset) is what is under test."""
    webm = tmp / "clip.webm"
    if not webm.exists():
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc=duration=120:size=160x90:rate=15",
            "-c:v", "libvpx", "-b:v", "100k", str(webm)],
            check=True, capture_output=True)
    body = webm.read_bytes()
    page.route("**/clips/*/video", lambda route: route.fulfill(
        status=200, body=body, content_type="video/webm"))


def _upload(page, name: str, video: Path):
    page.set_input_files("#file", files=[{
        "name": name, "mimeType": "video/mp4", "buffer": video.read_bytes()}])


def _open_first_job(page, filename: str):
    page.locator("#joblist .job", has=page.locator(".name", has_text=filename)).first.click()
    page.wait_for_selector(".clip video", state="attached", timeout=15000)


def test_text_is_escaped_after_reload(env, seeded, page):
    server, _, tmp = env
    page.goto(server.base)
    page.reload()
    name = page.locator("#joblist .job .name").first
    assert name.inner_text() == EVIL_NAME
    _open_first_job(page, EVIL_NAME)
    page.locator('.clip [data-field="note"]').first.fill(EVIL_NOTE)
    page.locator(".clip .approve").first.click()
    page.wait_for_selector(".toast.ok")

    page.reload()
    _open_first_job(page, EVIL_NAME)
    assert page.locator('.clip [data-field="note"]').first.input_value() == EVIL_NOTE
    # nothing was parsed as markup, and the layout is intact
    assert page.evaluate("document.querySelectorAll('img, b').length") == 0
    assert page.evaluate("document.querySelectorAll('[onerror]').length") == 0
    box = page.locator("#joblist .job").first.bounding_box()
    side = page.locator("#sidebar").bounding_box()
    assert box["width"] <= side["width"]
    assert page.evaluate("document.documentElement.scrollWidth") <= 1280
    assert page.dialogs == []


def test_background_upload_and_idle_polling(env, seeded, page):
    server, _, tmp = env
    _serve_webm_for_clips(page, tmp)
    second = _make_video(tmp / "b.mp4", 90)
    page.goto(server.base)
    _open_first_job(page, EVIL_NAME)

    # nothing is running: no polling requests at all for 15 s
    page.wait_for_timeout(500)
    page.api_calls.clear()
    page.wait_for_timeout(15000)
    assert page.api_calls == []

    # play a clip and half-type into the form fields
    page.evaluate("""() => {
        const v = document.querySelector('.clip video');
        v.muted = true; window.__video = v; return v.play();
    }""")
    page.locator('.clip [data-field="note"]').first.fill("half-typed note")
    page.locator('.clip [data-field="start"]').first.fill("1.5")
    page.wait_for_function("window.__video.currentTime > 0.5")

    # a second upload processes in the background
    _upload(page, "second.mp4", second)
    page.wait_for_selector("#joblist .job >> nth=1")
    assert page.locator("#detail .clip").count() > 0   # still on job 1
    t0 = page.evaluate("window.__video.currentTime")
    page.api_calls.clear()
    deadline = time.time() + 10
    last = t0
    while time.time() < deadline:
        page.wait_for_timeout(1000)
        now = page.evaluate("window.__video.currentTime")
        assert now > last, "playback stalled"
        last = now
    assert page.evaluate("window.__video.paused") is False
    assert page.evaluate("window.__video === document.querySelector('.clip video')")
    assert page.locator('.clip [data-field="note"]').first.input_value() == "half-typed note"
    assert page.locator('.clip [data-field="start"]').first.input_value() == "1.5"
    assert len(page.api_calls) >= 2, "page should poll while a job runs"

    # polling stops again once the job has finished
    server.wait_idle()
    page.wait_for_timeout(6000)       # let the final poll land
    page.api_calls.clear()
    page.wait_for_timeout(10000)
    assert page.api_calls == []
    assert page.dialogs == []


def test_inline_toasts_and_no_dialogs(env, seeded, page):
    server, _, _ = env
    page.goto(server.base)
    _open_first_job(page, EVIL_NAME)

    page.locator(".clip .approve").first.click()
    page.locator(".toast.ok", has_text="saved").wait_for(timeout=1000)

    page.locator('button[data-act="export"]').click()
    page.locator(".toast.ok", has_text="wrote").wait_for(timeout=3000)

    server.stop()
    page.locator(".clip .reject").first.click()
    page.locator(".toast.err").first.wait_for(timeout=3000)
    page.locator('button[data-act="export"]').click()
    page.wait_for_timeout(500)
    assert page.dialogs == []
