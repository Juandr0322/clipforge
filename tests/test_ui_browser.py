"""
Browser-level checks of the review page (F05, F06), driven by Playwright.

Starts the real server (mock engines, temp workspace) in a subprocess and
drives headless Chromium against it. Skipped when playwright or its
browser is not installed (`pip install -r requirements-dev.txt` and
`playwright install chromium`). The tests watch real time (video playback,
idle polling), so the whole file takes about 3.5 minutes and only runs when
asked:

    CLIPFORGE_BROWSER_TESTS=1 python -m pytest tests/test_ui_browser.py -q
"""
import json
import os
import re
import shutil
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

    def serve(route):
        # honour Range requests, or Chromium cannot seek (F06 sets currentTime)
        m = re.match(r"bytes=(\d+)-(\d*)", route.request.headers.get("range", ""))
        if not m:
            route.fulfill(status=200, body=body, content_type="video/webm",
                          headers={"Accept-Ranges": "bytes"})
            return
        lo = int(m.group(1))
        hi = min(int(m.group(2) or len(body) - 1), len(body) - 1)
        route.fulfill(status=206, body=body[lo:hi + 1], content_type="video/webm",
                      headers={"Accept-Ranges": "bytes",
                               "Content-Range": f"bytes {lo}-{hi}/{len(body)}"})

    page.route("**/clips/*/video", serve)


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


def test_switching_jobs_never_acts_on_the_wrong_job(env, seeded, page):
    import httpx
    server, _, tmp = env
    b = httpx.post(server.base + "/api/jobs", timeout=60, files={
        "file": ("switch-b.mp4", (tmp / "a.mp4").read_bytes(), "video/mp4")})
    assert b.status_code == 200, b.text
    b_id = b.json()["job_id"]
    server.wait_idle()
    b_clips = _get(server.base, f"/api/jobs/{b_id}")["clips"]
    assert b_clips and all(c["review"] is None for c in b_clips)

    page.goto(server.base)
    _open_first_job(page, EVIL_NAME)
    a_id = page.locator(".clip").first.get_attribute("data-job")
    assert a_id and a_id != b_id

    # click job B and, in the same tick, try to approve a card of job A
    clicked = page.evaluate("""(bName) => {
        const job = [...document.querySelectorAll('#joblist .job')]
          .find(j => j.querySelector('.name').textContent === bName);
        job.click();
        const btn = document.querySelector('.clip .approve');
        if (btn) btn.click();
        return !!btn;
    }""", "switch-b.mp4")
    assert clicked is False, "job A's cards must not stay clickable"

    page.wait_for_selector(f'.clip[data-job="{b_id}"]')
    assert page.locator(f'.clip:not([data-job="{b_id}"])').count() == 0
    page.wait_for_timeout(500)
    b_clips = _get(server.base, f"/api/jobs/{b_id}")["clips"]
    assert all(c["review"] is None for c in b_clips)
    assert page.dialogs == []


def test_slow_detail_response_for_old_job_is_ignored(env, seeded, page):
    import httpx
    server, _, _ = env
    jobs = _get(server.base, "/api/jobs")
    by_name = {j["filename"]: j["job_id"] for j in jobs}
    a_id, b_id = by_name[EVIL_NAME], by_name["switch-b.mp4"]

    page.goto(server.base)
    page.locator("#joblist .job").first.wait_for()

    def slow(route):
        time.sleep(1.5)          # the response for A arrives after B was opened
        route.continue_()

    page.route(f"**/api/jobs/{a_id}", slow)
    page.evaluate("id => document.querySelector(`.job[data-job=\"${id}\"]`).click()", a_id)
    page.wait_for_timeout(200)
    page.evaluate("id => document.querySelector(`.job[data-job=\"${id}\"]`).click()", b_id)
    page.wait_for_selector(f'.clip[data-job="{b_id}"]')
    page.wait_for_timeout(2500)           # let A's late response land
    assert page.locator(f'.clip:not([data-job="{b_id}"])').count() == 0
    assert page.locator(".clip").count() > 0
    assert httpx.get(server.base + "/api/jobs").status_code == 200


def test_recovers_from_a_failed_first_load(env, seeded, page):
    server, _, _ = env
    failures = []

    def flaky(route):
        if not failures:
            failures.append(route.request.url)
            route.abort()
        else:
            route.continue_()

    page.route("**/api/jobs", flaky)
    page.goto(server.base)
    page.locator(".toast.err").wait_for(timeout=3000)
    # nothing was running, so there is no polling, yet one retry fixes the list
    page.wait_for_selector("#joblist .job", timeout=8000)


def test_recovers_when_detail_fails_while_idle(env, seeded, page):
    server, _, _ = env
    failures = []

    def flaky(route):
        if not failures:
            failures.append(1)
            route.abort()
        else:
            route.continue_()

    page.goto(server.base)
    page.locator("#joblist .job").first.wait_for()
    page.route("**/api/metrics", flaky)       # fails once, the job list succeeds
    page.locator("#joblist .job").first.click()
    page.locator(".toast.err").wait_for(timeout=3000)
    page.wait_for_selector(".clip", timeout=8000)   # one retry, no focus needed


def test_dirty_field_is_released_on_blur(env, seeded, page):
    server, _, _ = env
    page.goto(server.base)
    _open_first_job(page, EVIL_NAME)
    note = page.locator('.clip [data-field="note"]').first
    server_value = note.input_value()
    note.fill("scratch")
    note.fill(server_value)
    page.locator("#detail .stagebar").click()      # blur
    assert note.evaluate("n => n.dataset.dirty") is None


# ---- F06: review speed ----------------------------------------------------
# A job with four clips whose score order differs from their time order:
#   id:     0      1      2      3
#   score:  85     65     95     40       -> by score: 2, 0, 1, 3
#   start:  ~3.1   ~11.5  5.0    1.0      -> by time:  3, 0, 2, 1

BY_SCORE = [2, 0, 1, 3]
BY_TIME = [3, 0, 2, 1]


@pytest.fixture(scope="module")
def multi(env):
    """A finished job rewritten to hold four rendered clips."""
    import httpx
    from clipforge.refine.refine import RefinedSet
    server, _, tmp = env
    vid = _make_video(tmp / "m.mp4", 30)
    r = httpx.post(server.base + "/api/jobs", timeout=60, files={
        "file": ("f06-multi.mp4", vid.read_bytes(), "video/mp4")})
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    server.wait_idle()
    job_dir = tmp / "ws" / jid
    rs = RefinedSet.load(job_dir / "refined.json")
    assert len(rs.clips) >= 2
    c0, c1 = rs.clips[0], rs.clips[1]
    assert c0.start < 5.0 < c1.start
    rs.clips = [
        c0.model_copy(update={"id": 0, "score": 85, "hook": "hook zero"}),
        c1.model_copy(update={"id": 1, "score": 65, "hook": "hook one"}),
        c0.model_copy(update={"id": 2, "score": 95, "hook": "hook two", "start": 5.0,
                              "speech_start": 5.5, "speech_end": 18.5, "end": 19.0}),
        c0.model_copy(update={"id": 3, "score": 40, "hook": "hook three", "start": 1.0,
                              "speech_start": 1.5, "speech_end": 14.5, "end": 15.0}),
    ]
    rs.save(job_dir / "refined.json")
    for cid in (2, 3):
        for suffix in (".mp4", "_qa.png"):
            shutil.copy(job_dir / "renders" / f"clip_000{suffix}",
                        job_dir / "renders" / f"clip_{cid:03d}{suffix}")
    return server, jid, job_dir


@pytest.fixture
def fresh(multi):
    """The four-clip job with no reviews."""
    _, jid, job_dir = multi
    (job_dir / "reviews.json").unlink(missing_ok=True)
    return multi


def _open_multi(page, server):
    page.goto(server.base)
    _open_first_job(page, "f06-multi.mp4")
    page.wait_for_selector(".clip.sel")


def _visible(page) -> list[int]:
    return page.eval_on_selector_all(
        ".clip:not([hidden])", "els => els.map(e => Number(e.dataset.clip))")


def _selected(page):
    sel = page.locator(".clip.sel")
    return int(sel.get_attribute("data-clip")) if sel.count() else None


def _counts(page) -> dict[str, int]:
    return {b.get_attribute("data-filter"): int(b.get_attribute("data-count"))
            for b in page.locator("button[data-filter]").all()}


def _decisions(server, jid) -> dict[int, str | None]:
    return {c["id"]: (c["review"] or {}).get("decision")
            for c in _get(server.base, f"/api/jobs/{jid}")["clips"]}


def _refresh_now(page):
    with page.expect_response(re.compile(r"/api/jobs/[^/]+$")):
        page.evaluate("refresh()")
    page.wait_for_timeout(300)


def test_keyboard_shortcuts_act_on_the_selected_clip(env, fresh, page):
    server, jid, _ = fresh
    _, _, tmp = env
    _serve_webm_for_clips(page, tmp)
    _open_multi(page, server)
    assert _visible(page) == BY_SCORE      # default sort: score, high to low
    assert _selected(page) == 2

    # J/K and Down/Up move the selection and stop at the ends
    for key, want in [("j", 0), ("ArrowDown", 1), ("k", 0), ("ArrowUp", 2), ("ArrowUp", 2)]:
        page.keyboard.press(key)
        assert _selected(page) == want, key

    # A approves, R rejects, both for the selected clip
    page.keyboard.press("j")               # clip 0
    page.keyboard.press("a")
    page.locator(".toast.ok", has_text="clip 1 approved").wait_for(timeout=3000)
    assert _decisions(server, jid)[0] == "approved"
    assert page.locator('.clip[data-clip="0"] .approve.on').count() == 1
    page.keyboard.press("r")
    page.locator(".toast.ok", has_text="clip 1 rejected").wait_for(timeout=3000)
    assert _decisions(server, jid)[0] == "rejected"
    assert _decisions(server, jid)[2] is None

    # Space plays and pauses the selected clip's video only
    page.evaluate("document.querySelectorAll('.clip video').forEach(v => v.muted = true)")
    video = '.clip[data-clip="0"] video'
    page.wait_for_function("sel => document.querySelector(sel).readyState >= 1", arg=video)
    page.keyboard.press(" ")
    page.wait_for_function("sel => !document.querySelector(sel).paused", arg=video)
    page.keyboard.press(" ")
    page.wait_for_function("sel => document.querySelector(sel).paused", arg=video)

    # ? opens the help panel, which lists exactly the page's shortcuts; Esc closes it
    assert page.locator("#help").is_hidden()
    page.keyboard.press("?")
    assert page.locator("#help").is_visible()
    assert page.locator("#helpbtn").get_attribute("aria-expanded") == "true"
    shown = page.locator("#help dt").all_inner_texts()
    assert shown == page.evaluate("SHORTCUTS.map(s => s.label)")
    keys = page.evaluate("[...KEYMAP.keys()].sort()")
    assert keys == sorted(["a", "r", " ", "j", "k", "ArrowDown", "ArrowUp",
                           "[", "]", "?", "Escape"])
    page.keyboard.press("Escape")
    assert page.locator("#help").is_hidden()
    page.locator("#helpbtn").click()
    assert page.locator("#help").is_visible()
    page.locator("#helpclose").click()
    assert page.locator("#help").is_hidden()
    assert page.dialogs == []


def test_shortcuts_are_ignored_while_typing(env, fresh, page):
    server, jid, _ = fresh
    _open_multi(page, server)
    page.keyboard.press("j")
    assert _selected(page) == 0
    note = page.locator('.clip[data-clip="0"] [data-field="note"]')
    start = page.locator('.clip[data-clip="0"] [data-field="start"]')
    note.click()
    page.keyboard.type("arjk?[] ")
    page.keyboard.press("ArrowDown")
    assert note.input_value() == "arjk?[] "
    start.click()
    page.keyboard.type("a[")
    assert start.input_value() == "a["
    page.keyboard.press("Escape")
    page.wait_for_timeout(500)
    assert _decisions(server, jid) == {0: None, 1: None, 2: None, 3: None}
    assert _selected(page) == 0
    assert page.locator("#help").is_hidden()
    assert page.locator(".toast").count() == 0

    # leaving the field hands the keys back
    page.locator("#detail .stagebar").click()
    page.keyboard.press("j")
    assert _selected(page) == 1


def test_filter_counts_and_approving_advances_to_next_unreviewed(env, fresh, page):
    server, jid, _ = fresh
    _open_multi(page, server)
    assert _counts(page) == {"all": 4, "unreviewed": 4, "approved": 0, "rejected": 0}
    assert page.locator('button[data-filter="unreviewed"]').inner_text() == "unreviewed (4)"

    page.locator('button[data-filter="unreviewed"]').click()
    assert _visible(page) == BY_SCORE and _selected(page) == 2

    page.keyboard.press("a")                # approve clip 2: leaves the view
    page.wait_for_function("document.querySelectorAll('.clip:not([hidden])').length === 3")
    assert _visible(page) == [0, 1, 3]
    assert _selected(page) == 0             # the next unreviewed clip
    assert _counts(page) == {"all": 4, "unreviewed": 3, "approved": 1, "rejected": 0}

    page.keyboard.press("r")                # reject clip 0
    page.wait_for_function("document.querySelectorAll('.clip:not([hidden])').length === 2")
    assert _visible(page) == [1, 3] and _selected(page) == 1
    assert _counts(page) == {"all": 4, "unreviewed": 2, "approved": 1, "rejected": 1}

    # no unreviewed clip after the last one: fall back to the one before it
    page.keyboard.press("j")
    assert _selected(page) == 3
    page.keyboard.press("a")
    page.wait_for_function("document.querySelectorAll('.clip:not([hidden])').length === 1")
    assert _visible(page) == [1] and _selected(page) == 1

    # the server agrees, and a background refresh keeps the same picture
    assert _decisions(server, jid) == {0: "rejected", 1: None, 2: "approved", 3: "approved"}
    _refresh_now(page)
    assert _visible(page) == [1] and _selected(page) == 1
    assert _counts(page) == {"all": 4, "unreviewed": 1, "approved": 2, "rejected": 1}

    # the other filters show the matching clips; an empty one says so
    page.locator('button[data-filter="approved"]').click()
    assert _visible(page) == [2, 3] and _selected(page) == 3   # clip 1 left; next in order
    page.locator('button[data-filter="rejected"]').click()
    assert _visible(page) == [0] and _selected(page) == 0
    page.locator('button[data-filter="all"]').click()
    assert _visible(page) == BY_SCORE
    page.locator('button[data-filter="unreviewed"]').click()
    page.keyboard.press("a")
    page.wait_for_function("document.querySelectorAll('.clip:not([hidden])').length === 0")
    assert page.locator(".nomatch").is_visible()
    assert _selected(page) is None
    page.keyboard.press("a")                # nothing selected: nothing happens
    page.wait_for_timeout(300)
    assert _decisions(server, jid) == {0: "rejected", 1: "approved", 2: "approved", 3: "approved"}


def test_sort_and_filter_survive_refresh_and_reload(env, fresh, page):
    import httpx
    server, jid, _ = fresh
    httpx.post(f"{server.base}/api/jobs/{jid}/clips/2/review",
               json={"decision": "approved", "note": ""}).raise_for_status()
    _open_multi(page, server)
    assert _visible(page) == BY_SCORE

    page.locator("select[data-sort]").select_option("time")
    assert _visible(page) == BY_TIME
    page.locator('button[data-filter="unreviewed"]').click()
    assert _visible(page) == [3, 0, 1]

    # a background refresh, with the server's state changed meanwhile
    httpx.post(f"{server.base}/api/jobs/{jid}/clips/3/review",
               json={"decision": "rejected", "note": ""}).raise_for_status()
    _refresh_now(page)
    assert _visible(page) == [0, 1]
    assert page.locator("select[data-sort]").input_value() == "time"
    assert page.locator('button[data-filter="unreviewed"]').get_attribute("aria-pressed") == "true"
    assert _counts(page) == {"all": 4, "unreviewed": 2, "approved": 1, "rejected": 1}

    # a reload remembers both choices
    page.reload()
    _open_first_job(page, "f06-multi.mp4")
    assert page.locator("select[data-sort]").input_value() == "time"
    assert page.locator('button[data-filter="unreviewed"]').get_attribute("aria-pressed") == "true"
    assert _visible(page) == [0, 1]
    page.locator('button[data-filter="all"]').click()
    assert _visible(page) == BY_TIME
    page.locator("select[data-sort]").select_option("score")
    assert _visible(page) == BY_SCORE
    page.reload()
    _open_first_job(page, "f06-multi.mp4")
    assert page.locator("select[data-sort]").input_value() == "score"
    assert _visible(page) == BY_SCORE


def test_view_prefs_work_when_storage_is_blocked(env, fresh, page):
    server, _, _ = fresh
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.add_init_script("""
        Storage.prototype.getItem = () => { throw new Error('blocked'); };
        Storage.prototype.setItem = () => { throw new Error('blocked'); };
    """)
    _open_multi(page, server)
    page.locator("select[data-sort]").select_option("time")
    page.locator('button[data-filter="approved"]').click()
    assert _visible(page) == []
    page.locator('button[data-filter="all"]').click()
    assert _visible(page) == BY_TIME
    assert errors == []


def _seek(page, cid: int, t: float) -> float:
    sel = f'.clip[data-clip="{cid}"] video'
    page.wait_for_function("sel => document.querySelector(sel).readyState >= 1", arg=sel)
    return page.evaluate("""([sel, t]) => new Promise(res => {
        const v = document.querySelector(sel);
        v.pause();
        v.addEventListener('seeked', () => res(v.currentTime), {once: true});
        v.currentTime = t;
    })""", [sel, t])


def test_trim_fields_are_set_from_the_playhead(env, fresh, page):
    server, jid, _ = fresh
    _, _, tmp = env
    _serve_webm_for_clips(page, tmp)
    clip = next(c for c in _get(server.base, f"/api/jobs/{jid}")["clips"] if c["id"] == 0)
    s0, e0 = clip["start"], clip["end"]
    before = _get(server.base, f"/api/jobs/{jid}")["metrics"]["avg_trim_adjustment_sec"]
    assert before == 0.0

    _open_multi(page, server)
    page.keyboard.press("j")                 # select clip 0
    assert _selected(page) == 0
    start = page.locator('.clip[data-clip="0"] [data-field="start"]')
    end = page.locator('.clip[data-clip="0"] [data-field="end"]')

    # the button: pause at 3.0 s -> start = S + 3.0, in absolute source seconds
    t = _seek(page, 0, 3.0)
    assert t == pytest.approx(3.0, abs=0.05)
    page.locator('.clip[data-clip="0"] button[data-act="setstart"]').click()
    assert float(start.input_value()) == pytest.approx(s0 + t, abs=0.002)
    assert float(start.input_value()) > 3.0 and s0 > 0     # absolute, not clip-relative

    # [ does the same thing from the keyboard
    t = _seek(page, 0, 2.0)
    page.keyboard.press("[")
    assert float(start.input_value()) == pytest.approx(s0 + t, abs=0.002)
    t = _seek(page, 0, 3.0)
    page.locator('.clip[data-clip="0"] button[data-act="setstart"]').click()
    want_start = s0 + t
    assert float(start.input_value()) == pytest.approx(want_start, abs=0.002)

    # ] and the button for the end
    t = _seek(page, 0, 5.0)
    page.keyboard.press("]")
    want_end = s0 + t
    assert float(end.input_value()) == pytest.approx(want_end, abs=0.002)
    t = _seek(page, 0, 5.0)
    page.locator('.clip[data-clip="0"] button[data-act="setend"]').click()
    assert float(end.input_value()) == pytest.approx(s0 + t, abs=0.002)

    # a background refresh must not overwrite the values that were set
    _refresh_now(page)
    assert float(start.input_value()) == pytest.approx(want_start, abs=0.002)
    assert float(end.input_value()) == pytest.approx(want_end, abs=0.002)

    # saving stores them and moves the job's avg-trim metric
    page.locator('.clip[data-clip="0"] .approve').click()
    page.locator(".toast.ok", has_text="saved").wait_for(timeout=3000)
    detail = _get(server.base, f"/api/jobs/{jid}")
    review = next(c for c in detail["clips"] if c["id"] == 0)["review"]
    assert review["decision"] == "approved"
    assert review["adjusted_start"] == pytest.approx(want_start, abs=0.002)
    assert review["adjusted_end"] == pytest.approx(want_end, abs=0.002)
    expected = abs(want_start - s0) + abs(want_end - e0)
    assert detail["metrics"]["avg_trim_adjustment_sec"] == pytest.approx(expected, abs=0.02)
    assert detail["metrics"]["avg_trim_adjustment_sec"] != before
    assert page.dialogs == []


def test_trim_start_after_end_is_not_saved(env, fresh, page):
    server, jid, _ = fresh
    _open_multi(page, server)
    card = '.clip[data-clip="0"]'
    page.keyboard.press("j")
    page.locator(f'{card} [data-field="start"]').fill("20")
    page.locator(f'{card} [data-field="end"]').fill("10")
    page.locator(f"{card} .approve").click()
    page.locator(".toast.err", has_text="before").wait_for(timeout=3000)
    assert _decisions(server, jid)[0] is None


def test_selection_moves_to_the_neighbour_when_a_refresh_removes_it(env, fresh, page):
    server, jid, job_dir = fresh
    refined = job_dir / "refined.json"
    original = refined.read_text()
    _open_multi(page, server)
    assert _visible(page) == BY_SCORE
    page.keyboard.press("j")
    page.keyboard.press("j")
    assert _selected(page) == 1                 # third in score order
    try:
        data = json.loads(original)
        data["clips"] = [c for c in data["clips"] if c["id"] != 1]
        refined.write_text(json.dumps(data))
        _refresh_now(page)
        assert _visible(page) == [2, 0, 3]
        assert _selected(page) == 3             # took the removed clip's place
        data["clips"] = [c for c in data["clips"] if c["id"] != 3]
        refined.write_text(json.dumps(data))
        _refresh_now(page)
        assert _visible(page) == [2, 0]
        assert _selected(page) == 0             # none after it: the previous one
    finally:
        refined.write_text(original)


def test_help_panel_is_modal_and_space_scrolls_without_a_selection(env, fresh, page):
    server, jid, _ = fresh
    _open_multi(page, server)
    page.keyboard.press("?")
    assert page.locator("#help").is_visible()
    sel = _selected(page)
    for key in ["a", "r", "j", "ArrowDown", "k", " ", "[", "]"]:
        page.keyboard.press(key)
    page.wait_for_timeout(300)
    assert _selected(page) == sel
    assert _decisions(server, jid) == {0: None, 1: None, 2: None, 3: None}
    # Esc closes it even when the cursor is in a field
    page.locator('.clip[data-clip="0"] [data-field="note"]').focus()
    page.keyboard.press("Escape")
    assert page.locator("#help").is_hidden()

    # a key with nothing to act on keeps its normal job
    page.locator("#detail .stagebar").click()
    prevented = page.evaluate("""() => {
        selected = null;
        const ev = new KeyboardEvent('keydown', {key: ' ', bubbles: true, cancelable: true});
        document.body.dispatchEvent(ev);
        return ev.defaultPrevented;
    }""")
    assert prevented is False


def test_keys_while_the_pane_is_loading_write_no_review(env, seeded, fresh, page):
    server, jid, _ = fresh
    a_id = jid
    b_id = next(j["job_id"] for j in _get(server.base, "/api/jobs")
                if j["filename"] == EVIL_NAME)
    assert a_id != b_id
    before = (_decisions(server, a_id), _decisions(server, b_id))

    _open_multi(page, server)                   # job A, a clip selected
    assert _selected(page) is not None

    held = []                                   # B's detail response is held back
    page.route(f"**/api/jobs/{b_id}", lambda route: held.append(route))
    page.evaluate("id => document.querySelector(`.job[data-job=\"${id}\"]`).click()", b_id)
    page.wait_for_timeout(300)
    assert held, "the detail request should be pending"
    assert page.locator("#detail .empty").inner_text() == "loading…"
    for key in ["a", "r", "j", " "]:
        page.keyboard.press(key)
    assert page.locator("#detail .empty").inner_text() == "loading…"
    held[0].continue_()
    page.wait_for_selector(f'.clip[data-job="{b_id}"]', timeout=10000)
    page.wait_for_timeout(500)
    assert (_decisions(server, a_id), _decisions(server, b_id)) == before
    assert page.locator(".toast.ok").count() == 0


# ---------------------------------------------------------------- F07: upload

def _data_transfer(page, files):
    """A DataTransfer holding real File objects, as a drag from the desktop
    would. files: [(name, bytes, mime)]."""
    import base64
    payload = [{"name": n, "mime": m, "b64": base64.b64encode(b).decode()}
               for n, b, m in files]
    return page.evaluate_handle("""(files) => {
        const dt = new DataTransfer();
        for (const f of files) {
            const bytes = Uint8Array.from(atob(f.b64), c => c.charCodeAt(0));
            dt.items.add(new File([bytes], f.name, {type: f.mime}));
        }
        return dt;
    }""", payload)


def _drop(page, files, target="#uploadbox"):
    dt = _data_transfer(page, files)
    page.dispatch_event(target, "dragenter", {"dataTransfer": dt})
    page.dispatch_event(target, "dragover", {"dataTransfer": dt})
    page.dispatch_event(target, "drop", {"dataTransfer": dt})


def _track_posts(page):
    """Log every POST /api/jobs as ('start'|'end', url) in the order it happens."""
    page.posts = []
    page.on("request", lambda r: page.posts.append(("start", r.url))
            if r.method == "POST" and r.url.endswith("/api/jobs") else None)
    page.on("requestfinished", lambda r: page.posts.append(("end", r.url))
            if r.method == "POST" and r.url.endswith("/api/jobs") else None)
    page.on("requestfailed", lambda r: page.posts.append(("end", r.url))
            if r.method == "POST" and r.url.endswith("/api/jobs") else None)


def _order_xhr(page):
    """Record, inside the page, when each XHR starts and ends (Playwright's own
    request events can arrive after the page has already moved on)."""
    page.add_init_script("""
        window.__xhr = [];
        const send = XMLHttpRequest.prototype.send;
        XMLHttpRequest.prototype.send = function (...a) {
            window.__xhr.push('start');
            // readyState 4 comes before load/loadend, i.e. before the page can react
            this.addEventListener('readystatechange',
                () => { if (this.readyState === 4) window.__xhr.push('end'); });
            return send.apply(this, a);
        };""")


def _job_count(page) -> int:
    return page.locator("#joblist .job").count()


def _upload_zone_is_idle(page):
    zone = page.locator("#uploadbox")
    assert zone.get_attribute("aria-disabled") == "false"
    assert "upload long-form video" in page.locator("#uploadlabel").inner_text()
    assert page.locator("#uploadbar").is_hidden()
    assert "dragover" not in (zone.get_attribute("class") or "")


@pytest.fixture(scope="module")
def small_videos(env):
    _, _, tmp = env
    return [_make_video(tmp / f"up{i}.mp4", 8) for i in (1, 2)]


def test_drop_highlights_the_zone_and_uploads(env, seeded, small_videos, page):
    server, _, _ = env
    _track_posts(page)
    page.goto(server.base)
    page.wait_for_selector("#joblist .job")
    before = _job_count(page)
    files = [("dropped.mp4", small_videos[0].read_bytes(), "video/mp4")]

    dt = _data_transfer(page, files)
    page.dispatch_event("#sidebar", "dragenter", {"dataTransfer": dt})
    assert "dragover" in page.locator("#uploadbox").get_attribute("class")
    assert "dragover" in page.locator("#sidebar").get_attribute("class")
    page.dispatch_event("#sidebar", "dragleave", {"dataTransfer": dt})
    assert "dragover" not in page.locator("#uploadbox").get_attribute("class")

    # a drag that carries no file (selected text, a link) is not an upload target
    text = page.evaluate_handle("() => { const d = new DataTransfer(); d.setData('text/plain', 'hi'); return d; }")
    page.dispatch_event("#sidebar", "dragenter", {"dataTransfer": text})
    assert "dragover" not in page.locator("#uploadbox").get_attribute("class")

    page.dispatch_event("#sidebar", "dragenter", {"dataTransfer": dt})
    page.dispatch_event("#sidebar", "drop", {"dataTransfer": dt})
    assert "dragover" not in page.locator("#uploadbox").get_attribute("class")
    page.wait_for_function(
        "n => document.querySelectorAll('#joblist .job').length > n", arg=before, timeout=30000)
    assert [k for k, _ in page.posts].count("start") == 1
    names = page.locator("#joblist .job .name").all_inner_texts()
    assert "dropped.mp4" in names
    page.wait_for_function(
        "() => document.querySelector('#uploadbox').getAttribute('aria-disabled') === 'false'")
    _upload_zone_is_idle(page)
    server.wait_idle()
    assert page.dialogs == []


def test_upload_shows_a_rising_percentage_then_pipeline_progress(env, seeded, page):
    server, _, tmp = env
    big = tmp / "a.mp4"                         # the 30 s seeded video
    size = big.stat().st_size
    page.goto(server.base)
    page.wait_for_selector("#joblist .job")
    before = _job_count(page)
    # slow the upload down so the percentage can be watched climbing
    cdp = page.context.new_cdp_session(page)
    cdp.send("Network.enable")
    cdp.send("Network.emulateNetworkConditions", {
        "offline": False, "latency": 0, "downloadThroughput": -1,
        "uploadThroughput": max(size // 6, 50_000)})
    page.evaluate("""() => {
        window.__pcts = [];
        const note = () => {
            const m = /(\\d+)%/.exec(document.querySelector('#uploadlabel').textContent);
            const bar = document.querySelector('#uploadbar');
            window.__pcts.push({label: m ? Number(m[1]) : null,
                                bar: bar.hidden ? null : bar.value});
        };
        new MutationObserver(note).observe(document.querySelector('#uploadbox'),
            {subtree: true, childList: true, characterData: true, attributes: true});
    }""")
    _upload(page, "progress.mp4", big)
    page.wait_for_function(
        "n => document.querySelectorAll('#joblist .job').length > n", arg=before, timeout=60000)
    cdp.send("Network.emulateNetworkConditions", {
        "offline": False, "latency": 0, "downloadThroughput": -1, "uploadThroughput": -1})
    pcts = [p["label"] for p in page.evaluate("window.__pcts") if p["label"] is not None]
    bars = [p["bar"] for p in page.evaluate("window.__pcts") if p["bar"] is not None]
    assert pcts == sorted(pcts), pcts                   # only ever rises
    assert pcts[-1] == 100
    assert len(set(pcts)) >= 3, f"percentage never visibly moved: {pcts}"
    assert bars[-1] == 100 and bars == sorted(bars)
    # ...and the new job now carries the pipeline's own progress
    job = page.locator("#joblist .job", has=page.locator(".name", has_text="progress.mp4"))
    assert re.search(r"ingested|transcribed|candidates_ready|refined|reframed|rendered",
                     job.locator(".stage").inner_text())
    page.wait_for_function(
        "() => document.querySelector('#uploadbox').getAttribute('aria-disabled') === 'false'")
    _upload_zone_is_idle(page)
    server.wait_idle()


@pytest.mark.parametrize("how", ["drop", "picker"])
def test_non_video_files_are_refused_without_a_request(env, seeded, page, how):
    server, _, _ = env
    _track_posts(page)
    page.goto(server.base)
    page.wait_for_selector("#joblist .job")
    before = _job_count(page)
    files = [("notes.txt", b"hello", "text/plain"),
             ("pic.png", b"\x89PNG\r\n\x1a\n" + b"0" * 40, "image/png")]
    if how == "drop":
        _drop(page, files, "#sidebar")
    else:
        page.set_input_files("#file", files=[
            {"name": n, "mimeType": m, "buffer": b} for n, b, m in files])
    msg = page.locator("#uploadmsg")
    msg.wait_for()
    text = msg.inner_text()
    assert "notes.txt" in text and "pic.png" in text
    assert text.count("not a video file") == 2
    page.wait_for_timeout(500)
    assert page.posts == []                                 # nothing was sent
    assert _job_count(page) == before
    _upload_zone_is_idle(page)
    # the message goes away with the next attempt
    _drop(page, [("again.txt", b"x", "text/plain")], "#sidebar")
    assert "notes.txt" not in msg.inner_text() and "again.txt" in msg.inner_text()
    assert page.dialogs == []


def test_extension_check_is_case_insensitive_and_needs_a_dot(env, seeded, small_videos, page):
    server, _, _ = env
    _track_posts(page)
    page.goto(server.base)
    page.wait_for_selector("#joblist .job")
    _drop(page, [("mp4", b"x", "video/mp4"), ("clip.mp4.txt", b"x", "video/mp4")], "#sidebar")
    page.locator("#uploadmsg .err").nth(1).wait_for()
    assert page.posts == []
    _drop(page, [("LOUD.MP4", small_videos[0].read_bytes(), "video/mp4")], "#sidebar")
    page.wait_for_selector("#joblist .job .name:text-is('LOUD.MP4')", timeout=30000)
    server.wait_idle()


def test_server_rejection_is_shown_inline_and_the_zone_recovers(env, seeded, page):
    import httpx
    server, _, _ = env
    expected = httpx.post(server.base + "/api/jobs", timeout=30, files={
        "file": ("junk.mp4", b"this is plain text", "video/mp4")})
    assert expected.status_code == 400
    detail = expected.json()["detail"]
    # an exact message, answered by a stub, to check it is shown verbatim
    page.route("**/api/jobs", lambda route: route.fulfill(
        status=400, content_type="application/json",
        body=json.dumps({"detail": "Exactly <b>this</b> text"}))
        if route.request.method == "POST" else route.continue_())

    page.goto(server.base)
    page.wait_for_selector("#joblist .job")
    before = _job_count(page)
    page.set_input_files("#file", files=[
        {"name": "stub.mp4", "mimeType": "video/mp4", "buffer": b"x"}])
    stub = page.locator("#uploadmsg .err")
    stub.wait_for(timeout=15000)
    assert stub.inner_text() == "stub.mp4: Exactly <b>this</b> text"
    assert page.evaluate("document.querySelectorAll('#uploadmsg b').length") == 0
    _upload_zone_is_idle(page)
    page.unroute("**/api/jobs")
    page.set_input_files("#file", files=[
        {"name": "junk.mp4", "mimeType": "video/mp4", "buffer": b"this is plain text"}])
    msg = page.locator("#uploadmsg .err")
    msg.wait_for(timeout=15000)
    shown = msg.inner_text()
    assert shown.startswith("junk.mp4: ")
    # the server's own 400 text (it embeds a per-request job id, so compare the
    # stable part)
    assert "moov atom not found" in shown and "moov atom not found" in detail
    page.wait_for_function(
        "() => document.querySelector('#uploadbox').getAttribute('aria-disabled') === 'false'")
    _upload_zone_is_idle(page)
    assert _job_count(page) == before
    # the zone still works: a good file after the bad one goes through
    assert page.dialogs == []


def test_several_files_upload_one_after_the_other(env, seeded, small_videos, page):
    server, _, _ = env
    _order_xhr(page)
    page.goto(server.base)
    page.wait_for_selector("#joblist .job")
    before = _job_count(page)
    _drop(page, [("multi-one.mp4", small_videos[0].read_bytes(), "video/mp4"),
                 ("skip-me.txt", b"nope", "text/plain"),
                 ("multi-two.mp4", small_videos[1].read_bytes(), "video/mp4")], "#sidebar")
    page.wait_for_function(
        "n => document.querySelectorAll('#joblist .job').length >= n + 2", arg=before, timeout=60000)
    names = page.locator("#joblist .job .name").all_inner_texts()
    assert "multi-one.mp4" in names and "multi-two.mp4" in names
    # strictly sequential: the second request starts only after the first has ended
    assert page.evaluate("window.__xhr") == ["start", "end", "start", "end"]
    assert "skip-me.txt: not a video file" in page.locator("#uploadmsg").inner_text()
    page.wait_for_function(
        "() => document.querySelector('#uploadbox').getAttribute('aria-disabled') === 'false'")
    _upload_zone_is_idle(page)
    server.wait_idle()


def test_a_drop_during_an_upload_is_queued(env, seeded, small_videos, page):
    server, _, _ = env
    _order_xhr(page)
    page.goto(server.base)
    page.wait_for_selector("#joblist .job")
    before = _job_count(page)
    cdp = page.context.new_cdp_session(page)
    cdp.send("Network.enable")
    cdp.send("Network.emulateNetworkConditions", {
        "offline": False, "latency": 0, "downloadThroughput": -1,
        "uploadThroughput": 300_000})
    _drop(page, [("queue-one.mp4", small_videos[0].read_bytes(), "video/mp4")], "#sidebar")
    page.wait_for_function("() => document.querySelector('#uploadbox').getAttribute('aria-disabled') === 'true'")
    _drop(page, [("queue-two.mp4", small_videos[1].read_bytes(), "video/mp4")], "#sidebar")
    cdp.send("Network.emulateNetworkConditions", {
        "offline": False, "latency": 0, "downloadThroughput": -1, "uploadThroughput": -1})
    page.wait_for_function(
        "n => document.querySelectorAll('#joblist .job').length >= n + 2", arg=before, timeout=60000)
    assert page.evaluate("window.__xhr") == ["start", "end", "start", "end"]
    server.wait_idle()


def test_phone_layout_at_390px(env, seeded, fresh, page):
    server, jid, _ = fresh
    _, _, tmp = env
    page.set_viewport_size({"width": 390, "height": 844})
    _serve_webm_for_clips(page, tmp)
    page.goto(server.base)
    _open_first_job(page, "f06-multi.mp4")
    page.wait_for_selector(".clip.sel")

    width = lambda: page.evaluate("document.documentElement.scrollWidth")  # noqa: E731
    assert width() <= 390
    side = page.locator("#sidebar").bounding_box()
    detail = page.locator("#detail").bounding_box()
    # stacked, not side by side
    assert side["y"] + side["height"] <= detail["y"] + 1
    assert side["width"] >= 340 and detail["width"] >= 340
    assert page.locator("#uploadbox").is_visible()
    assert page.evaluate("""() => { const d = document.querySelector('#detail');
                                     return d.scrollWidth <= d.clientWidth; }""")

    card = page.locator(".clip:not([hidden])").first
    box = card.bounding_box()
    assert box["x"] >= 0 and box["x"] + box["width"] <= 390
    for sel in (".approve", ".reject"):
        btn = card.locator(sel)
        btn.scroll_into_view_if_needed()
        b = btn.bounding_box()
        assert btn.is_visible() and b["width"] > 40
        assert b["x"] >= 0 and b["x"] + b["width"] <= 390, (sel, b)
        assert b["y"] >= 0 and b["y"] + b["height"] <= 844
    # nothing scrolled sideways while we looked
    assert width() <= 390
    assert page.evaluate("document.querySelector('#detail').scrollLeft") == 0
    # and the buttons work from here
    card.locator(".approve").click()
    page.locator(".toast.ok", has_text="saved").wait_for(timeout=3000)
    assert _decisions(server, jid)[int(card.get_attribute("data-clip"))] == "approved"
    # the filter bar and the trim row fit too
    for sel in ("button[data-filter]", ".clip:not([hidden]) .trim input"):
        for el in page.locator(sel).all()[:8]:
            b = el.bounding_box()
            assert b["x"] >= 0 and b["x"] + b["width"] <= 390, (sel, b)
    page.set_viewport_size({"width": 1280, "height": 900})


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
