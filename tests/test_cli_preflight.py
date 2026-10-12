"""
CLI preflight: missing ffmpeg/ffprobe and a busy port give a clear message
and exit code 2 (serve still starts without ffmpeg) instead of a traceback or
a late uvicorn bind error. Only localhost sockets are used.
"""
import socket
import sys

import pytest

from clipforge import cli


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(cli, "load_env", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["clipforge", *argv])
    return cli.main()


def no_ffmpeg(monkeypatch, missing=("ffmpeg", "ffprobe")):
    monkeypatch.setattr(
        cli.shutil, "which",
        lambda name, *a, **k: None if name in missing else f"/bin/{name}")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def busy_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    yield s.getsockname()[1]
    s.close()


@pytest.fixture
def uvicorn_calls(monkeypatch):
    import uvicorn
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append(k))
    return calls


# ------------------------------------------------------------- ffmpeg

@pytest.mark.parametrize("missing,named", [
    (("ffmpeg",), "ffmpeg"), (("ffprobe",), "ffprobe"),
    (("ffmpeg", "ffprobe"), "ffmpeg and ffprobe")])
def test_ffmpeg_problem_names_missing_tool_and_fix(monkeypatch, missing, named):
    no_ffmpeg(monkeypatch, missing)
    msg = cli.ffmpeg_problem()
    assert msg.startswith(f"{named} not found on PATH")
    assert "winget install Gyan.FFmpeg" in msg and "new terminal" in msg
    assert "brew install ffmpeg" in msg and "apt install ffmpeg" in msg


def test_ffmpeg_problem_none_when_both_present(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda n, *a, **k: f"/bin/{n}")
    assert cli.ffmpeg_problem() is None


def test_ingest_without_ffmpeg_exits_2_cleanly(tmp_path, monkeypatch, capsys):
    no_ffmpeg(monkeypatch, ("ffprobe",))
    called = []
    monkeypatch.setattr(cli, "ingest", lambda *a, **k: called.append(1))
    rc = run_cli(monkeypatch, "ingest", "v.mp4",
                 "--workspace", str(tmp_path / "ws"))
    err = capsys.readouterr().err
    assert rc == 2 and not called
    assert "ffprobe not found on PATH" in err and "winget install" in err
    assert "Traceback" not in err
    assert not (tmp_path / "ws").exists()


def test_pipeline_without_ffmpeg_exits_2_cleanly(tmp_path, monkeypatch, capsys):
    import clipforge.pipeline as pipeline
    no_ffmpeg(monkeypatch)
    called = []
    monkeypatch.setattr(pipeline, "run_pipeline",
                        lambda *a, **k: called.append(1))
    rc = run_cli(monkeypatch, "pipeline", "job1", "--workspace", str(tmp_path))
    err = capsys.readouterr().err
    assert rc == 2 and not called
    assert "ffmpeg and ffprobe not found on PATH" in err
    assert "brew install ffmpeg" in err and "Traceback" not in err


def test_serve_without_ffmpeg_warns_but_starts(tmp_path, monkeypatch, capsys,
                                               uvicorn_calls):
    no_ffmpeg(monkeypatch)
    port = free_port()
    rc = run_cli(monkeypatch, "serve", "--workspace", str(tmp_path),
                 "--port", str(port))
    out, err = capsys.readouterr()
    assert rc == 0 and len(uvicorn_calls) == 1
    assert "WARNING" in err and "ffmpeg and ffprobe not found" in err
    assert "winget install Gyan.FFmpeg" in err
    assert f"ClipForge running at http://127.0.0.1:{port}" in out


# ------------------------------------------------------- ffmpeg version

BANNERS = {
    "ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023": ("6.1.1-3ubuntu5", (6, 1)),
    "ffmpeg version 8.0 Copyright (c) 2000-2025": ("8.0", (8, 0)),
    "ffmpeg version n7.1.2 Copyright (c) 2000-2025": ("n7.1.2", (7, 1)),
    "ffmpeg version 5.1 Copyright": ("5.1", (5, 1)),
    "ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright": ("4.4.2-0ubuntu0.22.04.1", (4, 4)),
    "ffmpeg version 5.0.3-static https://johnvansickle.com": ("5.0.3-static", (5, 0)),
    # git builds carry no release number and count as current
    "ffmpeg version N-127388-g615b07170e-20261012 Copyright": ("N-127388-g615b07170e-20261012", None),
    "ffmpeg version 2024-05-13-git-37db0454e4-full_build-www.gyan.dev": ("2024-05-13-git-37db0454e4-full_build-www.gyan.dev", None),
    "something else entirely": ("unknown", None),
}


@pytest.mark.parametrize("banner", BANNERS)
def test_parse_ffmpeg_version(banner):
    assert cli.parse_ffmpeg_version(banner) == BANNERS[banner]


def fake_version(monkeypatch, banner):
    monkeypatch.setattr(cli.shutil, "which", lambda n, *a, **k: f"/bin/{n}")
    monkeypatch.setattr(cli, "ffmpeg_version",
                        lambda: cli.parse_ffmpeg_version(banner))


@pytest.mark.parametrize("banner", [
    "ffmpeg version 5.0.3 Copyright", "ffmpeg version 4.4.2-0ubuntu Copyright"])
def test_ffmpeg_older_than_5_1_is_refused_with_clear_message(monkeypatch, banner):
    fake_version(monkeypatch, banner)
    msg = cli.ffmpeg_problem()
    assert "too old" in msg and "5.1 or newer" in msg
    assert banner.split()[2] in msg and "winget install Gyan.FFmpeg" in msg


@pytest.mark.parametrize("banner", [
    "ffmpeg version 5.1 Copyright", "ffmpeg version 6.1.1-3ubuntu5 Copyright",
    "ffmpeg version 8.0 Copyright", "ffmpeg version N-127388-g615b07170e Copyright",
    "ffmpeg version 2024-05-13-git-37db0454e4-full_build Copyright"])
def test_ffmpeg_5_1_and_newer_and_git_builds_accepted(monkeypatch, banner):
    fake_version(monkeypatch, banner)
    assert cli.ffmpeg_problem() is None


def test_ingest_with_old_ffmpeg_exits_2_cleanly(tmp_path, monkeypatch, capsys):
    fake_version(monkeypatch, "ffmpeg version 4.4.2 Copyright")
    called = []
    monkeypatch.setattr(cli, "ingest", lambda *a, **k: called.append(1))
    rc = run_cli(monkeypatch, "ingest", "v.mp4", "--workspace", str(tmp_path))
    err = capsys.readouterr().err
    assert rc == 2 and not called
    assert "INGEST FAILED" in err and "ffmpeg 4.4.2 is too old" in err
    assert "Traceback" not in err


def test_ingest_prints_detected_ffmpeg_version(tmp_path, monkeypatch, capsys):
    fake_version(monkeypatch, "ffmpeg version 8.0 Copyright")
    monkeypatch.setattr(cli, "ingest", lambda *a, **k: (_ for _ in ()).throw(
        cli.IngestError("stop here")))
    run_cli(monkeypatch, "ingest", "v.mp4", "--workspace", str(tmp_path))
    assert "ffmpeg: 8.0" in capsys.readouterr().err


def test_serve_prints_detected_ffmpeg_version(tmp_path, monkeypatch, capsys,
                                              uvicorn_calls):
    fake_version(monkeypatch, "ffmpeg version 6.1.1-3ubuntu5 Copyright")
    rc = run_cli(monkeypatch, "serve", "--workspace", str(tmp_path),
                 "--port", str(free_port()))
    assert rc == 0 and "ffmpeg: 6.1.1-3ubuntu5" in capsys.readouterr().err


def test_serve_with_old_ffmpeg_warns_but_starts(tmp_path, monkeypatch, capsys,
                                                uvicorn_calls):
    fake_version(monkeypatch, "ffmpeg version 4.4.2 Copyright")
    rc = run_cli(monkeypatch, "serve", "--workspace", str(tmp_path),
                 "--port", str(free_port()))
    err = capsys.readouterr().err
    assert rc == 0 and len(uvicorn_calls) == 1
    assert "WARNING" in err and "too old" in err


def test_real_ffmpeg_is_detected_and_accepted():
    if cli.missing_ffmpeg_tools():
        pytest.skip("ffmpeg/ffprobe not installed")
    text, ver = cli.ffmpeg_version()
    assert text != "unknown"
    assert ver is None or ver >= cli.MIN_FFMPEG
    assert cli.ffmpeg_problem() is None


# --------------------------------------------------------------- port

def test_serve_with_busy_port_exits_2_before_starting(tmp_path, monkeypatch,
                                                      capsys, busy_port,
                                                      uvicorn_calls):
    rc = run_cli(monkeypatch, "serve", "--workspace", str(tmp_path),
                 "--port", str(busy_port))
    out, err = capsys.readouterr()
    assert rc == 2 and not uvicorn_calls
    assert err.strip().startswith(f"Port {busy_port} is already in use")
    assert "another ClipForge server may still be running" in err
    assert f"python -m clipforge.cli serve --port {busy_port + 1}" in err
    assert "running at" not in out and "Traceback" not in err


def test_serve_with_free_port_prints_running_line(tmp_path, monkeypatch,
                                                  capsys, uvicorn_calls):
    monkeypatch.setattr(cli.shutil, "which", lambda n, *a, **k: f"/bin/{n}")
    port = free_port()
    assert run_cli(monkeypatch, "serve", "--workspace", str(tmp_path),
                   "--port", str(port)) == 0
    out, err = capsys.readouterr()
    assert f"ClipForge running at http://127.0.0.1:{port}" in out
    assert "WARNING" not in err and len(uvicorn_calls) == 1


def test_port_problem_unit(busy_port):
    assert cli.port_problem("127.0.0.1", free_port()) is None
    assert "already in use" in cli.port_problem("127.0.0.1", busy_port)
    # the probe socket is closed again: the free port stays bindable
    p = free_port()
    assert cli.port_problem("127.0.0.1", p) is None
    assert cli.port_problem("127.0.0.1", p) is None


def test_port_problem_probes_every_resolved_address(monkeypatch, busy_port):
    """--host localhost can resolve to ::1 and 127.0.0.1: a server on the
    second must be noticed even when the first is free or unusable."""
    def fake_getaddrinfo(host, port, **kw):
        return [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", port, 0, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port)),
        ]
    monkeypatch.setattr(cli.socket, "getaddrinfo", fake_getaddrinfo)
    assert "already in use" in cli.port_problem("localhost", busy_port)
    monkeypatch.setattr(cli.socket, "getaddrinfo", fake_getaddrinfo)
    assert cli.port_problem("localhost", free_port()) is None


def test_port_problem_localhost_name_is_checked(busy_port):
    assert "already in use" in cli.port_problem("localhost", busy_port)


def test_port_problem_unresolvable_host_is_a_clean_message(monkeypatch):
    def fail(*a, **k):
        raise socket.gaierror(-2, "Name or service not known")
    monkeypatch.setattr(cli.socket, "getaddrinfo", fail)   # no real DNS lookup
    msg = cli.port_problem("no-such-host.invalid", 8000)
    assert msg.startswith("Cannot listen on no-such-host.invalid:8000")
