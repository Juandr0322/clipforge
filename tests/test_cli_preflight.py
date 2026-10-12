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
