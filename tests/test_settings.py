"""F01: .env loading happens only in the CLI, and real env vars win."""
import os
import re
import subprocess
import sys
from pathlib import Path

from clipforge import settings
from clipforge.settings import load_env

ROOT = Path(__file__).resolve().parent.parent


def test_loads_values_when_unset(tmp_path, monkeypatch):
    # setenv first so monkeypatch restores (removes) the vars at teardown
    monkeypatch.setenv("CLIPFORGE_HL", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.delenv("CLIPFORGE_HL")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    env = tmp_path / ".env"
    env.write_text("CLIPFORGE_HL=mock\nANTHROPIC_API_KEY=test-key\n")
    assert load_env(env) is True
    assert os.environ["CLIPFORGE_HL"] == "mock"
    assert os.environ["ANTHROPIC_API_KEY"] == "test-key"


def test_real_env_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPFORGE_HL", "anthropic")
    env = tmp_path / ".env"
    env.write_text("CLIPFORGE_HL=mock\n")
    load_env(env)
    assert os.environ["CLIPFORGE_HL"] == "anthropic"


def test_missing_env_is_not_an_error(tmp_path):
    before = dict(os.environ)
    assert load_env(tmp_path / "nope.env") is False
    assert dict(os.environ) == before


def test_default_location_is_repo_root_regardless_of_cwd(tmp_path, monkeypatch):
    assert settings.DEFAULT_ENV_PATH == ROOT / ".env"
    seen = {}
    monkeypatch.setattr(
        settings, "load_dotenv",
        lambda p, override: seen.update(p=Path(p), o=override) or True)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.chdir(tmp_path)
    load_env()
    assert seen["p"] == ROOT / ".env"
    assert seen["o"] is False


def test_importing_library_does_not_read_dotenv():
    code = (
        "import dotenv\n"
        "def boom(*a, **k): raise SystemExit('dotenv was called')\n"
        "dotenv.load_dotenv = boom\n"
        "import clipforge.pipeline, clipforge.server.app\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"


def test_cli_main_loads_env(monkeypatch):
    from clipforge import cli
    called = []
    monkeypatch.setattr(cli, "load_env", lambda *a, **k: called.append(1))
    monkeypatch.setattr(sys, "argv", ["clipforge", "status", "nojob"])
    try:
        cli.main()
    except Exception:
        pass  # the job does not exist; only the load order matters
    assert called == [1]


def test_env_example_lists_every_clipforge_var():
    names = set()
    for py in (ROOT / "clipforge").rglob("*.py"):
        names |= set(re.findall(r"CLIPFORGE_[A-Z0-9_]+", py.read_text()))
    example = (ROOT / ".env.example").read_text()
    missing = {n for n in names if n not in example}
    assert names and not missing, f"missing from .env.example: {missing}"


def test_python_dotenv_in_requirements():
    assert "python-dotenv" in (ROOT / "requirements.txt").read_text().lower()
