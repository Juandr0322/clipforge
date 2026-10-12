"""F01: .env loading happens only in the CLI, and real env vars win."""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

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
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    fake_root = tmp_path / "root"
    fake_root.mkdir()
    (fake_root / ".env").write_text("CLIPFORGE_DETECTOR=haar\n")
    monkeypatch.setattr(settings, "DEFAULT_ENV_PATH", fake_root / ".env")
    monkeypatch.setenv("CLIPFORGE_DETECTOR", "x")
    monkeypatch.delenv("CLIPFORGE_DETECTOR")
    monkeypatch.chdir(elsewhere)
    assert load_env() is True
    assert os.environ["CLIPFORGE_DETECTOR"] == "haar"


def test_empty_env_file_counts_as_loaded(tmp_path):
    env = tmp_path / ".env"
    env.write_text("")
    assert load_env(env) is True


def test_importing_library_does_not_read_dotenv():
    code = (
        "import os, sys\n"
        "before = dict(os.environ)\n"
        "import dotenv\n"
        "def boom(*a, **k): raise SystemExit('dotenv was called')\n"
        "dotenv.load_dotenv = boom\n"
        "dotenv.main.load_dotenv = boom\n"
        "import clipforge.pipeline, clipforge.server.app\n"
        "assert dict(os.environ) == before, 'environment changed on import'\n"
        "assert 'clipforge.settings' not in sys.modules\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"


def test_cli_main_loads_env_before_running_subcommand(tmp_path, monkeypatch):
    from clipforge import cli

    class Stop(Exception):
        pass

    calls = []
    monkeypatch.setattr(cli, "load_env", lambda *a, **k: calls.append("load_env"))

    def fake_manifest_load(job_dir):
        calls.append("subcommand")
        raise Stop

    monkeypatch.setattr(cli.Manifest, "load", fake_manifest_load)
    monkeypatch.setattr(
        sys, "argv",
        ["clipforge", "status", "nojob", "--workspace", str(tmp_path)])
    with pytest.raises(Stop):
        cli.main()
    assert calls == ["load_env", "subcommand"]


def test_env_example_lists_every_clipforge_var():
    pattern = r"CLIPFORGE_[A-Z0-9_]+"
    names = set()
    for py in (ROOT / "clipforge").rglob("*.py"):
        names |= set(re.findall(pattern, py.read_text()))
    listed = set(re.findall(pattern, (ROOT / ".env.example").read_text()))
    assert names, "no CLIPFORGE_* names found in source"
    assert names <= listed, f"missing from .env.example: {names - listed}"


def test_python_dotenv_in_requirements():
    assert "python-dotenv" in (ROOT / "requirements.txt").read_text().lower()
