"""Settings loading for the CLI.

Only the CLI entry point calls ``load_env``; importing the library (pipeline,
server) never reads a ``.env``, so tests cannot pick up a real API key.
"""
from pathlib import Path

from dotenv import load_dotenv

# The repo root: the parent of the ``clipforge`` package.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = REPO_ROOT / ".env"


def load_env(path: Path | str | None = None) -> bool:
    """Load ``KEY=value`` pairs from a .env file into ``os.environ``.

    Real environment variables always win (``override=False``). A missing file
    is not an error. Returns True if a file was found and read.
    """
    env_path = Path(path) if path is not None else DEFAULT_ENV_PATH
    if not env_path.is_file():
        return False
    return load_dotenv(env_path, override=False)
