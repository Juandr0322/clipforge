"""Shared test setup."""
import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_audio_library(tmp_path_factory):
    """Tests never read Diego's real ~/clipforge-audio (F48): every test
    starts with an empty library folder unless it sets its own."""
    mp = pytest.MonkeyPatch()
    mp.setenv("CLIPFORGE_AUDIO_LIBRARY",
              str(tmp_path_factory.mktemp("empty-audio-library")))
    yield
    mp.undo()
