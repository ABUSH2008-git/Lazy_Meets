import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def mock_api():
    import mock_groq

    port = _free_port()
    srv = mock_groq.serve(port)
    yield f"http://127.0.0.1:{port}/openai/v1"
    srv.shutdown()


@pytest.fixture
def settings(mock_api, tmp_path):
    from meetscribe.config import Settings

    s = Settings()
    s.api_key = "test"
    s.base_url = mock_api
    s.runs_dir = tmp_path / "runs"
    return s


@pytest.fixture(scope="session")
def demo_audio() -> Path:
    return ROOT / "samples" / "demo_meeting.mp3"
