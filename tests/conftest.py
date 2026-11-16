import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
TEST_ROOT = Path(tempfile.mkdtemp(prefix="milimo-tests-"))

os.environ.setdefault("MILIMO_DATABASE_URL", f"sqlite:///{TEST_ROOT / 'test.db'}")
os.environ.setdefault("MILIMO_PROJECTS_DIR", str(TEST_ROOT / "projects"))
os.environ.setdefault("MILIMO_HOST", "127.0.0.1")
os.environ.setdefault("MILIMO_MAX_UPLOAD_MB", "1")
os.environ.setdefault("MILIMO_DEBUG", "0")

for path in (
    BACKEND,
    ROOT / "LTX-2" / "packages" / "ltx-core" / "src",
    ROOT / "LTX-2" / "packages" / "ltx-pipelines" / "src",
    ROOT / "flux2" / "src",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(scope="session")
def app_client():
    from fastapi.testclient import TestClient
    from server import app

    with TestClient(app) as client:
        yield client


@pytest.fixture()
def project(app_client):
    response = app_client.post("/projects", json={"name": "Audit Test"})
    assert response.status_code == 200, response.text
    return response.json()
