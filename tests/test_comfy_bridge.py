"""ComfyUI bridge: workflow parsing, the HTTP client, and a live smoke test."""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import pytest

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xcf"
    b"\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)

EMPTY_IMAGE_WORKFLOW = {
    "1": {
        "class_type": "EmptyImage",
        "inputs": {"width": 64, "height": 48, "batch_size": 1, "color": 0},
    },
    "2": {
        "class_type": "SaveImage",
        "inputs": {"images": ["1", 0], "filename_prefix": "milimo_test"},
    },
}


class _FakeComfy(BaseHTTPRequestHandler):
    """Minimal ComfyUI surface: submit, history, view, queue, free."""

    prompt: dict = {}
    deleted: list = []
    freed: list = []

    def log_message(self, *args):  # silence the test output
        pass

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def do_POST(self):  # noqa: N802 - http.server API
        path = urlparse(self.path).path
        if path == "/prompt":
            payload = self._body()
            type(self).prompt = payload.get("prompt") or {}
            self._json({"prompt_id": "fake-prompt", "number": 0, "node_errors": {}})
        elif path == "/queue":
            payload = self._body()
            type(self).deleted.extend(payload.get("delete") or [])
            self._json({})
        elif path == "/interrupt":
            self._json({})
        elif path == "/free":
            payload = self._body()
            type(self).freed.append(payload)
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)

    def do_GET(self):  # noqa: N802 - http.server API
        path = urlparse(self.path).path
        if path == "/system_stats":
            self._json({
                "system": {
                    "os": "test", "ram_total": 16e9, "ram_free": 8e9,
                    "comfyui_version": "test-0.1",
                },
                "devices": [],
            })
        elif path == "/queue":
            self._json({"queue_running": [], "queue_pending": []})
        elif path == "/history/fake-prompt":
            self._json({"fake-prompt": {
                "status": {"status_str": "success", "completed": True, "messages": []},
                "outputs": {"2": {"images": [{
                    "filename": "fake_00001_.png",
                    "subfolder": "",
                    "type": "output",
                    "metadata": {"kind": "image", "width": 64, "height": 48},
                }]}},
            }})
        elif path == "/view":
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(PNG_BYTES)))
            self.end_headers()
            self.wfile.write(PNG_BYTES)
        else:
            self._json({"error": "not found"}, 404)


@pytest.fixture()
def fake_comfy():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeComfy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_parse_workflow_accepts_api_format_and_rejects_ui_json():
    from services.comfy import ComfyError, parse_workflow

    assert parse_workflow(json.dumps(EMPTY_IMAGE_WORKFLOW)) == EMPTY_IMAGE_WORKFLOW
    with pytest.raises(ComfyError, match="UI workflow"):
        parse_workflow({"nodes": [], "links": []})
    with pytest.raises(ComfyError, match="class_type"):
        parse_workflow({"1": {"inputs": {}}})


def test_overrides_prefix_and_validation():
    from services.comfy import (
        apply_filename_prefix,
        apply_overrides,
        parse_workflow,
        validate_workflow,
    )

    workflow = parse_workflow(EMPTY_IMAGE_WORKFLOW)
    merged = apply_overrides(workflow, {"1": {"width": 320, "height": 240}})
    assert merged["1"]["inputs"]["width"] == 320
    assert workflow["1"]["inputs"]["width"] == 64  # original untouched
    prefixed = apply_filename_prefix(merged, "job_42")
    assert prefixed["2"]["inputs"]["filename_prefix"] == "job_42"
    validation = validate_workflow(prefixed)
    assert validation["output_nodes"] == ["2"]
    assert (validation["width"], validation["height"]) == (320, 240)
    assert validation["warnings"] == []


def test_health_submit_history_and_download(fake_comfy):
    from services.comfy import ComfyClient

    with ComfyClient(base_url=fake_comfy, timeout=5) as client:
        health = client.health()
        assert health["reachable"] is True
        assert health["version"] == "test-0.1"
        assert health["queue_pending"] == 0

        prompt_id = client.submit(EMPTY_IMAGE_WORKFLOW)
        assert prompt_id == "fake-prompt"
        assert _FakeComfy.prompt["2"]["class_type"] == "SaveImage"

        entry = client.history(prompt_id)
        assert client.execution_error(entry) is None
        outputs = client.collect_outputs(entry)
        assert len(outputs) == 1
        assert outputs[0].kind == "image"
        assert client.download(outputs[0]) == PNG_BYTES


def test_client_reports_rejected_workflow():
    from services.comfy import ComfyClient, ComfyError

    with ComfyClient(base_url="http://127.0.0.1:9", timeout=0.5) as client:
        with pytest.raises(ComfyError):
            client.submit(EMPTY_IMAGE_WORKFLOW)
        assert client.health()["reachable"] is False


def test_progress_state_is_mapped_to_a_step():
    from services.comfy import ComfyClient

    payload = ComfyClient._progress_from_message("p1", EMPTY_IMAGE_WORKFLOW, {
        "type": "progress_state",
        "data": {
            "prompt_id": "p1",
            "nodes": {"9": {"value": 3.0, "max": 12.0, "state": "running"}},
        },
    })
    assert payload["stage"] == "denoising"
    assert payload["step"] == 3
    assert payload["total"] == 12
    assert 10 <= payload["progress"] <= 95

    other = ComfyClient._progress_from_message("p1", EMPTY_IMAGE_WORKFLOW, {
        "type": "progress", "data": {"prompt_id": "other", "value": 1, "max": 2},
    })
    assert other is None


def test_route_rejects_ui_workflows_and_outputless_graphs(app_client, project):
    response = app_client.post("/comfy/run", json={
        "project_id": project["id"],
        "workflow": {"nodes": [], "links": []},
    })
    assert response.status_code == 400
    assert "API Format" in response.json()["detail"]

    response = app_client.post("/comfy/run", json={
        "project_id": project["id"],
        "workflow": {"1": {"class_type": "EmptyImage", "inputs": {"width": 8, "height": 8}}},
    })
    assert response.status_code == 400
    assert "no Save" in response.json()["detail"]

    validated = app_client.post("/comfy/validate", json={"workflow": EMPTY_IMAGE_WORKFLOW})
    assert validated.status_code == 200
    assert validated.json()["output_nodes"] == ["2"]


def _comfy_reachable() -> bool:
    try:
        import httpx

        import config

        return httpx.get(f"{config.COMFY_URL}/system_stats", timeout=1.5).status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _comfy_reachable(), reason="ComfyUI is not running")
def test_live_comfyui_run_ingests_outputs(app_client, project):
    pid = project["id"]
    state = {
        "id": pid,
        "name": "Comfy Film",
        "fps": 25,
        "seed": 42,
        "scenes": [],
        "revision": project.get("revision", 0),
        "shots": [{
            "id": "comfy-shot",
            "project_id": pid,
            "index": 0,
            "sequence_index": 0,
            "prompt": "empty image smoke test",
            "seed": 1,
            "width": 64,
            "height": 48,
            "num_frames": 1,
            "track_index": 0,
            "timeline": [],
        }],
    }
    assert app_client.put(f"/projects/{pid}", json=state).status_code == 200

    response = app_client.post("/comfy/run", json={
        "project_id": pid,
        "shot_id": "comfy-shot",
        "workflow": EMPTY_IMAGE_WORKFLOW,
        "estimate_seconds": 5,
    })
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]

    status = {}
    for _ in range(90):
        status = app_client.get(f"/status/{job_id}").json()
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(1)
    assert status.get("status") == "completed", status
    assert status["video_url"].startswith(f"/projects/{pid}/generated/")
    assert status["progress"] == 100

    generated = os.path.join(os.environ["MILIMO_PROJECTS_DIR"], pid, "generated")
    files = [name for name in os.listdir(generated) if name.startswith(f"comfy_{job_id}")]
    assert files, "ComfyUI output was not copied into the project"
    assert os.path.getsize(os.path.join(generated, files[0])) > 0

    shot = next(
        item for item in app_client.get(f"/projects/{pid}").json()["shots"]
        if item["id"] == "comfy-shot"
    )
    assert shot["video_url"] == status["video_url"]
    assert shot["status"] == "completed"

    assets = app_client.get(f"/projects/{pid}/images").json()
    assert any(row.get("url") == status["video_url"] for row in assets)


def test_comfy_is_a_declared_external_slot():
    from job_utils import job_slot, row_priority
    from services.memory import costs

    estimate = costs.estimate("comfy", {"estimate_gb": 12})
    assert estimate.kind == "comfy"
    assert estimate.base_gb == 12
    assert estimate.total_gb >= 12
    assert job_slot("comfy") == "comfy"

    class Row:
        kind = "comfy"
        params_json = '{"priority": 7}'

    assert row_priority(Row()) == 7


def test_comfy_is_optional_and_can_be_disabled(app_client, project, monkeypatch):
    """Core Milimo routes must work with the bridge switched off."""
    import config

    monkeypatch.setattr(config, "COMFY_ENABLED", False)

    health = app_client.get("/comfy/health")
    assert health.status_code == 200
    assert health.json()["enabled"] is False
    assert health.json()["reachable"] is False
    assert "disabled" in health.json()["error"].lower()

    run = app_client.post("/comfy/run", json={
        "project_id": project["id"],
        "workflow": EMPTY_IMAGE_WORKFLOW,
    })
    assert run.status_code == 409
    assert "MILIMO_COMFY_ENABLED=0" in run.json()["detail"]

    validated = app_client.post("/comfy/validate", json={"workflow": EMPTY_IMAGE_WORKFLOW})
    assert validated.status_code == 200
    assert validated.json()["enabled"] is False
    assert validated.json()["output_nodes"] == ["2"]

    # The rest of the studio is independent of ComfyUI.
    assert app_client.get("/health").status_code == 200
    assert app_client.get(f"/projects/{project['id']}").status_code == 200
    preflight = app_client.post("/system/memory/preflight", json={
        "kind": "image", "params": {"width": 512, "height": 512},
    })
    assert preflight.status_code == 200
