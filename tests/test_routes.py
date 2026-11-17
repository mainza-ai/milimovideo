def test_no_duplicate_method_path_routes(app_client):
    from server import app

    seen = {}
    duplicates = []
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if not methods or not path:
            continue
        for method in methods:
            if method in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                key = (method, path)
                if key in seen:
                    duplicates.append(key)
                seen[key] = True
    assert duplicates == []


import asyncio

import pytest

from database import QueuedJob, engine
from sqlmodel import Session


@pytest.mark.asyncio
async def test_durable_queue_ledger_records_failure(app_client):
    from job_utils import enqueue_gpu_job

    queue_id = await enqueue_gpu_job("job_ledger_test", "unknown", {})
    status = None
    # The row may wait up to MAX_TURN_WAIT_SEC behind a higher-priority queue
    # entry before it is dispatched and fails.
    for _ in range(400):
        await asyncio.sleep(0.02)
        with Session(engine) as session:
            queued = session.get(QueuedJob, queue_id)
            status = queued.status if queued else None
        if status in ("failed", "completed"):
            break
    assert status == "failed"


def test_job_metrics_endpoint(app_client):
    response = app_client.get("/metrics/jobs")
    assert response.status_code == 200
    body = response.json()
    assert "total" in body and "by_status" in body and "recent_failures" in body


def test_static_model_routes_are_not_shadowed(app_client):
    queue = app_client.get("/models/queue")
    assert queue.status_code == 200, queue.text
    assert "queue" in queue.json()

    disk = app_client.get("/models/disk")
    assert disk.status_code == 200, disk.text
    assert "disk" in disk.json()

    search = app_client.get("/models/search", params={"q": ""})
    assert search.status_code in (200, 501), search.text
