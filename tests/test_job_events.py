"""Typed job events: envelope, sequence, replay, and terminal-once semantics."""

from __future__ import annotations

import json
import time

import pytest

import job_events
from events import EventManager, MESSAGE_TTL_SECONDS, event_manager


def _drain(queue):
    rows = []
    while not queue.empty():
        rows.append(queue.get_nowait())
    return rows


@pytest.fixture(autouse=True)
def _clean_events():
    event_manager._replay.clear()
    event_manager._seq = 0
    yield
    event_manager._replay.clear()
    event_manager._seq = 0


@pytest.mark.asyncio
async def test_event_manager_sequences_and_replays_after_last_event_id():
    first = EventManager()
    await first.broadcast("one", {"n": 1})
    await first.broadcast("two", {"n": 2})

    # Subscribe with a replay cursor: only seq > 1 should arrive.
    request = _FakeRequest()
    response = await first.subscribe(request, last_event_id="1")
    generator = response.body_iterator
    chunk = await generator.__anext__()
    assert str(chunk["id"]) == "2"
    assert chunk["event"] == "two"
    assert json.loads(chunk["data"]) == {"n": 2}
    await generator.aclose()


class _FakeRequest:
    def __init__(self):
        self._checks = 0

    async def is_disconnected(self) -> bool:
        self._checks += 1
        return self._checks > 2


@pytest.mark.asyncio
async def test_typed_progress_payload_carries_identity_and_stage(monkeypatch):
    calls = []

    async def fake_broadcast(event, payload):
        calls.append((event, payload))

    monkeypatch.setattr(event_manager, "broadcast", fake_broadcast)
    job_events.reset("job-abc")
    await job_events.stage(
        "job-abc", "denoising", progress=42, step=7, total=20, eta_seconds=31,
        message="Denoising 7/20",
    )
    event, payload = calls[-1]
    assert event == "job.stage"
    assert payload["job_id"] == "job-abc"
    assert payload["stage"] == "denoising"
    assert payload["progress"] == 42
    assert payload["step"] == 7
    assert payload["total"] == 20
    assert payload["eta_seconds"] == 31
    assert payload["message"] == "Denoising 7/20"
    assert payload["ts"] <= time.time()


@pytest.mark.asyncio
async def test_terminal_writes_once_and_ignores_duplicates(app_client, monkeypatch):
    """A second terminal must not overwrite the DB or emit a second event."""
    from sqlmodel import Session

    import job_utils
    from database import Job, engine

    job_id = "job-terminal-once"
    with Session(engine) as session:
        session.add(Job(id=job_id, type="video", status="processing", progress=90))
        session.commit()

    emitted = []

    async def fake_broadcast(event, payload):
        emitted.append((event, payload))

    monkeypatch.setattr(event_manager, "broadcast", fake_broadcast)
    monkeypatch.setattr(job_events, "_load_meta", lambda job_id: {})
    job_events.reset(job_id)

    first = await job_events.terminal(
        job_id, "completed", stage="completed", message="Done",
        output_path="/projects/p/a.mp4", progress=100,
    )
    second = await job_events.terminal(job_id, "failed", error="boom")
    assert first is True
    assert second is False
    assert [row[0] for row in emitted] == ["job.completed"]

    with Session(engine) as session:
        job = session.get(Job, job_id)
        assert job.status == "completed"
        assert job.stage == "completed"
        assert job.progress == 100
        assert job.output_path == "/projects/p/a.mp4"
        assert job.completed_at is not None


@pytest.mark.asyncio
async def test_recovery_marks_interrupted_jobs_failed_not_rerun(app_client, monkeypatch):
    """A restart must not silently reload a heavy model for a dead worker."""
    from sqlmodel import Session

    import job_utils
    from database import Job, QueuedJob, engine

    job_id = "job-interrupted"
    with Session(engine) as session:
        session.add(Job(id=job_id, type="video", status="processing", progress=40))
        session.add(
            QueuedJob(
                job_id=job_id, kind="video", params_json="{}", status="running",
                attempts=1,
            )
        )
        session.commit()

    launched = []
    monkeypatch.setattr(job_utils, "_spawn_background", lambda coro: launched.append(coro))
    monkeypatch.setattr(job_events, "_load_meta", lambda job_id: {})
    job_events.reset(job_id)

    await job_utils.resume_queued_jobs()

    assert launched == []  # nothing re-run
    with Session(engine) as session:
        job = session.get(Job, job_id)
        assert job.status == "failed"
        assert job.stage == "recovery"
        assert "restarted" in (job.error_message or "").lower()


@pytest.mark.asyncio
async def test_cancel_before_dispatch_is_terminal(app_client, monkeypatch):
    """Cancelling a queued job must cancel the Job *and* the dispatch row."""
    from sqlmodel import Session

    import job_events
    from database import Job, QueuedJob, engine

    job_id = "job-cancel-queued"
    with Session(engine) as session:
        session.add(Job(id=job_id, type="video", status="queued", stage="queued"))
        session.add(QueuedJob(job_id=job_id, kind="video", params_json="{}", status="queued"))
        session.commit()

    monkeypatch.setattr(job_events, "_load_meta", lambda job_id: {})
    job_events.reset(job_id)

    response = app_client.post(f"/jobs/{job_id}/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"

    with Session(engine) as session:
        job = session.get(Job, job_id)
        assert job.status == "cancelled"
        queued = session.exec(
            __import__("sqlmodel").select(QueuedJob).where(QueuedJob.job_id == job_id)
        ).first()
        assert queued.status == "cancelled"


def test_queue_priority_orders_interactive_before_bulk(app_client):
    from sqlmodel import Session, select

    import job_utils
    from database import QueuedJob, engine

    with Session(engine) as session:
        for row in session.exec(select(QueuedJob).where(QueuedJob.status == "queued")).all():
            session.delete(row)
        session.commit()
        session.add(QueuedJob(job_id="parse-1", kind="ai_parse", params_json="{}", status="queued"))
        session.add(QueuedJob(job_id="element-1", kind="element", params_json="{}", status="queued"))
        session.add(QueuedJob(job_id="video-1", kind="video", params_json="{}", status="queued"))
        session.commit()
        head = job_utils._queue_head(session)
        assert head.job_id == "video-1"

        # Once the interactive row is dispatched, the element job is next.
        head.status = "running"
        session.add(head)
        session.commit()
        assert job_utils._queue_head(session).job_id == "element-1"
    assert job_utils.job_priority("video") < job_utils.job_priority("element")
    assert job_utils.job_priority("element") < job_utils.job_priority("ai_parse")


def test_queue_head_prefers_the_slot_already_resident(app_client):
    from sqlmodel import Session, select

    import job_utils
    from database import QueuedJob, engine

    with Session(engine) as session:
        for row in session.exec(select(QueuedJob).where(QueuedJob.status == "queued")).all():
            session.delete(row)
        session.commit()
        session.add(QueuedJob(job_id="video-first", kind="video", params_json="{}", status="queued"))
        session.add(QueuedJob(job_id="image-second", kind="image", params_json="{}", status="queued"))
        session.commit()

        # Same priority tier: with Flux resident, group the image job next.
        assert job_utils._queue_head(session, active_slots={"image"}).job_id == "image-second"
        assert job_utils._queue_head(session, active_slots={"video"}).job_id == "video-first"
        # With nothing resident, FIFO wins inside the tier.
        assert job_utils._queue_head(session, active_slots=set()).job_id == "video-first"


@pytest.mark.asyncio
async def test_wait_for_turn_blocks_behind_a_higher_priority_job(app_client):
    import asyncio

    from sqlmodel import Session, select

    import job_utils
    from database import QueuedJob, engine

    with Session(engine) as session:
        for row in session.exec(select(QueuedJob).where(QueuedJob.status == "queued")).all():
            session.delete(row)
        session.commit()
        session.add(QueuedJob(job_id="bulk-1", kind="ai_parse", params_json="{}", status="queued"))
        session.add(QueuedJob(job_id="hot-1", kind="video", params_json="{}", status="queued"))
        session.commit()
        bulk_id = session.exec(
            select(QueuedJob).where(QueuedJob.job_id == "bulk-1")
        ).first().id

    task = asyncio.create_task(job_utils.wait_for_turn(bulk_id))
    await asyncio.sleep(0.4)
    assert not task.done()

    with Session(engine) as session:
        hot = session.exec(select(QueuedJob).where(QueuedJob.job_id == "hot-1")).first()
        hot.status = "running"
        session.add(hot)
        session.commit()

    assert await asyncio.wait_for(task, timeout=2.0) is True


@pytest.mark.asyncio
async def test_transient_memory_refusal_waits_and_requeues(app_client, monkeypatch):
    """A retryable refusal must not fail the job or hold the GPU semaphore."""
    import asyncio
    from sqlmodel import Session, select

    import job_events
    import job_utils
    from database import Job, QueuedJob, engine
    from services.memory.models import MemoryDeferred

    with Session(engine) as session:
        for row in session.exec(select(QueuedJob).where(QueuedJob.status == "queued")).all():
            session.delete(row)
        session.commit()
        session.add(Job(id="job-wait-memory", type="image", status="queued"))
        queued = QueuedJob(
            job_id="job-wait-memory", kind="image", params_json=json.dumps({"width": 1024}),
            status="queued",
        )
        session.add(queued)
        session.commit()
        queue_id = queued.id

    async def refused(job_id, params):
        raise MemoryDeferred("Needs 40.0 GB but only 12.0 GB is free")

    monkeypatch.setattr(job_utils, "queue_image_task", refused)
    monkeypatch.setattr(job_utils, "_spawn_background", lambda coro: coro.close())
    monkeypatch.setattr(job_events, "_load_meta", lambda job_id: {})
    job_events.reset("job-wait-memory")

    emitted = []

    async def fake_broadcast(event, payload):
        emitted.append((event, payload))

    monkeypatch.setattr(event_manager, "broadcast", fake_broadcast)

    await job_utils.run_queued_job(queue_id)

    with Session(engine) as session:
        row = session.get(QueuedJob, queue_id)
        assert row.status == "queued"
        assert row.started_at is None
        assert row.attempts == 0
        stored = json.loads(row.params_json)
        assert stored["_memory_waits"] == 1
        assert stored["_memory_not_before"] > time.time()
        job = session.get(Job, "job-wait-memory")
        assert job.status == "pending"
        assert job.stage == "waiting"
    names = [name for name, _ in emitted]
    assert "job.waiting" in names
    assert "job.failed" not in names


def test_apply_downgrade_persists_the_smaller_preset(app_client):
    from sqlmodel import Session

    import job_utils
    from database import Job, QueuedJob, engine

    with Session(engine) as session:
        session.add(Job(id="job-downgrade", type="video", status="queued",
                        params_json=json.dumps({"num_frames": 121, "upscale": True})))
        session.add(QueuedJob(job_id="job-downgrade", kind="video", status="queued",
                              params_json=json.dumps({"num_frames": 121, "upscale": True})))
        session.commit()

    params = {"num_frames": 121, "width": 768, "height": 512, "upscale": True}
    updated = job_utils.apply_downgrade("job-downgrade", params, {
        "params": {"num_frames": 93, "upscale": False},
        "label": "93 frames, upscale off",
    })
    assert updated["num_frames"] == 93
    assert updated["upscale"] is False

    with Session(engine) as session:
        job = session.get(Job, "job-downgrade")
        assert json.loads(job.params_json)["num_frames"] == 93
        queued = session.exec(
            __import__("sqlmodel").select(QueuedJob).where(QueuedJob.job_id == "job-downgrade")
        ).first()
        assert json.loads(queued.params_json)["upscale"] is False


def test_status_reports_stage_and_memory(app_client):
    from sqlmodel import Session

    from database import Job, engine

    job_id = "job-status-stage"
    with Session(engine) as session:
        session.add(
            Job(
                id=job_id, type="video", status="processing", progress=37,
                stage="denoising", status_message="Denoising 4/20",
                memory_json=json.dumps({"peak_gb": 88.4, "peak_stage": "denoising"}),
            )
        )
        session.commit()

    body = app_client.get(f"/status/{job_id}").json()
    assert body["progress"] == 37
    assert body["stage"] == "denoising"
    assert body["status_message"] == "Denoising 4/20"
    assert body["memory"]["peak_gb"] == 88.4


def test_memory_endpoints_expose_budget_preflight_and_guarded_unload(app_client):
    status = app_client.get("/system/memory")
    assert status.status_code == 200
    body = status.json()
    for key in ("total_gb", "ceiling_gb", "available_gb", "pressure", "slots", "external_engines"):
        assert key in body
    assert isinstance(body["external_engines"], list)

    preflight = app_client.post(
        "/system/memory/preflight",
        json={"kind": "video", "params": {"num_frames": 121, "width": 768, "height": 512, "upscale": True}},
    )
    assert preflight.status_code == 200
    decision = preflight.json()
    assert "ok" in decision
    assert decision["estimate"]["total_gb"] > 0
    assert "snapshot" in decision

    # Invalid unload target is rejected instead of silently doing nothing.
    assert app_client.post("/system/memory/unload", json={}).status_code == 400
    missing = app_client.post("/system/memory/unload", json={"slot": "not-a-slot"})
    assert missing.status_code == 404

    keep = app_client.post("/system/memory/keep-warm", json={"slot": "not-a-slot"})
    assert keep.status_code == 404


def test_progress_estimates_eta_from_the_stage_step_rate(app_client, monkeypatch):
    import time

    import job_events
    from job_utils import active_jobs, update_job_progress

    active_jobs.clear()
    active_jobs["job-eta"] = {
        "status": "processing", "progress": 0, "stage": "denoising",
        "stage_started_ts": time.time() - 10.0,
    }
    monkeypatch.setattr(job_events, "progress_threadsafe", lambda *a, **k: None)
    update_job_progress("job-eta", 50, "Denoising 5/10", stage="denoising", step=5, total=10)
    eta = active_jobs["job-eta"]["eta_seconds"]
    assert 5 <= eta <= 20
    active_jobs.clear()


def test_metrics_expose_memory_queue_and_event_stats(app_client):
    body = app_client.get("/metrics/jobs").json()
    assert set(body["memory"]) >= {"loads", "unloads", "evictions", "refusals"}
    assert set(body["queue"]) >= {"samples", "wait_p50_s", "wait_p95_s"}
    assert set(body["events"]) >= {"clients", "sequence", "dropped", "peak_pending"}


def test_health_includes_memory_and_sampler(app_client):
    body = app_client.get("/health").json()
    assert "memory" in body
    assert "ceiling_gb" in body["memory"]
    assert "sampler" in body
    assert "stage" in body["sampler"]


def test_llm_memory_route_reports_engines(app_client):
    body = app_client.get("/settings/llm/memory")
    assert body.status_code == 200
    assert "engines" in body.json()
    assert "snapshot" in body.json()
