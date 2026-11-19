"""P5/M0 platform truth: SQLite WAL, gzip JSON, cached health, memory sampler."""

from __future__ import annotations

import time

import pytest

from services.memory.models import MemorySnapshot
from services.memory.sampler import MemorySampler


def test_sqlite_runs_wal_with_a_busy_timeout(app_client):
    from database import sqlite_runtime

    runtime = sqlite_runtime()
    assert runtime["journal_mode"] == "wal"
    assert runtime["busy_timeout_ms"] == 5000
    assert runtime["synchronous"] == 1  # NORMAL
    assert isinstance(runtime["wal_reset_fix"], bool)


def test_health_exposes_the_process_cap_and_sqlite_truth(app_client):
    body = app_client.get("/health").json()
    memory = body["memory"]
    assert memory["ceiling_gb"] == pytest.approx(memory["process_cap_gb"])
    assert memory["process_cap_gb"] <= memory["device_recommended_gb"]
    assert "journal_mode" in body["sqlite"]
    assert body["sqlite"]["journal_mode"] == "wal"


def test_liveness_endpoint_is_cheap(app_client):
    started = time.perf_counter()
    body = app_client.get("/health/live").json()
    elapsed = time.perf_counter() - started
    assert body == {"status": "ok"}
    assert elapsed < 0.5


def test_json_responses_are_gzipped_above_the_threshold(app_client):
    for index in range(12):
        app_client.post("/projects", json={"name": f"gzip project {index} " + "x" * 120})
    response = app_client.get("/projects", headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200
    assert response.headers.get("content-encoding") == "gzip"
    # TestClient transparently decodes the body; the JSON is still valid.
    payload = response.json()
    assert len(payload) >= 12


def test_small_json_is_not_compressed(app_client):
    response = app_client.get("/health/live", headers={"Accept-Encoding": "gzip"})
    assert response.headers.get("content-encoding") is None


def test_memory_sampler_records_stage_peaks_without_real_models(monkeypatch):
    sampler = MemorySampler()
    readings = iter(
        [
            {"used": 10.0, "current": 8.0, "cache": 2.0, "physical": 50.0, "swap_percent": 1.0},
            {"used": 14.0, "current": 12.0, "cache": 2.0, "physical": 44.0, "swap_percent": 3.0},
            {"used": 9.0, "current": 8.0, "cache": 1.0, "physical": 48.0, "swap_percent": 2.0},
        ]
    )
    monkeypatch.setattr(sampler, "_sample", lambda: next(readings, {
        "used": 9.0, "current": 8.0, "cache": 1.0, "physical": 48.0, "swap_percent": 2.0,
    }))
    sampler.start("job-1")
    sampler.mark("load")
    time.sleep(0.6)
    sampler.mark("denoise")
    time.sleep(0.6)
    trace = sampler.stop()

    assert trace is not None
    assert trace["job_id"] == "job-1"
    assert trace["peak_gb"] >= 10.0
    assert trace["peak_stage"] in {"load", "denoise", "starting"}
    assert trace["stages"]
    assert trace["physical_min_gb"] <= 50.0
    assert trace["swap_peak_percent"] >= 1.0
    assert trace["samples"] >= 1
    sampler.stop()


def test_memory_snapshot_reports_cap_not_recommended():
    from services.memory import budget

    snapshot: MemorySnapshot = budget.snapshot({}, [])
    if snapshot.device == "mps":
        assert snapshot.process_cap_gb == pytest.approx(snapshot.ceiling_gb)
        assert snapshot.process_cap_gb <= snapshot.device_recommended_gb
    assert snapshot.allocator_cache_gb >= 0.0


def test_system_memory_includes_sampler_children_and_recent_jobs(app_client):
    body = app_client.get("/system/memory").json()
    assert "sampler" in body
    assert "recent_jobs" in body
    assert "children" in body
    assert body["children"]["count"] >= 0
