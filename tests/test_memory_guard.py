"""Runtime pressure guard: block, evict, cancel, and never leave state stuck."""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

import config
from memory_manager import memory_manager
from services.memory import children


class _Row(dict):
    """dict with attribute access is unnecessary; rows are plain dicts."""


@pytest.fixture(autouse=True)
def _reset_guard():
    memory_manager._pressure = "normal"
    memory_manager._admission_blocked_until = 0.0
    memory_manager._jobs.clear()
    memory_manager._slots.clear()
    yield
    memory_manager._pressure = "normal"
    memory_manager._admission_blocked_until = 0.0
    memory_manager._jobs.clear()
    memory_manager._slots.clear()


def test_warning_blocks_admission_then_recovery_unblocks(monkeypatch):
    result = memory_manager.guard_pressure("warning")
    assert result["pressure"] == "warning"
    assert memory_manager._pressure_blocked() is True

    decision = memory_manager.admit_job("guard-1", "video", {"estimate_gb": 1.0})
    assert decision.ok is False
    assert decision.retryable is True
    assert "pressure" in (decision.reason or "").lower()

    # Normal pressure clears the block immediately.
    monkeypatch.setattr(config, "MEMORY_PRESSURE_BLOCK_SEC", 0.0)
    memory_manager.guard_pressure("normal")
    assert memory_manager._pressure_blocked() is False


def test_critical_marks_the_oldest_active_job_for_cancellation(monkeypatch):
    import job_utils

    rows = job_utils.active_jobs
    rows.clear()
    rows["old-job"] = {"status": "processing", "stage": "denoising"}
    rows["new-job"] = {"status": "processing", "stage": "loading"}
    try:
        result = memory_manager.guard_pressure("critical")
        assert result["cancelled"] == ["old-job"]
        assert rows["old-job"]["cancelled"] is True
        assert rows["old-job"]["status"] == "cancelling"
        assert "pressure" in rows["old-job"]["status_message"].lower()
        # A second sweep does not re-cancel the same job or pick a new one
        # until the first is actually gone.
        again = memory_manager.guard_pressure("critical")
        assert again["cancelled"] == ["new-job"]
    finally:
        rows.clear()


def test_warning_evicts_idle_slots():
    from services.memory.models import SlotState

    state = SlotState(slot="image", model_id="flux:test", cost_gb=4.0)
    memory_manager._slots["image"] = state
    memory_manager._active_slots = set()
    result = memory_manager.guard_pressure("warning")
    assert "image" in result["evicted"]

    memory_manager._active_slots = set()
    memory_manager._slots.clear()


def test_child_process_budget_measures_real_descendants():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.6)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 2
        summary = children.snapshot(force=True)["summary"]
        while time.time() < deadline and summary["count"] == 0:
            time.sleep(0.05)
            summary = children.snapshot(force=True)["summary"]
        assert summary["count"] >= 1
        assert summary["total_gb"] >= 0.0
        assert summary["threads"] >= 1
    finally:
        process.kill()
        process.wait(timeout=5)
        children.snapshot(force=True)
