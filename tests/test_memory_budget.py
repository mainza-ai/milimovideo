"""Measured memory budget: costs, admission, leases, eviction, and engines.

These tests do not load models. They exercise the arithmetic and the policy
with the budget forced on, because `conftest` disables enforcement globally.
"""

from __future__ import annotations

import pytest

import config
from services.memory import budget, costs
from services.memory.models import EngineResidency, MemorySnapshot, SlotState


class _FakeEngine:
    """Minimal settings object shaped like ``services.llm.models.LLMSettings``."""

    revision = 1

    def __init__(self, providers):
        self.providers = providers


class _FakeProvider:
    def __init__(self, type_, *, base_url="http://127.0.0.1:9", memory_gb=0.0):
        self.type = type_
        self.name = type_
        self.base_url = base_url
        self.enabled = True
        self.is_local = True
        self.memory_gb = memory_gb


@pytest.fixture(scope="module", autouse=True)
def _database():
    """Engine adapters read the persisted LLM settings document."""
    from database import init_db

    init_db()


@pytest.fixture()
def enforced(monkeypatch):
    monkeypatch.setattr(config, "MEMORY_ENFORCE", True)
    monkeypatch.setattr(config, "SKIP_VRAM_CHECK", False)
    monkeypatch.setattr(config, "MEMORY_SWAP_LIMIT_PERCENT", 85.0)
    return True


def _snapshot(**overrides) -> MemorySnapshot:
    base = dict(
        total_gb=128.0,
        reserve_gb=19.2,
        ceiling_gb=115.4,
        in_process_gb=0.0,
        resident_gb=0.0,
        external_gb=0.0,
        sam_gb=0.0,
        used_gb=0.0,
        available_gb=115.4,
        swap_percent=0.0,
        pressure="normal",
        enforce=True,
        timestamp=0.0,
    )
    base.update(overrides)
    return MemorySnapshot(**base)


# ── Costs ────────────────────────────────────────────────────────────


def test_reference_121_frame_upscale_job_costs_more_than_the_mps_ceiling():
    estimate = costs.estimate("video", {
        "num_frames": 121, "width": 768, "height": 512, "upscale": True,
    })
    assert estimate.total_gb > 100
    # Upscaling must cost more than the same shot without it.
    plain = costs.estimate("video", {"num_frames": 121, "width": 768, "height": 512})
    assert estimate.total_gb > plain.total_gb
    assert plain.total_gb >= costs.VIDEO_BASE_GB


def test_cost_scales_with_frames_and_pixels():
    small = costs.estimate("video", {"num_frames": 49, "width": 512, "height": 320})
    large = costs.estimate("video", {"num_frames": 121, "width": 768, "height": 512})
    assert large.total_gb > small.total_gb


def test_image_cost_counts_references():
    one = costs.estimate("image", {"reference_count": 1})
    four = costs.estimate("image", {"reference_count": 4})
    assert four.total_gb >= one.total_gb
    assert one.total_gb > costs.IMAGE_BASE_GB


def test_llm_estimate_reads_param_count_and_quantization():
    small = costs.estimate_llm_gb("llama3.2:3b-q4_K_M")
    large = costs.estimate_llm_gb("llama3.1:70b")
    assert 0 < small < large
    # A 70B model in bf16 cannot fit beside an LTX-2 pipeline.
    assert large > costs.VIDEO_BASE_GB


# ── Admission ────────────────────────────────────────────────────────


def test_admission_allows_a_job_that_fits(enforced, monkeypatch):
    monkeypatch.setattr(budget, "snapshot", lambda *a, **k: _snapshot())
    decision = budget.admit("video", {"num_frames": 49, "width": 512, "height": 320})
    assert decision.ok is True
    assert decision.estimate is not None
    assert decision.reason is None


def test_admission_refuses_when_external_engines_eat_the_budget(enforced, monkeypatch):
    """The incident shape: Strata holds 33 GB and the job needs ~100 GB."""
    engine = EngineResidency(
        provider_id="strata", provider_type="openai_compatible", name="Strata",
        base_url="http://127.0.0.1:1234", resident_gb=33.0, can_unload=False,
        source="declared",
    )
    used = 33.0
    snap = _snapshot(
        external_gb=used, used_gb=used, available_gb=115.4 - used,
        external_engines=[engine],
    )
    monkeypatch.setattr(budget, "snapshot", lambda *a, **k: snap)
    decision = budget.admit(
        "video", {"num_frames": 121, "width": 768, "height": 512, "upscale": True}
    )
    assert decision.ok is False
    assert "only" in (decision.reason or "")
    assert decision.suggestions


def test_admission_refuses_when_free_ram_cannot_hold_the_peak(enforced, monkeypatch):
    """Budget headroom is not enough: the bytes must actually be free."""
    monkeypatch.setattr(
        budget, "snapshot", lambda *a, **k: _snapshot(
            physical_available_gb=50.0, swap_free_gb=20.0,
        )
    )
    decision = budget.admit(
        "video", {"num_frames": 121, "width": 768, "height": 512, "upscale": True}
    )
    assert decision.ok is False
    assert "ram is free" in (decision.reason or "").lower()


def test_admission_keeps_a_margin_when_swap_is_nearly_exhausted(enforced, monkeypatch):
    """A job that exactly fills free RAM must be refused when swap cannot absorb a miss."""
    monkeypatch.setattr(
        budget, "snapshot", lambda *a, **k: _snapshot(
            physical_available_gb=106.0, swap_free_gb=1.7,
        )
    )
    decision = budget.admit(
        "video", {"num_frames": 121, "width": 768, "height": 512, "upscale": True}
    )
    assert decision.ok is False
    assert "safety margin" in (decision.reason or "")


def test_admission_allows_the_same_job_with_healthy_swap(enforced, monkeypatch):
    monkeypatch.setattr(
        budget, "snapshot", lambda *a, **k: _snapshot(
            physical_available_gb=110.0, swap_free_gb=20.0,
        )
    )
    decision = budget.admit(
        "video", {"num_frames": 121, "width": 768, "height": 512, "upscale": True}
    )
    assert decision.ok is True


def test_admission_credits_unloading_idle_in_process_models(enforced, monkeypatch):
    idle_video = SlotState(slot="video", model_id="ltx", cost_gb=76.0, leases=0, last_used=0.0)
    snap = _snapshot(
        physical_available_gb=34.0, available_gb=40.0, resident_gb=76.0,
        used_gb=76.0, swap_free_gb=20.0, slots={"video": idle_video},
    )
    monkeypatch.setattr(budget, "snapshot", lambda *a, **k: snap)
    # 34 GB free + 76 GB reclaimed = 110 GB, enough for the 105.84 GB peak.
    decision = budget.admit(
        "video", {"num_frames": 121, "width": 768, "height": 512, "upscale": True},
        slots={"video": idle_video},
    )
    assert decision.ok is True


def test_admission_refuses_under_critical_pressure(enforced, monkeypatch):
    monkeypatch.setattr(
        budget, "snapshot", lambda *a, **k: _snapshot(pressure="critical", swap_percent=98.0)
    )
    decision = budget.admit("image", {})
    assert decision.ok is False
    assert "critical" in (decision.reason or "").lower()


def test_admission_refuses_when_swap_is_exhausted_and_ram_is_low(enforced, monkeypatch):
    monkeypatch.setattr(
        budget, "snapshot", lambda *a, **k: _snapshot(
            swap_percent=96.0, swap_free_gb=1.6,
            physical_available_gb=2.0, available_gb=10.0,
        )
    )
    decision = budget.admit("image", {})
    assert decision.ok is False
    assert "ram" in (decision.reason or "").lower()


def test_allocated_swap_alone_does_not_block_a_job(enforced, monkeypatch):
    """macOS keeps swap allocated after a heavy job; that is history, not pressure."""
    monkeypatch.setattr(
        budget, "snapshot", lambda *a, **k: _snapshot(
            swap_percent=96.0, swap_free_gb=1.6, physical_available_gb=48.0,
        )
    )
    decision = budget.admit("image", {})
    assert decision.ok is True
    assert any("swap" in warning for warning in decision.warnings)


def test_admission_counts_reclaimable_idle_slots(enforced, monkeypatch):
    idle_video = SlotState(
        slot="video", model_id="ltx", cost_gb=76.0, leases=0, last_used=0.0,
    )
    snap = _snapshot(
        resident_gb=76.0, used_gb=76.0, available_gb=115.4 - 76.0,
        slots={"video": idle_video},
    )
    monkeypatch.setattr(budget, "snapshot", lambda *a, **k: snap)
    # The image job conflicts with the idle video slot, so it can be reclaimed.
    decision = budget.admit("image", {}, slots={"video": idle_video})
    assert decision.reclaimable_gb >= 76.0
    assert any("video" in entry for entry in decision.will_unload)


def test_admission_never_reclaims_a_leased_slot(enforced, monkeypatch):
    busy_video = SlotState(
        slot="video", model_id="ltx", cost_gb=76.0, leases=1, last_used=0.0,
    )
    snap = _snapshot(
        resident_gb=76.0, used_gb=76.0, available_gb=115.4 - 76.0,
        slots={"video": busy_video},
    )
    monkeypatch.setattr(budget, "snapshot", lambda *a, **k: snap)
    decision = budget.admit("image", {}, slots={"video": busy_video})
    assert decision.reclaimable_gb == 0.0
    assert decision.will_unload == []


def test_admission_warns_near_the_ceiling(enforced, monkeypatch):
    monkeypatch.setattr(budget, "snapshot", lambda *a, **k: _snapshot())
    decision = budget.admit("video", {"num_frames": 121, "width": 768, "height": 512})
    if decision.ok:
        assert isinstance(decision.warnings, list)


def test_refusal_is_retryable_while_another_job_holds_memory(enforced, monkeypatch):
    monkeypatch.setattr(
        budget, "snapshot", lambda *a, **k: _snapshot(physical_available_gb=90.0, swap_free_gb=20.0)
    )
    decision = budget.admit(
        "video", {"num_frames": 121, "width": 768, "height": 512, "upscale": True},
        active_jobs=[{"job_id": "busy", "kind": "video", "started_at": 0}],
    )
    assert decision.ok is False
    assert decision.retryable is True
    assert decision.downgrade is not None


def test_refusal_is_permanent_when_the_peak_exceeds_the_machine(enforced, monkeypatch):
    monkeypatch.setattr(budget, "snapshot", lambda *a, **k: _snapshot())
    decision = budget.admit(
        "video", {"num_frames": 257, "width": 1216, "height": 704, "upscale": True},
        active_jobs=[{"job_id": "busy", "kind": "video", "started_at": 0}],
    )
    assert decision.ok is False
    assert decision.retryable is False


def test_external_residency_that_cannot_be_freed_is_not_retryable(enforced, monkeypatch):
    """Strata (33 GB) + oMLX (6.5 GB): waiting can never fit the 105.8 GB peak."""
    engines = [
        EngineResidency(
            provider_id="strata", provider_type="openai_compatible", name="Strata",
            base_url="http://127.0.0.1:1234", resident_gb=33.0,
            can_unload=False, source="declared",
        ),
        EngineResidency(
            provider_id="omlx", provider_type="openai_compatible", name="oMLX",
            base_url="http://127.0.0.1:8002", resident_gb=6.5,
            can_unload=True, source="measured",
        ),
    ]
    snap = _snapshot(
        external_gb=39.5, used_gb=39.5, available_gb=115.45 - 39.5,
        external_engines=engines,
    )
    monkeypatch.setattr(budget, "snapshot", lambda *a, **k: snap)
    decision = budget.admit(
        "video", {"num_frames": 121, "width": 768, "height": 512, "upscale": True},
        active_jobs=[{"job_id": "busy", "kind": "image", "started_at": 0}],
    )
    assert decision.ok is False
    assert decision.retryable is False
    assert decision.downgrade is None


def test_downgrade_chooses_a_preset_that_fits(enforced, monkeypatch):
    snap = _snapshot(physical_available_gb=104.0, swap_free_gb=20.0)
    monkeypatch.setattr(budget, "snapshot", lambda *a, **k: snap)
    decision = budget.admit(
        "video", {"num_frames": 121, "width": 768, "height": 512, "upscale": True}
    )
    assert decision.ok is False
    assert decision.downgrade is not None
    changes = decision.downgrade["params"]
    assert changes.get("upscale") is False
    assert decision.downgrade["estimate"]["total_gb"] <= 104.0
    assert decision.downgrade["label"]


def test_downgrade_is_absent_when_nothing_smaller_fits(enforced, monkeypatch):
    monkeypatch.setattr(budget, "snapshot", lambda *a, **k: _snapshot(physical_available_gb=8.0))
    decision = budget.admit(
        "video", {"num_frames": 121, "width": 768, "height": 512, "upscale": True}
    )
    assert decision.ok is False
    assert decision.downgrade is None


def test_enforcement_off_always_admits(monkeypatch):
    monkeypatch.setattr(config, "MEMORY_ENFORCE", False)
    decision = budget.admit("video", {"num_frames": 121, "width": 768, "height": 512})
    assert decision.ok is True
    assert decision.warnings


# ── CUDA ─────────────────────────────────────────────────────────────


def test_cuda_snapshot_sizes_against_vram(monkeypatch):
    monkeypatch.setattr(budget, "_cuda_state", lambda: (12.0, 16.0, 4.0))
    monkeypatch.setattr(budget, "_read_pressure", lambda: "normal")
    monkeypatch.setattr(budget, "_sam_gb", lambda: 0.0)
    monkeypatch.setattr(budget.engines, "collect", lambda *a, **k: [])
    monkeypatch.setattr(config, "CUDA_MEMORY_FRACTION", 0.90)
    monkeypatch.setattr(config, "CUDA_RESERVE_GB", 1.0)

    snap = budget.snapshot()

    assert snap.device == "cuda"
    assert snap.total_gb == pytest.approx(16.0)
    assert snap.physical_available_gb == pytest.approx(12.0)
    assert snap.in_process_gb == pytest.approx(4.0)
    assert snap.ceiling_gb == pytest.approx(14.4, abs=0.1)
    assert any("CUDA VRAM" in note for note in snap.notes)


def test_cuda_cost_uses_the_fp8_video_and_bf16_image_bases(monkeypatch):
    monkeypatch.setattr(costs, "_cuda_available", lambda: True)
    video = costs.estimate("video", {"num_frames": 121, "width": 768, "height": 512})
    image = costs.estimate("image", {})
    assert video.base_gb == costs.VIDEO_BASE_GB_CUDA
    assert image.base_gb == costs.IMAGE_BASE_GB_CUDA
    monkeypatch.setattr(costs, "_cuda_available", lambda: False)
    assert costs.estimate("video", {}).base_gb == costs.VIDEO_BASE_GB


def test_cuda_admission_refuses_a_16gb_card_with_a_vram_reason(monkeypatch):
    monkeypatch.setattr(config, "MEMORY_ENFORCE", True)
    monkeypatch.setattr(config, "SKIP_VRAM_CHECK", False)
    monkeypatch.setattr(costs, "_cuda_available", lambda: True)
    monkeypatch.setattr(budget, "_cuda_state", lambda: (12.0, 16.0, 4.0))
    monkeypatch.setattr(budget, "_read_pressure", lambda: "normal")
    monkeypatch.setattr(budget, "_sam_gb", lambda: 0.0)
    monkeypatch.setattr(budget.engines, "collect", lambda *a, **k: [])

    decision = budget.admit(
        "video", {"num_frames": 121, "width": 768, "height": 512, "upscale": True}
    )

    assert decision.ok is False
    assert "VRAM" in (decision.reason or "")
    assert decision.retryable is False
    assert any("offload" in hint for hint in decision.suggestions)


def test_cuda_ceiling_admits_a_card_that_can_hold_the_peak(monkeypatch):
    monkeypatch.setattr(config, "MEMORY_ENFORCE", True)
    monkeypatch.setattr(config, "SKIP_VRAM_CHECK", False)
    monkeypatch.setattr(costs, "_cuda_available", lambda: True)
    monkeypatch.setattr(budget, "_cuda_state", lambda: (78.0, 80.0, 2.0))
    monkeypatch.setattr(budget, "_read_pressure", lambda: "normal")
    monkeypatch.setattr(budget, "_sam_gb", lambda: 0.0)
    monkeypatch.setattr(budget.engines, "collect", lambda *a, **k: [])

    decision = budget.admit(
        "video", {"num_frames": 121, "width": 768, "height": 512, "upscale": True}
    )
    assert decision.ok is True
    assert decision.snapshot.device == "cuda"


def test_flux2_ae_points_at_the_real_flux2_repos():
    import json
    from pathlib import Path

    from flux2.util import FLUX2_MODEL_INFO

    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "backend/models_manifest.json").read_text()
    )
    models = {entry["id"]: entry for entry in manifest["models"]}
    assert models["flux2-klein-9b"]["huggingface"]["repo"] == "black-forest-labs/FLUX.2-klein-9B"
    assert models["flux2-ae"]["huggingface"]["repo"] == "black-forest-labs/FLUX.2-dev"
    assert models["flux2-ae"]["gated"] is True

    info = FLUX2_MODEL_INFO["flux.2-klein-9b"]
    assert info["repo_id"] == "black-forest-labs/FLUX.2-klein-9B"
    assert info["ae_repo_id"] == "black-forest-labs/FLUX.2-dev"


# ── Engines ──────────────────────────────────────────────────────────


def test_generic_engine_uses_a_declared_override(monkeypatch):
    from services.memory import engines

    engines.clear_cache()
    settings = _FakeEngine({
        "strata": _FakeProvider("openai_compatible", memory_gb=33.0),
    })
    rows = engines.collect(settings, force=True)
    assert len(rows) == 1
    assert rows[0].source == "declared"
    assert rows[0].resident_gb == pytest.approx(33.0)
    assert rows[0].can_unload is False
    engines.clear_cache()


def test_generic_engine_without_override_is_marked_unknown(monkeypatch):
    from services.memory import engines

    engines.clear_cache()

    class _Response:
        status_code = 200

        def json(self):
            return {"data": [{"id": "some-8b-model"}]}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, path):
            return _Response()

    monkeypatch.setattr(engines, "_client", lambda base_url: _Client())
    settings = _FakeEngine({
        "remote": _FakeProvider("openai_compatible", memory_gb=0.0),
    })
    rows = engines.collect(settings, force=True)
    assert len(rows) == 1
    assert rows[0].unknown is True
    assert rows[0].resident_gb > 0
    engines.clear_cache()


# ── Slots / leases / eviction ────────────────────────────────────────


@pytest.fixture()
def manager(monkeypatch):
    from memory_manager import MemoryManager

    monkeypatch.setattr(config, "MEMORY_ENFORCE", False)
    stateless = MemoryManager()
    # Never touch real pipelines in unit tests.
    monkeypatch.setattr(stateless, "_unload_ltx", lambda: None)
    monkeypatch.setattr(stateless, "_unload_flux", lambda: None)
    monkeypatch.setattr(stateless, "_unload_http_llm", lambda: None)
    monkeypatch.setattr(stateless, "_flush_memory", lambda: None)
    monkeypatch.setattr(stateless, "_notify", staticmethod(lambda *a, **k: None))
    return stateless


def test_lease_is_released_once(manager):
    manager.admit_job("job-1", "video", {"num_frames": 49})
    manager.prepare_for("video", model_id="ltx", cost_gb=76.0, job_id="job-1")
    state = manager._slots["video"]
    assert state.leases == 1
    assert state.job_ids == ["job-1"]
    manager.release_job("job-1")
    assert state.leases == 0
    assert state.job_ids == []
    # A second release is a no-op, not a negative lease.
    manager.release_job("job-1")
    assert state.leases == 0


def test_prepare_for_unloads_conflicting_idle_slot(manager):
    manager.mark_resident("image", "flux2", 36.0)
    manager.admit_job("job-2", "video", {"num_frames": 49})
    manager.prepare_for("video", model_id="ltx", cost_gb=76.0, job_id="job-2")
    assert "video" in manager._slots
    assert "image" not in manager._slots


def test_prepare_for_refuses_to_evict_a_leased_conflict(manager):
    manager.mark_resident("image", "flux2", 36.0)
    manager._slots["image"].leases = 1
    manager.admit_job("job-3", "video", {"num_frames": 49})
    with pytest.raises(MemoryError):
        manager.prepare_for("video", model_id="ltx", cost_gb=76.0, job_id="job-3")


def test_admission_unloads_idle_conflicts_before_the_load(manager, monkeypatch):
    manager.mark_resident("image", "flux2", 36.0)
    unloaded = []
    monkeypatch.setattr(manager, "_unload_slot", lambda slot: unloaded.append(slot))
    monkeypatch.setattr(config, "MEMORY_ENFORCE", True)
    import services.memory.budget as budget_module

    monkeypatch.setattr(
        budget_module, "admit",
        lambda *a, **k: type("D", (), {
            "ok": True, "reason": None, "estimate": None,
            "will_unload": ["image:flux2"], "suggestions": [],
            "snapshot": _snapshot(),
        })(),
    )
    manager.admit_job("job-reclaim", "video", {"num_frames": 49})
    assert unloaded == ["image"]


def test_set_keep_warm_blocks_eviction(manager):
    manager.mark_resident("video", "ltx", 76.0)
    assert manager.set_keep_warm("video", True) is True
    assert manager.evict_idle(force=True) == []
    assert "video" in manager._slots


def test_eviction_unloads_an_unleased_slot(manager):
    manager.mark_resident("image", "flux2", 36.0)
    assert manager.evict_idle(force=False) == []  # TTL is 300s; just used
    manager._slots["image"].last_used = 0.0
    assert manager.evict_idle(force=False) == ["image"]
    assert "image" not in manager._slots


def test_unload_sam_posts_to_the_lazy_service(monkeypatch):
    import httpx

    from memory_manager import MemoryManager

    calls = []

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url):
            calls.append(url)
            return type("R", (), {"status_code": 200})()

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: _Client())
    MemoryManager._unload_sam()
    assert calls == ["http://127.0.0.1:8002/unload"]


def test_active_slot_reports_through_status(manager):
    manager.admit_job("job-4", "image", {})
    manager.prepare_for("image", model_id="flux2", cost_gb=36.0, job_id="job-4")
    status = manager.status()
    assert "image" in status["active_slots"]
    assert status["slots"]["image"]["leases"] == 1
