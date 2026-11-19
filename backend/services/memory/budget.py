"""Measured memory budget, pressure detection, and admission control."""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Any, Dict, Iterable

import psutil

import config

from . import costs, engines
from .models import AdmissionDecision, CostBreakdown, EngineResidency, MemorySnapshot, SlotState

logger = logging.getLogger("memory.budget")

_last_pressure = "normal"


def _read_pressure() -> str:
    try:
        raw = subprocess.check_output(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            timeout=1.0, stderr=subprocess.DEVNULL,
        ).decode().strip()
        level = int(raw)
    except Exception:
        level = 1
    if level >= 3:
        return "critical"
    if level == 2:
        return "warning"
    return "normal"


def _mps_gb() -> float:
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            return float(torch.mps.driver_allocated_memory()) / 1e9
    except Exception:
        pass
    return 0.0


def _cuda_state() -> tuple[float, float, float] | None:
    """(free_gb, total_gb, allocated_gb) for device 0, or None."""
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            allocated = torch.cuda.memory_allocated(0)
            return free / 1e9, total / 1e9, allocated / 1e9
    except Exception:
        pass
    return None


def _mps_ceiling_gb() -> float | None:
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            return float(torch.mps.recommended_max_memory()) / 1e9
    except Exception:
        pass
    return None


def _allocator_state() -> tuple[float, float]:
    """(current allocated GB, cache GB) for the active accelerator."""
    try:
        import torch

        if torch.cuda.is_available():
            current = float(torch.cuda.memory_allocated(0)) / 1e9
            reserved = float(torch.cuda.memory_reserved(0)) / 1e9
            return current, max(0.0, reserved - current)
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            current = float(torch.mps.current_allocated_memory()) / 1e9
            driver = float(torch.mps.driver_allocated_memory()) / 1e9
            return current, max(0.0, driver - current)
    except Exception:
        pass
    return 0.0, 0.0


_sam_cache: dict[str, float] = {"stamp": 0.0, "gb": 0.0}


def _sam_gb() -> float:
    now = time.monotonic()
    if now - _sam_cache["stamp"] < 10.0:
        return _sam_cache["gb"]
    value = 0.0
    try:
        import httpx

        with httpx.Client(timeout=0.6) as client:
            response = client.get(f"http://127.0.0.1:{config.SAM_SERVICE_PORT}/health")
        if response.status_code < 400:
            data = response.json()
            value = costs.SAM_BASE_GB if data.get("model_loaded") else 0.5
    except Exception:
        value = 0.0
    _sam_cache.update({"stamp": now, "gb": value})
    return value


def snapshot(
    slots: Dict[str, SlotState] | None = None,
    active_jobs: Iterable[dict] | None = None,
) -> MemorySnapshot:
    global _last_pressure
    slots = slots or {}
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    cuda = _cuda_state()
    recommended = _mps_ceiling_gb() if not cuda else None
    allocator_current, allocator_cache = _allocator_state()
    device_recommended = 0.0
    process_cap = 0.0
    if cuda:
        free_vram, total, allocated = cuda
        fraction = max(0.1, min(0.99, float(config.CUDA_MEMORY_FRACTION)))
        reserve = max(float(config.CUDA_RESERVE_GB), total * (1.0 - fraction))
        device_recommended = total
        process_cap = max(0.5, min(total * fraction, total - reserve))
        ceiling = process_cap
        in_process = allocated
        device = "cuda"
        physical_available = free_vram
    else:
        total = vm.total / 1e9
        reserve = max(float(config.MEMORY_RESERVE_GB), total * 0.15)
        if recommended:
            fraction = max(0.1, min(0.95, float(config.MPS_MEMORY_FRACTION)))
            device_recommended = recommended
            # `set_per_process_memory_fraction` is relative to recommendedMaxWorkingSetSize.
            process_cap = max(0.5, min(recommended * fraction, total - reserve))
            ceiling = process_cap
            device = "mps"
        else:
            ceiling = total - reserve
            device = "cpu"
        in_process = _mps_gb()
        physical_available = vm.available / 1e9

    resident_non_llm = sum(
        state.cost_gb for slot, state in slots.items() if slot != "llm_http"
    )
    llm_slot = sum(state.cost_gb for slot, state in slots.items() if slot == "llm_http")
    try:
        external_rows = engines.collect()
    except Exception as exc:
        logger.warning("Could not read external engine residency: %s", exc)
        external_rows = []
    external = engines.total_gb(external_rows)
    # Do not count the same LLM twice (slot estimate + engine measurement).
    external = max(external, llm_slot)
    sam = _sam_gb()

    used = max(in_process, resident_non_llm) + external + sam
    available = max(0.0, ceiling - used)
    # The kernel pressure level is the authoritative signal. Swap occupancy is
    # reported separately: macOS keeps swap allocated after a heavy job, and
    # treating that as live pressure would refuse every job until reboot.
    pressure = _read_pressure()
    swap_free = swap.free / 1e9
    if cuda:
        # VRAM is the hard limit on CUDA; use it for pressure as well as sizing.
        if physical_available < 1.0:
            pressure = "critical"
        elif physical_available < total * 0.10 and pressure == "normal":
            pressure = "warning"
    elif physical_available < reserve * 0.25:
        pressure = "critical"
    _last_pressure = pressure

    notes: list[str] = []
    if recommended:
        notes.append(
            f"MPS process cap {process_cap:.1f} GB "
            f"({config.MPS_MEMORY_FRACTION:.0%} of {recommended:.1f} GB recommended)"
        )
    if allocator_cache > 0.05:
        notes.append(f"allocator cache holds {allocator_cache:.1f} GB")
    if cuda:
        notes.append(f"CUDA VRAM budget ({total:.1f} GB, {config.CUDA_MEMORY_FRACTION:.0%} cap)")
    if swap_free < 2.0:
        notes.append(f"swap is nearly exhausted ({swap_free:.1f} GB free)")
    if any(engine.unknown for engine in external_rows):
        notes.append("some engines report unknown residency; set their resident GB in AI & LLM")
    return MemorySnapshot(
        total_gb=total,
        reserve_gb=reserve,
        ceiling_gb=ceiling,
        in_process_gb=in_process,
        resident_gb=resident_non_llm + llm_slot,
        external_gb=external,
        sam_gb=sam,
        used_gb=used,
        available_gb=available,
        swap_percent=swap.percent,
        pressure=pressure,
        enforce=bool(config.MEMORY_ENFORCE),
        external_engines=list(external_rows),
        slots=dict(slots),
        active_jobs=list(active_jobs or []),
        timestamp=time.time(),
        notes=notes,
        physical_available_gb=physical_available,
        swap_free_gb=swap_free,
        device=device,
        device_recommended_gb=device_recommended,
        process_cap_gb=process_cap,
        allocator_current_gb=allocator_current,
        allocator_cache_gb=allocator_cache,
    )


def pressure() -> str:
    return _read_pressure()


def _conflicting(slot: str) -> set[str]:
    if slot == "video":
        return {"image", "llm_http", "comfy"}
    if slot == "image":
        return {"video", "llm_http", "comfy"}
    if slot == "llm_http":
        return {"comfy"}
    if slot == "comfy":
        return {"video", "image", "llm_http"}
    return set()


def reclaimable(
    snapshot_row: MemorySnapshot,
    kind: str,
    slots: Dict[str, SlotState],
    *,
    include_external: bool = True,
) -> tuple[float, list[str]]:
    """Idle/conflicting models that can be unloaded to make room."""
    target = (
        "video" if kind == "video"
        else "image" if kind == "image"
        else "llm_http" if kind == "llm"
        else "comfy" if kind == "comfy"
        else kind
    )
    freed = 0.0
    names: list[str] = []
    for slot, state in slots.items():
        if state.leases > 0 or state.keep_warm:
            continue
        if slot == target or slot in _conflicting(target):
            freed += state.cost_gb
            names.append(f"{slot}:{state.model_id}")
    if include_external and kind in ("video", "image", "llm"):
        for engine in snapshot_row.external_engines:
            if engine.can_unload and engine.resident_gb > 0:
                # oMLX/Ollama can be asked; Strata/generic cannot.
                freed += engine.resident_gb
                names.append(f"engine:{engine.provider_id}")
    return round(freed, 2), names


def _suggestions(
    decision_kind: str,
    estimate: CostBreakdown,
    snap: MemorySnapshot,
    will_unload: list[str],
) -> list[str]:
    hints: list[str] = []
    if will_unload:
        hints.append("Release and continue: " + ", ".join(will_unload[:4]))
    if estimate.kind == "video":
        hints.append("Reduce frames or resolution")
        if estimate.upscale:
            hints.append("Disable upscale for this shot")
    if estimate.kind == "image":
        hints.append("Use a smaller image preset")
    if any(engine.can_unload and engine.resident_gb for engine in snap.external_engines):
        hints.append("Unload local text models in Setup → AI & LLM")
    if snap.pressure != "normal":
        hints.append("Wait for accelerator memory pressure to clear")
    if snap.device == "cuda":
        hints.append("CUDA CPU offload is not implemented; use a larger GPU or a smaller checkpoint")
    return hints


def _fits(estimate: CostBreakdown, snap: MemorySnapshot, effective: float, physical_room: float) -> bool:
    margin = max(4.0, estimate.total_gb * 0.05) if snap.swap_free_gb < 4.0 else 0.5
    budget_ok = estimate.total_gb <= effective + 0.01
    physical_ok = (
        not snap.physical_available_gb
        or estimate.total_gb + margin <= physical_room
    )
    return budget_ok and physical_ok


def suggest_downgrade(
    kind: str,
    params: Dict[str, Any] | None,
    snap: MemorySnapshot,
    slots: Dict[str, SlotState] | None,
    effective: float,
    physical_room: float,
) -> Dict[str, Any] | None:
    """Largest smaller preset that fits the same budget, or None.

    The ladder preserves the request's intent as long as possible: drop the
    upscale pass first, then frames, then resolution.
    """
    params = dict(params or {})
    options: list[tuple[dict, str]] = []
    if kind == "video":
        cur_frames = int(params.get("num_frames") or 121)
        cur_w = int(params.get("width") or 768)
        cur_h = int(params.get("height") or 512)
        area = cur_w * cur_h
        upscale = bool(params.get("upscale"))

        def video(changes: dict, label: str) -> None:
            if int(changes.get("num_frames") or 121) > cur_frames:
                return
            if int(changes.get("width") or 768) * int(changes.get("height") or 512) > area:
                return
            options.append((changes, label))

        if upscale:
            video({"upscale": False}, "upscale off")
        frames = (121, 93, 81, 65, 49)
        for frame in frames:
            if frame < cur_frames:
                video({"num_frames": frame, "upscale": False}, f"{frame} frames, upscale off")
        for width, height in ((768, 512), (640, 448), (512, 320), (384, 256)):
            if width * height >= area:
                continue
            for frame in (121, 93, 65, 49):
                if frame > cur_frames:
                    continue
                video(
                    {"num_frames": frame, "width": width, "height": height, "upscale": False},
                    f"{frame} frames, {width}×{height}, upscale off",
                )
    elif kind == "image":
        cur_w = int(params.get("width") or 1024)
        cur_h = int(params.get("height") or 1024)
        area = cur_w * cur_h
        for width, height in ((1024, 1024), (768, 768), (640, 640), (512, 512)):
            if width * height >= area:
                continue
            options.append(({"width": width, "height": height}, f"{width}×{height}"))
    else:
        return None

    for changes, label in options:
        candidate = {**params, **changes}
        estimate = costs.estimate(kind, candidate)
        if _fits(estimate, snap, effective, physical_room):
            return {
                "params": changes,
                "label": label,
                "estimate": estimate.as_dict(),
            }
    return None


def admit(
    kind: str,
    params: Dict[str, Any] | None,
    slots: Dict[str, SlotState] | None = None,
    active_jobs: Iterable[dict] | None = None,
    *,
    job_id: str | None = None,
    force: bool = False,
    include_external: bool = True,
) -> AdmissionDecision:
    estimate = costs.estimate(kind, params)
    snap = snapshot(slots, active_jobs)
    if not snap.enforce or force or config.SKIP_VRAM_CHECK:
        return AdmissionDecision(
            ok=True, kind=kind, reason=None, estimate=estimate, snapshot=snap,
            warnings=["memory enforcement disabled"],
        )

    freed, will_unload = reclaimable(snap, kind, slots or {}, include_external=include_external)
    effective = snap.available_gb + freed
    # Only in-process slots become physical RAM again; an external engine that
    # reports it can unload also frees, but Strata-style engines may not.
    reclaim_inproc = 0.0
    reclaim_external = 0.0
    for entry in will_unload:
        name = entry.split(":", 1)[0]
        if name == "engine":
            reclaim_external += next(
                (engine.resident_gb for engine in snap.external_engines
                 if entry.endswith(engine.provider_id) and engine.can_unload),
                0.0,
            )
        else:
            state = (slots or {}).get(name)
            if state:
                reclaim_inproc += state.cost_gb
    physical_room = snap.physical_available_gb + reclaim_inproc + reclaim_external

    active = list(active_jobs or [])
    # Leased slots are not reclaimable yet, but the job holding them will
    # release them; that is the only way a wait can enlarge the budget.
    leased = sum(
        state.cost_gb for state in (slots or {}).values() if state.leases > 0
    )
    best_case = effective + leased
    transient = (
        bool(active)
        or snap.pressure != "normal"
        or reclaim_external > 0
        or (snap.physical_available_gb and snap.physical_available_gb < 4.0)
    )
    retryable = (
        transient
        and estimate.total_gb <= best_case + 0.01
        and estimate.total_gb <= snap.ceiling_gb
        and estimate.total_gb <= snap.total_gb
    )
    downgrade = suggest_downgrade(kind, params, snap, slots, effective, physical_room)

    def refuse(reason: str) -> AdmissionDecision:
        return AdmissionDecision(
            ok=False, kind=kind, reason=reason,
            estimate=estimate, snapshot=snap, reclaimable_gb=freed,
            will_unload=will_unload,
            suggestions=_suggestions(kind, estimate, snap, will_unload),
            retryable=retryable, downgrade=downgrade,
        )

    if snap.pressure == "critical":
        return refuse(f"System memory pressure is critical ({snap.swap_percent:.0f}% swap).")
    if snap.physical_available_gb and snap.physical_available_gb < 4.0:
        return refuse(
            f"Only {snap.physical_available_gb:.1f} GB of RAM is free. "
            "Close other apps or unload models before starting this job."
        )
    if (
        snap.device != "cuda"
        and snap.swap_free_gb
        and snap.swap_free_gb < 2.0
        and snap.physical_available_gb < snap.total_gb * 0.25
    ):
        return refuse(
            f"Swap is {snap.swap_percent:.0f}% used with only {snap.swap_free_gb:.1f} GB free, "
            f"and {snap.physical_available_gb:.1f} GB of RAM is free. "
            "Restart the machine or free memory before this job."
        )
    # With little swap left there is no fallback for an estimate miss, so keep
    # a real margin instead of admitting a job that exactly fills free RAM.
    if snap.device == "cuda":
        safety_margin = max(1.0, estimate.total_gb * 0.05)
    else:
        safety_margin = max(4.0, estimate.total_gb * 0.05) if snap.swap_free_gb < 4.0 else 0.5
    if snap.physical_available_gb and estimate.total_gb + safety_margin > physical_room:
        pool = "VRAM" if snap.device == "cuda" else "RAM"
        swap_note = (
            "" if snap.device == "cuda"
            else f", only {snap.swap_free_gb:.1f} GB swap free"
        )
        return refuse(
            f"Needs ~{estimate.total_gb:.1f} GB (+{safety_margin:.1f} GB safety margin{swap_note}) "
            f"but only {physical_room:.1f} GB of {pool} is free even after releasing idle models "
            f"({snap.physical_available_gb:.1f} GB now)."
        )
    if estimate.total_gb > effective + 0.01:
        return refuse(
            f"Needs ~{estimate.total_gb:.1f} GB but only {effective:.1f} GB can be made "
            f"available ({snap.used_gb:.1f} GB resident, {snap.external_gb:.1f} GB external)."
        )
    warnings: list[str] = []
    if estimate.total_gb > snap.ceiling_gb * 0.95:
        warnings.append("estimated peak is within 5% of the MPS ceiling")
    if snap.pressure == "warning":
        warnings.append("accelerator memory pressure is elevated")
    if snap.swap_percent > float(config.MEMORY_SWAP_LIMIT_PERCENT):
        warnings.append(
            f"swap is {snap.swap_percent:.0f}% used ({snap.swap_free_gb:.1f} GB free); "
            "an interrupted job may take longer to recover"
        )
    return AdmissionDecision(
        ok=True, kind=kind, reason=None, estimate=estimate, snapshot=snap,
        reclaimable_gb=freed, will_unload=will_unload, warnings=warnings,
    )

def set_mps_cap() -> float:
    """Cap this process so PyTorch fails before the OS thrashes (MPS or CUDA)."""
    try:
        import torch

        if torch.cuda.is_available():
            fraction = max(0.1, min(0.99, float(config.CUDA_MEMORY_FRACTION)))
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            logger.info("CUDA per-process memory fraction set to %.2f", fraction)
            return fraction
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            fraction = max(0.1, min(0.95, float(config.MPS_MEMORY_FRACTION)))
            torch.mps.set_per_process_memory_fraction(fraction)
            logger.info("MPS per-process memory fraction set to %.2f", fraction)
            return fraction
    except Exception as exc:
        logger.debug("Could not set accelerator memory fraction: %s", exc)
    return 0.0
