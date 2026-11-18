"""
Central Memory Manager — measured residency, budget, and eviction.

On Apple Silicon, GPU models share physical memory. This manager now budgets
bytes (weights + workspace + external engines), admits or refuses jobs before a
load, leases slots to active jobs, and evicts idle models on a TTL or under
memory pressure.

Slots:
  - "video"    → LTX-2 pipeline (ModelManager)
  - "image"    → Flux 2 pipeline (FluxInpainter)
  - "llm_http" → a local HTTP LLM engine (Ollama/oMLX/Strata/...)
  - "sam"      → SAM 3 microservice (external process)
  - "depth"    → Depth Anything (in-process, small)

Usage:
    memory_manager.admit_job(job_id, "video", params)
    memory_manager.prepare_for("video")          # inside the GPU lease
    memory_manager.release_job(job_id)
"""
import gc
import logging
import threading
import time
from typing import Any, Dict

import torch

import config
from services.memory import budget, costs, engines
from services.memory.models import AdmissionDecision, MemorySnapshot, SlotState

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self):
        self._active_slots: set[str] = set()
        self._slots: Dict[str, SlotState] = {}
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._current_job: str | None = None
        self._lock = threading.RLock()
        self._monitor: threading.Thread | None = None
        self._monitor_stop = threading.Event()
        self._counters = {"loads": 0, "unloads": 0, "evictions": 0, "refusals": 0}

    # ── Slot Conflict Rules ──────────────────────────────────────
    # These slots cannot coexist in memory simultaneously.
    CONFLICTS = {
        "video": {"image", "llm_http", "comfy"},
        "image": {"video", "llm_http", "comfy"},
        "llm_http": {"comfy"},
        "comfy": {"video", "image", "llm_http"},
        "sam": set(),
        "depth": set(),
    }

    # ── Admission / leases ──────────────────────────────────────
    def admit_job(self, job_id: str, kind: str, params: dict | None = None) -> AdmissionDecision:
        """Budget a job before it loads anything. Refuses with an actionable reason."""
        with self._lock:
            if job_id in self._jobs:
                return self._jobs[job_id]["decision"]
            decision = budget.admit(
                kind,
                params or {},
                slots=self._slots,
                active_jobs=self._job_rows(),
                job_id=job_id,
            )
            if not decision.ok:
                self._counters["refusals"] += 1
                logger.warning("Memory admission refused %s (%s): %s", job_id, kind, decision.reason)
                return decision

            # Release what the decision said it would.
            for entry in decision.will_unload:
                if entry.startswith("engine:"):
                    provider_id = entry.split(":", 1)[1]
                    try:
                        result = engines.unload(provider_id=provider_id)
                        logger.info("Memory admission unloaded engine %s: %s", provider_id, result)
                    except Exception as exc:
                        logger.warning("Could not unload engine %s: %s", provider_id, exc)
                elif ":" in entry:
                    # Entries look like "video:ltx"; the slot is the left side.
                    slot_name = entry.split(":", 1)[0]
                    self._unload_slot(slot_name)

            self._jobs[job_id] = {
                "kind": kind,
                "slots": set(),
                "decision": decision,
                "started_at": time.time(),
            }
            self._current_job = job_id
            return decision

    def release_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.pop(job_id, None)
            if not job:
                return
            for slot in list(job["slots"]):
                state = self._slots.get(slot)
                if state:
                    state.leases = max(0, state.leases - 1)
                    state.job_ids = [item for item in state.job_ids if item != job_id]
                    state.last_used = time.time()
            if self._current_job == job_id:
                self._current_job = None

    def _job_rows(self) -> list[dict[str, Any]]:
        return [
            {"job_id": job_id, "kind": job["kind"], "started_at": job["started_at"]}
            for job_id, job in self._jobs.items()
        ]

    def _job_for(self, job_id: str | None) -> str | None:
        return job_id or self._current_job

    # ── Prepare / release ───────────────────────────────────────
    def prepare_for(
        self,
        slot: str,
        *,
        model_id: str | None = None,
        cost_gb: float | None = None,
        job_id: str | None = None,
        kind: str | None = None,
        allow_admission: bool = True,
    ) -> float:
        """Ensure a slot is ready. Unloads conflicts, admits the cost, leases it."""
        with self._lock:
            model_id = model_id or f"{slot}:default"
            cost = float(cost_gb if cost_gb is not None else costs.slot_base_gb(slot))
            existing = self._slots.get(slot)
            if existing and existing.model_id == model_id and slot in self._active_slots:
                existing.last_used = time.time()
                self._lease(slot, self._job_for(job_id))
                return existing.cost_gb

            admitted_job = self._job_for(job_id)
            if allow_admission and not (admitted_job and admitted_job in self._jobs):
                decision = budget.admit(
                    kind or slot,
                    {"estimate_gb": cost},
                    slots=self._slots,
                    active_jobs=self._job_rows(),
                )
                if not decision.ok:
                    self.evict_idle(force=False)
                    decision = budget.admit(
                        kind or slot,
                        {"estimate_gb": cost},
                        slots=self._slots,
                        active_jobs=self._job_rows(),
                    )
                if not decision.ok:
                    raise MemoryError(
                        f"{decision.reason} "
                        + ("Suggestions: " + "; ".join(decision.suggestions) if decision.suggestions else "")
                    )

            for conflict in self.CONFLICTS.get(slot, set()):
                if conflict in self._active_slots:
                    state = self._slots.get(conflict)
                    if state and state.leases > 0:
                        raise MemoryError(
                            f"Cannot load {slot}: {conflict} is in use by an active job."
                        )
                    self._unload_slot(conflict)

            if existing and existing.model_id != model_id:
                self._unload_slot(slot)

            self._slots[slot] = SlotState(
                slot=slot,
                model_id=model_id,
                cost_gb=cost,
                leases=0,
                acquired_at=time.time(),
                last_used=time.time(),
                keep_warm=bool(config.MEMORY_KEEP_WARM),
            )
            self._active_slots.add(slot)
            self._lease(slot, self._job_for(job_id))
            self._counters["loads"] += 1
            logger.info("Memory: resident %s (%s, %.1f GB)", slot, model_id, cost)
            return cost

    def _lease(self, slot: str, job_id: str | None) -> None:
        if not job_id:
            return
        job = self._jobs.get(job_id)
        if not job:
            return
        state = self._slots.get(slot)
        if not state:
            return
        if slot not in job["slots"]:
            state.leases += 1
            state.job_ids.append(job_id)
            job["slots"].add(slot)

    def mark_resident(self, slot: str, model_id: str, cost_gb: float, keep_warm: bool = False) -> None:
        with self._lock:
            self._slots[slot] = SlotState(
                slot=slot,
                model_id=model_id,
                cost_gb=float(cost_gb),
                acquired_at=time.time(),
                last_used=time.time(),
                keep_warm=keep_warm or bool(config.MEMORY_KEEP_WARM),
            )
            self._active_slots.add(slot)

    def note_llm(self, provider_id: str, model_id: str, cost_gb: float, job_id: str | None = None) -> float:
        # If the engine already reports this residency, it is already counted in
        # the external budget; do not double-charge or evict the engine we need.
        with self._lock:
            existing = self._slots.get("llm_http")
            if existing and existing.model_id == f"{provider_id}/{model_id}":
                existing.last_used = time.time()
                return existing.cost_gb
        snapshot = budget.snapshot(self._slots, self._job_rows())
        external = {engine.provider_id: engine for engine in snapshot.external_engines}
        engine = external.get(provider_id)
        if engine and engine.resident_gb > 0:
            return self.prepare_for(
                "llm_http",
                model_id=f"{provider_id}/{model_id}",
                cost_gb=float(engine.resident_gb),
                job_id=job_id,
                kind="llm",
                allow_admission=False,
            )
        decision = budget.admit(
            "llm",
            {"estimate_gb": cost_gb, "model": model_id},
            slots=self._slots,
            active_jobs=self._job_rows(),
            include_external=False,
        )
        if not decision.ok:
            raise MemoryError(decision.reason or "no memory for local LLM")
        return self.prepare_for(
            "llm_http",
            model_id=f"{provider_id}/{model_id}",
            cost_gb=cost_gb,
            job_id=job_id,
            kind="llm",
            allow_admission=False,
        )

    def has_active_job(self) -> bool:
        with self._lock:
            return bool(self._jobs)

    def unload_slot(self, slot: str) -> None:
        self._unload_slot(slot)

    def touch(self, slot: str) -> None:
        state = self._slots.get(slot)
        if state:
            state.last_used = time.time()

    def release(self, slot: str) -> None:
        """Mark a slot as no longer active (model has been unloaded)."""
        with self._lock:
            self._slots.pop(slot, None)
            self._active_slots.discard(slot)
        logger.info("Memory: released slot '%s'. Active: %s", slot, self._active_slots)

    def _unload_slot(self, slot: str) -> None:
        """Unload the model in the given slot."""
        with self._lock:
            state = self._slots.get(slot)
            if slot == "video":
                self._unload_ltx()
            elif slot == "image":
                self._unload_flux()
            elif slot == "llm_http":
                self._unload_http_llm()
            elif slot == "sam":
                self._unload_sam()
            elif slot == "comfy":
                self._unload_comfy()
            self._slots.pop(slot, None)
            self._active_slots.discard(slot)
        self._counters["unloads"] += 1
        logger.info("Memory: unloaded slot '%s' (%s)", slot, state.model_id if state else "unknown")
        self._flush_memory()

    def _unload_ltx(self) -> None:
        try:
            from model_engine import manager

            if manager._pipeline is not None:
                logger.info("MemoryManager: Unloading LTX-2 pipeline...")
                del manager._pipeline
                manager._pipeline = None
                manager._pipeline_type = None
        except Exception as exc:
            logger.warning("MemoryManager: Failed to unload LTX: %s", exc)

    def _unload_flux(self) -> None:
        try:
            from models.flux_wrapper import flux_inpainter

            if flux_inpainter.model_loaded:
                logger.info("MemoryManager: Unloading Flux 2 pipeline...")
                flux_inpainter.unload()
        except Exception as exc:
            logger.warning("MemoryManager: Failed to unload Flux: %s", exc)

    @staticmethod
    def _unload_sam() -> None:
        """Release the external SAM 3 image model, which now loads lazily."""
        try:
            import httpx

            with httpx.Client(timeout=5.0) as client:
                response = client.post("http://127.0.0.1:8001/unload")
            logger.info("MemoryManager: SAM unload -> %s", response.status_code)
        except Exception as exc:
            logger.debug("MemoryManager: SAM unload skipped: %s", exc)

    @staticmethod
    def _unload_comfy() -> None:
        """Ask ComfyUI to drop cached models; skip while it is executing."""
        if not config.COMFY_ENABLED:
            return
        try:
            from services.comfy import ComfyClient

            with ComfyClient(timeout=3.0) as client:
                queue = client.queue()
                if queue.get("queue_running"):
                    logger.info("MemoryManager: ComfyUI is running; not freeing")
                    return
                result = client.free(unload_models=True)
            logger.info("MemoryManager: ComfyUI free -> %s", result)
        except Exception as exc:
            logger.debug("MemoryManager: ComfyUI free skipped: %s", exc)

    def _unload_http_llm(self) -> None:
        try:
            result = engines.unload()
            logger.info("MemoryManager: Released local HTTP LLM models: %s", result)
        except Exception as exc:
            logger.warning("MemoryManager: Failed to release HTTP LLM models: %s", exc)

    def _flush_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ── Eviction / status ───────────────────────────────────────
    def evict_idle(self, force: bool = False) -> list[str]:
        """Unload unleased, non-pinned slots past their TTL (or all if forced)."""
        evicted: list[str] = []
        now = time.time()
        with self._lock:
            for slot, state in list(self._slots.items()):
                if state.leases > 0 or state.keep_warm:
                    continue
                ttl = int(config.MEMORY_IDLE_TTL_SEC.get(slot, 300))
                if force or (ttl > 0 and now - state.last_used >= ttl):
                    evicted.append(slot)
            for slot in evicted:
                self._unload_slot(slot)
            self._counters["evictions"] += len(evicted)
        if evicted:
            logger.info("Memory: evicted idle slots %s", evicted)
            self._notify("memory.evicted", {"slots": evicted, "force": force})
        return evicted

    def unload_all(self) -> list[str]:
        with self._lock:
            slots = list(self._slots)
        for slot in slots:
            self._unload_slot(slot)
        result = engines.unload()
        if result.get("unloaded") or result.get("errors"):
            self._notify("memory.evicted", {"slots": slots, "engines": result, "force": True})
        return slots

    def set_keep_warm(self, slot: str, keep_warm: bool) -> bool:
        state = self._slots.get(slot)
        if not state:
            return False
        state.keep_warm = bool(keep_warm)
        return True

    def status(self) -> dict[str, Any]:
        with self._lock:
            snapshot: MemorySnapshot = budget.snapshot(self._slots, self._job_rows())
            return {
                **snapshot.as_dict(),
                "active_slots": sorted(self._active_slots),
                "jobs": self._job_rows(),
                "counters": self.counters(),
            }

    def snapshot(self) -> MemorySnapshot:
        with self._lock:
            return budget.snapshot(self._slots, self._job_rows())

    def note_pressure(self) -> str:
        return budget.pressure()

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    # ── Monitor ─────────────────────────────────────────────────
    def start_monitor(self) -> None:
        if self._monitor and self._monitor.is_alive():
            return
        self._monitor_stop.clear()
        self._monitor = threading.Thread(target=self._monitor_loop, name="memory-monitor", daemon=True)
        self._monitor.start()
        logger.info("Memory monitor started (interval %.1fs)", config.MEMORY_MONITOR_INTERVAL_SEC)

    def stop_monitor(self) -> None:
        self._monitor_stop.set()

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.wait(config.MEMORY_MONITOR_INTERVAL_SEC):
            try:
                self.evict_idle(force=False)
                pressure = budget.pressure()
                if pressure in ("warning", "critical"):
                    self.evict_idle(force=True)
                    self._notify(
                        "memory.pressure",
                        {"pressure": pressure, "snapshot": self.snapshot().as_dict()},
                    )
            except Exception as exc:
                logger.warning("Memory monitor failed: %s", exc)

    @staticmethod
    def _notify(event: str, payload: dict) -> None:
        try:
            import asyncio

            import job_utils
            from events import event_manager

            loop = job_utils.global_loop
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(event_manager.broadcast(event, payload), loop)
        except Exception:
            pass

    @property
    def active(self) -> set[str]:
        return self._active_slots.copy()


# Singleton
memory_manager = MemoryManager()
