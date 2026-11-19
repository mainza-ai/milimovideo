"""Per-job memory sampler: peak by stage, system headroom, persisted with the job."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import psutil

logger = logging.getLogger("memory.sampler")


class MemorySampler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._job_id: str | None = None
        self._stage = "starting"
        self._started_at = 0.0
        self._peak = 0.0
        self._peak_stage = "starting"
        self._stages: dict[str, float] = {}
        self._stage_min: dict[str, float] = {}
        self._swap_peak = 0.0
        self._physical_min = 0.0
        self._current = 0.0
        self._cache = 0.0
        self._samples = 0

    def _sample(self) -> dict[str, float]:
        driver = 0.0
        current = 0.0
        try:
            import torch

            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                driver = float(torch.mps.driver_allocated_memory()) / 1e9
                current = float(torch.mps.current_allocated_memory()) / 1e9
            elif torch.cuda.is_available():
                driver = float(torch.cuda.memory_reserved(0)) / 1e9
                current = float(torch.cuda.memory_allocated(0)) / 1e9
        except Exception:
            pass
        if driver <= 0.0:
            try:
                driver = psutil.Process().memory_info().rss / 1e9
                current = driver
            except Exception:
                driver = 0.0
        try:
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            physical = vm.available / 1e9
            swap_percent = swap.percent
        except Exception:
            physical = 0.0
            swap_percent = 0.0
        return {
            "used": driver,
            "current": current,
            "cache": max(0.0, driver - current),
            "physical": physical,
            "swap_percent": swap_percent,
        }

    def start(self, job_id: str) -> None:
        with self._lock:
            self._job_id = job_id
            self._stage = "starting"
            self._started_at = time.time()
            self._peak = 0.0
            self._peak_stage = "starting"
            self._stages = {}
            self._stage_min = {}
            self._swap_peak = 0.0
            self._physical_min = 0.0
            self._current = 0.0
            self._cache = 0.0
            self._samples = 0
            self._stop.clear()
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="memory-sampler", daemon=True)
        self._thread.start()

    def mark(self, stage: str) -> None:
        with self._lock:
            self._stage = stage or self._stage

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                job_id = self._job_id
                stage = self._stage
            if job_id:
                sample = self._sample()
                value = sample["used"]
                with self._lock:
                    self._samples += 1
                    self._current = sample["current"]
                    self._cache = sample["cache"]
                    self._swap_peak = max(self._swap_peak, sample["swap_percent"])
                    if sample["physical"] > 0:
                        self._physical_min = (
                            sample["physical"] if self._physical_min <= 0
                            else min(self._physical_min, sample["physical"])
                        )
                    self._stages[stage] = max(self._stages.get(stage, 0.0), value)
                    previous_min = self._stage_min.get(stage)
                    self._stage_min[stage] = (
                        value if previous_min is None else min(previous_min, value)
                    )
                    if value > self._peak:
                        self._peak = value
                        self._peak_stage = stage
            self._stop.wait(0.5)

    def stop(self) -> dict[str, Any] | None:
        with self._lock:
            job_id = self._job_id
            if not job_id:
                return None
            trace = {
                "job_id": job_id,
                "started_at": self._started_at,
                "ended_at": time.time(),
                "peak_gb": round(self._peak, 2),
                "peak_stage": self._peak_stage,
                "stages": {stage: round(value, 2) for stage, value in self._stages.items()},
                "stage_min": {stage: round(value, 2) for stage, value in self._stage_min.items()},
                "swap_peak_percent": round(self._swap_peak, 1),
                "physical_min_gb": round(self._physical_min, 2),
                "allocator_cache_peak_gb": round(self._cache, 2),
                "samples": self._samples,
            }
            self._job_id = None
            self._stop.set()
            return trace

    def current(self) -> dict[str, Any]:
        with self._lock:
            return {
                "job_id": self._job_id,
                "stage": self._stage,
                "peak_gb": round(self._peak, 2),
                "peak_stage": self._peak_stage,
                "current_gb": round(self._current, 2),
                "cache_gb": round(self._cache, 2),
                "physical_min_gb": round(self._physical_min, 2),
                "swap_peak_percent": round(self._swap_peak, 1),
                "samples": self._samples,
            }


sampler = MemorySampler()
