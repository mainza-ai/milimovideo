"""Per-job memory sampler: peak by stage, persisted with the job."""

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
        self._swap_peak = 0.0
        self._samples = 0

    def _sample(self) -> float:
        try:
            import torch

            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                return float(torch.mps.driver_allocated_memory()) / 1e9
        except Exception:
            pass
        try:
            return psutil.Process().memory_info().rss / 1e9
        except Exception:
            return 0.0

    def start(self, job_id: str) -> None:
        with self._lock:
            self._job_id = job_id
            self._stage = "starting"
            self._started_at = time.time()
            self._peak = 0.0
            self._peak_stage = "starting"
            self._stages = {}
            self._swap_peak = 0.0
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
                value = self._sample()
                try:
                    swap = psutil.swap_memory().percent
                except Exception:
                    swap = 0.0
                with self._lock:
                    self._samples += 1
                    self._swap_peak = max(self._swap_peak, swap)
                    self._stages[stage] = max(self._stages.get(stage, 0.0), value)
                    if value > self._peak:
                        self._peak = value
                        self._peak_stage = stage
            self._stop.wait(1.0)

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
                "swap_peak_percent": round(self._swap_peak, 1),
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
                "swap_peak_percent": round(self._swap_peak, 1),
                "samples": self._samples,
            }


sampler = MemorySampler()
