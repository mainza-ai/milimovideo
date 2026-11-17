"""Typed memory-budget, residency, and admission models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CostBreakdown:
    kind: str
    base_gb: float
    workspace_gb: float
    total_gb: float
    upscale: bool = False
    frames: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "base_gb": round(self.base_gb, 2),
            "workspace_gb": round(self.workspace_gb, 2),
            "total_gb": round(self.total_gb, 2),
            "upscale": self.upscale,
            "frames": self.frames,
            "width": self.width,
            "height": self.height,
            "notes": self.notes,
        }


@dataclass
class EngineResidency:
    provider_id: str
    provider_type: str
    name: str
    base_url: str
    resident_gb: float = 0.0
    models: List[str] = field(default_factory=list)
    can_unload: bool = False
    source: str = "unknown"  # measured | declared | estimated | unknown
    error: Optional[str] = None
    unknown: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_type": self.provider_type,
            "name": self.name,
            "base_url": self.base_url,
            "resident_gb": round(self.resident_gb, 2),
            "models": list(self.models),
            "can_unload": self.can_unload,
            "source": self.source,
            "error": self.error,
            "unknown": self.unknown,
        }


@dataclass
class SlotState:
    slot: str
    model_id: str
    cost_gb: float
    leases: int = 0
    acquired_at: float = 0.0
    last_used: float = 0.0
    keep_warm: bool = False
    job_ids: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "slot": self.slot,
            "model_id": self.model_id,
            "cost_gb": round(self.cost_gb, 2),
            "leases": self.leases,
            "acquired_at": self.acquired_at,
            "last_used": self.last_used,
            "keep_warm": self.keep_warm,
            "job_ids": list(self.job_ids),
        }


@dataclass
class MemorySnapshot:
    total_gb: float
    reserve_gb: float
    ceiling_gb: float
    in_process_gb: float
    resident_gb: float
    external_gb: float
    sam_gb: float
    used_gb: float
    available_gb: float
    swap_percent: float
    pressure: str
    enforce: bool
    external_engines: List[EngineResidency] = field(default_factory=list)
    slots: Dict[str, SlotState] = field(default_factory=dict)
    active_jobs: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = 0.0
    notes: List[str] = field(default_factory=list)
    # Physical free RAM and free swap are separate signals: swap can stay
    # allocated long after a heavy job without indicating current pressure.
    physical_available_gb: float = 0.0
    swap_free_gb: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_gb": round(self.total_gb, 2),
            "reserve_gb": round(self.reserve_gb, 2),
            "ceiling_gb": round(self.ceiling_gb, 2),
            "in_process_gb": round(self.in_process_gb, 2),
            "resident_gb": round(self.resident_gb, 2),
            "external_gb": round(self.external_gb, 2),
            "sam_gb": round(self.sam_gb, 2),
            "used_gb": round(self.used_gb, 2),
            "available_gb": round(self.available_gb, 2),
            "swap_percent": round(self.swap_percent, 1),
            "pressure": self.pressure,
            "enforce": self.enforce,
            "external_engines": [engine.as_dict() for engine in self.external_engines],
            "slots": {slot: state.as_dict() for slot, state in self.slots.items()},
            "active_jobs": list(self.active_jobs),
            "timestamp": self.timestamp,
            "notes": list(self.notes),
            "physical_available_gb": round(self.physical_available_gb, 2),
            "swap_free_gb": round(self.swap_free_gb, 2),
        }


class MemoryDeferred(RuntimeError):
    """A job was refused for a transient reason and may be retried later."""


@dataclass
class AdmissionDecision:
    ok: bool
    kind: str
    reason: Optional[str]
    estimate: Optional[CostBreakdown]
    snapshot: MemorySnapshot
    reclaimable_gb: float = 0.0
    will_unload: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # True when waiting for other work to finish can plausibly free the bytes.
    retryable: bool = False
    # A smaller preset that would fit, when one exists.
    downgrade: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "kind": self.kind,
            "reason": self.reason,
            "estimate": self.estimate.as_dict() if self.estimate else None,
            "snapshot": self.snapshot.as_dict(),
            "reclaimable_gb": round(self.reclaimable_gb, 2),
            "will_unload": list(self.will_unload),
            "suggestions": list(self.suggestions),
            "warnings": list(self.warnings),
            "retryable": self.retryable,
            "downgrade": self.downgrade,
        }
