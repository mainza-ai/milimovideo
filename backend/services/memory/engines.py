"""Adapters that ask local LLM engines what they hold in memory.

Ollama and oMLX report residency and can unload. Strata and generic
OpenAI-compatible servers may only expose a model list; they are budgeted from a
declared override or a conservative estimate and marked unknown.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Iterable, List

import httpx

from .costs import estimate_llm_gb
from .models import EngineResidency

logger = logging.getLogger("memory.engines")

CACHE_TTL_SEC = 4.0
TIMEOUT_SEC = 2.5

_cache_lock = threading.Lock()
_cache: dict[str, Any] = {"stamp": 0.0, "revision": None, "engines": []}


def _client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url.rstrip("/"), timeout=TIMEOUT_SEC)


def _ollama(provider_id: str, provider: Any) -> EngineResidency:
    try:
        with _client(provider.base_url) as client:
            response = client.get("/api/ps")
        if response.status_code >= 400:
            return EngineResidency(
                provider_id=provider_id, provider_type=provider.type, name=provider.name,
                base_url=provider.base_url, error=f"HTTP {response.status_code}", unknown=True,
            )
        rows = response.json().get("models") or []
        models = [str(row.get("name") or row.get("model")) for row in rows]
        resident = sum(
            float(row.get("size_vram") or row.get("size") or 0) for row in rows
        ) / 1e9
        return EngineResidency(
            provider_id=provider_id, provider_type=provider.type, name=provider.name,
            base_url=provider.base_url, resident_gb=resident, models=models,
            can_unload=True, source="measured",
        )
    except Exception as exc:
        return EngineResidency(
            provider_id=provider_id, provider_type=provider.type, name=provider.name,
            base_url=provider.base_url, error=str(exc)[:160], unknown=True,
        )


def _openai_status(provider_id: str, provider: Any) -> EngineResidency | None:
    """oMLX exposes per-model residency at /v1/models/status."""
    try:
        with _client(provider.base_url) as client:
            response = client.get("/v1/models/status")
        if response.status_code >= 400:
            return None
        data = response.json()
    except Exception:
        return None
    # Only oMLX-shaped payloads are residency reports; a bare OpenAI model list
    # belongs to the generic adapter.
    if not isinstance(data, dict) or ("models" not in data and "current_model_memory" not in data):
        return None
    rows = data.get("models") or []
    loaded = [row for row in rows if row.get("loaded") or row.get("is_loading")]
    resident = float(data.get("current_model_memory") or 0) / 1e9
    if resident <= 0:
        resident = sum(
            float(row.get("actual_size") or row.get("resident_estimated_size") or 0)
            for row in loaded
        ) / 1e9
    return EngineResidency(
        provider_id=provider_id, provider_type=provider.type, name=provider.name,
        base_url=provider.base_url, resident_gb=resident,
        models=[str(row.get("id")) for row in loaded],
        can_unload=True, source="measured",
    )


def _generic(provider_id: str, provider: Any) -> EngineResidency:
    models: list[str] = []
    error = None
    try:
        with _client(provider.base_url) as client:
            response = client.get("/v1/models")
        if response.status_code < 400:
            rows = response.json().get("data") or []
            models = [str(row.get("id")) for row in rows if row.get("id")]
        else:
            error = f"HTTP {response.status_code}"
    except Exception as exc:
        error = str(exc)[:160]

    declared = float(getattr(provider, "memory_gb", 0.0) or 0.0)
    if declared > 0:
        return EngineResidency(
            provider_id=provider_id, provider_type=provider.type, name=provider.name,
            base_url=provider.base_url, resident_gb=declared, models=models,
            can_unload=False, source="declared", error=error,
        )
    if models:
        # A loaded engine is likely serving one model; budget the largest estimate
        # and mark it unknown so the UI asks for an override.
        largest = max(estimate_llm_gb(model) for model in models)
        return EngineResidency(
            provider_id=provider_id, provider_type=provider.type, name=provider.name,
            base_url=provider.base_url, resident_gb=largest, models=models,
            can_unload=False, source="estimated", error=error, unknown=True,
        )
    return EngineResidency(
        provider_id=provider_id, provider_type=provider.type, name=provider.name,
        base_url=provider.base_url, resident_gb=0.0, models=[],
        can_unload=False, source="unknown", error=error, unknown=bool(error),
    )


def collect(settings: Any | None = None, force: bool = False) -> List[EngineResidency]:
    """Return residency for every enabled local text engine, cached briefly."""
    from services.llm.store import llm_settings

    settings = settings or llm_settings.get()
    with _cache_lock:
        if (
            not force
            and _cache["revision"] == settings.revision
            and time.monotonic() - _cache["stamp"] < CACHE_TTL_SEC
        ):
            return list(_cache["engines"])

    engines: list[EngineResidency] = []
    for provider_id, provider in settings.providers.items():
        if not provider.enabled or not provider.is_local or provider.type == "gemma":
            continue
        if provider.type == "ollama":
            engines.append(_ollama(provider_id, provider))
        elif provider.type in ("openai_compatible", "lmstudio"):
            engines.append(_openai_status(provider_id, provider) or _generic(provider_id, provider))
        else:
            # Local cloud-like slots (none today) stay out of the external budget.
            continue

    with _cache_lock:
        _cache.update({"stamp": time.monotonic(), "revision": settings.revision, "engines": engines})
    return engines


def total_gb(engines: Iterable[EngineResidency]) -> float:
    return round(sum(max(0.0, engine.resident_gb) for engine in engines), 2)


def clear_cache() -> None:
    with _cache_lock:
        _cache.update({"stamp": 0.0, "revision": None, "engines": []})


def unload(settings: Any | None = None, provider_id: str | None = None) -> dict[str, Any]:
    """Best-effort unload on engines that support it. Never assumes success."""
    from services.llm.store import llm_settings

    settings = settings or llm_settings.get()
    targets = [
        (pid, provider) for pid, provider in settings.providers.items()
        if provider.enabled and provider.is_local and provider.type != "gemma"
        and (provider_id is None or pid == provider_id)
    ]
    result: dict[str, Any] = {"unloaded": [], "skipped": [], "errors": []}
    for pid, provider in targets:
        try:
            if provider.type == "ollama":
                with _client(provider.base_url) as client:
                    rows = client.get("/api/ps").json().get("models") or []
                    for row in rows:
                        model = row.get("name") or row.get("model")
                        client.post("/api/generate", json={"model": model, "stream": False, "keep_alive": 0})
                        result["unloaded"].append(f"{pid}:{model}")
            elif provider.type in ("openai_compatible", "lmstudio"):
                status = _openai_status(pid, provider)
                if status is None or not status.models:
                    result["skipped"].append(pid)
                    continue
                with _client(provider.base_url) as client:
                    for model in status.models:
                        response = client.post(f"/v1/models/{model}/unload")
                        if response.status_code < 400:
                            result["unloaded"].append(f"{pid}:{model}")
                        else:
                            result["errors"].append(f"{pid}:{model}: HTTP {response.status_code}")
            else:
                result["skipped"].append(pid)
        except Exception as exc:
            result["errors"].append(f"{pid}: {str(exc)[:160]}")
    clear_cache()
    return result


def resident_gb(settings: Any, provider_id: str) -> float:
    for engine in collect(settings):
        if engine.provider_id == provider_id:
            return engine.resident_gb
    return 0.0
