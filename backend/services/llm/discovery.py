"""Local loopback discovery for inference servers.

Milimo must find the Ollama, LM Studio, oMLX, Strata, vLLM, or llama.cpp server
the user already runs without asking them to type a URL. Probes are
loopback-only, short-timeout, and parallel.

Identification is by response shape, not by "something answered on the port":
a Docker console on :8080 or another API must not be offered as an LLM.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from .instances import instance_id, normalize_base_url

logger = logging.getLogger("llm.discovery")

PROBE_TIMEOUT = 1.5

# (port, provider type, display name, wire)
KNOWN_SERVERS = [
    (11434, "ollama", "Ollama", "ollama_native"),
    (1234, "lmstudio", "LM Studio", "openai_chat"),
    (8787, "openai_compatible", "oMLX", "openai_chat"),
    (47831, "openai_compatible", "Strata", "openai_chat"),
    (8081, "openai_compatible", "llama.cpp server", "openai_chat"),
    (8080, "openai_compatible", "OpenAI-compatible (8080)", "openai_chat"),
    (8899, "openai_compatible", "OpenAI-compatible (8899)", "openai_chat"),
    (8000, "openai_compatible", "vLLM / OpenAI-compatible (8000)", "openai_chat"),
]


def _catalog_shape(wire: str, payload: Any) -> list[str] | None:
    """Return model ids when the payload looks like a real catalog, else None."""
    if not isinstance(payload, dict):
        return None
    rows = payload.get("models") if wire == "ollama_native" else payload.get("data")
    if rows is None and wire != "ollama_native":
        rows = payload.get("models")
    if not isinstance(rows, list):
        return None
    ids: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            value = row.get("name") or row.get("id") or row.get("model")
        else:
            value = row
        if isinstance(value, str) and value.strip():
            ids.append(value.strip())
    return ids


def _probe(port: int, provider: str, name: str, wire: str) -> dict[str, Any] | None:
    root = f"http://127.0.0.1:{port}"
    path = "/api/tags" if wire == "ollama_native" else "/v1/models"
    base_url = root if wire == "ollama_native" else f"{root}/v1"
    try:
        with httpx.Client(timeout=PROBE_TIMEOUT) as client:
            response = client.get(f"{root}{path}")
    except Exception:
        return None

    if response.status_code in (401, 403):
        return {
            "provider": provider,
            "name": name,
            "wire": wire,
            "base_url": base_url,
            "port": port,
            "models": [],
            "source": "loopback",
            "needs_key": True,
        }
    if response.status_code >= 400:
        return None
    try:
        payload = response.json()
    except Exception:
        return None
    models = _catalog_shape(wire, payload)
    if models is None:
        return None
    return {
        "provider": provider,
        "name": name,
        "wire": wire,
        "base_url": base_url,
        "port": port,
        "models": models,
        "source": "loopback",
        "needs_key": False,
    }


def discover_local_servers(existing: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Scan the known loopback ports in parallel and return identified engines.

    `existing` maps a normalized base URL to a configured provider instance id;
    matching servers are marked attached so a rescan is idempotent.
    """
    existing = existing or {}
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(KNOWN_SERVERS)) as pool:
        futures = [
            pool.submit(_probe, port, provider, name, wire)
            for port, provider, name, wire in KNOWN_SERVERS
        ]
        for future in futures:
            try:
                found = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Discovery probe failed: %s", exc)
                continue
            if not found:
                continue
            key = normalize_base_url(found["base_url"])
            attached_id = existing.get(key)
            found["instance_id"] = attached_id or instance_id(found["provider"], found["base_url"])
            found["attached"] = bool(attached_id)
            results.append(found)
    results.sort(key=lambda item: (item["provider"], item["port"]))
    return results
