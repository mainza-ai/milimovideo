"""Local loopback discovery for inference servers.

Milimo must be able to find an Ollama, LM Studio, vLLM, llama.cpp, or
OpenAI-compatible server the user already runs, without asking them to type a
URL. Probes are loopback-only, short-timeout, and parallel.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

logger = logging.getLogger("llm.discovery")

PROBE_TIMEOUT = 1.5

# (port, provider type, display name, wire)
KNOWN_SERVERS = [
    (11434, "ollama", "Ollama", "ollama_native"),
    (1234, "lmstudio", "LM Studio", "openai_chat"),
    (8081, "openai_compatible", "llama.cpp server", "openai_chat"),
    (8080, "openai_compatible", "OpenAI-compatible (8080)", "openai_chat"),
    (8899, "openai_compatible", "OpenAI-compatible (8899)", "openai_chat"),
    (8000, "openai_compatible", "vLLM / OpenAI-compatible (8000)", "openai_chat"),
    (8787, "openai_compatible", "oMLX (8787)", "openai_chat"),
    (47831, "openai_compatible", "Strata (47831)", "openai_chat"),
]


def _models_for(wire: str, base_url: str) -> list[str]:
    try:
        with httpx.Client(timeout=PROBE_TIMEOUT) as client:
            if wire == "ollama_native":
                response = client.get(f"{base_url}/api/tags")
                if response.status_code >= 400:
                    return []
                rows = response.json().get("models") or []
                return [str(row.get("name") or row.get("model")) for row in rows if row.get("name") or row.get("model")]
            response = client.get(f"{base_url}/v1/models")
            if response.status_code >= 400:
                return []
            rows = response.json().get("data") or []
            return [str(row["id"]) for row in rows if row.get("id")]
    except Exception:
        return []


def _probe(port: int, provider: str, name: str, wire: str) -> dict[str, Any] | None:
    base_url = f"http://127.0.0.1:{port}"
    root = base_url if wire != "openai_chat" else f"{base_url}/v1"
    try:
        with httpx.Client(timeout=PROBE_TIMEOUT) as client:
            # A 401/403 still proves a server is listening; only transport
            # failures and 404 mean "not this kind of engine".
            if wire == "ollama_native":
                response = client.get(f"{base_url}/api/tags")
            else:
                response = client.get(f"{base_url}/v1/models")
            if response.status_code >= 500:
                return None
            if response.status_code == 404:
                return None
    except Exception:
        return None

    models = _models_for(wire, base_url)
    return {
        "provider": provider,
        "name": name,
        "wire": wire,
        "base_url": root,
        "port": port,
        "models": models,
        "source": "loopback",
    }


def discover_local_servers() -> list[dict[str, Any]]:
    """Scan the known loopback ports in parallel and return live engines."""
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(KNOWN_SERVERS)) as pool:
        futures = [
            pool.submit(_probe, port, provider, name, wire)
            for port, provider, name, wire in KNOWN_SERVERS
        ]
        for future in futures:
            try:
                found = future.result()
            except Exception as exc:  # pragma: no cover - probe is defensive
                logger.debug("Discovery probe failed: %s", exc)
                continue
            if found:
                results.append(found)
    results.sort(key=lambda item: (item["provider"], item["port"]))
    return results
