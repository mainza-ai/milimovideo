"""Provider instance identity and endpoint helpers.

A provider *type* is a wire family (`ollama`, `openai_compatible`, ...). A
provider *instance* is one reachable endpoint of that type. A local machine can
run oMLX on :8787 and Strata on :47831 at the same time; both are
`openai_compatible` instances with different ids.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

INSTANCE_ID_SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.@:-")


def normalize_base_url(url: str) -> str:
    """Canonical form used to match an engine across discovery and settings."""
    raw = (url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "http").lower()
    host = (parsed.hostname or "").lower()
    port = parsed.port
    if port is None:
        port = 443 if scheme == "https" else 80
    path = (parsed.path or "").rstrip("/")
    return f"{scheme}://{host}:{port}{path}"


def host_port(url: str) -> str:
    parsed = urlparse((url or "").strip())
    host = (parsed.hostname or "local").lower()
    port = parsed.port
    return f"{host}:{port}" if port else host


CANONICAL_URLS = {
    "ollama": {"http://127.0.0.1:11434", "http://localhost:11434"},
    "lmstudio": {"http://127.0.0.1:1234/v1", "http://localhost:1234/v1"},
    "openai": {"https://api.openai.com/v1"},
    "anthropic": {"https://api.anthropic.com"},
}


def instance_id(provider_type: str, base_url: str) -> str:
    """Stable id for one endpoint.

    The canonical endpoint for a type keeps the friendly id (`ollama`,
    `lmstudio`); additional endpoints of the same type get `type@host:port`.
    """
    if provider_type == "gemma" or not (base_url or "").strip():
        return provider_type
    normalized = normalize_base_url(base_url)
    canonical = {normalize_base_url(url) for url in CANONICAL_URLS.get(provider_type, set())}
    if normalized in canonical:
        return provider_type
    return f"{provider_type}@{host_port(base_url)}"


def is_local_host(host: str) -> bool:
    h = (host or "").strip().lower().rstrip(".")
    if h in {"localhost", "host.docker.internal"} or h.endswith(".localhost") or h.endswith(".local"):
        return True
    try:
        addr = ipaddress.ip_address(h)
        return addr.is_loopback or addr.is_private or addr.is_link_local
    except ValueError:
        return False


def valid_instance_id(value: str) -> bool:
    return bool(value) and all(char in INSTANCE_ID_SAFE for char in value)
