"""Wire adapters for the configured LLM providers.

The runtime never talks to a provider directly; every request goes through one
of these adapters so endpoint validation, credential placement, timeout, and
error taxonomy stay in one place. Both a sync (`httpx.Client`) and an async
(`httpx.AsyncClient`) path exist: GPU tasks run in worker threads, while the
script-parse route must not block the event loop.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import httpx

logger = logging.getLogger("llm.providers")


class ProviderError(RuntimeError):
    """A provider call failed with an actionable, non-secret reason."""

    def __init__(self, provider: str, fault: str, message: str):
        self.provider = provider
        self.fault = fault
        self.message = message
        super().__init__(f"{provider}: {message}")


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class Completion:
    text: str
    provider: str
    model: str
    latency_ms: int
    usage: dict[str, int] = field(default_factory=dict)


# ── payload builders ────────────────────────────────────────────────


def _ollama_payload(
    model: str,
    messages: Sequence[ChatMessage],
    options: dict,
    keep_alive: str | None,
    json_mode: bool = False,
) -> dict:
    system = "\n".join(m.content for m in messages if m.role == "system").strip()
    prompt = "\n".join(m.content for m in messages if m.role != "system").strip()
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    if system:
        payload["system"] = system
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    if json_mode:
        # Ollama's native grammar constraint; other wires rely on the prompt.
        payload["format"] = "json"
    return payload


def _openai_payload(model: str, messages: Sequence[ChatMessage], options: dict) -> dict:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "stream": False,
    }
    payload.update(options)
    return payload


def _anthropic_payload(model: str, messages: Sequence[ChatMessage], options: dict) -> dict:
    system = "\n".join(m.content for m in messages if m.role == "system").strip()
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": options.pop("max_tokens", 1024),
        "messages": [
            {"role": m.role if m.role in {"user", "assistant"} else "user", "content": m.content}
            for m in messages
            if m.role != "system"
        ],
    }
    if system:
        payload["system"] = system
    payload.update(options)
    return payload


def _openai_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _anthropic_headers(api_key: str | None) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _fault_for_status(status: int) -> str:
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "not_found"
    if status == 429:
        return "rate_limited"
    if 400 <= status < 500:
        return "bad_request"
    return "server_error"


def _error_from_response(provider: str, response: httpx.Response) -> ProviderError:
    detail = ""
    try:
        body = response.json()
        if isinstance(body, dict):
            detail = str(body.get("error") or body.get("message") or body.get("detail") or "")
    except Exception:
        detail = response.text[:200]
    fault = _fault_for_status(response.status_code)
    if response.status_code == 404 and provider == "ollama":
        fault = "model_missing"
    return ProviderError(provider, fault, f"HTTP {response.status_code}: {detail or 'no detail'}")


def _wrap_httpx(provider: str, exc: Exception) -> ProviderError:
    if isinstance(exc, httpx.TimeoutException):
        return ProviderError(provider, "timeout", "request timed out")
    if isinstance(exc, httpx.ConnectError):
        return ProviderError(provider, "unreachable", "cannot reach the endpoint")
    return ProviderError(provider, "transport", str(exc)[:200])


# ── completion ──────────────────────────────────────────────────────


def _parse_completion(provider: str, wire: str, data: dict) -> tuple[str, dict[str, int]]:
    usage: dict[str, int] = {}
    if wire == "ollama_native":
        text = str(data.get("response") or "")
        for src, dst in (("prompt_eval_count", "prompt_tokens"), ("eval_count", "completion_tokens")):
            if isinstance(data.get(src), int):
                usage[dst] = data[src]
        return text, usage
    if wire == "anthropic_messages":
        blocks = data.get("content") or []
        text = "".join(str(b.get("text", "")) for b in blocks if isinstance(b, dict))
        raw_usage = data.get("usage") or {}
        for src, dst in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
            if isinstance(raw_usage.get(src), int):
                usage[dst] = raw_usage[src]
        return text, usage
    choices = data.get("choices") or []
    if not choices:
        return "", usage
    message = choices[0].get("message") or {}
    text = message.get("content")
    if isinstance(text, list):
        text = "".join(str(part.get("text", "")) for part in text if isinstance(part, dict))
    raw_usage = data.get("usage") or {}
    for src, dst in (("prompt_tokens", "prompt_tokens"), ("completion_tokens", "completion_tokens")):
        if isinstance(raw_usage.get(src), int):
            usage[dst] = raw_usage[src]
    return str(text or ""), usage


def _request_spec(
    wire: str,
    provider: str,
    api_key: str | None,
    model: str,
    messages,
    options,
    keep_alive,
    json_mode: bool = False,
):
    if wire == "ollama_native":
        return (
            "POST",
            "/api/generate",
            _ollama_headers(),
            _ollama_payload(model, messages, options, keep_alive, json_mode),
        )
    if wire == "anthropic_messages":
        return "POST", "/v1/messages", _anthropic_headers(api_key), _anthropic_payload(model, messages, options)
    return "POST", "/chat/completions", _openai_headers(api_key), _openai_payload(model, messages, options)


def _ollama_headers() -> dict[str, str]:
    return {"Content-Type": "application/json"}


def complete(
    *,
    provider: str,
    wire: str,
    base_url: str,
    api_key: str | None,
    model: str,
    messages: Sequence[ChatMessage],
    options: dict,
    timeout_sec: int,
    keep_alive: str | None = None,
    json_mode: bool = False,
) -> Completion:
    """Synchronous completion for worker threads."""
    if wire == "gemma_native":
        raise ProviderError(provider, "in_process", "Gemma requires the loaded LTX text encoder")
    method, path, headers, payload = _request_spec(
        wire, provider, api_key, model, messages, dict(options), keep_alive, json_mode
    )
    started = time.monotonic()
    try:
        with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_sec) as client:
            response = client.request(method, path, headers=headers, json=payload)
    except Exception as exc:  # httpx transport errors
        raise _wrap_httpx(provider, exc) from exc
    if response.status_code >= 400:
        raise _error_from_response(provider, response)
    try:
        data = response.json()
    except Exception as exc:
        raise ProviderError(provider, "bad_response", "endpoint returned non-JSON") from exc
    text, usage = _parse_completion(provider, wire, data)
    return Completion(
        text=text,
        provider=provider,
        model=model,
        latency_ms=int((time.monotonic() - started) * 1000),
        usage=usage,
    )


async def acomplete(
    *,
    provider: str,
    wire: str,
    base_url: str,
    api_key: str | None,
    model: str,
    messages: Sequence[ChatMessage],
    options: dict,
    timeout_sec: int,
    keep_alive: str | None = None,
    json_mode: bool = False,
) -> Completion:
    """Async completion for request handlers."""
    if wire == "gemma_native":
        raise ProviderError(provider, "in_process", "Gemma requires the loaded LTX text encoder")
    method, path, headers, payload = _request_spec(
        wire, provider, api_key, model, messages, dict(options), keep_alive, json_mode
    )
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_sec) as client:
            response = await client.request(method, path, headers=headers, json=payload)
    except Exception as exc:
        raise _wrap_httpx(provider, exc) from exc
    if response.status_code >= 400:
        raise _error_from_response(provider, response)
    try:
        data = response.json()
    except Exception as exc:
        raise ProviderError(provider, "bad_response", "endpoint returned non-JSON") from exc
    text, usage = _parse_completion(provider, wire, data)
    return Completion(
        text=text,
        provider=provider,
        model=model,
        latency_ms=int((time.monotonic() - started) * 1000),
        usage=usage,
    )


# ── model catalogs ──────────────────────────────────────────────────


def _ollama_models(base_url: str, timeout_sec: int) -> list[dict]:
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_sec) as client:
        response = client.get("/api/tags")
    if response.status_code >= 400:
        raise _error_from_response("ollama", response)
    rows = response.json().get("models") or []
    models: list[dict] = []
    for row in rows:
        details = row.get("details") or {}
        families = details.get("families") or []
        family = str(details.get("family") or "")
        blob = f"{row.get('name', '')} {family} {' '.join(families)}".lower()
        models.append({
            "id": row.get("name") or row.get("model"),
            "name": row.get("name") or row.get("model"),
            "size": row.get("size"),
            "modified_at": row.get("modified_at"),
            "parameter_size": details.get("parameter_size"),
            "quantization": details.get("quantization_level"),
            "family": family or None,
            "vision": any(token in blob for token in ("vision", "llava", "mllama", "clip", "qwen2.5-vl", "gemma3")),
            "context_length": None,
        })
    return models


def _openai_models(base_url: str, api_key: str | None, timeout_sec: int) -> list[dict]:
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_sec) as client:
        response = client.get("/models", headers=_openai_headers(api_key))
    if response.status_code >= 400:
        raise _error_from_response("openai_compatible", response)
    rows = response.json().get("data") or []
    return [
        {
            "id": row.get("id"),
            "name": row.get("id"),
            "size": None,
            "modified_at": None,
            "parameter_size": None,
            "quantization": None,
            "family": None,
            "vision": any(token in str(row.get("id", "")).lower() for token in ("vl", "vision")),
            "context_length": row.get("context_length"),
        }
        for row in rows
        if row.get("id")
    ]


def _anthropic_models(base_url: str, api_key: str | None, timeout_sec: int) -> list[dict]:
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_sec) as client:
        response = client.get("/v1/models", headers=_anthropic_headers(api_key))
    if response.status_code >= 400:
        raise _error_from_response("anthropic", response)
    rows = response.json().get("data") or []
    return [
        {
            "id": row.get("id"),
            "name": row.get("display_name") or row.get("id"),
            "size": None,
            "modified_at": row.get("created_at"),
            "parameter_size": None,
            "quantization": None,
            "family": None,
            "vision": True,
            "context_length": None,
        }
        for row in rows
        if row.get("id")
    ]


def list_models(
    *,
    provider: str,
    wire: str,
    base_url: str,
    api_key: str | None,
    timeout_sec: int,
) -> list[dict]:
    if wire == "gemma_native":
        return [{
            "id": "ltx-gemma3",
            "name": "LTX Gemma 3 text encoder",
            "size": None,
            "modified_at": None,
            "parameter_size": None,
            "quantization": None,
            "family": "gemma",
            "vision": True,
            "context_length": None,
        }]
    try:
        if wire == "ollama_native":
            return _ollama_models(base_url, timeout_sec)
        if wire == "anthropic_messages":
            return _anthropic_models(base_url, api_key, timeout_sec)
        return _openai_models(base_url, api_key, timeout_sec)
    except ProviderError:
        raise
    except Exception as exc:
        raise _wrap_httpx(provider, exc) from exc


# ── Gemma in-process adapter ────────────────────────────────────────


def complete_gemma(
    text_encoder,
    messages: Sequence[ChatMessage],
    *,
    max_new_tokens: int,
    seed: int,
    image_path: str | None = None,
    has_input_image: bool = False,
) -> str:
    """Run the loaded LTX Gemma text encoder over a chat transcript."""
    if text_encoder is None:
        raise ProviderError("gemma", "in_process", "LTX text encoder is not loaded")
    system = "\n".join(m.content for m in messages if m.role == "system").strip()
    prompt = "\n".join(m.content for m in messages if m.role != "system").strip()
    try:
        if image_path and has_input_image:
            import torch
            from ltx_pipelines.utils.media_io import decode_image, resize_aspect_ratio_preserving

            encoded = decode_image(image_path=image_path)
            encoded = torch.tensor(encoded)
            encoded = resize_aspect_ratio_preserving(encoded, 896).to(torch.uint8)
            return text_encoder.enhance_i2v(prompt, encoded, seed=seed, system_prompt=system)
        return text_encoder.enhance_t2v(prompt, seed=seed, system_prompt=system)
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError("gemma", "in_process", str(exc)[:200]) from exc


def extract_json_object(text: str) -> dict | None:
    """Best-effort JSON object extraction from a model reply."""
    import re

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
