"""Wire adapters for the configured LLM providers.

Every request goes through one of these adapters so endpoint validation,
credential placement, timeout, capability parsing, and the error taxonomy stay
in one place. Sync (`httpx.Client`) serves GPU worker threads; async
(`httpx.AsyncClient`) serves request handlers and streaming.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Sequence

import httpx

from .capabilities import classify_model, extract_context

logger = logging.getLogger("llm.providers")


class ProviderCancelled(Exception):
    """The caller cancelled an in-flight HTTP request."""


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


# ── images ──────────────────────────────────────────────────────────


def encode_image(path: str) -> tuple[str, str]:
    """Return (mime, base64) for a local image. Raises ProviderError on failure."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise ProviderError("image", "input", f"cannot read {path}: {exc}") from exc
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return mime, base64.b64encode(raw).decode("ascii")


def _image_data_url(mime: str, b64: str) -> str:
    return f"data:{mime};base64,{b64}"


# ── payload builders ────────────────────────────────────────────────


def _ollama_payload(
    model: str,
    messages: Sequence[ChatMessage],
    options: dict,
    keep_alive: str | None,
    json_mode: bool = False,
    image: tuple[str, str] | None = None,
    stream: bool = False,
) -> dict:
    system = "\n".join(m.content for m in messages if m.role == "system").strip()
    prompt = "\n".join(m.content for m in messages if m.role != "system").strip()
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": options,
    }
    if system:
        payload["system"] = system
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    if image is not None:
        payload["images"] = [image[1]]
    if json_mode:
        # Ollama's native grammar constraint; other wires rely on the prompt.
        payload["format"] = "json"
    return payload


def _openai_messages(messages: Sequence[ChatMessage], image: tuple[str, str] | None) -> list[dict]:
    rows: list[dict] = []
    for index, message in enumerate(messages):
        is_last_user = message.role != "system" and image is not None and index == len(messages) - 1
        if is_last_user:
            rows.append({
                "role": message.role,
                "content": [
                    {"type": "text", "text": message.content},
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(image[0], image[1])},
                    },
                ],
            })
        else:
            rows.append({"role": message.role, "content": message.content})
    return rows


def _openai_payload(
    model: str,
    messages: Sequence[ChatMessage],
    options: dict,
    image: tuple[str, str] | None = None,
    stream: bool = False,
) -> dict:
    payload: dict[str, Any] = {
        "model": model,
        "messages": _openai_messages(messages, image),
        "stream": stream,
    }
    payload.update(options)
    return payload


def _anthropic_payload(
    model: str,
    messages: Sequence[ChatMessage],
    options: dict,
    image: tuple[str, str] | None = None,
    stream: bool = False,
) -> dict:
    system = "\n".join(m.content for m in messages if m.role == "system").strip()
    rows: list[dict] = []
    chat = [m for m in messages if m.role != "system"]
    for index, message in enumerate(chat):
        is_last = image is not None and index == len(chat) - 1
        if is_last:
            content: Any = [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": image[0], "data": image[1]},
                },
                {"type": "text", "text": message.content},
            ]
        else:
            content = message.content
        rows.append({"role": message.role if message.role in {"user", "assistant"} else "user", "content": content})
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": options.pop("max_tokens", 1024),
        "messages": rows,
    }
    if system:
        payload["system"] = system
    payload.update(options)
    if stream:
        payload["stream"] = True
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


def _ollama_headers() -> dict[str, str]:
    return {"Content-Type": "application/json"}


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
    image: tuple[str, str] | None = None,
    stream: bool = False,
):
    if wire == "ollama_native":
        return (
            "POST",
            "/api/generate",
            _ollama_headers(),
            _ollama_payload(model, messages, options, keep_alive, json_mode, image, stream),
        )
    if wire == "anthropic_messages":
        return (
            "POST",
            "/v1/messages",
            _anthropic_headers(api_key),
            _anthropic_payload(model, messages, dict(options), image, stream),
        )
    return (
        "POST",
        "/chat/completions",
        _openai_headers(api_key),
        _openai_payload(model, messages, dict(options), image, stream),
    )


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
    image_path: str | None = None,
) -> Completion:
    """Synchronous completion for worker threads."""
    if wire == "gemma_native":
        raise ProviderError(provider, "in_process", "Gemma requires the loaded LTX text encoder")
    image = encode_image(image_path) if image_path else None
    method, path, headers, payload = _request_spec(
        wire, provider, api_key, model, messages, dict(options), keep_alive, json_mode, image
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
    image_path: str | None = None,
    cancel_check=None,
) -> Completion:
    """Async completion for request handlers."""
    if wire == "gemma_native":
        raise ProviderError(provider, "in_process", "Gemma requires the loaded LTX text encoder")
    image = encode_image(image_path) if image_path else None
    method, path, headers, payload = _request_spec(
        wire, provider, api_key, model, messages, dict(options), keep_alive, json_mode, image
    )
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_sec) as client:
            if cancel_check is None:
                response = await client.request(method, path, headers=headers, json=payload)
            else:
                request = asyncio.create_task(
                    client.request(method, path, headers=headers, json=payload)
                )
                try:
                    while True:
                        done, _ = await asyncio.wait({request}, timeout=0.1)
                        if done:
                            response = request.result()
                            break
                        if cancel_check():
                            request.cancel()
                            with suppress(asyncio.CancelledError, Exception):
                                await request
                            raise ProviderCancelled(f"{provider}: request cancelled")
                finally:
                    if not request.done():
                        request.cancel()
                        with suppress(asyncio.CancelledError, Exception):
                            await request
    except ProviderCancelled:
        raise
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


# ── streaming ───────────────────────────────────────────────────────


def _sse_delta(wire: str, payload: str) -> str | None:
    """Extract content deltas; reasoning deltas are ignored on purpose."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if wire == "ollama_native":
        return str(data.get("response") or "") or None
    if wire == "anthropic_messages":
        if data.get("type") == "content_block_delta":
            return str((data.get("delta") or {}).get("text") or "") or None
        return None
    choices = data.get("choices") or []
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        text = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        return text or None
    return None


async def astream(
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
    image_path: str | None = None,
) -> AsyncIterator[str]:
    """Yield text deltas. Callers enforce a first-byte timeout and fall back."""
    if wire == "gemma_native":
        raise ProviderError(provider, "in_process", "Gemma cannot stream")
    image = encode_image(image_path) if image_path else None
    method, path, headers, payload = _request_spec(
        wire, provider, api_key, model, messages, dict(options), keep_alive, json_mode, image, stream=True
    )
    try:
        async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_sec) as client:
            async with client.stream(method, path, headers=headers, json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise _error_from_response(provider, response)
                async for line in response.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if wire == "ollama_native":
                        if line.startswith("data:"):
                            line = line[5:].strip()
                        delta = _sse_delta(wire, line)
                        if delta:
                            yield delta
                        try:
                            if json.loads(line).get("done"):
                                return
                        except Exception:
                            pass
                        continue
                    if line.startswith("event:"):
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        return
                    delta = _sse_delta(wire, line)
                    if delta:
                        yield delta
    except ProviderError:
        raise
    except Exception as exc:
        raise _wrap_httpx(provider, exc) from exc


# ── warm / unload ───────────────────────────────────────────────────


def warm(
    *,
    provider: str,
    wire: str,
    base_url: str,
    api_key: str | None,
    model: str,
    timeout_sec: int,
    keep_alive: str | None = None,
) -> dict[str, Any]:
    """Load the model into memory and return the measured load time."""
    if wire == "gemma_native":
        return {"ok": True, "load_ms": 0, "in_process": True}
    started = time.monotonic()
    try:
        with httpx.Client(base_url=base_url.rstrip("/"), timeout=max(timeout_sec, 1)) as client:
            if wire == "ollama_native":
                response = client.post("/api/generate", json={
                    "model": model,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": keep_alive or "5m",
                })
            else:
                path = "/v1/messages" if wire == "anthropic_messages" else "/chat/completions"
                headers = _anthropic_headers(api_key) if wire == "anthropic_messages" else _openai_headers(api_key)
                body: dict[str, Any] = {
                    "model": model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                }
                if wire == "anthropic_messages":
                    headers = {**headers, "Content-Type": "application/json"}
                    body["max_tokens"] = 8
                else:
                    body["max_tokens"] = 8
                response = client.post(path, headers=headers, json=body)
    except Exception as exc:
        raise _wrap_httpx(provider, exc) from exc
    if response.status_code >= 400:
        raise _error_from_response(provider, response)
    return {"ok": True, "load_ms": int((time.monotonic() - started) * 1000)}


def unload(*, provider: str, wire: str, base_url: str, model: str, timeout_sec: int = 5) -> bool:
    """Release a model where the wire supports it (Ollama does)."""
    if wire != "ollama_native" or not model:
        return False
    try:
        with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_sec) as client:
            # Omitting `prompt` is the documented unload form; an empty prompt
            # still runs a generation on some Ollama versions.
            response = client.post("/api/generate", json={
                "model": model, "stream": False, "keep_alive": 0,
            })
        return response.status_code < 400
    except Exception:
        return False


# ── model catalogs ──────────────────────────────────────────────────


def _ollama_show(base_url: str, model: str) -> dict[str, Any]:
    try:
        with httpx.Client(base_url=base_url.rstrip("/"), timeout=2.0) as client:
            response = client.post("/api/show", json={"model": model})
        if response.status_code >= 400:
            return {}
        return response.json()
    except Exception:
        return {}


def _ollama_context(show: dict[str, Any]) -> tuple[int | None, bool]:
    info = show.get("model_info") or {}
    context = None
    for key, value in info.items():
        if key.endswith(".context_length") and isinstance(value, int) and value > 0:
            context = value
            break
    capabilities = [str(item).lower() for item in (show.get("capabilities") or [])]
    vision = "vision" in capabilities or any(
        "vision" in str(key).lower() and bool(value) for key, value in info.items()
    )
    return context, vision


def _ollama_models(base_url: str, timeout_sec: int, deep: bool = False) -> list[dict]:
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
        name = str(row.get("name") or row.get("model") or "")
        flags = classify_model(name, family=family, families=families)
        models.append({
            "id": name,
            "name": name,
            "size": row.get("size"),
            "modified_at": row.get("modified_at"),
            "parameter_size": details.get("parameter_size"),
            "quantization": details.get("quantization_level"),
            "family": family or None,
            "context_length": None,
            **flags,
            "source": "ollama",
        })
    if deep:
        from concurrent.futures import ThreadPoolExecutor

        def enrich(model: dict) -> None:
            show = _ollama_show(base_url, model["id"])
            context, vision = _ollama_context(show)
            if context:
                model["context_length"] = context
            if vision:
                model["vision"] = True

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(enrich, models[:32]))
    return models


def _openai_models(base_url: str, api_key: str | None, timeout_sec: int, deep: bool = False) -> list[dict]:
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_sec) as client:
        response = client.get("/models", headers=_openai_headers(api_key))
    if response.status_code >= 400:
        raise _error_from_response("openai_compatible", response)
    rows = response.json().get("data") or []
    results: list[dict] = []
    for row in rows:
        model_id = row.get("id")
        if not model_id:
            continue
        meta = row.get("meta") or {}
        extra = [str(meta.get("label") or ""), *[str(a) for a in (meta.get("aliases") or [])]]
        declared = bool(row.get("vision")) or "vision" in [str(c).lower() for c in (row.get("capabilities") or [])]
        flags = classify_model(str(model_id), extra=extra, declared_vision=declared)
        results.append({
            "id": model_id,
            "name": str(meta.get("label") or model_id),
            "size": row.get("size"),
            "modified_at": row.get("created") or row.get("created_at"),
            "parameter_size": meta.get("parameter_size") or row.get("parameter_size"),
            "quantization": meta.get("quantization") or row.get("quantization"),
            "family": row.get("owned_by") or meta.get("family") or None,
            "context_length": extract_context(row, meta),
            **flags,
            "source": "openai_compatible",
        })
    return results


def _anthropic_models(base_url: str, api_key: str | None, timeout_sec: int, deep: bool = False) -> list[dict]:
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_sec) as client:
        response = client.get("/v1/models", headers=_anthropic_headers(api_key))
    if response.status_code >= 400:
        raise _error_from_response("anthropic", response)
    rows = response.json().get("data") or []
    results = []
    for row in rows:
        model_id = row.get("id")
        if not model_id:
            continue
        results.append({
            "id": model_id,
            "name": row.get("display_name") or model_id,
            "size": None,
            "modified_at": row.get("created_at"),
            "parameter_size": None,
            "quantization": None,
            "family": "anthropic",
            "context_length": extract_context(row, row.get("meta") or {}),
            "chat": True,
            "vision": True,
            "embedding_only": False,
            "source": "anthropic",
        })
    return results


def list_models(
    *,
    provider: str,
    wire: str,
    base_url: str,
    api_key: str | None,
    timeout_sec: int,
    deep: bool = False,
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
            "context_length": None,
            "chat": True,
            "vision": True,
            "embedding_only": False,
            "source": "gemma",
        }]
    try:
        if wire == "ollama_native":
            return _ollama_models(base_url, timeout_sec, deep)
        if wire == "anthropic_messages":
            return _anthropic_models(base_url, api_key, timeout_sec, deep)
        return _openai_models(base_url, api_key, timeout_sec, deep)
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
