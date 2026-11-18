"""Role-based LLM execution over the configured provider plans.

Every call resolves a role (`prompt_video`, `prompt_image`, `script_parse`,
...) to an ordered list of provider/model hops, then tries them in order. The
runtime owns context preflight, budget checks, cancellation, streaming
negotiation, health recording, and provenance.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Sequence

from . import budget, providers
from .models import LLMProviderConfig
from services.memory import costs as memory_costs
from services.memory import engines as memory_engines
from .store import LLMSettingsError, llm_settings

logger = logging.getLogger("llm.runtime")

STREAM_FIRST_BYTE_SEC = 8.0
CATALOG_TTL_SEC = 300.0

_catalog: dict[str, tuple[float, list[dict]]] = {}
_catalog_lock = threading.Lock()


def _cached_catalog(provider_id: str) -> list[dict] | None:
    with _catalog_lock:
        row = _catalog.get(provider_id)
        if not row:
            return None
        stamp, models = row
        if time.monotonic() - stamp > CATALOG_TTL_SEC:
            return None
        return models


def cache_catalog(provider_id: str, models: list[dict]) -> None:
    with _catalog_lock:
        _catalog[provider_id] = (time.monotonic(), list(models))


def clear_catalog(provider_id: str | None = None) -> None:
    with _catalog_lock:
        if provider_id is None:
            _catalog.clear()
        else:
            _catalog.pop(provider_id, None)


def cached_model(provider_id: str, model: str) -> dict | None:
    for row in _cached_catalog(provider_id) or []:
        if str(row.get("id")) == model or str(row.get("name")) == model:
            return row
    return None


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    role: str
    latency_ms: int
    usage: dict[str, int] = field(default_factory=dict)

    def provenance(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "tokens": self.usage or None,
        }


class LLMCancelled(RuntimeError):
    pass


def _messages(system: str | None, prompt: str) -> list[providers.ChatMessage]:
    rows: list[providers.ChatMessage] = []
    if system:
        rows.append(providers.ChatMessage(role="system", content=system))
    rows.append(providers.ChatMessage(role="user", content=prompt))
    return rows


def _options(hop: dict[str, Any]) -> dict[str, Any]:
    wire = hop["wire"]
    temperature = hop.get("temperature")
    top_p = hop.get("top_p")
    max_tokens = hop.get("max_tokens")
    seed = hop.get("seed")

    if wire == "ollama_native":
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if max_tokens:
            options["num_predict"] = max_tokens
        if seed is not None:
            options["seed"] = seed
        return options

    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if max_tokens:
        options["max_tokens"] = max_tokens
    if seed is not None and wire == "openai_chat":
        options["seed"] = seed
    return options


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text or "") / 4))


def _plan(role: str) -> list[dict[str, Any]]:
    return llm_settings.plans(role)


def _hop_settings(hop: dict[str, Any]) -> dict[str, Any]:
    provider = llm_settings.get().providers.get(hop["provider"])
    if provider is None:
        raise LLMSettingsError(f"role {hop.get('role')}: provider {hop['provider']!r} disappeared")
    return {
        "wire": provider.wire,
        "base_url": provider.base_url,
        "timeout_sec": provider.timeout_sec,
        "keep_alive": provider.keep_alive,
        "is_local": provider.is_local,
        "type": provider.type,
    }


def _context_guard(hop: dict[str, Any], messages: Sequence[providers.ChatMessage], max_tokens: int) -> None:
    info = cached_model(hop["provider"], hop["model"])
    if not info:
        return
    limit = info.get("context_length")
    if not isinstance(limit, int) or limit <= 0:
        return
    estimate = sum(estimate_tokens(m.content) for m in messages) + max(int(max_tokens or 0), 0)
    if estimate > int(limit * 0.95):
        raise LLMSettingsError(
            f"{hop['provider']}/{hop['model']}: prompt is ~{estimate} tokens but the model "
            f"allows {limit}. Shorten the prompt or pick a model with a larger context."
        )
    if not info.get("chat", True):
        raise LLMSettingsError(
            f"{hop['provider']}/{hop['model']}: this model is not a chat model "
            f"({'embedding' if info.get('embedding_only') else 'tool/audio/image'}). "
            "Pick a chat model for this role."
        )


def _reserve_llm(provider_id: str, model: str, is_local: bool) -> None:
    """Reserve budget before a local HTTP model runs.

    Raises LLMSettingsError when the model cannot fit; the caller then falls
    back or returns the original prompt. Cloud providers are skipped.
    """
    if not is_local:
        return
    try:
        from memory_manager import memory_manager

        settings = llm_settings.get()
        provider = settings.providers.get(provider_id)
        cost = float(getattr(provider, "memory_gb", 0.0) or 0.0) if provider else 0.0
        if cost <= 0:
            cost = memory_engines.resident_gb(settings, provider_id)
        if cost <= 0:
            cost = memory_costs.estimate_llm_gb(model)
        memory_manager.note_llm(provider_id, model, cost)
    except MemoryError as exc:
        raise LLMSettingsError(f"{provider_id}/{model}: {exc}") from exc
    except Exception as exc:
        logger.debug("LLM memory reservation skipped for %s: %s", provider_id, exc)


def _local_models_in_use() -> dict[str, set[str]]:
    """Every local (instance, model) pair a role or default could have loaded."""
    settings = llm_settings.get()
    used: dict[str, set[str]] = {}
    for provider_id, provider in settings.providers.items():
        if not provider.is_local or provider.type == "gemma":
            continue
        if provider.default_model:
            used.setdefault(provider_id, set()).add(provider.default_model)
    for assignment in settings.roles.values():
        if not assignment.assigned:
            continue
        provider = settings.providers.get(assignment.provider)
        if provider and provider.is_local and provider.type != "gemma":
            used.setdefault(assignment.provider, set()).add(assignment.model)
        for hop in assignment.fallback:
            fallback_provider = settings.providers.get(hop.provider)
            if fallback_provider and fallback_provider.is_local and fallback_provider.type != "gemma":
                used.setdefault(hop.provider, set()).add(hop.model)
    return used


def release_local() -> list[str]:
    """Unload local HTTP models that support it and release the slot.

    Called after prompt enhancement so a warm 27B does not sit through
    denoising. Unloads every model a role could have used, not just the
    provider default. Safe to call when nothing is loaded.
    """
    result = memory_engines.unload()
    released = list(dict.fromkeys(item.split(":", 1)[0] for item in result.get("unloaded", [])))
    try:
        from memory_manager import memory_manager

        memory_manager.release("llm_http")
    except Exception:
        pass
    return released


def unload_local_models() -> None:
    """MemoryManager hook: drop local LLM residency before a heavy model loads."""
    memory_engines.unload()

    def release() -> None:
        try:
            from memory_manager import memory_manager

            memory_manager.release("llm_http")
        except Exception:
            pass

    release()


def resolve(role: str) -> dict[str, Any]:
    """Return the effective plan for a role without calling a provider."""
    try:
        hops = _plan(role)
    except LLMSettingsError as exc:
        return {"role": role, "usable": False, "reason": str(exc), "plan": []}
    return {
        "role": role,
        "usable": True,
        "reason": None,
        "plan": [
            {
                "provider": hop["provider"],
                "model": hop["model"],
                "is_local": hop.get("is_local"),
                "wire": hop.get("wire"),
            }
            for hop in hops
        ],
    }


def describe_route(role: str) -> dict[str, Any] | None:
    resolution = resolve(role)
    if not resolution["usable"] or not resolution["plan"]:
        return None
    return resolution["plan"][0]


def _record_failure(provider: str, exc: Exception) -> None:
    message = getattr(exc, "message", None) or str(exc)
    llm_settings.record_failure(provider, message)


def _prepare_hop(hop: dict[str, Any]) -> dict[str, Any]:
    settings = _hop_settings(hop)
    merged = {**hop, **settings}
    merged["api_key"] = llm_settings.provider_key(hop["provider"])
    return merged


def _budget_precheck(hop: dict[str, Any], messages, max_tokens: int) -> float:
    settings = llm_settings.get()
    prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
    try:
        return budget.precheck(
            settings,
            provider=hop["provider"],
            is_local=bool(hop.get("is_local")),
            model=hop["model"],
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
        )
    except ValueError as exc:
        raise LLMSettingsError(str(exc)) from exc


def _usage_after(hop: dict[str, Any], messages, usage: dict[str, int]) -> None:
    budget.record(
        provider=hop["provider"],
        is_local=bool(hop.get("is_local")),
        model=hop["model"],
        usage=usage,
        estimated_prompt_tokens=sum(estimate_tokens(m.content) for m in messages),
    )


def _check_cancel(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check and cancel_check():
        raise LLMCancelled("cancelled")


def complete(
    role: str,
    prompt: str,
    *,
    system: str | None = None,
    text_encoder: Any = None,
    image_path: str | None = None,
    has_input_image: bool = False,
    max_tokens: int | None = None,
    seed: int | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> LLMResult:
    """Run a role synchronously, trying the primary hop then fallbacks."""
    hops = _plan(role)
    messages = _messages(system, prompt)
    errors: list[str] = []

    for hop in hops:
        _check_cancel(cancel_check)
        hop = _prepare_hop(hop)
        provider_id = hop["provider"]
        model = hop["model"]
        if max_tokens is not None:
            hop["max_tokens"] = max_tokens
        if seed is not None:
            hop["seed"] = seed
        if temperature is not None:
            hop["temperature"] = temperature
        call_max = int(hop.get("max_tokens") or 400)

        try:
            _context_guard(hop, messages, call_max)
            _budget_precheck(hop, messages, call_max)

            if hop["wire"] == "gemma_native":
                raw = providers.complete_gemma(
                    text_encoder,
                    messages,
                    max_new_tokens=call_max,
                    seed=hop.get("seed") if hop.get("seed") is not None else 42,
                    image_path=image_path,
                    has_input_image=has_input_image,
                )
                result = LLMResult(text=raw, provider=provider_id, model=model, role=role, latency_ms=0)
            else:
                _reserve_llm(provider_id, model, bool(hop.get("is_local")))
                done = providers.complete(
                    provider=provider_id,
                    wire=hop["wire"],
                    base_url=hop["base_url"],
                    api_key=hop["api_key"],
                    model=model,
                    messages=messages,
                    options=_options(hop),
                    timeout_sec=hop["timeout_sec"],
                    keep_alive=hop["keep_alive"],
                    json_mode=json_mode,
                    image_path=image_path if has_input_image else None,
                    cancel_check=cancel_check,
                )
                result = LLMResult(
                    text=done.text,
                    provider=provider_id,
                    model=model,
                    role=role,
                    latency_ms=done.latency_ms,
                    usage=done.usage,
                )
            if not result.text.strip():
                raise providers.ProviderError(provider_id, "empty", "provider returned an empty response")
            llm_settings.record_success(provider_id, result.latency_ms, result.usage)
            _usage_after(hop, messages, result.usage)
            return result
        except LLMCancelled:
            raise
        except providers.ProviderCancelled as exc:
            raise LLMCancelled(str(exc)) from exc
        except Exception as exc:
            _record_failure(provider_id, exc)
            errors.append(f"{provider_id}: {getattr(exc, 'message', str(exc))}")
            logger.warning("LLM role %s hop %s/%s failed: %s", role, provider_id, model, exc)

    raise LLMSettingsError(f"role {role}: every provider failed ({'; '.join(errors)})")


async def acomplete(
    role: str,
    prompt: str,
    *,
    system: str | None = None,
    text_encoder: Any = None,
    image_path: str | None = None,
    has_input_image: bool = False,
    max_tokens: int | None = None,
    seed: int | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> LLMResult:
    """Run a role asynchronously. Gemma hops execute in a worker thread."""
    hops = _plan(role)
    messages = _messages(system, prompt)
    errors: list[str] = []

    for hop in hops:
        _check_cancel(cancel_check)
        hop = _prepare_hop(hop)
        provider_id = hop["provider"]
        model = hop["model"]
        if max_tokens is not None:
            hop["max_tokens"] = max_tokens
        if seed is not None:
            hop["seed"] = seed
        if temperature is not None:
            hop["temperature"] = temperature
        call_max = int(hop.get("max_tokens") or 400)

        try:
            _context_guard(hop, messages, call_max)
            _budget_precheck(hop, messages, call_max)

            if hop["wire"] == "gemma_native":
                raw = await asyncio.to_thread(
                    providers.complete_gemma,
                    text_encoder,
                    messages,
                    max_new_tokens=call_max,
                    seed=hop.get("seed") if hop.get("seed") is not None else 42,
                    image_path=image_path,
                    has_input_image=has_input_image,
                )
                result = LLMResult(text=raw, provider=provider_id, model=model, role=role, latency_ms=0)
            else:
                _reserve_llm(provider_id, model, bool(hop.get("is_local")))
                done = await providers.acomplete(
                    provider=provider_id,
                    wire=hop["wire"],
                    base_url=hop["base_url"],
                    api_key=hop["api_key"],
                    model=model,
                    messages=messages,
                    options=_options(hop),
                    timeout_sec=hop["timeout_sec"],
                    keep_alive=hop["keep_alive"],
                    json_mode=json_mode,
                    image_path=image_path if has_input_image else None,
                    cancel_check=cancel_check,
                )
                result = LLMResult(
                    text=done.text,
                    provider=provider_id,
                    model=model,
                    role=role,
                    latency_ms=done.latency_ms,
                    usage=done.usage,
                )
            if not result.text.strip():
                raise providers.ProviderError(provider_id, "empty", "provider returned an empty response")
            llm_settings.record_success(provider_id, result.latency_ms, result.usage)
            _usage_after(hop, messages, result.usage)
            return result
        except LLMCancelled:
            raise
        except providers.ProviderCancelled as exc:
            raise LLMCancelled(str(exc)) from exc
        except Exception as exc:
            _record_failure(provider_id, exc)
            errors.append(f"{provider_id}: {getattr(exc, 'message', str(exc))}")
            logger.warning("LLM role %s hop %s/%s failed: %s", role, provider_id, model, exc)

    raise LLMSettingsError(f"role {role}: every provider failed ({'; '.join(errors)})")


async def astream(
    role: str,
    prompt: str,
    *,
    system: str | None = None,
    image_path: str | None = None,
    has_input_image: bool = False,
    max_tokens: int | None = None,
    temperature: float | None = None,
    json_mode: bool = False,
    cancel_check: Callable[[], bool] | None = None,
) -> AsyncIterator[str]:
    """Stream a role, falling back to non-stream when an engine cannot stream.

    oMLX accepts `stream: true` and then sends nothing; the first-byte timeout
    catches that and the call completes through the normal path.
    """
    hops = _plan(role)
    messages = _messages(system, prompt)
    errors: list[str] = []

    for hop in hops:
        _check_cancel(cancel_check)
        hop = _prepare_hop(hop)
        provider_id = hop["provider"]
        model = hop["model"]
        if max_tokens is not None:
            hop["max_tokens"] = max_tokens
        if temperature is not None:
            hop["temperature"] = temperature
        call_max = int(hop.get("max_tokens") or 400)

        if hop["wire"] == "gemma_native":
            try:
                result = await acomplete(
                    role, prompt, system=system, image_path=image_path,
                    has_input_image=has_input_image, max_tokens=max_tokens,
                    temperature=temperature, json_mode=json_mode, cancel_check=cancel_check,
                )
                yield result.text
                return
            except Exception as exc:
                _record_failure(provider_id, exc)
                errors.append(f"{provider_id}: {getattr(exc, 'message', str(exc))}")
                continue

        try:
            _context_guard(hop, messages, call_max)
            _budget_precheck(hop, messages, call_max)
        except Exception as exc:
            _record_failure(provider_id, exc)
            errors.append(f"{provider_id}: {getattr(exc, 'message', str(exc))}")
            continue

        _reserve_llm(provider_id, model, bool(hop.get("is_local")))
        started = time.monotonic()
        generator = providers.astream(
            provider=provider_id,
            wire=hop["wire"],
            base_url=hop["base_url"],
            api_key=hop["api_key"],
            model=model,
            messages=messages,
            options=_options(hop),
            timeout_sec=hop["timeout_sec"],
            keep_alive=hop["keep_alive"],
            json_mode=json_mode,
            image_path=image_path if has_input_image else None,
        )
        first: str | None = None
        try:
            first = await asyncio.wait_for(generator.__anext__(), timeout=STREAM_FIRST_BYTE_SEC)
        except StopAsyncIteration:
            first = None
        except (asyncio.TimeoutError, asyncio.CancelledError):
            try:
                await generator.aclose()
            except Exception:
                pass
            logger.info("%s did not stream within %.0fs; using non-stream", provider_id, STREAM_FIRST_BYTE_SEC)
            try:
                _reserve_llm(provider_id, model, bool(hop.get("is_local")))
                done = await providers.acomplete(
                    provider=provider_id,
                    wire=hop["wire"],
                    base_url=hop["base_url"],
                    api_key=hop["api_key"],
                    model=model,
                    messages=messages,
                    options=_options(hop),
                    timeout_sec=hop["timeout_sec"],
                    keep_alive=hop["keep_alive"],
                    json_mode=json_mode,
                    image_path=image_path if has_input_image else None,
                    cancel_check=cancel_check,
                )
            except providers.ProviderCancelled as exc2:
                raise LLMCancelled(str(exc2)) from exc2
            except Exception as exc2:
                _record_failure(provider_id, exc2)
                errors.append(f"{provider_id}: {getattr(exc2, 'message', str(exc2))}")
                continue
            latency = int((time.monotonic() - started) * 1000)
            llm_settings.record_success(provider_id, latency, done.usage)
            _usage_after(hop, messages, done.usage)
            yield done.text
            return
        except Exception as exc:
            try:
                await generator.aclose()
            except Exception:
                pass
            _record_failure(provider_id, exc)
            errors.append(f"{provider_id}: {getattr(exc, 'message', str(exc))}")
            continue

        chunks: list[str] = []
        if first:
            chunks.append(first)
            yield first
        cancelled = False
        try:
            async for delta in generator:
                if cancel_check and cancel_check():
                    cancelled = True
                    break
                chunks.append(delta)
                yield delta
        finally:
            try:
                await generator.aclose()
            except Exception:
                pass
        text = "".join(chunks)
        latency = int((time.monotonic() - started) * 1000)
        usage = {"completion_tokens": estimate_tokens(text)}
        llm_settings.record_success(provider_id, latency, usage)
        _usage_after(hop, messages, usage)
        if cancelled:
            raise LLMCancelled("cancelled")
        if not text.strip():
            errors.append(f"{provider_id}: empty stream")
            continue
        return

    raise LLMSettingsError(f"role {role}: every provider failed ({'; '.join(errors)})")


# ── model catalogs ──────────────────────────────────────────────────


def provider_models(provider_id: str, deep: bool = False) -> list[dict]:
    settings = llm_settings.get()
    config = settings.providers.get(provider_id)
    if config is None:
        raise LLMSettingsError(f"unknown provider {provider_id!r}")
    if settings.air_gapped and not config.is_local:
        raise LLMSettingsError(f"{provider_id}: blocked by air-gap")
    api_key = llm_settings.provider_key(provider_id)
    models = providers.list_models(
        provider=provider_id,
        wire=config.wire,
        base_url=config.base_url,
        api_key=api_key,
        timeout_sec=min(config.timeout_sec, 10),
        deep=deep,
    )
    cache_catalog(provider_id, models)
    return models


def cap_missing_models() -> list[dict[str, str]]:
    """Role assignments whose model is absent from a loaded catalog."""
    rows: list[dict[str, str]] = []
    settings = llm_settings.get()
    for role, assignment in settings.roles.items():
        if not assignment.assigned:
            continue
        catalog = _cached_catalog(assignment.provider)
        if catalog is None:
            continue
        if not any(str(m.get("id")) == assignment.model for m in catalog):
            rows.append({"role": role, "provider": assignment.provider, "model": assignment.model})
    return rows


# ── prompt enhancement ──────────────────────────────────────────────


def parse_enhancement(text: str, fallback_prompt: str) -> tuple[str, bool]:
    data = providers.extract_json_object(text)
    if data:
        for key in ("enhanced_prompt", "prompt", "text", "image_prompt", "result", "output"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), bool(data.get("is_reference_only", False))
        strings = [value for value in data.values() if isinstance(value, str) and len(value.strip()) > 3]
        if strings:
            return max(strings, key=len).strip(), bool(data.get("is_reference_only", False))
        return fallback_prompt, False
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    if cleaned.startswith("{") or cleaned.startswith("["):
        return fallback_prompt, False
    return (cleaned or fallback_prompt), False


def enhance(
    role: str,
    prompt: str,
    *,
    system: str | None,
    text_encoder: Any = None,
    image_path: str | None = None,
    has_input_image: bool = False,
    seed: int = 42,
    max_tokens: int | None = None,
    json_mode: bool | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[str, bool, LLMResult | None]:
    """Enhance a prompt for the given role. Returns the original on failure."""
    try:
        result = complete(
            role,
            f"Raw Input Prompt: {prompt}",
            system=system,
            text_encoder=text_encoder,
            image_path=image_path,
            has_input_image=has_input_image,
            seed=seed,
            max_tokens=max_tokens,
            json_mode=(role == "prompt_video") if json_mode is None else json_mode,
            cancel_check=cancel_check,
        )
    except LLMCancelled:
        raise
    except Exception as exc:
        logger.error("Prompt enhancement role %s failed: %s", role, exc)
        return prompt, False, None

    enhanced, is_reference_only = parse_enhancement(result.text, prompt)
    return enhanced, is_reference_only, result


# ── verification / test / warm ──────────────────────────────────────


def test_target(
    prompt: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    role: str = "prompt_video",
) -> LLMResult:
    """Send a test prompt to an explicit provider/model or the role plan."""
    if provider:
        settings = llm_settings.get()
        config = settings.providers.get(provider)
        if config is None:
            raise LLMSettingsError(f"unknown provider {provider!r}")
        if config.wire == "gemma_native":
            raise LLMSettingsError("Gemma has no standalone endpoint; test it through a generation")
        if settings.air_gapped and not config.is_local:
            raise LLMSettingsError(f"{provider}: blocked by air-gap")
        target_model = model or config.default_model
        if not target_model:
            raise LLMSettingsError(f"{provider}: no model selected")
        api_key = llm_settings.provider_key(provider)
        messages = _messages(None, prompt)
        hop = {
            "provider": provider,
            "model": target_model,
            "is_local": config.is_local,
            "wire": config.wire,
            "base_url": config.base_url,
            "timeout_sec": config.timeout_sec,
            "keep_alive": config.keep_alive,
            "max_tokens": 200,
            "temperature": 0.2,
            "api_key": api_key,
        }
        _budget_precheck(hop, messages, 200)
        done = providers.complete(
            provider=provider,
            wire=config.wire,
            base_url=config.base_url,
            api_key=api_key,
            model=target_model,
            messages=messages,
            options=_options(hop),
            timeout_sec=config.timeout_sec,
            keep_alive=config.keep_alive,
        )
        llm_settings.record_success(provider, done.latency_ms, done.usage)
        _usage_after(hop, messages, done.usage)
        return LLMResult(
            text=done.text, provider=provider, model=target_model, role=role,
            latency_ms=done.latency_ms, usage=done.usage,
        )

    return complete(role, prompt, max_tokens=200, temperature=0.2)


def verify_provider(provider_id: str) -> dict[str, Any]:
    """Live connectivity + model catalog check for one provider."""
    settings = llm_settings.get()
    config: LLMProviderConfig | None = settings.providers.get(provider_id)
    if config is None:
        raise LLMSettingsError(f"unknown provider {provider_id!r}")
    if settings.air_gapped and not config.is_local:
        raise LLMSettingsError(f"{provider_id}: blocked by air-gap")
    if config.wire == "gemma_native":
        return {
            "ok": True, "provider": provider_id, "wire": config.wire, "latency_ms": 0,
            "models": ["ltx-gemma3"], "error": None, "fault": "in_process", "needs_key": False,
        }
    started = time.monotonic()
    try:
        models = provider_models(provider_id)
    except Exception as exc:
        message = getattr(exc, "message", str(exc))
        llm_settings.record_failure(provider_id, message)
        return {
            "ok": False,
            "provider": provider_id,
            "wire": config.wire,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "models": [],
            "error": message,
            "fault": getattr(exc, "fault", "unknown"),
            "needs_key": getattr(exc, "fault", "") == "auth",
        }
    llm_settings.record_success(provider_id, int((time.monotonic() - started) * 1000))
    return {
        "ok": True,
        "provider": provider_id,
        "wire": config.wire,
        "latency_ms": int((time.monotonic() - started) * 1000),
        "models": [str(row.get("id")) for row in models],
        "error": None,
        "fault": None,
        "needs_key": False,
    }


def warm_provider(provider_id: str) -> dict[str, Any]:
    settings = llm_settings.get()
    config = settings.providers.get(provider_id)
    if config is None:
        raise LLMSettingsError(f"unknown provider {provider_id!r}")
    if not config.default_model:
        raise LLMSettingsError(f"{provider_id}: set a default model before warming")
    if settings.air_gapped and not config.is_local:
        raise LLMSettingsError(f"{provider_id}: blocked by air-gap")
    from memory_manager import memory_manager

    cost = float(getattr(config, "memory_gb", 0.0) or 0.0) or memory_costs.estimate_llm_gb(
        config.default_model
    )
    decision = memory_manager.snapshot()
    if decision.enforce and cost > decision.available_gb:
        raise LLMSettingsError(
            f"{provider_id}: warming {config.default_model} needs ~{cost:.1f} GB but only "
            f"{decision.available_gb:.1f} GB is available (swap {decision.swap_percent:.0f}%)."
        )
    result = providers.warm(
        provider=provider_id,
        wire=config.wire,
        base_url=config.base_url,
        api_key=llm_settings.provider_key(provider_id),
        model=config.default_model,
        timeout_sec=config.timeout_sec,
        keep_alive=config.keep_alive,
    )
    memory_manager.note_llm(
        provider_id, config.default_model,
        float(getattr(config, "memory_gb", 0.0) or 0.0) or memory_costs.estimate_llm_gb(config.default_model),
    )
    llm_settings.record_load(provider_id, int(result.get("load_ms") or 0))
    return {"provider": provider_id, "model": config.default_model, **result}


__all__ = [
    "LLMCancelled",
    "LLMResult",
    "astream",
    "cache_catalog",
    "cached_model",
    "cap_missing_models",
    "clear_catalog",
    "complete",
    "acomplete",
    "describe_route",
    "enhance",
    "parse_enhancement",
    "provider_models",
    "release_local",
    "resolve",
    "test_target",
    "unload_local_models",
    "verify_provider",
    "warm_provider",
]
