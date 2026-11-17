"""Role-based LLM execution over the configured provider plan.

Every call resolves a role (`prompt_video`, `prompt_image`, `script_parse`,
...) to an ordered list of provider/model hops, then tries them in order. The
runtime owns retry, provenance, health recording, and air-gap enforcement.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import providers
from .models import LLMProviderConfig
from .store import LLMSettingsError, llm_settings

logger = logging.getLogger("llm.runtime")


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


def _plan(role: str) -> list[dict[str, Any]]:
    return llm_settings.plans(role)


def _hop_settings(hop: dict[str, Any]) -> dict[str, Any]:
    provider_id = hop["provider"]
    provider = llm_settings.get().providers.get(provider_id)
    if provider is None:
        raise LLMSettingsError(f"role {hop.get('role')}: provider {provider_id!r} disappeared")
    return {
        "provider": provider_id,
        "wire": provider.wire,
        "base_url": provider.base_url,
        "timeout_sec": provider.timeout_sec,
        "keep_alive": provider.keep_alive,
        "is_local": provider.is_local,
    }


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


def _record_failure(provider: str, exc: Exception) -> None:
    message = getattr(exc, "message", None) or str(exc)
    llm_settings.record_failure(provider, message)


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
) -> LLMResult:
    """Run a role synchronously, trying the primary hop then fallbacks."""
    hops = _plan(role)
    messages = _messages(system, prompt)
    errors: list[str] = []

    for hop in hops:
        provider_id = hop["provider"]
        model = hop["model"]
        settings = _hop_settings(hop)
        hop = {**hop, **settings}
        if max_tokens is not None:
            hop["max_tokens"] = max_tokens
        if seed is not None:
            hop["seed"] = seed
        if temperature is not None:
            hop["temperature"] = temperature

        try:
            if hop["wire"] == "gemma_native":
                raw = providers.complete_gemma(
                    text_encoder,
                    messages,
                    max_new_tokens=hop.get("max_tokens") or 400,
                    seed=hop.get("seed") if hop.get("seed") is not None else 42,
                    image_path=image_path,
                    has_input_image=has_input_image,
                )
                result = LLMResult(text=raw, provider=provider_id, model=model, role=role, latency_ms=0)
            else:
                api_key = llm_settings.provider_key(provider_id)
                done = providers.complete(
                    provider=provider_id,
                    wire=hop["wire"],
                    base_url=settings["base_url"],
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    options=_options(hop),
                    timeout_sec=settings["timeout_sec"],
                    keep_alive=settings["keep_alive"],
                    json_mode=json_mode,
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
            llm_settings.record_success(provider_id)
            return result
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
) -> LLMResult:
    """Run a role asynchronously. Gemma hops are executed in a worker thread."""
    hops = _plan(role)
    messages = _messages(system, prompt)
    errors: list[str] = []

    for hop in hops:
        provider_id = hop["provider"]
        model = hop["model"]
        settings = _hop_settings(hop)
        hop = {**hop, **settings}
        if max_tokens is not None:
            hop["max_tokens"] = max_tokens
        if seed is not None:
            hop["seed"] = seed
        if temperature is not None:
            hop["temperature"] = temperature

        try:
            if hop["wire"] == "gemma_native":
                raw = await asyncio.to_thread(
                    providers.complete_gemma,
                    text_encoder,
                    messages,
                    max_new_tokens=hop.get("max_tokens") or 400,
                    seed=hop.get("seed") if hop.get("seed") is not None else 42,
                    image_path=image_path,
                    has_input_image=has_input_image,
                )
                result = LLMResult(text=raw, provider=provider_id, model=model, role=role, latency_ms=0)
            else:
                api_key = llm_settings.provider_key(provider_id)
                done = await providers.acomplete(
                    provider=provider_id,
                    wire=hop["wire"],
                    base_url=settings["base_url"],
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    options=_options(hop),
                    timeout_sec=settings["timeout_sec"],
                    keep_alive=settings["keep_alive"],
                    json_mode=json_mode,
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
            llm_settings.record_success(provider_id)
            return result
        except Exception as exc:
            _record_failure(provider_id, exc)
            errors.append(f"{provider_id}: {getattr(exc, 'message', str(exc))}")
            logger.warning("LLM role %s hop %s/%s failed: %s", role, provider_id, model, exc)

    raise LLMSettingsError(f"role {role}: every provider failed ({'; '.join(errors)})")


# ── prompt enhancement ──────────────────────────────────────────────


def parse_enhancement(text: str, fallback_prompt: str) -> tuple[str, bool]:
    data = providers.extract_json_object(text)
    if data and isinstance(data.get("enhanced_prompt"), str) and data["enhanced_prompt"].strip():
        return data["enhanced_prompt"].strip(), bool(data.get("is_reference_only", False))
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
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
        )
    except Exception as exc:
        logger.error("Prompt enhancement role %s failed: %s", role, exc)
        return prompt, False, None

    enhanced, is_reference_only = parse_enhancement(result.text, prompt)
    return enhanced, is_reference_only, result


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
        done = providers.complete(
            provider=provider,
            wire=config.wire,
            base_url=config.base_url,
            api_key=api_key,
            model=target_model,
            messages=_messages(None, prompt),
            options=_options({"wire": config.wire, "temperature": 0.2, "max_tokens": 200}),
            timeout_sec=config.timeout_sec,
            keep_alive=config.keep_alive,
        )
        return LLMResult(text=done.text, provider=provider, model=target_model, role=role, latency_ms=done.latency_ms, usage=done.usage)

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
            "ok": True,
            "provider": provider_id,
            "wire": config.wire,
            "latency_ms": 0,
            "models": ["ltx-gemma3"],
            "error": None,
            "fault": "in_process",
        }
    api_key = llm_settings.provider_key(provider_id)
    started = time.monotonic()
    try:
        models = providers.list_models(
            provider=provider_id,
            wire=config.wire,
            base_url=config.base_url,
            api_key=api_key,
            timeout_sec=min(config.timeout_sec, 10),
        )
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
        }
    llm_settings.record_success(provider_id)
    return {
        "ok": True,
        "provider": provider_id,
        "wire": config.wire,
        "latency_ms": int((time.monotonic() - started) * 1000),
        "models": [row["id"] for row in models],
        "error": None,
        "fault": None,
    }


def provider_models(provider_id: str) -> list[dict]:
    settings = llm_settings.get()
    config = settings.providers.get(provider_id)
    if config is None:
        raise LLMSettingsError(f"unknown provider {provider_id!r}")
    if settings.air_gapped and not config.is_local:
        raise LLMSettingsError(f"{provider_id}: blocked by air-gap")
    api_key = llm_settings.provider_key(provider_id)
    return providers.list_models(
        provider=provider_id,
        wire=config.wire,
        base_url=config.base_url,
        api_key=api_key,
        timeout_sec=min(config.timeout_sec, 10),
    )


__all__ = [
    "LLMResult",
    "resolve",
    "complete",
    "acomplete",
    "enhance",
    "test_target",
    "verify_provider",
    "provider_models",
]
