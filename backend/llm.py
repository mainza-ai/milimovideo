"""Prompt enhancement facade over the role-based LLM runtime.

The public signature is unchanged (`enhance_prompt(...) -> (text, is_ref)`), so
GPU tasks keep working. Provider choice now comes from the persisted settings
plane; call sites no longer branch on `config.LLM_PROVIDER`.

Usage:
    from llm import enhance_prompt, prepare_text_encoder
    text_encoder = prepare_text_encoder(pipeline, is_video=True)
    enhanced = enhance_prompt(prompt, system_prompt="...", text_encoder=text_encoder)
"""

import logging
import os
import threading
from collections import OrderedDict
from typing import Callable

from services.llm import runtime
from services.llm.prompts import IMAGE_SYSTEM_PROMPT, get_video_system_prompt
from services.llm.store import llm_settings

logger = logging.getLogger(__name__)

__all__ = [
    "IMAGE_SYSTEM_PROMPT",
    "describe_route",
    "enhance_prompt",
    "get_video_system_prompt",
    "last_enhancement",
    "module_role",
    "prepare_text_encoder",
    "release_local_models",
    "role_uses_gemma",
]

_local = threading.local()

# Reuse successful rewrites for identical prompt/seed/settings. The value is
# small text, not model tensors; cap it so a long studio session cannot grow
# without bound. Settings revision invalidates the cache on any plan change.
_ENHANCE_CACHE: "OrderedDict[tuple, tuple[str, bool, dict]]" = OrderedDict()
_ENHANCE_CACHE_MAX = 128
_ENHANCE_CACHE_LOCK = threading.Lock()


def _image_signature(path: str | None) -> tuple:
    if not path:
        return ("", 0, 0)
    try:
        stat = os.stat(path)
        return (path, stat.st_size, int(stat.st_mtime))
    except OSError:
        return (path, 0, 0)


def clear_prompt_cache() -> None:
    with _ENHANCE_CACHE_LOCK:
        _ENHANCE_CACHE.clear()


def role_for(is_video: bool) -> str:
    """Map the caller's media kind to the role that owns prompt rewriting."""
    return "prompt_video" if is_video else "prompt_image"


# Backwards-compatible name used by older call sites.
module_role = role_for


def role_uses_gemma(role: str) -> bool:
    """True when the first usable hop for a role is the in-process Gemma."""
    resolution = runtime.resolve(role)
    if not resolution["usable"] or not resolution["plan"]:
        return False
    return resolution["plan"][0]["provider"] == "gemma"


def prepare_text_encoder(pipeline, is_video: bool = True):
    """Return the LTX text encoder only when the resolved role needs Gemma."""
    if not role_uses_gemma(role_for(is_video)):
        return None
    ledger = getattr(pipeline, "stage_1_model_ledger", None)
    if ledger is None:
        return None
    try:
        return ledger.text_encoder()
    except Exception as exc:
        logger.warning("Could not retrieve the LTX text encoder: %s", exc)
        return None


def describe_route(role: str) -> dict | None:
    return runtime.describe_route(role)


def describe_runtime() -> dict:
    """Compact runtime description used in logs and job provenance."""
    settings = llm_settings.get()
    return {
        "active_provider": settings.active_provider,
        "active_model": settings.active_model,
        "air_gapped": settings.air_gapped,
        "prompt_video": runtime.resolve("prompt_video"),
        "prompt_image": runtime.resolve("prompt_image"),
    }


def last_enhancement() -> dict | None:
    """Provenance for the last successful enhancement on this thread."""
    return getattr(_local, "provenance", None)


def release_local_models() -> list[str]:
    """Unload local HTTP LLMs after enhancement, before denoising."""
    try:
        return runtime.release_local()
    except Exception as exc:
        logger.warning("Could not release local LLM models: %s", exc)
        return []


def enhance_prompt(
    prompt: str,
    system_prompt: str | None = None,
    is_video: bool = True,
    text_encoder=None,
    image_path: str | None = None,
    seed: int = 42,
    duration_seconds: float = 4.8,
    has_input_image: bool = False,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[str, bool]:
    """Enhance a generation prompt through the resolved role plan.

    Returns ``(enhanced_prompt, is_reference_only)``. On any provider failure
    the original prompt is returned unchanged so generation still proceeds.
    """
    role = role_for(is_video)
    json_mode = system_prompt is None
    settings_revision = llm_settings.get().revision
    cache_key = (
        settings_revision,
        role,
        prompt,
        system_prompt or "",
        seed,
        has_input_image,
        round(float(duration_seconds), 3),
        _image_signature(image_path),
    )
    with _ENHANCE_CACHE_LOCK:
        cached = _ENHANCE_CACHE.get(cache_key)
        if cached is not None:
            _ENHANCE_CACHE.move_to_end(cache_key)
    if cached is not None:
        text, is_reference, provenance = cached
        _local.provenance = {**provenance, "cached": True}
        return text, is_reference

    if not system_prompt:
        system_prompt = (
            get_video_system_prompt(duration_seconds, has_input_image)
            if is_video
            else IMAGE_SYSTEM_PROMPT
        )

    enhanced, is_reference_only, result = runtime.enhance(
        role,
        prompt,
        system=system_prompt,
        text_encoder=text_encoder,
        image_path=image_path,
        has_input_image=has_input_image,
        seed=seed,
        max_tokens=400 if is_video else 300,
        json_mode=json_mode,
        cancel_check=cancel_check,
    )

    if result is None:
        _local.provenance = {"role": role, "provider": None, "model": None, "failed": True}
        return prompt, False

    logger.info(
        "Enhanced %s prompt via %s/%s in %sms",
        role, result.provider, result.model, result.latency_ms,
    )
    provenance = result.provenance()
    _local.provenance = provenance
    with _ENHANCE_CACHE_LOCK:
        _ENHANCE_CACHE[cache_key] = (enhanced, is_reference_only, provenance)
        _ENHANCE_CACHE.move_to_end(cache_key)
        while len(_ENHANCE_CACHE) > _ENHANCE_CACHE_MAX:
            _ENHANCE_CACHE.popitem(last=False)
    return enhanced, is_reference_only
