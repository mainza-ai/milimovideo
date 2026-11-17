"""Default Milimo LLM settings and film-production role catalog."""

from __future__ import annotations

import os
from typing import Any

from .capabilities import classify_model, pick_default_model
from .instances import instance_id
from .models import (
    LLMBudgetConfig,
    LLMProviderConfig,
    LLMRoleAssignment,
    SCHEMA_VERSION,
    LLMSettings,
)


ROLE_CATALOG = [
    {"role": "prompt_video", "group": "prompt", "label": "Video prompt",
     "description": "Camera, action, continuity, and audio rewrite for video generation"},
    {"role": "prompt_image", "group": "prompt", "label": "Image prompt",
     "description": "Still/preview prompt rewrite (Flux has no LTX text encoder)"},
    {"role": "script_parse", "group": "story", "label": "Script parser",
     "description": "Screenplay/treatment to scenes and shots"},
    {"role": "agent_plan", "group": "agent", "label": "Agent planner",
     "description": "Future natural-language plan proposals in the command dock"},
    {"role": "review_critic", "group": "review", "label": "Review critic",
     "description": "Future frame/take critique and continuity notes"},
    {"role": "continuity", "group": "review", "label": "Continuity checker",
     "description": "Future cross-shot identity, wardrobe, and prop checks"},
]

# Roles that must never resolve to the in-process LTX text encoder.
HTTP_ONLY_ROLES = {"prompt_image", "script_parse", "agent_plan", "review_critic", "continuity"}


def provider_env_key(provider: str) -> str:
    return {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }.get(provider, "")


def _provider(
    provider_type: str,
    name: str,
    base_url: str,
    default_model: str,
    wire: str,
    *,
    is_local: bool,
    enabled: bool,
    timeout: int = 120,
    keep_alive: str = "0",
) -> LLMProviderConfig:
    return LLMProviderConfig(
        id=instance_id(provider_type, base_url),
        type=provider_type,
        name=name,
        base_url=base_url,
        default_model=default_model,
        wire=wire,
        is_local=is_local,
        enabled=enabled,
        timeout_sec=timeout,
        keep_alive=keep_alive,
    )


def _chat_models(providers: dict[str, LLMProviderConfig], catalog: dict[str, list[dict]]) -> list[tuple[str, str]]:
    """Chat-capable (instance, model) pairs, local first, in provider order."""
    rows: list[tuple[str, str]] = []
    for provider_id, provider in providers.items():
        if provider.type == "gemma" or not provider.enabled:
            continue
        chat = [
            str(model.get("id"))
            for model in catalog.get(provider_id, [])
            if model.get("chat", True) and model.get("id")
        ]
        ordered = ([provider.default_model] if provider.default_model else []) + [
            model for model in chat if model != provider.default_model
        ]
        rows.extend((provider_id, model) for model in ordered if model)
    return rows


def build_defaults(
    discovered: list[dict[str, Any]] | None = None,
    catalog: dict[str, list[dict]] | None = None,
) -> LLMSettings:
    """Build a first-boot document from what is actually installed.

    No phantom models: an unset `MILIMO_OLLAMA_MODEL` no longer seeds
    `llama3.1`. Roles that cannot be satisfied are explicitly unassigned.
    """
    catalog = catalog or {}
    env_provider = os.environ.get("MILIMO_LLM_PROVIDER", "gemma").strip().lower()
    if env_provider not in {"gemma", "ollama"}:
        env_provider = "gemma"
    env_model = os.environ.get("MILIMO_OLLAMA_MODEL", "").strip()

    providers: dict[str, LLMProviderConfig] = {
        "gemma": _provider(
            "gemma", "Gemma 3 (LTX text encoder)", "", "ltx-gemma3",
            "gemma_native", is_local=True, enabled=True,
        ),
    }

    if discovered:
        for server in discovered:
            provider_type = str(server.get("provider") or "openai_compatible")
            base_url = str(server.get("base_url") or "")
            pid = str(server.get("instance_id") or instance_id(provider_type, base_url))
            models = [str(m) for m in (server.get("models") or []) if m]
            chat = [
                model for model in models
                if classify_model(model, declared_vision=False)["chat"]
            ]
            providers[pid] = _provider(
                provider_type,
                str(server.get("name") or pid),
                base_url,
                pick_default_model([{"id": model, **classify_model(model)} for model in models]) or (
                    chat[0] if chat else (models[0] if models else "")
                ),
                str(server.get("wire") or ("ollama_native" if provider_type == "ollama" else "openai_chat")),
                is_local=True,
                enabled=True,
                keep_alive=os.environ.get("MILIMO_OLLAMA_KEEP_ALIVE", "0") if provider_type == "ollama" else "0",
            )
    else:
        providers["ollama"] = _provider(
            "ollama", "Ollama", os.environ.get("MILIMO_OLLAMA_URL", "http://127.0.0.1:11434"),
            env_model, "ollama_native", is_local=True, enabled=True,
            keep_alive=os.environ.get("MILIMO_OLLAMA_KEEP_ALIVE", "0"),
        )

    providers["openai"] = _provider(
        "openai", "OpenAI", "https://api.openai.com/v1", "",
        "openai_chat", is_local=False, enabled=False,
    )
    providers["anthropic"] = _provider(
        "anthropic", "Anthropic", "https://api.anthropic.com", "",
        "anthropic_messages", is_local=False, enabled=False,
    )

    chat_rows = _chat_models(providers, catalog)
    if not chat_rows:
        # Fall back to the declared default models when no catalog is available.
        chat_rows = [
            (pid, p.default_model) for pid, p in providers.items()
            if p.type != "gemma" and p.enabled and p.default_model
        ]
    primary = chat_rows[0] if chat_rows else None

    roles: dict[str, LLMRoleAssignment] = {}
    for meta in ROLE_CATALOG:
        role = meta["role"]
        if role == "prompt_video":
            if env_provider == "ollama" and primary and providers[primary[0]].type == "ollama":
                roles[role] = LLMRoleAssignment(role=role, provider=primary[0], model=primary[1])
            else:
                roles[role] = LLMRoleAssignment(role=role, provider="gemma", model="ltx-gemma3")
            continue
        if primary:
            roles[role] = LLMRoleAssignment(role=role, provider=primary[0], model=primary[1])
        else:
            roles[role] = LLMRoleAssignment(role=role, provider="", model="")

    if env_provider == "ollama" and primary:
        active_provider, active_model = primary
    else:
        active_provider, active_model = "gemma", "ltx-gemma3"

    return LLMSettings(
        schema_version=SCHEMA_VERSION,
        active_provider=active_provider,
        active_model=active_model,
        air_gapped=True,
        providers=providers,
        roles=roles,
        budget=LLMBudgetConfig(daily_usd=0.0, eval_gate=False),
    )


def default_settings(discovered: list[dict[str, Any]] | None = None) -> LLMSettings:
    """Backwards-compatible entry point."""
    return build_defaults(discovered=discovered)
