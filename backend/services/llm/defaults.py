"""Default Milimo LLM settings and film-production role catalog."""

from __future__ import annotations

import os

from .models import LLMProviderConfig, LLMRoleAssignment, LLMModelTarget, LLMSettings


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


def default_settings() -> LLMSettings:
    ollama_model = os.environ.get("MILIMO_OLLAMA_MODEL", "llama3.1").strip()
    env_provider = os.environ.get("MILIMO_LLM_PROVIDER", "gemma").strip().lower()
    if env_provider not in {"gemma", "ollama"}:
        env_provider = "gemma"
    active_model = ollama_model if env_provider == "ollama" else "ltx-gemma3"

    providers = {
        "gemma": _provider(
            "gemma", "Gemma 3 (LTX text encoder)", "", "ltx-gemma3",
            "gemma_native", is_local=True, enabled=True,
        ),
        "ollama": _provider(
            "ollama", "Ollama", os.environ.get("MILIMO_OLLAMA_URL", "http://127.0.0.1:11434"),
            ollama_model, "ollama_native", is_local=True, enabled=True,
            keep_alive=os.environ.get("MILIMO_OLLAMA_KEEP_ALIVE", "0"),
        ),
        "lmstudio": _provider(
            "lmstudio", "LM Studio", "http://127.0.0.1:1234/v1", "",
            "openai_chat", is_local=True, enabled=False,
        ),
        "openai_compatible": _provider(
            "openai_compatible", "OpenAI-compatible local", "http://127.0.0.1:8080/v1", "",
            "openai_chat", is_local=True, enabled=False,
        ),
        "openai": _provider(
            "openai", "OpenAI", "https://api.openai.com/v1", "",
            "openai_chat", is_local=False, enabled=False,
        ),
        "anthropic": _provider(
            "anthropic", "Anthropic", "https://api.anthropic.com", "",
            "anthropic_messages", is_local=False, enabled=False,
        ),
    }

    roles = {
        "prompt_video": LLMRoleAssignment(
            role="prompt_video", provider="gemma", model="ltx-gemma3",
        ),
        "prompt_image": LLMRoleAssignment(
            role="prompt_image", provider="ollama", model=ollama_model,
        ),
        "script_parse": LLMRoleAssignment(
            role="script_parse", provider="ollama", model=ollama_model,
        ),
        "agent_plan": LLMRoleAssignment(
            role="agent_plan", provider="ollama", model=ollama_model,
        ),
        "review_critic": LLMRoleAssignment(
            role="review_critic", provider="ollama", model=ollama_model,
        ),
        "continuity": LLMRoleAssignment(
            role="continuity", provider="ollama", model=ollama_model,
        ),
    }

    return LLMSettings(
        active_provider=env_provider,
        active_model=active_model,
        air_gapped=True,
        providers=providers,
        roles=roles,
    )
