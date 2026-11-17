"""Staged local/cloud profiles.

Profiles are transformations over the current settings document. They never
invent endpoints: they only re-route roles across providers the user already
has, and they keep local-first the default.
"""

from __future__ import annotations

from typing import Any

from . import runtime
from .models import LLMSettings, LLMRoleAssignment
from .store import LLMSettingsError, llm_settings

PROFILES = {
    "sovereign": {
        "label": "Sovereign local",
        "description": "Air-gapped. Every role on a local engine; cloud disabled and budget 0.",
    },
    "local_preferred": {
        "label": "Local preferred",
        "description": "Air-gapped. Cloud providers may stay configured but no role is routed to them.",
    },
    "hybrid": {
        "label": "Hybrid (opt-in)",
        "description": "Cloud allowed for agent/review roles when a key, budget, and eval exist.",
    },
}


def _local_candidates(settings: LLMSettings) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for provider_id, provider in settings.providers.items():
        if not provider.is_local or provider.type == "gemma" or not provider.enabled:
            continue
        cached = runtime.cached_model(provider_id, provider.default_model) if provider.default_model else None
        if cached and not cached.get("chat", True):
            continue
        if provider.default_model:
            rows.append((provider_id, provider.default_model))
    rows.sort(key=lambda row: 0 if settings.providers[row[0]].type == "ollama" else 1)
    return rows


def _first_cloud(settings: LLMSettings) -> tuple[str, str] | None:
    for provider_id, provider in settings.providers.items():
        if provider.is_local or not provider.enabled or not provider.default_model:
            continue
        if llm_settings.provider_key(provider_id):
            return provider_id, provider.default_model
    return None


def apply_profile(profile: str, expected_revision: int | None = None) -> dict[str, Any]:
    if profile not in PROFILES:
        raise LLMSettingsError(f"unknown profile {profile!r}")
    settings = llm_settings.get()
    doc = settings.model_copy(deep=True)
    local = _local_candidates(doc)
    local_primary = local[0] if local else None

    if profile in {"sovereign", "local_preferred"}:
        doc.air_gapped = True
        for assignment in doc.roles.values():
            provider = doc.providers.get(assignment.provider)
            if provider is None:
                continue
            if provider.is_local:
                continue
            if local_primary:
                assignment.provider, assignment.model = local_primary
                assignment.fallback = [hop for hop in assignment.fallback if doc.providers[hop.provider].is_local]
            else:
                assignment.provider = ""
                assignment.model = ""
                assignment.fallback = []
        if profile == "sovereign":
            for provider in doc.providers.values():
                if not provider.is_local:
                    provider.enabled = False
            doc.budget.daily_usd = 0.0
    else:  # hybrid
        cloud = _first_cloud(doc)
        if cloud is None:
            raise LLMSettingsError(
                "hybrid needs an enabled cloud provider with a stored key and default model"
            )
        if doc.budget.daily_usd <= 0:
            raise LLMSettingsError("hybrid needs a positive daily cloud budget")
        if not doc.last_eval:
            raise LLMSettingsError("hybrid needs a passing local eval on record")
        doc.air_gapped = False
        for role in ("agent_plan", "review_critic"):
            assignment = doc.roles.get(role)
            if assignment is None:
                assignment = LLMRoleAssignment(role=role)
                doc.roles[role] = assignment
            assignment.fallback = [
                *[hop for hop in assignment.fallback if hop.provider != cloud[0]],
                *([{"provider": cloud[0], "model": cloud[1]}] if assignment.assigned else []),
            ]
            if not assignment.assigned and local_primary:
                assignment.provider, assignment.model = local_primary

    return llm_settings.save(doc, expected_revision=expected_revision)
