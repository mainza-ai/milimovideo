"""Persisted, revisioned LLM settings store with redacted snapshots."""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from . import budget, discovery, secretbox
from .capabilities import classify_model, pick_default_model
from .defaults import build_defaults, provider_env_key
from .instances import instance_id, is_local_host, normalize_base_url, valid_instance_id
from .models import (
    LLMProviderConfig,
    LLMRoleAssignment,
    LLMSettings,
    PROVIDER_TYPES,
    ROLE_IDS,
    SCHEMA_VERSION,
    WIRE_TYPES,
)

logger = logging.getLogger("llm.settings")

KEY = "llm"
CLOUD_ENV = "MILIMO_LLM_CLOUD"


class LLMSettingsError(ValueError):
    pass


class LLMRevisionConflict(LLMSettingsError):
    def __init__(self, expected: int | None, current: int):
        self.expected = expected
        self.current = current
        super().__init__(f"settings changed since revision {expected} (current {current}): re-read and re-apply")


def _validate_url(provider: LLMProviderConfig) -> None:
    if provider.type == "gemma":
        return
    url = (provider.base_url or "").strip()
    if not url:
        raise LLMSettingsError(f"provider {provider.id or provider.type}: base URL is required")
    parsed = urlparse(url)
    if parsed.username or parsed.password:
        raise LLMSettingsError(f"provider {provider.id or provider.type}: credentials must not be embedded in the URL")
    if parsed.query or parsed.fragment:
        raise LLMSettingsError(f"provider {provider.id or provider.type}: base URL must not contain query or fragment")
    scheme = (parsed.scheme or "").lower()
    host = parsed.hostname or ""
    if scheme not in {"http", "https"}:
        raise LLMSettingsError(f"provider {provider.id or provider.type}: only http/https endpoints are supported")
    local = is_local_host(host)
    if scheme == "http" and not local:
        raise LLMSettingsError(f"provider {provider.id or provider.type}: remote plaintext HTTP is refused")
    if not local and scheme != "https":
        raise LLMSettingsError(f"provider {provider.id or provider.type}: remote providers require HTTPS")


def validate_settings(settings: LLMSettings) -> None:
    if not settings.providers:
        raise LLMSettingsError("at least one provider must be configured")
    if settings.active_provider not in settings.providers:
        raise LLMSettingsError(f"active provider {settings.active_provider!r} is not configured")
    if not settings.active_model.strip():
        raise LLMSettingsError("active model must not be empty")
    if not (0 <= settings.temperature <= 2):
        raise LLMSettingsError("temperature must be between 0 and 2")
    if settings.max_tokens < 0:
        raise LLMSettingsError("max_tokens must be >= 0")
    if settings.budget.daily_usd < 0:
        raise LLMSettingsError("budget.daily_usd must be >= 0")

    for provider_id, provider in settings.providers.items():
        if not valid_instance_id(provider_id):
            raise LLMSettingsError(f"invalid provider instance id {provider_id!r}")
        if provider.type not in PROVIDER_TYPES:
            raise LLMSettingsError(f"provider {provider_id}: unknown type {provider.type!r}")
        if provider.wire not in WIRE_TYPES:
            raise LLMSettingsError(f"provider {provider_id}: unknown wire {provider.wire!r}")
        if provider.timeout_sec <= 0:
            raise LLMSettingsError(f"provider {provider_id}: timeout_sec must be positive")
        if provider.id and provider.id != provider_id:
            raise LLMSettingsError(f"provider {provider_id}: id mismatch ({provider.id!r})")
        if provider.type == "gemma" and not provider.is_local:
            raise LLMSettingsError("provider gemma must be local")
        _validate_url(provider)

    for role, assignment in settings.roles.items():
        if role not in ROLE_IDS:
            raise LLMSettingsError(f"unknown LLM role {role!r}")
        if assignment.role != role:
            raise LLMSettingsError(f"role {role}: assigned role id must match")
        if not assignment.provider:
            if assignment.model.strip():
                raise LLMSettingsError(f"role {role}: a model requires a provider")
            if assignment.fallback:
                raise LLMSettingsError(f"role {role}: unassigned roles cannot have fallbacks")
            continue
        if assignment.provider not in settings.providers:
            raise LLMSettingsError(f"role {role}: provider {assignment.provider!r} is not configured")
        if not assignment.model.strip():
            raise LLMSettingsError(f"role {role}: model must not be empty")
        if assignment.temperature is not None and not (0 <= assignment.temperature <= 2):
            raise LLMSettingsError(f"role {role}: temperature must be between 0 and 2")
        if assignment.max_tokens is not None and assignment.max_tokens < 0:
            raise LLMSettingsError(f"role {role}: max_tokens must be >= 0")
        for hop in assignment.fallback:
            if hop.provider not in settings.providers:
                raise LLMSettingsError(f"role {role}: fallback provider {hop.provider!r} is not configured")
            if not hop.model.strip():
                raise LLMSettingsError(f"role {role}: fallback model must not be empty")


def _cloud_enabled() -> bool:
    return os.environ.get(CLOUD_ENV, "0").strip() == "1"


class LLMSettingsStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._settings: LLMSettings | None = None
        self._key_sources: dict[str, str] = {}
        self._telemetry: dict[str, dict[str, Any]] = {}

    # ── load/save ────────────────────────────────────────────────
    def _read_row(self) -> tuple[str | None, datetime | None]:
        from sqlmodel import Session
        from database import engine, AppSetting

        with Session(engine) as session:
            row = session.get(AppSetting, KEY)
            return (row.value_json, row.updated_at) if row else (None, None)

    def _write_row(self, value: str) -> None:
        from sqlmodel import Session
        from database import engine, AppSetting

        with Session(engine) as session:
            row = session.get(AppSetting, KEY)
            if row:
                row.value_json = value
                row.updated_at = datetime.utcnow()
            else:
                row = AppSetting(key=KEY, value_json=value, updated_at=datetime.utcnow())
            session.add(row)
            session.commit()

    def _normalize(self, settings: LLMSettings) -> LLMSettings:
        for provider_id, provider in settings.providers.items():
            provider.id = provider_id
        return settings

    def _migrate(self, settings: LLMSettings) -> bool:
        """Upgrade an older document in memory. Returns True when changed."""
        changed = False
        if settings.schema_version > SCHEMA_VERSION:
            raise LLMSettingsError(
                f"settings schema {settings.schema_version} is newer than this build ({SCHEMA_VERSION})"
            )
        if settings.schema_version < SCHEMA_VERSION:
            settings.schema_version = SCHEMA_VERSION
            changed = True
        # v1 documents may have non-Gemma roles pointing at a provider type;
        # those are still valid instance ids and need no rewrite.
        for provider_id, provider in settings.providers.items():
            if not provider.id:
                provider.id = provider_id
                changed = True
        return changed

    def _hydrate(self, raw: str) -> tuple[LLMSettings, bool]:
        data = json.loads(raw)
        settings = LLMSettings.model_validate(data)
        migrated = self._migrate(settings)
        sources: dict[str, str] = {}
        for provider_id, provider in settings.providers.items():
            stored = provider.api_key
            source = secretbox.source(stored)
            if stored:
                try:
                    provider.api_key = secretbox.unseal(stored)
                    sources[provider_id] = source
                except Exception as exc:
                    logger.error("Could not unseal credential for %s: %s", provider_id, exc)
                    provider.api_key = None
                    sources[provider_id] = "invalid"
            else:
                sources[provider_id] = "none"
            provider.has_key = bool(provider.api_key or self._env_key(provider_id, provider.type))
            provider.secret_source = (
                "env" if not provider.api_key and self._env_key(provider_id, provider.type) else sources.get(provider_id, "none")
            )
            provider.key_status = "set" if provider.has_key else ("none" if provider.is_local else "missing")
        self._key_sources = sources
        return self._normalize(settings), migrated

    def _note_source(self, provider_id: str, api_key: str | None) -> None:
        if not api_key:
            self._key_sources[provider_id] = "none"
        else:
            self._key_sources[provider_id] = secretbox.source(secretbox.seal(api_key))

    def _env_key(self, provider_id: str, provider_type: str | None = None) -> str | None:
        lookup = provider_type
        if not lookup and self._settings is not None:
            provider = self._settings.providers.get(provider_id)
            lookup = provider.type if provider else None
        env = provider_env_key(lookup or provider_id)
        return os.environ.get(env, "").strip() if env else None

    def _seed_settings(self) -> LLMSettings:
        try:
            servers = discovery.discover_local_servers()
        except Exception as exc:
            logger.warning("Discovery during first-boot seeding failed: %s", exc)
            servers = []
        catalog = {
            str(server.get("instance_id")): [
                {"id": model, **classify_model(model)} for model in server.get("models", [])
            ]
            for server in servers
        }
        return self._normalize(build_defaults(discovered=servers, catalog=catalog))

    def _load_locked(self) -> LLMSettings:
        if self._settings is not None:
            return self._settings
        raw, _ = self._read_row()
        if raw:
            try:
                settings, migrated = self._hydrate(raw)
                self._settings = settings
                if migrated:
                    self._persist_locked(settings)
                return settings
            except Exception as exc:
                logger.error("Persisted LLM settings are unreadable; using defaults: %s", exc)
        settings = self._seed_settings()
        for provider_id, provider in settings.providers.items():
            provider.has_key = bool(self._env_key(provider_id))
            provider.secret_source = "env" if provider.has_key else "none"
            provider.key_status = "set" if provider.has_key else ("none" if provider.is_local else "missing")
        self._settings = settings
        self._persist_locked(settings)
        return settings

    def _serializable(self, settings: LLMSettings) -> str:
        data = settings.model_dump()
        for provider in data["providers"].values():
            provider["api_key"] = secretbox.seal(provider.get("api_key"))
            provider["has_key"] = False
            provider["clear_key"] = False
        return json.dumps(data, separators=(",", ":"))

    def _persist_locked(self, settings: LLMSettings) -> None:
        self._write_row(self._serializable(settings))

    # ── public API ───────────────────────────────────────────────
    def get(self) -> LLMSettings:
        with self._lock:
            return self._load_locked().model_copy(deep=True)

    def _provider_public(self, provider_id: str, provider: LLMProviderConfig) -> dict[str, Any]:
        env = self._env_key(provider_id)
        provider.id = provider_id
        provider.has_key = bool(provider.api_key or env)
        provider.secret_source = "env" if not provider.api_key and env else self._key_sources.get(provider_id, "none")
        provider.key_status = "set" if provider.has_key else ("none" if provider.is_local else "missing")
        return {
            **provider.model_dump(),
            "api_key": None,
            "clear_key": False,
            "telemetry": self._telemetry.get(provider_id, {}),
        }

    def public(self) -> dict[str, Any]:
        with self._lock:
            settings = self._load_locked().model_copy(deep=True)
            data = settings.model_dump()
            data["providers"] = {
                provider_id: self._provider_public(provider_id, provider)
                for provider_id, provider in settings.providers.items()
            }
            data["role_catalog"] = __import__("services.llm.defaults", fromlist=["ROLE_CATALOG"]).ROLE_CATALOG
            data["cloud_enabled"] = _cloud_enabled()
            data["spend"] = budget.summary()
            data["schema_version"] = SCHEMA_VERSION
            return data

    def _eval_gate(self, current: LLMSettings, merged: LLMSettings) -> None:
        if not merged.budget.eval_gate:
            return
        evidence = merged.last_eval or current.last_eval
        for role, assignment in merged.roles.items():
            if not assignment.assigned:
                continue
            provider = merged.providers.get(assignment.provider)
            if provider is None or provider.is_local:
                continue
            row = (evidence.roles.get(role) if evidence else None) or {}
            passed = (
                row.get("provider") == assignment.provider
                and row.get("model") == assignment.model
                and float(row.get("pass_rate") or 0.0) >= 0.8
            )
            if not passed:
                raise LLMSettingsError(
                    f"eval gate: role {role} is assigned to cloud model "
                    f"{assignment.provider}/{assignment.model}, which has no passing local eval. "
                    "Run the role eval and save before enabling cloud routing."
                )

    def save(self, update: LLMSettings, expected_revision: int | None = None) -> dict[str, Any]:
        with self._lock:
            current = self._load_locked()
            if expected_revision is not None and expected_revision != current.revision:
                raise LLMRevisionConflict(expected_revision, current.revision)

            merged = update.model_copy(deep=True)
            merged.revision = current.revision + 1
            merged.schema_version = SCHEMA_VERSION
            # Eval evidence is server-owned; a client cannot forge a pass to
            # unlock cloud routing.
            merged.last_eval = current.last_eval
            self._normalize(merged)

            for provider_id, provider in merged.providers.items():
                existing = current.providers.get(provider_id)
                if provider.clear_key:
                    provider.api_key = None
                elif provider.api_key:
                    pass
                elif existing:
                    provider.api_key = existing.api_key
                provider.has_key = bool(provider.api_key or self._env_key(provider_id))
                self._note_source(provider_id, provider.api_key)

            validate_settings(merged)
            self._eval_gate(current, merged)
            self._settings = merged
            self._persist_locked(merged)
            logger.info(
                "LLM settings revision %s saved (active=%s/%s, providers=%s)",
                merged.revision, merged.active_provider, merged.active_model,
                ",".join(merged.providers),
            )
            return self.public()

    def patch_air_gap(self, value: bool) -> dict[str, Any]:
        with self._lock:
            current = self._load_locked().model_copy(deep=True)
            current.air_gapped = bool(value)
            current.revision += 1
            self._settings = current
            self._persist_locked(current)
            return self.public()

    def set_budget(self, daily_usd: float, eval_gate: bool) -> dict[str, Any]:
        with self._lock:
            current = self._load_locked().model_copy(deep=True)
            current.budget.daily_usd = max(0.0, float(daily_usd))
            current.budget.eval_gate = bool(eval_gate)
            current.revision += 1
            self._settings = current
            self._persist_locked(current)
            return self.public()

    def set_last_eval(self, summary: dict[str, Any]) -> None:
        from .models import LLMEvalSummary

        with self._lock:
            current = self._load_locked().model_copy(deep=True)
            current.last_eval = LLMEvalSummary.model_validate(summary)
            self._settings = current
            self._persist_locked(current)

    def clear_key(self, provider_id: str) -> dict[str, Any]:
        with self._lock:
            current = self._load_locked().model_copy(deep=True)
            provider = current.providers.get(provider_id)
            if not provider:
                raise LLMSettingsError(f"unknown provider {provider_id!r}")
            provider.api_key = None
            provider.clear_key = False
            self._note_source(provider_id, None)
            current.revision += 1
            self._settings = current
            self._persist_locked(current)
            return self.public()

    def add_instance(self, server: dict[str, Any]) -> dict[str, Any]:
        """Create or refresh one discovered provider instance."""
        with self._lock:
            current = self._load_locked().model_copy(deep=True)
            provider_type = str(server.get("provider") or "openai_compatible")
            base_url = str(server.get("base_url") or "")
            models = [str(m) for m in (server.get("models") or []) if m]
            chat = [m for m in models if classify_model(m)["chat"]]
            pid = str(server.get("instance_id") or instance_id(provider_type, base_url))
            existing = current.providers.get(pid)
            if existing is None:
                for other_id, other in current.providers.items():
                    if normalize_base_url(other.base_url) == normalize_base_url(base_url) and other.type == provider_type:
                        pid, existing = other_id, other
                        break
            if existing is None:
                current.providers[pid] = LLMProviderConfig(
                    id=pid,
                    type=provider_type,
                    name=str(server.get("name") or pid),
                    base_url=base_url,
                    default_model=pick_default_model(
                        [{"id": model, **classify_model(model)} for model in models]
                    ) or (chat[0] if chat else (models[0] if models else "")),
                    wire=str(server.get("wire") or ("ollama_native" if provider_type == "ollama" else "openai_chat")),
                    enabled=True,
                    is_local=True,
                    keep_alive="0",
                )
            else:
                existing.base_url = base_url or existing.base_url
                existing.name = existing.name or str(server.get("name") or pid)
                existing.enabled = True
                if not existing.default_model and (chat or models):
                    existing.default_model = pick_default_model(
                        [{"id": model, **classify_model(model)} for model in models]
                    ) or (chat[0] if chat else models[0])
            if not current.active_model and current.providers[pid].default_model:
                current.active_provider = pid
                current.active_model = current.providers[pid].default_model
            current.revision += 1
            validate_settings(current)
            self._settings = current
            self._persist_locked(current)
            return self.public()

    def remove_instance(self, provider_id: str) -> dict[str, Any]:
        with self._lock:
            current = self._load_locked().model_copy(deep=True)
            if provider_id == "gemma":
                raise LLMSettingsError("the Gemma provider is built in and cannot be removed")
            if provider_id not in current.providers:
                raise LLMSettingsError(f"unknown provider {provider_id!r}")
            in_use = [
                role for role, assignment in current.roles.items()
                if assignment.provider == provider_id
                or any(hop.provider == provider_id for hop in assignment.fallback)
            ]
            if in_use:
                raise LLMSettingsError(
                    f"provider {provider_id} is used by role(s) {', '.join(sorted(in_use))}; unassign them first"
                )
            del current.providers[provider_id]
            if current.active_provider == provider_id or not current.active_model:
                candidates = [
                    (pid, provider) for pid, provider in current.providers.items()
                    if provider.default_model
                ]
                if candidates:
                    pid, provider = candidates[0]
                    current.active_provider = pid
                    current.active_model = provider.default_model
                else:
                    current.active_provider = "gemma"
                    current.active_model = "ltx-gemma3"
            current.revision += 1
            self._settings = current
            self._persist_locked(current)
            return self.public()

    def provider_key(self, provider_id: str) -> str | None:
        with self._lock:
            settings = self._load_locked()
            provider = settings.providers.get(provider_id)
            if not provider:
                return None
            return provider.api_key or self._env_key(provider_id)

    def role_assignment(self, role: str) -> LLMRoleAssignment | None:
        with self._lock:
            settings = self._load_locked()
            return settings.roles.get(role)

    def role_provider(self, role: str) -> str | None:
        assignment = self.role_assignment(role)
        if assignment is not None:
            return assignment.provider or None
        settings = self.get()
        return settings.active_provider

    def plans(self, role: str) -> list[dict[str, Any]]:
        """Return usable primary+fallback hops, or raise with actionable reasons."""
        with self._lock:
            settings = self._load_locked()
            assignment = settings.roles.get(role)
            if assignment is not None:
                if not assignment.assigned:
                    raise LLMSettingsError(
                        f"role {role}: unassigned. Assign a provider and model in Studio → AI & LLM."
                    )
                hops = [{"provider": assignment.provider, "model": assignment.model}]
                hops.extend({"provider": hop.provider, "model": hop.model} for hop in assignment.fallback)
                temperature = assignment.temperature if assignment.temperature is not None else settings.temperature
                top_p = assignment.top_p
                seed = assignment.seed
                max_tokens = assignment.max_tokens if assignment.max_tokens is not None else settings.max_tokens
                reasoning = assignment.reasoning_effort
            else:
                hops = [{"provider": settings.active_provider, "model": settings.active_model}]
                temperature = settings.temperature
                top_p = None
                seed = None
                max_tokens = settings.max_tokens
                reasoning = None

            usable: list[dict[str, Any]] = []
            reasons: list[str] = []
            for hop in hops:
                provider_id = hop["provider"]
                provider = settings.providers.get(provider_id)
                if not provider:
                    reasons.append(f"{provider_id}: not configured")
                    continue
                if not provider.enabled:
                    reasons.append(f"{provider_id}: disabled")
                    continue
                if not hop["model"].strip():
                    reasons.append(f"{provider_id}: no model")
                    continue
                if settings.air_gapped and not provider.is_local:
                    reasons.append(f"{provider_id}: blocked by air-gap")
                    continue
                if not provider.is_local and not _cloud_enabled():
                    reasons.append(f"{provider_id}: cloud is disabled (set {CLOUD_ENV}=1)")
                    continue
                usable.append({
                    **hop,
                    "provider_type": provider.type,
                    "wire": provider.wire,
                    "base_url": provider.base_url,
                    "timeout_sec": provider.timeout_sec,
                    "keep_alive": provider.keep_alive,
                    "is_local": provider.is_local,
                    "api_key": provider.api_key or self._env_key(provider_id),
                })
            if not usable:
                raise LLMSettingsError(f"role {role}: no usable provider ({'; '.join(reasons) or 'unassigned'})")
            return [{
                **hop,
                "role": role,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
                "max_tokens": max_tokens,
                "reasoning_effort": reasoning,
            } for hop in usable]

    # ── telemetry ────────────────────────────────────────────────
    def record_load(self, provider_id: str, load_ms: int) -> None:
        with self._lock:
            row = self._telemetry.setdefault(provider_id, {})
            row["last_load_ms"] = int(load_ms)

    def record_success(
        self,
        provider_id: str,
        latency_ms: int | None = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        with self._lock:
            settings = self._load_locked()
            provider = settings.providers.get(provider_id)
            if provider:
                provider.last_success_at = datetime.now(timezone.utc).isoformat()
                provider.last_fault = None
                provider.key_status = "valid" if provider.has_key or provider.is_local else "set"
            row = self._telemetry.setdefault(provider_id, {})
            row["calls"] = int(row.get("calls") or 0) + 1
            row["last_success_at"] = datetime.now(timezone.utc).isoformat()
            if latency_ms is not None:
                row["last_latency_ms"] = int(latency_ms)
                completion = int((usage or {}).get("completion_tokens") or 0)
                if completion and latency_ms > 0:
                    row["last_tokens_per_sec"] = round(completion / (latency_ms / 1000.0), 2)
            if usage:
                row["last_usage"] = dict(usage)

    def record_failure(self, provider_id: str, error: str) -> None:
        with self._lock:
            settings = self._load_locked()
            provider = settings.providers.get(provider_id)
            if provider:
                provider.last_fault = error[:300]
                if not provider.is_local:
                    provider.key_status = "invalid"
            row = self._telemetry.setdefault(provider_id, {})
            row["failures"] = int(row.get("failures") or 0) + 1
            row["last_fault"] = error[:300]

    def telemetry(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {key: dict(value) for key, value in self._telemetry.items()}

    def local_instances(self) -> list[tuple[str, LLMProviderConfig]]:
        with self._lock:
            settings = self._load_locked()
            return [
                (provider_id, provider.model_copy(deep=True))
                for provider_id, provider in settings.providers.items()
                if provider.is_local and provider.type != "gemma" and provider.enabled
            ]


llm_settings = LLMSettingsStore()
