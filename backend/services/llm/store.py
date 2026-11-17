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

from . import secretbox
from .defaults import default_settings, provider_env_key
from .models import LLMProviderConfig, LLMRoleAssignment, LLMSettings, WIRE_TYPES, PROVIDER_TYPES, ROLE_IDS

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


def _is_local_host(host: str) -> bool:
    h = (host or "").strip().lower().rstrip(".")
    if h in {"localhost", "host.docker.internal"} or h.endswith(".localhost") or h.endswith(".local"):
        return True
    try:
        addr = ipaddress.ip_address(h)
        return addr.is_loopback or addr.is_private or addr.is_link_local
    except ValueError:
        return False


def _validate_url(provider: LLMProviderConfig) -> None:
    if provider.type == "gemma":
        return
    url = (provider.base_url or "").strip()
    if not url:
        raise LLMSettingsError(f"provider {provider.type}: base URL is required")
    parsed = urlparse(url)
    if parsed.username or parsed.password:
        raise LLMSettingsError(f"provider {provider.type}: credentials must not be embedded in the URL")
    if parsed.query or parsed.fragment:
        raise LLMSettingsError(f"provider {provider.type}: base URL must not contain query or fragment")
    scheme = (parsed.scheme or "").lower()
    host = parsed.hostname or ""
    if scheme not in {"http", "https"}:
        raise LLMSettingsError(f"provider {provider.type}: only http/https endpoints are supported")
    local = _is_local_host(host)
    if scheme == "http" and not local:
        raise LLMSettingsError(f"provider {provider.type}: remote plaintext HTTP is refused")
    if not local and scheme != "https":
        raise LLMSettingsError(f"provider {provider.type}: remote providers require HTTPS")


def validate_settings(settings: LLMSettings) -> None:
    if settings.active_provider not in settings.providers:
        raise LLMSettingsError(f"active provider {settings.active_provider!r} is not configured")
    if settings.active_provider not in PROVIDER_TYPES:
        raise LLMSettingsError(f"unknown provider type {settings.active_provider!r}")
    if not settings.active_model.strip():
        raise LLMSettingsError("active model must not be empty")
    if not (0 <= settings.temperature <= 2):
        raise LLMSettingsError("temperature must be between 0 and 2")
    if settings.max_tokens < 0:
        raise LLMSettingsError("max_tokens must be >= 0")

    for provider_id, provider in settings.providers.items():
        if provider_id != provider.type or provider_id not in PROVIDER_TYPES:
            raise LLMSettingsError(f"unknown provider {provider_id!r}")
        if provider.wire not in WIRE_TYPES:
            raise LLMSettingsError(f"provider {provider_id}: unknown wire {provider.wire!r}")
        if provider.timeout_sec <= 0:
            raise LLMSettingsError(f"provider {provider_id}: timeout_sec must be positive")
        _validate_url(provider)

    for role, assignment in settings.roles.items():
        if role not in ROLE_IDS:
            raise LLMSettingsError(f"unknown LLM role {role!r}")
        if assignment.role != role:
            raise LLMSettingsError(f"role {role}: assigned role id must match")
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


def _cloud_enabled() -> bool:
    return os.environ.get(CLOUD_ENV, "0").strip() == "1"


class LLMSettingsStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._settings: LLMSettings | None = None
        self._key_sources: dict[str, str] = {}

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

    def _hydrate(self, raw: str) -> LLMSettings:
        data = json.loads(raw)
        settings = LLMSettings.model_validate(data)
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
            provider.has_key = bool(provider.api_key or self._env_key(provider_id))
            provider.secret_source = (
                "env" if not provider.api_key and self._env_key(provider_id) else sources.get(provider_id, "none")
            )
            provider.key_status = "set" if provider.has_key else ("none" if provider.is_local else "missing")
        self._key_sources = sources
        return settings

    def _note_source(self, provider_id: str, api_key: str | None) -> None:
        """Track where the stored credential lives for redacted reads."""
        if not api_key:
            self._key_sources[provider_id] = "none"
        else:
            self._key_sources[provider_id] = secretbox.source(secretbox.seal(api_key))

    def _env_key(self, provider_id: str) -> str | None:
        env = provider_env_key(provider_id)
        return os.environ.get(env, "").strip() if env else None

    def _load_locked(self) -> LLMSettings:
        if self._settings is not None:
            return self._settings
        raw, _ = self._read_row()
        if raw:
            try:
                self._settings = self._hydrate(raw)
                return self._settings
            except Exception as exc:
                logger.error("Persisted LLM settings are unreadable; using defaults: %s", exc)
        settings = default_settings()
        # Seed status fields before the first save.
        for provider_id, provider in settings.providers.items():
            provider.has_key = bool(self._env_key(provider_id))
            provider.secret_source = "env" if provider.has_key else "none"
            provider.key_status = "set" if provider.has_key else ("none" if provider.is_local else "missing")
        self._settings = settings
        self._persist_locked(settings)
        return settings

    def _serializable(self, settings: LLMSettings) -> str:
        data = settings.model_dump()
        for provider_id, provider in data["providers"].items():
            key = provider.get("api_key")
            provider["api_key"] = secretbox.seal(key)
            provider["has_key"] = False
            provider["clear_key"] = False
        return json.dumps(data, separators=(",", ":"))

    def _persist_locked(self, settings: LLMSettings) -> None:
        self._write_row(self._serializable(settings))

    # ── public API ───────────────────────────────────────────────
    def get(self) -> LLMSettings:
        with self._lock:
            return self._load_locked().model_copy(deep=True)

    def public(self) -> dict[str, Any]:
        with self._lock:
            settings = self._load_locked().model_copy(deep=True)
            for provider_id, provider in settings.providers.items():
                env = self._env_key(provider_id)
                provider.has_key = bool(provider.api_key or env)
                provider.secret_source = "env" if not provider.api_key and env else self._key_sources.get(provider_id, "none")
                provider.key_status = "set" if provider.has_key else ("none" if provider.is_local else "missing")
                provider.api_key = None
                provider.clear_key = False
            data = settings.model_dump()
            data["role_catalog"] = __import__("services.llm.defaults", fromlist=["ROLE_CATALOG"]).ROLE_CATALOG
            data["cloud_enabled"] = _cloud_enabled()
            return data

    def save(self, update: LLMSettings, expected_revision: int | None = None) -> dict[str, Any]:
        with self._lock:
            current = self._load_locked()
            if expected_revision is not None and expected_revision != current.revision:
                raise LLMRevisionConflict(expected_revision, current.revision)

            merged = update.model_copy(deep=True)
            merged.revision = current.revision + 1
            merged.schema_version = current.schema_version

            # Preserve or rotate credentials explicitly. Blank input never
            # erases a stored key; clear_key does.
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
            self._settings = merged
            self._persist_locked(merged)
            logger.info("LLM settings revision %s saved (active=%s/%s)", merged.revision, merged.active_provider, merged.active_model)
            return self.public()

    def patch_air_gap(self, value: bool) -> dict[str, Any]:
        with self._lock:
            current = self._load_locked().model_copy(deep=True)
            current.air_gapped = bool(value)
            current.revision += 1
            self._settings = current
            self._persist_locked(current)
            return self.public()

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
        if assignment:
            return assignment.provider
        settings = self.get()
        return settings.active_provider

    def plans(self, role: str) -> list[dict[str, Any]]:
        """Return usable primary+fallback hops, or raise with actionable reasons."""
        with self._lock:
            settings = self._load_locked()
            assignment = settings.roles.get(role)
            if assignment:
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

    def record_success(self, provider_id: str) -> None:
        with self._lock:
            settings = self._load_locked()
            provider = settings.providers.get(provider_id)
            if provider:
                provider.last_success_at = datetime.now(timezone.utc).isoformat()
                provider.last_fault = None
                provider.key_status = "valid" if provider.has_key or provider.is_local else "set"

    def record_failure(self, provider_id: str, error: str) -> None:
        with self._lock:
            settings = self._load_locked()
            provider = settings.providers.get(provider_id)
            if provider:
                provider.last_fault = error[:300]
                if not provider.is_local:
                    provider.key_status = "invalid"


llm_settings = LLMSettingsStore()
