"""Persisted ComfyUI connection settings.

Environment variables remain the boot defaults. The UI can override URL, enabled
state, declared residency, and an optional bearer token in the local SQLite
`app_setting` row. Tokens are never returned by the read API.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import Any

import config

KEY = "comfy"
_SCHEMA_VERSION = 1
_lock = threading.RLock()


class ComfySettingsError(ValueError):
    pass


class ComfyRevisionConflict(ComfySettingsError):
    def __init__(self, expected: int, current: int):
        super().__init__(f"revision {expected} does not match current {current}")
        self.expected = expected
        self.current = current


def _defaults() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "revision": 0,
        "url": config.COMFY_URL,
        "enabled": True,
        "declared_gb": float(config.COMFY_DECLARED_GB),
        "token": "",
    }


def _read() -> dict[str, Any]:
    from sqlmodel import Session

    from database import AppSetting, engine

    row = None
    try:
        with Session(engine) as session:
            row = session.get(AppSetting, KEY)
    except Exception:
        row = None
    defaults = _defaults()
    if not row:
        return defaults
    try:
        data = json.loads(row.value_json or "{}")
    except json.JSONDecodeError:
        return defaults
    merged = {**defaults, **data}
    merged["schema_version"] = _SCHEMA_VERSION
    return merged


def _write(data: dict[str, Any]) -> None:
    from sqlmodel import Session

    from database import AppSetting, engine

    with Session(engine) as session:
        row = session.get(AppSetting, KEY)
        payload = json.dumps(data)
        if row:
            row.value_json = payload
            row.updated_at = datetime.utcnow()
        else:
            row = AppSetting(key=KEY, value_json=payload, updated_at=datetime.utcnow())
        session.add(row)
        session.commit()


def _validate(data: dict[str, Any]) -> dict[str, Any]:
    url = str(data.get("url") or "").strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise ComfySettingsError("ComfyUI URL must start with http:// or https://")
    enabled = bool(data.get("enabled", True))
    try:
        declared = float(data.get("declared_gb", config.COMFY_DECLARED_GB))
    except (TypeError, ValueError) as exc:
        raise ComfySettingsError("declared_gb must be a number") from exc
    if not 0 <= declared <= 256:
        raise ComfySettingsError("declared_gb must be between 0 and 256")
    token = str(data.get("token") or "").strip()
    return {"url": url, "enabled": enabled, "declared_gb": declared, "token": token}


class ComfySettingsStore:
    def get(self) -> dict[str, Any]:
        with _lock:
            return _read()

    def public(self) -> dict[str, Any]:
        data = self.get()
        return {
            "schema_version": data.get("schema_version", _SCHEMA_VERSION),
            "revision": int(data.get("revision") or 0),
            "url": data.get("url"),
            "enabled": bool(data.get("enabled", True)),
            "declared_gb": float(data.get("declared_gb") or 0),
            "has_token": bool(data.get("token")),
            "hard_disabled": not config.COMFY_ENABLED,
        }

    def save(self, payload: dict[str, Any], expected_revision: int | None = None) -> dict[str, Any]:
        with _lock:
            current = _read()
            if expected_revision is not None and int(expected_revision) != int(current.get("revision") or 0):
                raise ComfyRevisionConflict(int(expected_revision), int(current.get("revision") or 0))
            clean = _validate({**current, **payload})
            next_data = {
                **current,
                **clean,
                "schema_version": _SCHEMA_VERSION,
                "revision": int(current.get("revision") or 0) + 1,
            }
            _write(next_data)
            return self.public()

    def effective(self) -> dict[str, Any]:
        data = self.get()
        return {
            **data,
            "enabled": bool(config.COMFY_ENABLED and data.get("enabled", True)),
        }

    def effective_values(self) -> dict[str, Any]:
        data = self.effective()
        return {
            "url": data["url"],
            "enabled": bool(data["enabled"]),
            "declared_gb": float(data["declared_gb"]),
            "token": data.get("token") or "",
        }


comfy_settings = ComfySettingsStore()
