"""Cloud spend guard and ledger.

Local providers are never budgeted. Cloud calls refuse unless the operator has
enabled cloud, set a positive daily budget, and the model has a known rate.
Unknown prices fail closed: an unmetered paid call is worse than a refusal.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any

from .models import LLMSettings

logger = logging.getLogger("llm.budget")

SPEND_KEY = "llm_spend"
RATE_ENV = "MILIMO_LLM_RATE_CARD"
MAX_ENTRIES = 200

# Conservative reference rates in USD per 1M tokens. Override with a JSON file
# at MILIMO_LLM_RATE_CARD: {"model-prefix": [input_per_mtok, output_per_mtok]}.
DEFAULT_RATES: dict[str, tuple[float, float]] = {
    "gpt-6.1": (5.0, 20.0),
    "gpt-6": (5.0, 15.0),
    "gpt-5": (5.0, 15.0),
    "o4": (3.0, 12.0),
    "o3": (3.0, 12.0),
    "claude-4": (5.0, 25.0),
    "claude-3-7": (3.0, 15.0),
    "claude-3-5": (3.0, 15.0),
    "claude-3-opus": (15.0, 75.0),
}

_lock = threading.RLock()
_memory_ledger: dict[str, Any] | None = None


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _rates() -> dict[str, tuple[float, float]]:
    rates = dict(DEFAULT_RATES)
    path = os.environ.get(RATE_ENV, "").strip()
    if not path:
        return rates
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        for key, value in raw.items():
            if isinstance(value, list) and len(value) == 2:
                rates[str(key)] = (float(value[0]), float(value[1]))
    except Exception as exc:
        logger.warning("Could not read %s: %s", RATE_ENV, exc)
    return rates


def rates() -> dict[str, tuple[float, float]]:
    return _rates()


def rate_for(model: str) -> tuple[float, float] | None:
    blob = (model or "").lower()
    for prefix in sorted(_rates(), key=len, reverse=True):
        if blob.startswith(prefix.lower()) or prefix.lower() in blob:
            return _rates()[prefix]
    return None


def _read_ledger() -> dict[str, Any]:
    global _memory_ledger
    try:
        from sqlmodel import Session
        from database import engine, AppSetting

        with Session(engine) as session:
            row = session.get(AppSetting, SPEND_KEY)
            if row and row.value_json:
                data = json.loads(row.value_json)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return _memory_ledger or {"date": _today(), "total_usd": 0.0, "entries": []}


def _write_ledger(data: dict[str, Any]) -> None:
    global _memory_ledger
    _memory_ledger = data
    try:
        from sqlmodel import Session
        from database import engine, AppSetting
        from datetime import datetime as _dt

        blob = json.dumps(data, separators=(",", ":"))
        with Session(engine) as session:
            row = session.get(AppSetting, SPEND_KEY)
            if row:
                row.value_json = blob
                row.updated_at = _dt.utcnow()
            else:
                row = AppSetting(key=SPEND_KEY, value_json=blob, updated_at=_dt.utcnow())
            session.add(row)
            session.commit()
    except Exception as exc:
        logger.warning("Could not persist LLM spend ledger: %s", exc)


def summary() -> dict[str, Any]:
    with _lock:
        data = _read_ledger()
        if data.get("date") != _today():
            data = {"date": _today(), "total_usd": 0.0, "entries": []}
        return {
            "date": data.get("date"),
            "total_usd": round(float(data.get("total_usd") or 0.0), 6),
            "entries": list(data.get("entries") or [])[-20:],
        }


def spend_today() -> float:
    return float(summary()["total_usd"])


def estimate(model: str, prompt_tokens: int, max_tokens: int) -> float | None:
    rate = rate_for(model)
    if rate is None:
        return None
    input_cost = (max(prompt_tokens, 0) / 1_000_000) * rate[0]
    output_cost = (max(max_tokens, 0) / 1_000_000) * rate[1]
    return round(input_cost + output_cost, 8)


def precheck(
    settings: LLMSettings,
    *,
    provider: str,
    is_local: bool,
    model: str,
    prompt_tokens: int,
    max_tokens: int,
) -> float:
    """Raise with a precise reason when a cloud call must not be attempted."""
    if is_local:
        return 0.0
    budget = float(settings.budget.daily_usd or 0.0)
    if budget <= 0:
        raise ValueError(
            f"{provider}: cloud budget is 0. Set a daily budget in Studio → AI & LLM "
            "or keep this role local."
        )
    estimated = estimate(model, prompt_tokens, max_tokens)
    if estimated is None:
        raise ValueError(
            f"{provider}: no price for model {model!r}. Add it to MILIMO_LLM_RATE_CARD "
            "or use a local model."
        )
    spent = spend_today()
    if spent + estimated > budget:
        raise ValueError(
            f"{provider}: daily cloud budget ${budget:.2f} would be exceeded "
            f"(${spent:.4f} spent, ${estimated:.4f} estimated). Use a local role or raise the budget."
        )
    return estimated


def record(
    *,
    provider: str,
    is_local: bool,
    model: str,
    usage: dict[str, int] | None,
    estimated_prompt_tokens: int = 0,
) -> None:
    if is_local:
        return
    usage = usage or {}
    prompt_tokens = int(usage.get("prompt_tokens") or estimated_prompt_tokens or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    cost = estimate(model, prompt_tokens, completion_tokens) or 0.0
    with _lock:
        data = _read_ledger()
        if data.get("date") != _today():
            data = {"date": _today(), "total_usd": 0.0, "entries": []}
        data["total_usd"] = round(float(data.get("total_usd") or 0.0) + cost, 8)
        entries = list(data.get("entries") or [])
        entries.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "usd": round(cost, 8),
        })
        data["entries"] = entries[-MAX_ENTRIES:]
        _write_ledger(data)
