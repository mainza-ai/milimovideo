"""Local role evaluation harness.

Measures the contracts the studio relies on per role: wrapper-free prompts,
action-shaped video rewrites, and parseable scene lists. Results are recorded in
the settings document and can gate cloud routing.
"""

from __future__ import annotations

import logging
import statistics
import time
from datetime import datetime, timezone
from typing import Any

from . import runtime
from .fixtures import DEFAULT_EVAL_ROLES, FIXTURES, FIXTURE_VERSION
from .prompts import IMAGE_SYSTEM_PROMPT, get_video_system_prompt
from .store import LLMSettingsError, llm_settings

logger = logging.getLogger("llm.eval")

MAX_TOKENS = {"prompt_video": 300, "prompt_image": 200, "script_parse": 900}


def _is_wrapper(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    if stripped.startswith("{") or stripped.startswith("["):
        return True
    for marker in ('"enhanced_prompt"', '"prompt":', '"image_prompt"'):
        if marker in stripped:
            return True
    return False


def _video_ok(text: str) -> tuple[bool, str]:
    if _is_wrapper(text):
        return False, "returned a JSON wrapper"
    if len(text) < 30:
        return False, f"too short ({len(text)} chars)"
    if len(text) > 1500:
        return False, f"too long ({len(text)} chars)"
    return True, ""


def _image_ok(text: str) -> tuple[bool, str]:
    if _is_wrapper(text):
        return False, "returned a JSON wrapper"
    if len(text) < 20:
        return False, f"too short ({len(text)} chars)"
    if len(text) > 1200:
        return False, f"too long ({len(text)} chars)"
    return True, ""


def _script_ok(text: str) -> tuple[bool, str]:
    from services.ai_storyboard import _extract_json, _validate_scenes

    try:
        parsed = _extract_json(text)
    except Exception as exc:
        return False, f"unparseable JSON: {exc}"
    if not isinstance(parsed, list) or not parsed:
        return False, "no scene list"
    try:
        scenes = _validate_scenes(parsed)
    except Exception as exc:
        return False, f"invalid scenes: {exc}"
    shots = sum(len(scene.get("shots") or []) for scene in scenes)
    if not scenes or not shots:
        return False, "no shots"
    return True, ""


def _run_case(role: str, fixture: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    try:
        if role == "prompt_video":
            text, _, result = runtime.enhance(
                role,
                fixture["prompt"],
                system=get_video_system_prompt(4.8),
                max_tokens=MAX_TOKENS[role],
                json_mode=True,
            )
            ok, reason = _video_ok(text)
        elif role == "prompt_image":
            text, _, result = runtime.enhance(
                role,
                fixture["prompt"],
                system=IMAGE_SYSTEM_PROMPT,
                max_tokens=MAX_TOKENS[role],
                json_mode=False,
            )
            ok, reason = _image_ok(text)
        else:
            result = runtime.complete(
                role,
                f"Break this into a storyboard:\n\n{fixture['text']}",
                system=_script_system(),
                max_tokens=MAX_TOKENS[role],
                json_mode=True,
                seed=42,
            )
            text = result.text
            ok, reason = _script_ok(text)
    except Exception as exc:
        elapsed = int((time.monotonic() - started) * 1000)
        return {
            "role": role,
            "fixture": fixture["id"],
            "ok": False,
            "ms": elapsed,
            "reason": str(getattr(exc, "message", exc))[:240],
            "provider": None,
            "model": None,
            "sample": None,
        }
    elapsed = int((time.monotonic() - started) * 1000)
    return {
        "role": role,
        "fixture": fixture["id"],
        "ok": ok,
        "ms": elapsed,
        "reason": reason or None,
        "provider": result.provider if result else None,
        "model": result.model if result else None,
        "sample": (text or "")[:240],
    }


def _script_system() -> str:
    from services.ai_storyboard import AI_STORYBOARD_SYSTEM_PROMPT

    return AI_STORYBOARD_SYSTEM_PROMPT


def run_eval(roles: list[str] | None = None, limit: int = 2) -> dict[str, Any]:
    """Run fixtures for the requested roles through the saved role plan."""
    requested = roles or DEFAULT_EVAL_ROLES
    limit = max(1, min(int(limit or 2), 4))
    settings = llm_settings.get()
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for role in requested:
        fixtures = FIXTURES.get(role)
        if not fixtures:
            skipped.append({"role": role, "reason": "no fixtures for this role"})
            continue
        resolution = runtime.resolve(role)
        if not resolution["usable"]:
            skipped.append({"role": role, "reason": resolution["reason"]})
            continue
        for fixture in fixtures[:limit]:
            row = _run_case(role, fixture)
            results.append(row)
            logger.info(
                "eval %s/%s -> %s (%sms)", role, fixture["id"], "pass" if row["ok"] else "fail", row["ms"],
            )

    by_role: dict[str, dict[str, Any]] = {}
    for role in requested:
        rows = [row for row in results if row["role"] == role]
        if not rows:
            continue
        passed = [row for row in rows if row["ok"]]
        p50 = int(statistics.median([row["ms"] for row in rows])) if rows else 0
        first = rows[0]
        by_role[role] = {
            "provider": first.get("provider"),
            "model": first.get("model"),
            "cases": len(rows),
            "passed": len(passed),
            "pass_rate": round(len(passed) / len(rows), 3),
            "p50_ms": p50,
        }

    total = len(results)
    passed_total = len([row for row in results if row["ok"]])
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "revision": settings.revision,
        "fixture_version": FIXTURE_VERSION,
        "cases": total,
        "pass_rate": round(passed_total / total, 3) if total else 0.0,
        "p50_ms": int(statistics.median([row["ms"] for row in results])) if results else 0,
        "roles": by_role,
    }
    if results:
        try:
            llm_settings.set_last_eval(summary)
        except Exception as exc:
            logger.warning("Could not persist eval summary: %s", exc)

    return {
        "summary": summary,
        "results": results,
        "skipped": skipped,
        "fixture_version": FIXTURE_VERSION,
    }


def recommended_plan(summary: dict[str, Any] | None = None) -> dict[str, Any]:
    """Role -> provider/model for the best passing local model on record."""
    evidence = summary or (llm_settings.get().last_eval.model_dump() if llm_settings.get().last_eval else None)
    if not evidence:
        raise LLMSettingsError("no eval on record; run the local eval first")
    settings = llm_settings.get()
    roles: dict[str, dict[str, str]] = {}
    for role, row in (evidence.get("roles") or {}).items():
        if float(row.get("pass_rate") or 0) < 0.5:
            continue
        provider_id = row.get("provider")
        model = row.get("model")
        provider = settings.providers.get(provider_id or "")
        if not provider or not model:
            continue
        roles[role] = {"provider": provider_id, "model": model}
    if not roles:
        raise LLMSettingsError("eval has no passing role to recommend; fix the role plan and rerun")
    return {"roles": roles, "source": evidence}
