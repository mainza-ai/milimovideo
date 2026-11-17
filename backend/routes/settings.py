"""LLM configuration plane: persisted settings, discovery, evaluation.

Local-first: the default plan runs entirely on loopback. Cloud providers stay
disabled unless `MILIMO_LLM_CLOUD=1`, the air-gap switch is off, a positive
daily budget exists, the model has a known rate, and (optionally) a passing eval
is on record. Reads are redacted; writes are revision-checked whole-document
updates.
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from services.llm import budget, discovery, eval as eval_engine, profiles, runtime
from services.llm.instances import instance_id, normalize_base_url
from services.llm.models import (
    LLMDiscoverAttach,
    LLMEvalRequest,
    LLMSettings,
    LLMTestRequest,
    LLMVerifyResult,
)
from services.llm.store import LLMSettingsError, LLMRevisionConflict, llm_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("/llm")
def get_llm_settings():
    """Redacted LLM settings snapshot. Keys are never returned."""
    return llm_settings.public()


@router.put("/llm")
def update_llm_settings(update: LLMSettings, expected_revision: Optional[int] = None):
    """Whole-document update guarded by the revision the client read."""
    try:
        snapshot = llm_settings.save(update, expected_revision=expected_revision)
    except LLMRevisionConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "expected_revision": exc.expected, "revision": exc.current,
                    "settings": llm_settings.public()},
        )
    except LLMSettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    snapshot["warnings"] = runtime.cap_missing_models()
    logger.info(
        "LLM settings saved: revision=%s active=%s/%s providers=%s",
        snapshot["revision"], snapshot["active_provider"], snapshot["active_model"],
        ",".join(snapshot["providers"]),
    )
    return snapshot


@router.patch("/llm")
def patch_llm_settings(body: dict):
    """Field-level update. Only the air-gap switch is patched independently."""
    if set(body) - {"air_gapped"}:
        raise HTTPException(400, detail="only 'air_gapped' can be patched; use PUT for the full document")
    if "air_gapped" not in body:
        raise HTTPException(400, detail="air_gapped is required")
    logger.info("LLM air-gap set to %s", bool(body["air_gapped"]))
    return llm_settings.patch_air_gap(bool(body["air_gapped"]))


@router.post("/llm/budget")
def set_llm_budget(body: dict):
    if set(body) - {"daily_usd", "eval_gate"}:
        raise HTTPException(400, detail="only daily_usd and eval_gate can be set here")
    try:
        return llm_settings.set_budget(
            float(body.get("daily_usd", 0.0)),
            bool(body.get("eval_gate", False)),
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, detail=str(exc))


@router.get("/llm/models")
def list_llm_models(
    provider: Optional[str] = Query(default=None),
    deep: bool = Query(default=False),
):
    """Model catalog with context, vision, and chat flags."""
    provider_id = provider or llm_settings.role_provider("prompt_video")
    if not provider_id:
        raise HTTPException(400, detail="no provider selected and no role is assigned")
    try:
        models = runtime.provider_models(provider_id, deep=deep)
    except LLMSettingsError as exc:
        raise HTTPException(400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(503, detail=f"Cannot reach {provider_id}: {getattr(exc, 'message', str(exc))}")
    return {"provider": provider_id, "models": models}


@router.post("/llm/discover")
def discover_llm_servers():
    """Scan loopback ports and identify real inference engines by response shape."""
    existing = {
        normalize_base_url(provider.base_url): provider_id
        for provider_id, provider in llm_settings.get().providers.items()
        if provider.base_url
    }
    servers = discovery.discover_local_servers(existing=existing)
    return {"servers": servers, "count": len(servers)}


@router.post("/llm/attach")
def attach_llm_server(server: LLMDiscoverAttach):
    payload = server.model_dump()
    if not payload.get("instance_id"):
        payload["instance_id"] = instance_id(payload["provider"], payload["base_url"])
    try:
        snapshot = llm_settings.add_instance(payload)
    except LLMSettingsError as exc:
        raise HTTPException(400, detail=str(exc))
    runtime.clear_catalog()
    logger.info("LLM provider attached: %s (%s)", payload["instance_id"], payload["base_url"])
    return snapshot


@router.delete("/llm/providers/{provider_id}")
def remove_llm_provider(provider_id: str):
    try:
        snapshot = llm_settings.remove_instance(provider_id)
    except LLMSettingsError as exc:
        raise HTTPException(400, detail=str(exc))
    runtime.clear_catalog(provider_id)
    return snapshot


@router.post("/llm/providers/{provider_id}/verify", response_model=LLMVerifyResult)
def verify_llm_provider(provider_id: str):
    try:
        result = runtime.verify_provider(provider_id)
    except LLMSettingsError as exc:
        raise HTTPException(400, detail=str(exc))
    return LLMVerifyResult(**result)


@router.post("/llm/providers/{provider_id}/warm")
def warm_llm_provider(provider_id: str):
    try:
        return runtime.warm_provider(provider_id)
    except LLMSettingsError as exc:
        raise HTTPException(400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(503, detail=getattr(exc, "message", str(exc)))


@router.delete("/llm/providers/{provider_id}/key")
def clear_llm_provider_key(provider_id: str):
    try:
        return llm_settings.clear_key(provider_id)
    except LLMSettingsError as exc:
        raise HTTPException(404, detail=str(exc))


@router.post("/llm/test")
def test_llm(req: LLMTestRequest):
    """Send a test prompt through the role plan or an explicit provider/model."""
    try:
        result = runtime.test_target(
            req.prompt, provider=req.provider, model=req.model, role=req.role,
        )
    except LLMSettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=getattr(exc, "message", str(exc)))
    return {
        "reply": result.text,
        "provider": result.provider,
        "model": result.model,
        "role": result.role,
        "latency_ms": result.latency_ms,
        "usage": result.usage or None,
    }


@router.post("/llm/test/stream")
async def test_llm_stream(req: LLMTestRequest):
    """Stream a test prompt as SSE, with a non-stream fallback inside the runtime."""
    async def events():
        def emit(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        if req.provider:
            # Explicit targets stream through the role plan only when they match
            # it; otherwise run the non-stream call once and emit it whole.
            try:
                result = await asyncio.to_thread(
                    runtime.test_target,
                    req.prompt,
                    provider=req.provider,
                    model=req.model,
                    role=req.role,
                )
            except Exception as exc:
                yield emit("error", {"message": str(getattr(exc, "message", exc))})
                return
            yield emit("start", {"provider": result.provider, "model": result.model, "role": result.role})
            yield emit("delta", {"text": result.text})
            yield emit("done", {
                "provider": result.provider, "model": result.model, "role": result.role,
                "latency_ms": result.latency_ms, "usage": result.usage or None,
            })
            return

        plan = runtime.resolve(req.role)
        if not plan["usable"]:
            yield emit("error", {"message": plan["reason"]})
            return
        primary = plan["plan"][0]
        yield emit("start", {"provider": primary["provider"], "model": primary["model"], "role": req.role})
        try:
            async for delta in runtime.astream(req.role, req.prompt):
                yield emit("delta", {"text": delta})
        except Exception as exc:
            yield emit("error", {"message": str(getattr(exc, "message", exc))})
            return
        yield emit("done", {"provider": primary["provider"], "model": primary["model"], "role": req.role})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/llm/health")
def llm_health(check: bool = Query(default=False)):
    """Resolution status for every role plus provider health and budget.

    `check=true` also reads local catalogs so missing role models are reported;
    it is explicit because it makes one request per local engine.
    """
    snapshot = llm_settings.public()
    if check:
        for provider_id, provider in snapshot["providers"].items():
            if provider["is_local"] and provider["enabled"]:
                try:
                    runtime.provider_models(provider_id)
                except Exception:
                    continue
    roles = {role: runtime.resolve(role) for role in snapshot["roles"]}
    providers = {
        provider_id: {
            "enabled": config["enabled"],
            "is_local": config["is_local"],
            "key_status": config["key_status"],
            "last_success_at": config["last_success_at"],
            "last_fault": config["last_fault"],
            "telemetry": config.get("telemetry") or {},
        }
        for provider_id, config in snapshot["providers"].items()
    }
    return {
        "revision": snapshot["revision"],
        "air_gapped": snapshot["air_gapped"],
        "cloud_enabled": snapshot["cloud_enabled"],
        "active_provider": snapshot["active_provider"],
        "active_model": snapshot["active_model"],
        "roles": roles,
        "providers": providers,
        "missing_models": runtime.cap_missing_models(),
        "last_eval": snapshot.get("last_eval"),
        "budget": snapshot.get("budget"),
        "spend": snapshot.get("spend"),
        "profiles": profiles.PROFILES,
    }


@router.post("/llm/eval")
def run_llm_eval(req: LLMEvalRequest):
    """Run local role fixtures. This calls the saved plan and can take minutes."""
    try:
        return eval_engine.run_eval(req.roles or None, req.limit)
    except LLMSettingsError as exc:
        raise HTTPException(400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))


@router.post("/llm/eval/apply")
def apply_llm_recommendation(expected_revision: Optional[int] = None):
    try:
        plan = eval_engine.recommended_plan()
    except LLMSettingsError as exc:
        raise HTTPException(400, detail=str(exc))
    doc = llm_settings.get()
    for role, target in plan["roles"].items():
        assignment = doc.roles.get(role)
        if assignment is None:
            continue
        assignment.provider = target["provider"]
        assignment.model = target["model"]
    try:
        snapshot = llm_settings.save(doc, expected_revision=expected_revision)
    except LLMRevisionConflict as exc:
        raise HTTPException(409, detail={"message": str(exc), "revision": exc.current})
    except LLMSettingsError as exc:
        raise HTTPException(400, detail=str(exc))
    snapshot["warnings"] = runtime.cap_missing_models()
    return snapshot


@router.post("/llm/profiles/{profile}")
def apply_llm_profile(profile: str, expected_revision: Optional[int] = None):
    try:
        return profiles.apply_profile(profile, expected_revision=expected_revision)
    except LLMRevisionConflict as exc:
        raise HTTPException(409, detail={"message": str(exc), "revision": exc.current})
    except LLMSettingsError as exc:
        raise HTTPException(400, detail=str(exc))


@router.get("/llm/spend")
def llm_spend():
    return {**budget.summary(), "rates": {model: list(rate) for model, rate in budget.rates().items()}}
