"""LLM configuration plane: persisted settings, discovery, verification.

Local-first: the default plan runs entirely on loopback. Cloud providers stay
disabled unless `MILIMO_LLM_CLOUD=1` and the air-gap switch is off. Reads are
redacted; writes are revision-checked whole-document updates.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from services.llm import discovery, runtime
from services.llm.models import LLMSettings, LLMTestRequest, LLMVerifyResult
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
    logger.info(
        "LLM settings saved: revision=%s active=%s/%s", snapshot["revision"],
        snapshot["active_provider"], snapshot["active_model"],
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


@router.get("/llm/models")
def list_llm_models(provider: Optional[str] = Query(default=None)):
    """Model catalog for one provider (defaults to the active role's provider)."""
    provider_id = provider or llm_settings.role_provider("prompt_video")
    try:
        models = runtime.provider_models(provider_id)
    except LLMSettingsError as exc:
        raise HTTPException(400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(503, detail=f"Cannot reach {provider_id}: {getattr(exc, 'message', str(exc))}")
    return {"provider": provider_id, "models": models}


@router.post("/llm/discover")
def discover_llm_servers():
    """Scan loopback ports for running local inference servers."""
    servers = discovery.discover_local_servers()
    return {"servers": servers, "count": len(servers)}


@router.post("/llm/providers/{provider_id}/verify", response_model=LLMVerifyResult)
def verify_llm_provider(provider_id: str):
    try:
        result = runtime.verify_provider(provider_id)
    except LLMSettingsError as exc:
        raise HTTPException(400, detail=str(exc))
    return LLMVerifyResult(**result)


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


@router.get("/llm/health")
def llm_health():
    """Resolution status for every role plus per-provider last success/fault."""
    snapshot = llm_settings.public()
    roles = {role: runtime.resolve(role) for role in snapshot["roles"]}
    providers = {
        provider_id: {
            "enabled": config["enabled"],
            "is_local": config["is_local"],
            "key_status": config["key_status"],
            "last_success_at": config["last_success_at"],
            "last_fault": config["last_fault"],
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
    }
