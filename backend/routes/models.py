"""
Model Management API Routes.

Provides endpoints for discovering, downloading, and managing AI models.
"""

import logging
import os
import shutil
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from typing import Optional

import config
from events import event_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])


# ── Response Schemas ──────────────────────────────────────────────

class DownloadRequest(BaseModel):
    """Request to start a model download."""
    force: bool = False  # Re-download even if exists


class DeleteResponse(BaseModel):
    model_id: str
    deleted: bool
    message: str


# ── Helper ──────────────────────────────────────────────────────

def _get_registry():
    """Lazy import to avoid circular imports at module level."""
    from services.model_registry import model_registry
    return model_registry


# ── Endpoints ──────────────────────────────────────────────────

@router.get("/models")
def list_models(
    pipeline: Optional[str] = Query(None, description="Filter by pipeline: video, image, segmentation"),
    family: Optional[str] = Query(None, description="Filter by family: ltx2, flux2, sam3"),
    status: Optional[str] = Query(None, description="Filter by status: downloaded, not_downloaded, etc."),
):
    """
    List all known models with their current status.
    
    Combines the static manifest with real-time disk scan results.
    """
    registry = _get_registry()
    models = registry.get_all_models(pipeline=pipeline, family=family, status=status)
    return {"models": models, "count": len(models)}


@router.get("/models/pipelines")
def pipeline_readiness(
    pipeline: Optional[str] = Query(None, description="Specific pipeline to check"),
):
    """
    Check which pipelines are ready for use.
    
    Returns readiness status per pipeline, listing any missing required models.
    """
    registry = _get_registry()
    readiness = registry.get_pipeline_readiness(pipeline=pipeline)
    return readiness


@router.get("/models/disk")
def disk_usage():
    """
    Get disk usage summary for model storage.
    
    Returns total expected size, total downloaded size, and free disk space.
    """
    registry = _get_registry()
    return registry.get_disk_usage()


@router.get("/models/memory")
def memory_status():
    """
    Get current GPU memory manager status.
    
    Returns which model slots are active and conflict rules.
    """
    from memory_manager import memory_manager
    return memory_manager.status()


@router.get("/models/{model_id}")
def get_model(model_id: str):
    """Get details for a specific model."""
    registry = _get_registry()
    model = registry.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in manifest")
    return model


@router.post("/models/{model_id}/download")
async def start_download(model_id: str, request: DownloadRequest, background_tasks: BackgroundTasks):
    """
    Start downloading a model from HuggingFace.
    
    Downloads are performed in the background with progress reported via SSE.
    Supports resume for interrupted downloads.
    """
    registry = _get_registry()
    model = registry.get_model_info(model_id)

    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    if model.status.value == "incompatible":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is not compatible with your device ({registry._device})"
        )

    if model.status.value in ("downloaded", "active") and not request.force:
        raise HTTPException(
            status_code=409,
            detail=f"Model '{model_id}' is already downloaded. Use force=true to re-download."
        )

    if model.status.value == "downloading":
        raise HTTPException(
            status_code=409,
            detail=f"Model '{model_id}' is already being downloaded."
        )

    # Check disk space
    if model.size_bytes:
        try:
            usage = shutil.disk_usage(config.PROJECT_ROOT)
            if usage.free < model.size_bytes * 1.1:  # 10% buffer
                raise HTTPException(
                    status_code=507,
                    detail=f"Insufficient disk space. Need {model.size_bytes / 1e9:.1f} GB, "
                           f"only {usage.free / 1e9:.1f} GB free."
                )
        except OSError:
            pass  # Can't check, proceed anyway

    # Start async download
    from services.model_downloader import model_downloader
    background_tasks.add_task(model_downloader.download_model, model_id)

    # Mark as downloading immediately
    from services.model_registry import ModelStatus
    registry.update_model_status(model_id, ModelStatus.DOWNLOADING)

    return {
        "model_id": model_id,
        "status": "downloading",
        "message": f"Download started for {model.name}"
    }


@router.delete("/models/{model_id}/download")
async def cancel_download(model_id: str):
    """Cancel an active download."""
    from services.model_downloader import model_downloader
    cancelled = model_downloader.cancel_download(model_id)

    if not cancelled:
        raise HTTPException(
            status_code=404,
            detail=f"No active download found for '{model_id}'"
        )

    return {"model_id": model_id, "status": "cancelled"}


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """
    Delete a downloaded model from disk.
    
    Will refuse to delete if the model is currently active in GPU memory.
    """
    registry = _get_registry()
    model = registry.get_model_info(model_id)

    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    if model.status.value == "active":
        raise HTTPException(
            status_code=409,
            detail=f"Model '{model_id}' is currently active in GPU memory. Unload it first."
        )

    if model.status.value == "not_downloaded":
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' is not downloaded."
        )

    # Delete from disk
    try:
        if os.path.isfile(model.absolute_path):
            os.remove(model.absolute_path)
        elif os.path.isdir(model.absolute_path):
            shutil.rmtree(model.absolute_path)
        else:
            raise HTTPException(status_code=404, detail="Model file not found on disk")

        # Also clean up any temp files
        temp_path = model.absolute_path + ".downloading"
        if os.path.exists(temp_path):
            os.remove(temp_path)

        logger.info(f"Deleted model '{model_id}' from {model.absolute_path}")

        # Refresh status
        from services.model_registry import ModelStatus
        registry.update_model_status(model_id, ModelStatus.NOT_DOWNLOADED, downloaded_bytes=0)

        return DeleteResponse(
            model_id=model_id,
            deleted=True,
            message=f"Model '{model.name}' deleted successfully"
        )

    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied — cannot delete model file")
    except Exception as e:
        logger.error(f"Failed to delete model '{model_id}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/scan")
def scan_models():
    """
    Force re-scan disk for model files.
    
    Useful after manually placing model files or after a download completes
    outside of Milimo.
    """
    registry = _get_registry()
    registry.scan_disk()
    models = registry.get_all_models()
    return {
        "message": "Disk scan complete",
        "models": models,
        "count": len(models),
    }


# ── Activate / Deactivate ─────────────────────────────────────────

@router.post("/models/{model_id}/activate")
async def activate_model(model_id: str):
    """
    Load a model into GPU memory.
    
    For base models: triggers pipeline load via MemoryManager.
    For LoRAs: requires base to be active first.
    """
    registry = _get_registry()
    model = registry.get_model_info(model_id)

    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    if model.status.value not in ("downloaded", "active"):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is not downloaded (status: {model.status.value})"
        )

    if model.status.value == "active":
        return {"model_id": model_id, "status": "active", "message": "Already active"}

    # For LoRAs: check base is loaded
    if model.requires_base and model.base_repo_id:
        base = registry.get_model_info(model.base_repo_id)
        if not base or base.status.value != "active":
            raise HTTPException(
                status_code=400,
                detail=f"Base model '{model.base_repo_id}' must be active before loading LoRA '{model_id}'"
            )

    # For base models: use MemoryManager to unload conflicts, then mark active
    from memory_manager import memory_manager
    from services.model_registry import ModelStatus

    if not model.requires_base:
        slot = "video" if model.pipeline == "video" else "image"
        memory_manager.prepare_for(slot)

    registry.update_model_status(model_id, ModelStatus.ACTIVE)

    # Broadcast status change
    await event_manager.broadcast("model_status_change", {
        "model_id": model_id,
        "new_status": "active",
    })

    return {
        "model_id": model_id,
        "status": "active",
        "message": f"Model '{model.name}' activated"
    }


@router.post("/models/{model_id}/deactivate")
async def deactivate_model(model_id: str):
    """
    Unload a model from GPU memory.
    """
    registry = _get_registry()
    model = registry.get_model_info(model_id)

    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    if model.status.value != "active":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is not currently active"
        )

    from memory_manager import memory_manager
    from services.model_registry import ModelStatus

    if not model.requires_base:
        slot = "video" if model.pipeline == "video" else "image"
        memory_manager.release(slot)

    registry.update_model_status(model_id, ModelStatus.DOWNLOADED)

    await event_manager.broadcast("model_status_change", {
        "model_id": model_id,
        "new_status": "downloaded",
    })

    return {
        "model_id": model_id,
        "status": "downloaded",
        "message": f"Model '{model.name}' deactivated"
    }


# ── HuggingFace Token ─────────────────────────────────────────────

class HFTokenRequest(BaseModel):
    token: str


@router.patch("/settings/hf-token")
async def set_hf_token(request: HFTokenRequest):
    """
    Set the HuggingFace auth token for downloading gated models.
    
    Stored in environment variable for the current session.
    """
    os.environ["HF_TOKEN"] = request.token
    logger.info("HuggingFace token updated")
    return {"message": "HuggingFace token set successfully", "has_token": True}


@router.get("/settings/hf-token")
def get_hf_token_status():
    """Check if a HuggingFace token is configured."""
    has_token = bool(os.environ.get("HF_TOKEN"))
    return {"has_token": has_token}


# ── Hub Search ─────────────────────────────────────────────────────

@router.get("/models/search")
async def search_hub(
    q: str = Query("", description="Search query"),
    family: Optional[str] = Query(None, description="Filter by family: ltx2, flux2, sam3"),
    limit: int = Query(25, description="Max results per author"),
):
    """
    Search HuggingFace Hub for compatible models.
    
    Returns live Hub results cross-referenced with local manifest.
    """
    from services.model_search import model_search
    try:
        results = await model_search.search(query=q, family=family, limit=limit)
        return {"results": results, "count": len(results)}
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e))


@router.get("/models/search/{repo_id:path}")
async def get_hub_model_info(repo_id: str):
    """Get detailed info for a specific HuggingFace repo."""
    from services.model_search import model_search
    info = await model_search.get_model_info(repo_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Could not retrieve info for '{repo_id}'")
    return info


# ── LoRA Management ────────────────────────────────────────────────

class LoRARequest(BaseModel):
    strength: float = 1.0


@router.post("/models/{model_id}/lora")
async def add_lora(model_id: str, request: LoRARequest):
    """
    Add a LoRA to the active pipeline.
    
    For LTX-2: triggers pipeline reconstruction with updated LoRA stack.
    Requires the base model to be active first.
    """
    from services.model_loader import model_loader
    try:
        result = await model_loader.add_lora(model_id, strength=request.strength)
        await event_manager.broadcast("lora_updated", result)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except MemoryError as e:
        raise HTTPException(status_code=507, detail=str(e))


@router.delete("/models/{model_id}/lora")
async def remove_lora(model_id: str):
    """Remove a LoRA from the active pipeline."""
    from services.model_loader import model_loader
    try:
        result = await model_loader.remove_lora(model_id)
        await event_manager.broadcast("lora_updated", result)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/models/{model_id}/lora")
async def update_lora_strength(model_id: str, request: LoRARequest):
    """Update the strength of an active LoRA (0.0 — 2.0)."""
    from services.model_loader import model_loader
    try:
        result = await model_loader.update_lora_strength(model_id, request.strength)
        await event_manager.broadcast("lora_updated", result)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/models/loras/{slot}")
def get_active_loras(slot: str):
    """Get currently active LoRAs for a pipeline slot (video/image)."""
    from services.model_loader import model_loader
    if slot not in ("video", "image"):
        raise HTTPException(status_code=400, detail="Slot must be 'video' or 'image'")
    return {"slot": slot, "loras": model_loader.get_active_loras(slot)}


@router.get("/models/vram-check/{model_id}")
def check_vram(model_id: str):
    """Pre-check if there's enough VRAM to load a model."""
    registry = _get_registry()
    model = registry.get_model_info(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    from services.model_loader import model_loader
    can_load, reason = model_loader.can_load_model(model.vram_estimate_gb)
    return {
        "model_id": model_id,
        "can_load": can_load,
        "vram_estimate_gb": model.vram_estimate_gb,
        "reason": reason,
    }


# ── Download Queue ─────────────────────────────────────────────────

class EnqueueRequest(BaseModel):
    priority: Optional[int] = None


@router.post("/models/{model_id}/enqueue")
async def enqueue_download(model_id: str, request: EnqueueRequest = EnqueueRequest()):
    """Add a model to the priority download queue."""
    from services.model_downloader import model_downloader
    try:
        result = await model_downloader.enqueue_download(model_id, priority=request.priority)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/models/{model_id}/enqueue")
async def dequeue_download(model_id: str):
    """Remove a model from the download queue before it starts."""
    from services.model_downloader import model_downloader
    removed = await model_downloader.remove_from_queue(model_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Model not in queue")
    return {"model_id": model_id, "removed": True}


@router.get("/models/queue")
def get_download_queue():
    """Get the current download queue with priorities."""
    from services.model_downloader import model_downloader
    return {
        "queue": model_downloader.get_queue_status(),
        "active": model_downloader.get_active_downloads(),
    }


# ── Disk Space Dashboard ──────────────────────────────────────────

@router.get("/models/disk")
def get_disk_usage():
    """Get disk usage breakdown: system disk + per-family model sizes."""
    from services.model_settings import model_settings
    return model_settings.get_disk_usage()


# ── Model Version Tracking ────────────────────────────────────────

@router.get("/models/{model_id}/version")
async def check_model_version(model_id: str):
    """Check if a model has a newer version on HuggingFace."""
    from services.model_settings import model_settings
    try:
        result = await model_settings.check_model_version(model_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/models/versions")
async def check_all_versions():
    """Check version status for all downloaded models."""
    from services.model_settings import model_settings
    results = await model_settings.check_all_versions()
    updates_available = sum(1 for r in results if r.get("update_available"))
    return {
        "models": results,
        "total_checked": len(results),
        "updates_available": updates_available,
    }


# ── HF Transfer Acceleration ──────────────────────────────────────

class HFTransferRequest(BaseModel):
    enabled: bool


@router.patch("/settings/hf-transfer")
def set_hf_transfer(request: HFTransferRequest):
    """Toggle HF_HUB_ENABLE_HF_TRANSFER for accelerated downloads."""
    from services.model_settings import model_settings
    return model_settings.set_hf_transfer(request.enabled)


@router.get("/settings/hf-transfer")
def get_hf_transfer():
    """Check if HF Transfer acceleration is enabled."""
    from services.model_settings import model_settings
    return {"enabled": model_settings.get_hf_transfer_enabled()}


# ── Combined Settings ─────────────────────────────────────────────

@router.get("/settings")
def get_all_settings():
    """Get all model management settings."""
    from services.model_settings import model_settings
    return model_settings.get_all_settings()
