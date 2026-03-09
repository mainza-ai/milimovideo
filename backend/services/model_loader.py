"""
Model Loader Service — LoRA stacking + pipeline lifecycle.

Bridges the model registry with the actual pipeline loaders (ModelManager, FluxInpainter),
handling LoRA attachment, VRAM pre-checks, and MPS dtype guards.

LTX-2 LoRAs are passed at pipeline construction time via LoraPathStrengthAndSDOps.
Flux2 does not currently support LoRAs (custom pipeline).

Usage:
    from services.model_loader import model_loader
    await model_loader.activate("ltx2-ic-lora-union", strength=0.8)
    active = model_loader.get_active_loras("video")
"""

import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import config

logger = logging.getLogger(__name__)


@dataclass
class ActiveLoRA:
    """Represents a LoRA currently stacked on a pipeline."""
    model_id: str
    name: str
    absolute_path: str
    strength: float = 1.0
    role: str = "ic_lora"  # ic_lora | camera_lora


class ModelLoaderService:
    """
    Coordinates model activation with actual pipeline loading.
    
    Responsibilities:
      - VRAM pre-check before activation
      - Track which LoRAs are active on each pipeline slot
      - Reconstruct LTX-2 pipelines when LoRA config changes
      - Manage activate/deactivate lifecycle with MemoryManager
    """

    def __init__(self):
        self._active_loras: dict[str, list[ActiveLoRA]] = {
            "video": [],
            "image": [],
        }
        self._device = self._detect_device()

    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ── VRAM Guard ────────────────────────────────────────────────

    def can_load_model(self, vram_estimate_gb: float) -> tuple[bool, str]:
        """
        Check if there's enough memory to load a model.
        
        Returns (can_load, reason).
        """
        if vram_estimate_gb <= 0:
            return True, "No VRAM estimate — skipping check"

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_mem / 1e9
            allocated_gb = torch.cuda.memory_allocated(0) / 1e9
            available_gb = total_gb - allocated_gb

            if available_gb >= vram_estimate_gb * 1.1:
                return True, f"CUDA: {available_gb:.1f} GB available, need {vram_estimate_gb:.1f} GB"
            return False, (
                f"Insufficient CUDA VRAM: {available_gb:.1f} GB available, "
                f"need {vram_estimate_gb:.1f} GB (+ 10% safety)"
            )

        elif torch.backends.mps.is_available():
            # MPS uses unified memory — use psutil if available
            try:
                import psutil
                available_gb = psutil.virtual_memory().available / 1e9
                # MPS needs more headroom due to unified memory sharing
                if available_gb >= vram_estimate_gb * 1.2:
                    return True, f"MPS: {available_gb:.1f} GB system RAM available"
                return False, (
                    f"Insufficient system memory for MPS: {available_gb:.1f} GB available, "
                    f"need {vram_estimate_gb:.1f} GB (+ 20% safety for unified memory)"
                )
            except ImportError:
                logger.warning("psutil not installed — skipping VRAM pre-check for MPS")
                return True, "psutil not available — cannot check MPS memory"

        return False, "No GPU device available"

    # ── LoRA Management ───────────────────────────────────────────

    def get_active_loras(self, slot: str) -> list[dict]:
        """Return currently active LoRAs for a pipeline slot."""
        return [
            {
                "model_id": lora.model_id,
                "name": lora.name,
                "strength": lora.strength,
                "role": lora.role,
            }
            for lora in self._active_loras.get(slot, [])
        ]

    async def add_lora(self, model_id: str, strength: float = 1.0) -> dict:
        """
        Add a LoRA to the active pipeline.
        
        For LTX-2: triggers pipeline reconstruction with new LoRA list.
        Returns the updated LoRA stack.
        """
        from services.model_registry import model_registry, ModelStatus

        model = model_registry.get_model_info(model_id)
        if not model:
            raise ValueError(f"Model '{model_id}' not found in registry")

        if model.type not in ("ic_lora", "camera_lora"):
            raise ValueError(f"Model '{model_id}' is not a LoRA (type: {model.type})")

        if model.status not in (ModelStatus.DOWNLOADED, ModelStatus.ACTIVE):
            raise ValueError(f"Model '{model_id}' is not downloaded")

        # Determine which slot this LoRA belongs to
        slot = "video" if model.pipeline == "video" else "image"

        # Check if already active
        existing = [l for l in self._active_loras[slot] if l.model_id == model_id]
        if existing:
            # Update strength
            existing[0].strength = strength
            logger.info(f"Updated LoRA '{model_id}' strength to {strength}")
        else:
            # Add new LoRA
            lora = ActiveLoRA(
                model_id=model_id,
                name=model.name,
                absolute_path=model.absolute_path,
                strength=strength,
                role=model.role,
            )
            self._active_loras[slot].append(lora)
            logger.info(f"Added LoRA '{model_id}' (strength={strength}) to {slot} pipeline")

        # Mark as active in registry
        model_registry.update_model_status(model_id, ModelStatus.ACTIVE)

        # Trigger pipeline reconstruction for LTX-2
        if slot == "video":
            await self._reconstruct_ltx_pipeline()

        return {
            "slot": slot,
            "loras": self.get_active_loras(slot),
            "message": f"LoRA '{model.name}' added to {slot} pipeline",
        }

    async def remove_lora(self, model_id: str) -> dict:
        """
        Remove a LoRA from the active pipeline.
        
        Triggers pipeline reconstruction without the removed LoRA.
        """
        from services.model_registry import model_registry, ModelStatus

        model = model_registry.get_model_info(model_id)
        slot = "video" if model and model.pipeline == "video" else "image"

        removed = False
        self._active_loras[slot] = [
            l for l in self._active_loras[slot]
            if l.model_id != model_id or not (removed := True)  # noqa: remove and flag
        ]

        # Simpler removal
        original_len = len(self._active_loras.get(slot, []))
        self._active_loras[slot] = [
            l for l in self._active_loras.get(slot, [])
            if l.model_id != model_id
        ]
        removed = len(self._active_loras[slot]) < original_len

        if not removed:
            raise ValueError(f"LoRA '{model_id}' is not currently active")

        # Revert to downloaded in registry
        if model:
            model_registry.update_model_status(model_id, ModelStatus.DOWNLOADED)

        # Trigger pipeline reconstruction
        if slot == "video":
            await self._reconstruct_ltx_pipeline()

        return {
            "slot": slot,
            "loras": self.get_active_loras(slot),
            "message": f"LoRA '{model_id}' removed from {slot} pipeline",
        }

    async def update_lora_strength(self, model_id: str, strength: float) -> dict:
        """Update the strength of an active LoRA."""
        for slot in ("video", "image"):
            for lora in self._active_loras.get(slot, []):
                if lora.model_id == model_id:
                    lora.strength = max(0.0, min(2.0, strength))
                    logger.info(f"Updated LoRA '{model_id}' strength to {lora.strength}")

                    # For LTX-2, strength changes require pipeline reconstruction
                    if slot == "video":
                        await self._reconstruct_ltx_pipeline()

                    return {
                        "model_id": model_id,
                        "strength": lora.strength,
                        "loras": self.get_active_loras(slot),
                    }

        raise ValueError(f"LoRA '{model_id}' is not currently active")

    def clear_loras(self, slot: str) -> None:
        """Remove all LoRAs from a pipeline slot."""
        from services.model_registry import model_registry, ModelStatus

        for lora in self._active_loras.get(slot, []):
            model = model_registry.get_model_info(lora.model_id)
            if model:
                model_registry.update_model_status(lora.model_id, ModelStatus.DOWNLOADED)

        self._active_loras[slot] = []
        logger.info(f"Cleared all LoRAs from {slot} pipeline")

    # ── Pipeline Reconstruction ───────────────────────────────────

    async def _reconstruct_ltx_pipeline(self) -> None:
        """
        Reconstruct the LTX-2 pipeline with current LoRA configuration.
        
        LTX-2 uses LoraPathStrengthAndSDOps at construction time.
        When LoRAs change, we must reload the pipeline with updated LoRA list.
        """
        from model_engine import manager
        from ltx_core.loader import LoraPathStrengthAndSDOps

        active_loras = self._active_loras.get("video", [])

        # Build LoRA objects
        lora_objs = []
        for lora in active_loras:
            if os.path.exists(lora.absolute_path):
                lora_objs.append(
                    LoraPathStrengthAndSDOps(lora.absolute_path, lora.strength, None)
                )
                logger.info(f"  LoRA: {lora.name} (strength={lora.strength})")
            else:
                logger.warning(f"  LoRA file not found: {lora.absolute_path}")

        # Determine pipeline type based on active LoRAs
        has_ic_lora = any(l.role == "ic_lora" for l in active_loras)
        pipeline_type = "ic_lora" if has_ic_lora else "ti2vid"

        logger.info(
            f"Reconstructing LTX-2 pipeline as '{pipeline_type}' "
            f"with {len(lora_objs)} LoRA(s)"
        )

        # Force unload current pipeline
        if manager._pipeline is not None:
            del manager._pipeline
            manager._pipeline = None
            manager._pipeline_type = None
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Reload with new LoRA configuration
        await manager.load_pipeline(pipeline_type, loras=lora_objs)

    # ── Activation Helpers ────────────────────────────────────────

    async def activate_base_model(self, model_id: str) -> dict:
        """
        Activate a base model (loads the full pipeline).
        
        Checks VRAM, unloads conflicts via MemoryManager, then loads.
        """
        from services.model_registry import model_registry, ModelStatus
        from memory_manager import memory_manager

        model = model_registry.get_model_info(model_id)
        if not model:
            raise ValueError(f"Model '{model_id}' not found")

        # VRAM pre-check
        can_load, reason = self.can_load_model(model.vram_estimate_gb)
        if not can_load:
            raise MemoryError(reason)

        slot = "video" if model.pipeline == "video" else "image"

        # Clear existing LoRAs for this slot
        self.clear_loras(slot)

        # Prepare memory (unloads conflicts)
        memory_manager.prepare_for(slot)

        # Load the appropriate pipeline
        if model.family in ("ltx2", "ltx23"):
            from model_engine import manager
            await manager.load_pipeline("ti2vid")
        elif model.family == "flux2":
            from models.flux_wrapper import flux_inpainter
            flux_inpainter.load_model()

        # Mark as active
        model_registry.update_model_status(model_id, ModelStatus.ACTIVE)

        return {
            "model_id": model_id,
            "status": "active",
            "vram_check": reason,
            "message": f"Model '{model.name}' activated",
        }

    async def deactivate_base_model(self, model_id: str) -> dict:
        """Deactivate a base model (unloads pipeline)."""
        from services.model_registry import model_registry, ModelStatus
        from memory_manager import memory_manager

        model = model_registry.get_model_info(model_id)
        if not model:
            raise ValueError(f"Model '{model_id}' not found")

        slot = "video" if model.pipeline == "video" else "image"

        # Clear LoRAs first
        self.clear_loras(slot)

        # Unload
        memory_manager.release(slot)
        if slot == "video":
            memory_manager._unload_ltx()
        elif slot == "image":
            memory_manager._unload_flux()

        # Flush memory
        memory_manager._flush_memory()

        # Mark as downloaded
        model_registry.update_model_status(model_id, ModelStatus.DOWNLOADED)

        return {
            "model_id": model_id,
            "status": "downloaded",
            "message": f"Model '{model.name}' deactivated",
        }


# ── Singleton ──────────────────────────────────────────────────────
model_loader = ModelLoaderService()
