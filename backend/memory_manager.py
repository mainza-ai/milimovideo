"""
Central Memory Manager — coordinates model lifecycle to prevent OOM.

On Apple Silicon with unified memory, Flux (~36GB), LTX (~76GB), and Ollama (~49GB)
can collectively exceed 128GB. This manager enforces mutual exclusion between
heavy GPU models.

Slot system:
  - "video"  → LTX-2 pipeline (model_engine.ModelManager)
  - "image"  → Flux 2 pipeline (flux_wrapper.FluxInpainter)
  - "llm"    → Ollama (managed via keep_alive, not loaded in this process)

Usage:
    from memory_manager import memory_manager
    memory_manager.prepare_for("video")  # Unloads Flux if loaded
    memory_manager.prepare_for("image")  # Unloads LTX if loaded
"""
import gc
import logging
import torch

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self):
        self._active_slots: set[str] = set()
    
    # ── Slot Conflict Rules ──────────────────────────────────────
    # These slots cannot coexist in memory simultaneously.
    CONFLICTS = {
        "video": {"image"},   # LTX and Flux are mutually exclusive
        "image": {"video"},   # Flux and LTX are mutually exclusive
    }
    
    def prepare_for(self, slot: str) -> None:
        """
        Prepare memory for loading a model into the given slot.
        Unloads any conflicting models first.
        """
        conflicts = self.CONFLICTS.get(slot, set())
        to_unload = conflicts & self._active_slots
        
        if to_unload:
            logger.info(f"MemoryManager: Preparing for '{slot}' — unloading {to_unload}")
            for conflicting_slot in to_unload:
                self._unload_slot(conflicting_slot)
        
        self._active_slots.add(slot)
    
    def release(self, slot: str) -> None:
        """Mark a slot as no longer active (model has been unloaded)."""
        self._active_slots.discard(slot)
        logger.info(f"MemoryManager: Released slot '{slot}'. Active: {self._active_slots}")
    
    def _unload_slot(self, slot: str) -> None:
        """Unload the model in the given slot."""
        if slot == "video":
            self._unload_ltx()
        elif slot == "image":
            self._unload_flux()
        
        self._active_slots.discard(slot)
        self._flush_memory()
    
    def _unload_ltx(self) -> None:
        """Unload LTX-2 pipeline via ModelManager."""
        try:
            from model_engine import manager
            if manager._pipeline is not None:
                logger.info("MemoryManager: Unloading LTX-2 pipeline...")
                del manager._pipeline
                manager._pipeline = None
                manager._pipeline_type = None
                logger.info("MemoryManager: LTX-2 pipeline unloaded")
        except Exception as e:
            logger.warning(f"MemoryManager: Failed to unload LTX: {e}")
    
    def _unload_flux(self) -> None:
        """Unload Flux 2 pipeline via FluxInpainter."""
        try:
            from models.flux_wrapper import flux_inpainter
            if flux_inpainter.model_loaded:
                logger.info("MemoryManager: Unloading Flux 2 pipeline...")
                flux_inpainter.unload()
                logger.info("MemoryManager: Flux 2 pipeline unloaded")
        except Exception as e:
            logger.warning(f"MemoryManager: Failed to unload Flux: {e}")
    
    def _flush_memory(self) -> None:
        """Force garbage collection and clear GPU caches."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logger.info("MemoryManager: Memory flushed (gc + cache cleared)")
    
    @property
    def active(self) -> set[str]:
        """Currently active model slots."""
        return self._active_slots.copy()
    
    def status(self) -> dict:
        """Return memory manager status for debugging/API."""
        return {
            "active_slots": list(self._active_slots),
            "conflicts": {k: list(v) for k, v in self.CONFLICTS.items()},
        }


# Singleton
memory_manager = MemoryManager()
