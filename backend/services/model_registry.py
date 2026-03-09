"""
Model Registry — Central source of truth for model status.

Cross-references the static models_manifest.json against the filesystem
to provide real-time model availability, download status, and pipeline
readiness information.

Usage:
    from services.model_registry import model_registry
    models = model_registry.get_all_models()
    ready = model_registry.get_pipeline_readiness("video")
"""

import json
import logging
import os
import shutil
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import config

logger = logging.getLogger(__name__)

MANIFEST_PATH = os.path.join(config.BACKEND_DIR, "models_manifest.json")
DOWNLOAD_TEMP_SUFFIX = ".downloading"


class ModelStatus(str, Enum):
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    ACTIVE = "active"
    ERROR = "error"
    INCOMPATIBLE = "incompatible"


@dataclass
class ModelInfo:
    """Runtime-resolved model information (v2 — extended with HF integration fields)."""
    id: str
    name: str
    family: str
    role: str
    pipeline: str
    required: bool
    size_bytes: Optional[int]
    relative_path: str
    absolute_path: str
    description: str
    device_compatibility: list
    depends_on: list
    alternatives: list
    huggingface: dict

    # v2 fields
    type: str = "base"                          # base | quantized | ic_lora | camera_lora | adapter
    pipeline_tag: str = ""                       # HF pipeline tag: image-to-video, text-to-image, etc.
    requires_base: bool = False                  # True for LoRAs/adapters
    base_repo_id: Optional[str] = None           # Which base model must be loaded first
    vram_estimate_gb: float = 0.0                # Estimated GPU memory
    recommended_dtype: str = "bfloat16"          # bfloat16 | float32 | float8_e4m3fn
    gated: bool = False                          # True if HF repo requires auth
    license: str = ""                            # License identifier

    # Runtime state
    status: ModelStatus = ModelStatus.NOT_DOWNLOADED
    downloaded_bytes: int = 0
    download_progress: float = 0.0
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Serializable dict for API responses."""
        d = asdict(self)
        d["status"] = self.status.value
        return d


class ModelRegistry:
    """
    Singleton service: loads manifest, scans disk, tracks model state.
    
    Does NOT modify how models are loaded — that remains in ModelManager
    and FluxInpainter. This is purely a visibility/discovery layer.
    """

    def __init__(self):
        self._manifest: list[dict] = []
        self._models: dict[str, ModelInfo] = {}
        self._device = self._detect_device()
        self._load_manifest()
        self.scan_disk()

    # ── Manifest Loading ──────────────────────────────────────────

    def _load_manifest(self) -> None:
        """Load the models manifest JSON."""
        if not os.path.exists(MANIFEST_PATH):
            logger.warning(f"Model manifest not found at {MANIFEST_PATH}. Model registry will be empty.")
            return

        try:
            with open(MANIFEST_PATH, "r") as f:
                data = json.load(f)
            self._manifest = data.get("models", [])
            version = data.get("version", "1.0")
            logger.info(f"Model manifest v{version} loaded: {len(self._manifest)} models defined")
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse model manifest: {e}")
            self._manifest = []

    # ── Device Detection ──────────────────────────────────────────

    def _detect_device(self) -> str:
        """Detect current compute device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ── Disk Scanning ──────────────────────────────────────────────

    def _resolve_path(self, relative_path: str) -> str:
        """Resolve a manifest relative_path to an absolute filesystem path."""
        return os.path.join(config.PROJECT_ROOT, relative_path)

    def _check_file_exists(self, absolute_path: str) -> bool:
        """Check if a model file or directory exists on disk."""
        return os.path.exists(absolute_path)

    def _get_file_size(self, absolute_path: str) -> int:
        """Get the size of a file or directory on disk."""
        if os.path.isfile(absolute_path):
            return os.path.getsize(absolute_path)
        elif os.path.isdir(absolute_path):
            total = 0
            for dirpath, _dirnames, filenames in os.walk(absolute_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.isfile(fp):
                        total += os.path.getsize(fp)
            return total
        return 0

    def scan_disk(self) -> None:
        """
        Refresh model status by scanning the filesystem.
        
        Cross-references manifest entries against actual files on disk.
        Detects:
          - Fully downloaded models
          - Partially downloaded (*.downloading) temp files
          - Device incompatibilities (e.g., FP8 on MPS)
        """
        self._models.clear()

        for entry in self._manifest:
            model_id = entry["id"]
            abs_path = self._resolve_path(entry["relative_path"])
            temp_path = abs_path + DOWNLOAD_TEMP_SUFFIX

            info = ModelInfo(
                id=model_id,
                name=entry["name"],
                family=entry["family"],
                role=entry["role"],
                pipeline=entry["pipeline"],
                required=entry.get("required", False),
                size_bytes=entry.get("size_bytes"),
                relative_path=entry["relative_path"],
                absolute_path=abs_path,
                description=entry.get("description", ""),
                device_compatibility=entry.get("device_compatibility", []),
                depends_on=entry.get("depends_on", []),
                alternatives=entry.get("alternatives", []),
                huggingface=entry.get("huggingface", {}),
                # v2 fields
                type=entry.get("type", "base"),
                pipeline_tag=entry.get("pipeline_tag", ""),
                requires_base=entry.get("requires_base", False),
                base_repo_id=entry.get("base_repo_id"),
                vram_estimate_gb=entry.get("vram_estimate_gb", 0.0),
                recommended_dtype=entry.get("recommended_dtype", "bfloat16"),
                gated=entry.get("gated", False),
                license=entry.get("license", ""),
            )

            # Check device compatibility
            if self._device not in info.device_compatibility:
                info.status = ModelStatus.INCOMPATIBLE
                info.error_message = f"Not compatible with {self._device.upper()} device"
            # Check if fully downloaded
            elif self._check_file_exists(abs_path):
                info.status = ModelStatus.DOWNLOADED
                info.downloaded_bytes = self._get_file_size(abs_path)
            # Check for partial download
            elif os.path.exists(temp_path):
                info.status = ModelStatus.DOWNLOADING
                info.downloaded_bytes = self._get_file_size(temp_path)
                if info.size_bytes and info.size_bytes > 0:
                    info.download_progress = info.downloaded_bytes / info.size_bytes
                info.error_message = "Download interrupted — resume available"
            else:
                info.status = ModelStatus.NOT_DOWNLOADED

            self._models[model_id] = info

        # Update active status from memory manager
        self._sync_active_status()

        counts: dict[str, int] = {}
        for m in self._models.values():
            counts[m.status.value] = counts.get(m.status.value, 0) + 1
        logger.info(f"Disk scan complete: {counts}")

    def _sync_active_status(self) -> None:
        """Mark models as ACTIVE if their pipeline is currently loaded in GPU."""
        try:
            from memory_manager import memory_manager
            active_slots = memory_manager.active

            if "video" in active_slots:
                for m in self._models.values():
                    if m.family in ("ltx2", "ltx23") and m.status == ModelStatus.DOWNLOADED:
                        m.status = ModelStatus.ACTIVE

            if "image" in active_slots:
                for m in self._models.values():
                    if m.family == "flux2" and m.status == ModelStatus.DOWNLOADED:
                        m.status = ModelStatus.ACTIVE
        except Exception as e:
            logger.debug(f"Could not sync active status: {e}")

    # ── Public API ──────────────────────────────────────────────────

    def get_all_models(self, pipeline: Optional[str] = None, family: Optional[str] = None,
                       status: Optional[str] = None) -> list[dict]:
        """
        Return all models, optionally filtered.
        
        Args:
            pipeline: Filter by pipeline (video, image, segmentation)
            family: Filter by family (ltx2, ltx23, flux2, sam3)
            status: Filter by status (downloaded, not_downloaded, etc.)
        """
        models = list(self._models.values())

        if pipeline:
            models = [m for m in models if m.pipeline == pipeline]
        if family:
            models = [m for m in models if m.family == family]
        if status:
            models = [m for m in models if m.status.value == status]

        return [m.to_dict() for m in models]

    def get_model(self, model_id: str) -> Optional[dict]:
        """Get a single model by ID."""
        model = self._models.get(model_id)
        return model.to_dict() if model else None

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get raw ModelInfo object (internal use)."""
        return self._models.get(model_id)

    def get_pipeline_readiness(self, pipeline: Optional[str] = None) -> dict:
        """
        Check pipeline readiness.
        
        Returns:
            {
                "video": {"ready": True, "missing": [], "total": 6, "downloaded": 6},
                "image": {"ready": False, "missing": ["flux2-klein-9b"], "total": 5, "downloaded": 3},
                ...
            }
        """
        pipelines_to_check = [pipeline] if pipeline else ["video", "image", "segmentation"]
        result = {}

        for pipe in pipelines_to_check:
            models = [m for m in self._models.values() if m.pipeline == pipe]
            required = [m for m in models if m.required]
            missing = [
                m.id for m in required
                if m.status in (ModelStatus.NOT_DOWNLOADED, ModelStatus.ERROR)
            ]
            # Mark incompatible-but-has-alternative as not-missing
            resolved_missing = []
            for mid in missing:
                model = self._models[mid]
                alt_ok = any(
                    self._models.get(alt_id) and
                    self._models[alt_id].status in (ModelStatus.DOWNLOADED, ModelStatus.ACTIVE)
                    for alt_id in model.alternatives
                )
                if not alt_ok:
                    resolved_missing.append(mid)

            downloaded = [
                m for m in models
                if m.status in (ModelStatus.DOWNLOADED, ModelStatus.ACTIVE)
            ]

            result[pipe] = {
                "ready": len(resolved_missing) == 0,
                "missing": resolved_missing,
                "total": len(models),
                "downloaded": len(downloaded),
            }

        return result

    def get_disk_usage(self) -> dict:
        """Get disk usage summary for model storage."""
        total_expected = 0
        total_downloaded = 0

        for m in self._models.values():
            if m.size_bytes:
                total_expected += m.size_bytes
            if m.status in (ModelStatus.DOWNLOADED, ModelStatus.ACTIVE):
                total_downloaded += m.downloaded_bytes

        try:
            usage = shutil.disk_usage(config.PROJECT_ROOT)
            free_bytes = usage.free
        except Exception:
            free_bytes = None

        return {
            "total_expected_bytes": total_expected,
            "total_downloaded_bytes": total_downloaded,
            "disk_free_bytes": free_bytes,
        }

    def update_model_status(self, model_id: str, status: ModelStatus,
                            progress: float = 0.0, error: Optional[str] = None,
                            downloaded_bytes: int = 0) -> None:
        """Update a model's status (used by download service)."""
        model = self._models.get(model_id)
        if model:
            model.status = status
            model.download_progress = progress
            model.error_message = error
            model.downloaded_bytes = downloaded_bytes


# ── Singleton ──────────────────────────────────────────────────────
model_registry = ModelRegistry()
