"""
Model Settings & Version Tracking Service.

Manages HF_TRANSFER toggle, model version checking via HfApi.model_info(),
and disk space reporting.

Usage:
    from services.model_settings import model_settings
    model_settings.set_hf_transfer(True)
    await model_settings.check_model_version("ltx2-base")
"""

import logging
import os
import shutil
from typing import Optional

import config

logger = logging.getLogger(__name__)


class ModelSettingsService:
    """Manages model-related settings and system info."""

    # ── HF Transfer Toggle ────────────────────────────────────────

    def get_hf_transfer_enabled(self) -> bool:
        """Check if HF_HUB_ENABLE_HF_TRANSFER is active."""
        return os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") == "1"

    def set_hf_transfer(self, enabled: bool) -> dict:
        """
        Toggle accelerated downloads via hf_transfer (Rust-based, 5-10x faster).
        
        Requires: pip install hf_transfer
        """
        if enabled:
            # Check if hf_transfer is actually installed
            try:
                import importlib
                importlib.import_module("hf_transfer")
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
                logger.info("HF Transfer enabled (Rust accelerator active)")
                return {"enabled": True, "message": "HF Transfer enabled (Rust accelerated downloads)"}
            except ImportError:
                return {
                    "enabled": False,
                    "message": "hf_transfer package not installed. Run: pip install hf_transfer",
                    "install_command": "pip install hf_transfer",
                }
        else:
            os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
            logger.info("HF Transfer disabled")
            return {"enabled": False, "message": "HF Transfer disabled"}

    # ── Disk Space ────────────────────────────────────────────────

    def get_disk_usage(self) -> dict:
        """Report disk usage for the project root and models directory."""
        try:
            project_usage = shutil.disk_usage(config.PROJECT_ROOT)

            # Calculate model directory sizes
            model_dirs = self._get_model_dirs()
            models_total = 0
            model_breakdown = {}

            for name, path in model_dirs.items():
                if os.path.exists(path):
                    size = self._dir_size(path)
                    models_total += size
                    model_breakdown[name] = {
                        "path": path,
                        "size_bytes": size,
                        "size_gb": round(size / 1e9, 2),
                    }

            return {
                "disk": {
                    "total_bytes": project_usage.total,
                    "used_bytes": project_usage.used,
                    "free_bytes": project_usage.free,
                    "total_gb": round(project_usage.total / 1e9, 1),
                    "used_gb": round(project_usage.used / 1e9, 1),
                    "free_gb": round(project_usage.free / 1e9, 1),
                    "usage_pct": round(project_usage.used / project_usage.total * 100, 1),
                },
                "models": {
                    "total_bytes": models_total,
                    "total_gb": round(models_total / 1e9, 2),
                    "breakdown": model_breakdown,
                },
            }
        except OSError as e:
            logger.error(f"Disk usage check failed: {e}")
            return {"error": str(e)}

    def _get_model_dirs(self) -> dict[str, str]:
        """Return known model directory paths."""
        return {
            "ltx2": os.path.join(config.PROJECT_ROOT, "ltx2", "models"),
            "flux2": getattr(config, "FLUX_WEIGHTS_PATH", os.path.join(config.PROJECT_ROOT, "flux2_models")),
            "sam3": os.path.join(config.PROJECT_ROOT, "sam3", "checkpoints"),
            "hf_cache": os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
        }

    def _dir_size(self, path: str) -> int:
        """Calculate total size of a directory recursively."""
        total = 0
        try:
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.isfile(fp):
                        total += os.path.getsize(fp)
        except (OSError, PermissionError):
            pass
        return total

    # ── Version Tracking ──────────────────────────────────────────

    async def check_model_version(self, model_id: str) -> dict:
        """
        Check if a downloaded model has a newer version on HuggingFace.
        
        Compares the HF repo's last_modified against the local file's mtime.
        """
        import asyncio

        from services.model_registry import model_registry

        model = model_registry.get_model_info(model_id)
        if not model:
            raise ValueError(f"Model '{model_id}' not found")

        hf_info = model.huggingface
        if not hf_info or not hf_info.get("repo"):
            return {"model_id": model_id, "update_available": False, "reason": "No HF repo configured"}

        # Get local file modification time
        local_mtime = None
        if os.path.exists(model.absolute_path):
            local_mtime = os.path.getmtime(model.absolute_path)

        # Query HuggingFace for repo last_modified
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            info = await asyncio.to_thread(
                api.model_info,
                hf_info["repo"],
                token=os.environ.get("HF_TOKEN"),
            )

            remote_modified = info.last_modified
            if remote_modified and local_mtime:
                remote_ts = remote_modified.timestamp()
                update_available = remote_ts > local_mtime

                return {
                    "model_id": model_id,
                    "update_available": update_available,
                    "local_modified": local_mtime,
                    "remote_modified": remote_ts,
                    "remote_date": str(remote_modified),
                    "repo_id": hf_info["repo"],
                }

            return {
                "model_id": model_id,
                "update_available": False,
                "reason": "Could not compare timestamps",
                "local_exists": local_mtime is not None,
            }

        except Exception as e:
            logger.warning(f"Version check failed for '{model_id}': {e}")
            return {
                "model_id": model_id,
                "update_available": False,
                "error": str(e),
            }

    async def check_all_versions(self) -> list[dict]:
        """Check version status for all downloaded models."""
        from services.model_registry import model_registry, ModelStatus

        results = []
        for model_dict in model_registry.get_all_models():
            if model_dict.get("status") in ("downloaded", "active"):
                try:
                    result = await self.check_model_version(model_dict["id"])
                    results.append(result)
                except Exception as e:
                    results.append({
                        "model_id": model_dict["id"],
                        "update_available": False,
                        "error": str(e),
                    })

        return results

    # ── All Settings ──────────────────────────────────────────────

    def get_all_settings(self) -> dict:
        """Return all model-related settings."""
        return {
            "hf_token_configured": bool(os.environ.get("HF_TOKEN")),
            "hf_transfer_enabled": self.get_hf_transfer_enabled(),
            "disk_usage": self.get_disk_usage(),
        }


# ── Singleton ──────────────────────────────────────────────────────
model_settings = ModelSettingsService()
