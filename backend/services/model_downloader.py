"""
Model Downloader v2 — tqdm_class-based SSE progress streaming.

Uses huggingface_hub's built-in tqdm_class override to pipe real-time
download progress directly to SSE broadcasts, replacing manual byte counting.

Usage:
    from services.model_downloader import model_downloader
    await model_downloader.download_model("ltx2-base")
"""

import asyncio
import logging
import os
import shutil
import time
from dataclasses import dataclass
from functools import partial
from typing import Optional

import config
from events import event_manager

logger = logging.getLogger(__name__)

PROGRESS_INTERVAL = 2.0  # Seconds between SSE broadcasts


@dataclass
class DownloadState:
    model_id: str
    total_bytes: int
    downloaded_bytes: int = 0
    speed_mbps: float = 0.0
    eta_seconds: float = 0.0
    cancelled: bool = False
    error: Optional[str] = None
    started_at: float = 0.0
    last_broadcast: float = 0.0


@dataclass
class QueuedDownload:
    """A download waiting in the priority queue."""
    model_id: str
    name: str
    priority: int = 2       # 0=required, 1=LoRA, 2=optional
    size_bytes: int = 0
    queued_at: float = 0.0


class SSEProgressBar:
    """
    Custom tqdm-compatible class that broadcasts progress via SSE.
    
    huggingface_hub calls tqdm_class(total=N, ...) then .update(n) on each chunk.
    We mimic the tqdm interface but pipe to SSE instead of terminal.
    """

    def __init__(self, *args, total=None, model_id="unknown",
                 download_state=None, broadcast_fn=None, **kwargs):
        self.total = total or 0
        self.n = 0
        self.start_t = time.time()
        self._model_id = model_id
        self._state = download_state
        self._broadcast_fn = broadcast_fn
        self._last_broadcast = 0.0

    def update(self, n=1):
        """Called by huggingface_hub on each downloaded chunk."""
        self.n += n

        if self._state:
            self._state.downloaded_bytes = self.n
            if self.total > 0:
                self._state.total_bytes = self.total

            now = time.time()
            elapsed = max(0.01, now - self.start_t)
            self._state.speed_mbps = (self.n / (1024 * 1024)) / elapsed

            if self.total > 0 and self._state.speed_mbps > 0:
                remaining = self.total - self.n
                self._state.eta_seconds = remaining / (self._state.speed_mbps * 1024 * 1024)

            # Check cancellation
            if self._state.cancelled:
                raise InterruptedError("Download cancelled")

            # Throttled SSE broadcast
            if now - self._last_broadcast >= PROGRESS_INTERVAL and self._broadcast_fn:
                self._last_broadcast = now
                self._broadcast_fn(self._state)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def set_description(self, *args, **kwargs):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def refresh(self):
        pass


class ModelDownloader:
    """
    Handles async model downloads from HuggingFace with resume support.
    
    Uses tqdm_class override for real-time SSE progress streaming.
    Downloads are queued with priority ordering and processed sequentially.
    """

    # Priority levels (lower = higher priority)
    PRIORITY_REQUIRED = 0   # Required base models
    PRIORITY_LORA = 1       # LoRAs and adapters
    PRIORITY_OPTIONAL = 2   # Optional / nice-to-have

    def __init__(self):
        self.active_downloads: dict[str, DownloadState] = {}
        self._download_lock = asyncio.Lock()
        self._queue: list[QueuedDownload] = []
        self._queue_lock = asyncio.Lock()
        self._processing = False

    def _priority_for_model(self, model) -> int:
        """Determine download priority based on model type."""
        if getattr(model, 'required', False):
            return self.PRIORITY_REQUIRED
        if getattr(model, 'type', 'base') in ('ic_lora', 'camera_lora', 'adapter'):
            return self.PRIORITY_LORA
        return self.PRIORITY_OPTIONAL

    async def enqueue_download(self, model_id: str, priority: Optional[int] = None) -> dict:
        """
        Add a model to the download queue.
        
        If no priority specified, it's inferred from model type.
        Returns the queue position and total queue size.
        """
        from services.model_registry import model_registry, ModelStatus

        model = model_registry.get_model_info(model_id)
        if not model:
            raise ValueError(f"Model '{model_id}' not found")

        # Don't re-queue if already downloading or queued
        if model_id in self.active_downloads:
            return {"model_id": model_id, "status": "already_downloading"}

        async with self._queue_lock:
            if any(q.model_id == model_id for q in self._queue):
                return {"model_id": model_id, "status": "already_queued"}

            if priority is None:
                priority = self._priority_for_model(model)

            queued = QueuedDownload(
                model_id=model_id,
                name=model.name,
                priority=priority,
                size_bytes=model.size_bytes or 0,
                queued_at=time.time(),
            )
            self._queue.append(queued)
            # Sort by priority, then by queue time
            self._queue.sort(key=lambda q: (q.priority, q.queued_at))

        model_registry.update_model_status(model_id, ModelStatus.DOWNLOADING)

        # Start processing if not already running
        if not self._processing:
            asyncio.ensure_future(self._process_queue())

        position = next(
            (i for i, q in enumerate(self._queue) if q.model_id == model_id), -1
        )

        await event_manager.broadcast("download_queued", {
            "model_id": model_id,
            "position": position,
            "queue_size": len(self._queue),
            "priority": priority,
        })

        return {
            "model_id": model_id,
            "status": "queued",
            "position": position,
            "queue_size": len(self._queue),
            "priority": priority,
        }

    async def _process_queue(self) -> None:
        """Process downloads from the priority queue, one at a time."""
        self._processing = True
        try:
            while True:
                async with self._queue_lock:
                    if not self._queue:
                        break
                    item = self._queue.pop(0)

                logger.info(
                    f"Queue: starting download '{item.model_id}' "
                    f"(priority={item.priority}, remaining={len(self._queue)})"
                )
                await self.download_model(item.model_id)

                # Broadcast queue update
                await event_manager.broadcast("queue_updated", {
                    "queue": self.get_queue_status(),
                })
        finally:
            self._processing = False

    def get_queue_status(self) -> list[dict]:
        """Return the current download queue."""
        return [
            {
                "model_id": q.model_id,
                "name": q.name,
                "priority": q.priority,
                "size_bytes": q.size_bytes,
                "position": i,
                "queued_at": q.queued_at,
            }
            for i, q in enumerate(self._queue)
        ]

    async def remove_from_queue(self, model_id: str) -> bool:
        """Remove a model from the download queue (before it starts)."""
        from services.model_registry import model_registry, ModelStatus

        async with self._queue_lock:
            original_len = len(self._queue)
            self._queue = [q for q in self._queue if q.model_id != model_id]
            removed = len(self._queue) < original_len

        if removed:
            model_registry.update_model_status(model_id, ModelStatus.NOT_DOWNLOADED)
            return True
        return False

    async def download_model(self, model_id: str) -> None:
        """
        Download a model file from HuggingFace.
        
        1. Resolves model info from registry
        2. Downloads using huggingface_hub (tqdm_class → SSE)
        3. Updates registry on completion/failure
        """
        from services.model_registry import model_registry, ModelStatus

        model = model_registry.get_model_info(model_id)
        if not model:
            logger.error(f"Download requested for unknown model: {model_id}")
            return

        hf_info = model.huggingface
        if not hf_info or not hf_info.get("repo"):
            logger.error(f"Model '{model_id}' has no HuggingFace source configured")
            model_registry.update_model_status(model_id, ModelStatus.ERROR,
                                                error="No HuggingFace source configured")
            await self._broadcast_error(model_id, "No HuggingFace source configured")
            return

        # Disk space pre-check
        if model.size_bytes:
            try:
                usage = shutil.disk_usage(config.PROJECT_ROOT)
                if usage.free < model.size_bytes * 1.1:
                    error_msg = (f"Insufficient disk space. Need {model.size_bytes / 1e9:.1f} GB, "
                                 f"only {usage.free / 1e9:.1f} GB free.")
                    model_registry.update_model_status(model_id, ModelStatus.ERROR, error=error_msg)
                    await self._broadcast_error(model_id, error_msg)
                    return
            except OSError:
                pass

        state = DownloadState(
            model_id=model_id,
            total_bytes=model.size_bytes or 0,
            started_at=time.time(),
        )
        self.active_downloads[model_id] = state

        try:
            target_dir = os.path.dirname(model.absolute_path)
            os.makedirs(target_dir, exist_ok=True)

            if hf_info.get("filename"):
                await self._download_single_file(model, state)
            elif hf_info.get("subdir"):
                await self._download_directory(model, state)
            else:
                raise ValueError(f"Model '{model_id}' has neither filename nor subdir")

            if not state.cancelled:
                model_registry.update_model_status(model_id, ModelStatus.DOWNLOADED,
                                                    progress=1.0,
                                                    downloaded_bytes=state.total_bytes)
                model_registry.scan_disk()
                await self._broadcast_complete(model_id)
                logger.info(f"Download complete: {model.name}")

        except InterruptedError:
            logger.info(f"Download cancelled: {model_id}")
            model_registry.update_model_status(model_id, ModelStatus.NOT_DOWNLOADED,
                                                error="Download cancelled by user")
            await self._broadcast_error(model_id, "Cancelled", resumable=True)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Download failed for '{model_id}': {error_msg}")
            model_registry.update_model_status(model_id, ModelStatus.ERROR, error=error_msg)

            # Detect gated repo errors
            is_gated = "gated" in error_msg.lower() or "access" in error_msg.lower()
            await self._broadcast_error(model_id, error_msg, resumable=True, gated=is_gated)
        finally:
            self.active_downloads.pop(model_id, None)

    async def _download_single_file(self, model, state: DownloadState) -> None:
        """Download a single model file with tqdm_class SSE progress."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub required. Install with: pip install huggingface_hub")

        hf_info = model.huggingface
        repo_id = hf_info["repo"]
        filename = hf_info["filename"]
        target_dir = os.path.dirname(model.absolute_path)

        logger.info(f"Downloading {repo_id}/{filename} → {target_dir}")

        # Build the SSE-streaming tqdm class
        loop = asyncio.get_event_loop()

        def sync_broadcast(download_state: DownloadState):
            """Sync callback → async SSE broadcast."""
            try:
                loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(self._broadcast_progress(download_state))
                )
            except RuntimeError:
                pass

        progress_cls = partial(
            SSEProgressBar,
            model_id=model.id,
            download_state=state,
            broadcast_fn=sync_broadcast,
        )

        await loop.run_in_executor(None, lambda: hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            token=os.environ.get("HF_TOKEN"),
            tqdm_class=progress_cls,
        ))

    async def _download_directory(self, model, state: DownloadState) -> None:
        """Download a model directory (snapshot) with tqdm_class SSE progress."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError("huggingface_hub required. Install with: pip install huggingface_hub")

        hf_info = model.huggingface
        repo_id = hf_info["repo"]
        subdir = hf_info["subdir"]
        target_dir = os.path.dirname(model.absolute_path)

        logger.info(f"Downloading {repo_id}/{subdir}/ → {target_dir}")

        loop = asyncio.get_event_loop()

        def sync_broadcast(download_state: DownloadState):
            try:
                loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(self._broadcast_progress(download_state))
                )
            except RuntimeError:
                pass

        progress_cls = partial(
            SSEProgressBar,
            model_id=model.id,
            download_state=state,
            broadcast_fn=sync_broadcast,
        )

        await loop.run_in_executor(None, lambda: snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{subdir}/**"],
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            token=os.environ.get("HF_TOKEN"),
            tqdm_class=progress_cls,
        ))

    def cancel_download(self, model_id: str) -> bool:
        """Cancel an active download."""
        state = self.active_downloads.get(model_id)
        if state:
            state.cancelled = True
            logger.info(f"Cancellation requested for download: {model_id}")
            return True
        return False

    def get_active_downloads(self) -> dict:
        """Return all active download states."""
        return {
            mid: {
                "model_id": s.model_id,
                "total_bytes": s.total_bytes,
                "downloaded_bytes": s.downloaded_bytes,
                "speed_mbps": round(s.speed_mbps, 2),
                "eta_seconds": round(s.eta_seconds),
                "progress": round(s.downloaded_bytes / s.total_bytes, 4) if s.total_bytes > 0 else 0,
            }
            for mid, s in self.active_downloads.items()
        }

    # ── SSE Broadcasting ──────────────────────────────────────────

    async def _broadcast_progress(self, state: DownloadState) -> None:
        """Broadcast download progress via SSE."""
        progress = 0.0
        if state.total_bytes > 0:
            progress = round(state.downloaded_bytes / state.total_bytes, 4)

        await event_manager.broadcast("download_progress", {
            "model_id": state.model_id,
            "progress": progress,
            "speed_mbps": round(state.speed_mbps, 2),
            "eta_seconds": round(state.eta_seconds),
            "downloaded_bytes": state.downloaded_bytes,
            "total_bytes": state.total_bytes,
        })

    async def _broadcast_complete(self, model_id: str) -> None:
        """Broadcast download completion via SSE."""
        await event_manager.broadcast("download_complete", {
            "model_id": model_id,
        })

    async def _broadcast_error(self, model_id: str, error: str,
                               resumable: bool = False, gated: bool = False) -> None:
        """Broadcast download error via SSE."""
        await event_manager.broadcast("download_error", {
            "model_id": model_id,
            "error": error,
            "resumable": resumable,
            "gated": gated,
        })


# ── Singleton ──────────────────────────────────────────────────────
model_downloader = ModelDownloader()
