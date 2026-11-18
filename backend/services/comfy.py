"""ComfyUI bridge: submit an API-format workflow, relay progress, fetch outputs.

ComfyUI is treated as an external GPU engine. We submit the graph ComfyUI
already understands (``Save (API Format)``), watch it over the websocket when
available, fall back to queue/history polling, then copy every produced file
into the project workspace.

No model names are hardcoded: whatever checkpoints the local ComfyUI instance
has are the engine's business. A model-free ``EmptyImage -> SaveImage`` graph is
therefore a valid smoke test.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

import httpx

import config

logger = logging.getLogger("services.comfy")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".avif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"}
TEXT_EXTENSIONS = {".txt", ".json", ".md", ".srt", ".vtt"}
SAFE_EXTENSION = re.compile(r"^\.[a-z0-9]{1,8}$")


class ComfyError(RuntimeError):
    """A workflow was rejected or failed inside ComfyUI."""


class ComfyUnavailable(ComfyError):
    """The local ComfyUI server did not answer."""


class ComfyCancelled(ComfyError):
    """The user cancelled the workflow."""


@dataclass(frozen=True)
class ComfyOutput:
    node_id: str
    filename: str
    subfolder: str = ""
    type: str = "output"
    kind: str = "image"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "filename": self.filename,
            "subfolder": self.subfolder,
            "type": self.type,
            "kind": self.kind,
        }


def _media_kind(key: str, filename: str) -> str:
    extension = os.path.splitext(filename)[1].lower()
    if extension in IMAGE_EXTENSIONS:
        return "image"
    if extension in VIDEO_EXTENSIONS:
        return "video"
    if extension in AUDIO_EXTENSIONS:
        return "audio"
    if extension in TEXT_EXTENSIONS:
        return "text"
    lowered = (key or "").lower()
    if any(token in lowered for token in ("image", "gif", "mask")):
        return "image"
    if any(token in lowered for token in ("video", "gif")):
        return "video"
    if "audio" in lowered:
        return "audio"
    if any(token in lowered for token in ("text", "string", "json")):
        return "text"
    guessed, _ = mimetypes.guess_type(filename)
    if guessed:
        if guessed.startswith("image/"):
            return "image"
        if guessed.startswith("video/"):
            return "video"
        if guessed.startswith("audio/"):
            return "audio"
    return "file"


def parse_workflow(workflow: Any) -> Dict[str, Any]:
    """Accept API-format JSON as a dict or a string and validate its shape."""
    if isinstance(workflow, str):
        try:
            workflow = json.loads(workflow)
        except json.JSONDecodeError as exc:
            raise ComfyError(f"Workflow is not valid JSON: {exc}") from exc
    if not isinstance(workflow, dict):
        raise ComfyError("Workflow must be a JSON object in ComfyUI API format")
    if "nodes" in workflow and "links" in workflow:
        raise ComfyError(
            "This looks like a UI workflow. In ComfyUI use Workflow → Save (API Format) "
            "and import that JSON instead."
        )
    if not workflow:
        raise ComfyError("Workflow is empty")
    for node_id, node in workflow.items():
        if not isinstance(node, dict) or "class_type" not in node:
            raise ComfyError(
                f"Node {node_id} is missing class_type; use ComfyUI API format"
            )
        if not isinstance(node.get("inputs"), dict):
            raise ComfyError(f"Node {node_id} is missing inputs")
    # Detach from the request object: callers may reuse it after a job starts.
    return json.loads(json.dumps(workflow))


def apply_overrides(workflow: Dict[str, Any], overrides: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Merge per-node input overrides into a parsed workflow."""
    if not overrides:
        return workflow
    merged = json.loads(json.dumps(workflow))
    for node_id, inputs in overrides.items():
        node = merged.get(str(node_id))
        if not node:
            raise ComfyError(f"Cannot override missing node {node_id}")
        if not isinstance(inputs, dict):
            raise ComfyError(f"Overrides for node {node_id} must be an object")
        node.setdefault("inputs", {}).update(inputs)
    return merged


def apply_filename_prefix(workflow: Dict[str, Any], prefix: Optional[str]) -> Dict[str, Any]:
    """Point every SaveImage node at a job-scoped filename prefix."""
    if not prefix:
        return workflow
    for node in workflow.values():
        if node.get("class_type") == "SaveImage":
            node.setdefault("inputs", {})["filename_prefix"] = prefix
    return workflow


def validate_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Cheap structural checks and a readout for the UI."""
    outputs = [
        node_id for node_id, node in workflow.items()
        if str(node.get("class_type", "")).startswith(("Save", "Preview"))
    ]
    warnings = []
    if not outputs:
        warnings.append("No Save/Preview node found; the run may produce no files.")
    width, height = workflow_size(workflow)
    return {
        "nodes": len(workflow),
        "output_nodes": outputs,
        "width": width,
        "height": height,
        "warnings": warnings,
    }


def workflow_size(workflow: Dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    """Best-effort output size from the first node declaring width/height."""
    for node in workflow.values():
        inputs = node.get("inputs") or {}
        width, height = inputs.get("width"), inputs.get("height")
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            return width, height
    return None, None


def _prompt_ids(rows: Iterable[Any]) -> list[str]:
    ids: list[str] = []
    for row in rows or []:
        if isinstance(row, (list, tuple)) and len(row) > 1:
            ids.append(str(row[1]))
        elif isinstance(row, dict) and row.get("prompt_id"):
            ids.append(str(row["prompt_id"]))
    return ids


class ComfyClient:
    """Thin synchronous HTTP client plus an async runner with progress."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.base_url = (base_url or config.COMFY_URL).rstrip("/")
        self.timeout = float(timeout or config.COMFY_TIMEOUT_SEC)
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ComfyClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ── HTTP primitives ─────────────────────────────────────────
    def _get(self, path: str, **kwargs) -> httpx.Response:
        try:
            return self._client.get(path, **kwargs)
        except httpx.HTTPError as exc:
            raise ComfyUnavailable(
                f"ComfyUI is not reachable at {self.base_url}: {exc}"
            ) from exc

    def _post(self, path: str, **kwargs) -> httpx.Response:
        try:
            return self._client.post(path, **kwargs)
        except httpx.HTTPError as exc:
            raise ComfyUnavailable(
                f"ComfyUI is not reachable at {self.base_url}: {exc}"
            ) from exc

    # ── Readiness ───────────────────────────────────────────────
    def health(self) -> Dict[str, Any]:
        """Return a normalized readiness row; never raises for an offline server."""
        payload: Dict[str, Any] = {"reachable": False, "url": self.base_url}
        try:
            response = self._get("/system_stats", timeout=min(self.timeout, 3.0))
            response.raise_for_status()
            stats = response.json()
            system = stats.get("system") or {}
            payload.update({
                "reachable": True,
                "version": system.get("comfyui_version"),
                "os": system.get("os"),
                "ram_total_gb": round(float(system.get("ram_total") or 0) / 1e9, 2),
                "ram_free_gb": round(float(system.get("ram_free") or 0) / 1e9, 2),
                "devices": stats.get("devices") or [],
            })
            try:
                queue = self.queue()
                payload["queue_running"] = len(_prompt_ids(queue.get("queue_running")))
                payload["queue_pending"] = len(_prompt_ids(queue.get("queue_pending")))
            except ComfyError:
                payload["queue_running"] = None
                payload["queue_pending"] = None
        except (ComfyError, ValueError, httpx.HTTPStatusError) as exc:
            payload["error"] = str(exc)
        return payload

    def system_stats(self) -> Dict[str, Any]:
        response = self._get("/system_stats")
        response.raise_for_status()
        return response.json()

    def queue(self) -> Dict[str, Any]:
        response = self._get("/queue")
        response.raise_for_status()
        return response.json()

    def queue_position(self, prompt_id: str) -> Optional[int]:
        """1-based pending position, 0 when running, None when absent."""
        try:
            queue = self.queue()
        except ComfyError:
            return None
        if prompt_id in _prompt_ids(queue.get("queue_running")):
            return 0
        pending = _prompt_ids(queue.get("queue_pending"))
        if prompt_id in pending:
            return pending.index(prompt_id) + 1
        return None

    def object_info(self) -> Dict[str, Any]:
        response = self._get("/object_info")
        response.raise_for_status()
        return response.json()

    # ── Submit / history / outputs ──────────────────────────────
    def submit(self, workflow: Dict[str, Any], client_id: Optional[str] = None) -> str:
        prompt = parse_workflow(workflow)
        body = {"prompt": prompt, "client_id": client_id or uuid.uuid4().hex}
        response = self._post("/prompt", json=body)
        if response.status_code >= 400:
            detail = response.text[:600]
            try:
                parsed = response.json()
                detail = parsed.get("error") or parsed.get("detail") or detail
                node_errors = parsed.get("node_errors")
                if node_errors:
                    detail = f"{detail}; {json.dumps(node_errors)[:500]}"
            except ValueError:
                pass
            raise ComfyError(f"ComfyUI rejected the workflow: {detail}")
        data = response.json()
        node_errors = data.get("node_errors") or {}
        if node_errors:
            raise ComfyError(f"ComfyUI node errors: {json.dumps(node_errors)[:600]}")
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ComfyError("ComfyUI did not return a prompt_id")
        return str(prompt_id)

    def history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        response = self._get(f"/history/{prompt_id}")
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            return None
        return data.get(prompt_id)

    @staticmethod
    def execution_error(entry: Optional[Dict[str, Any]]) -> Optional[str]:
        if not entry:
            return None
        status = entry.get("status") or {}
        for message in status.get("messages") or []:
            if isinstance(message, (list, tuple)) and message and message[0] == "execution_error":
                data = message[1] if len(message) > 1 and isinstance(message[1], dict) else {}
                return str(
                    data.get("exception_message")
                    or data.get("exception_type")
                    or "ComfyUI execution error"
                )
        if status.get("status_str") == "error":
            return "ComfyUI execution error"
        return None

    @staticmethod
    def collect_outputs(entry: Optional[Dict[str, Any]]) -> list[ComfyOutput]:
        outputs: list[ComfyOutput] = []
        seen: set[tuple[str, str, str]] = set()
        for node_id, payload in ((entry or {}).get("outputs") or {}).items():
            if not isinstance(payload, dict):
                continue
            for key, items in payload.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict) or not item.get("filename"):
                        continue
                    filename = str(item["filename"])
                    subfolder = str(item.get("subfolder") or "")
                    kind = str(item.get("type") or "output")
                    identity = (filename, subfolder, kind)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    outputs.append(ComfyOutput(
                        node_id=str(node_id),
                        filename=filename,
                        subfolder=subfolder,
                        type=kind,
                        kind=_media_kind(str(key), filename),
                    ))
        return outputs

    def download(self, output: ComfyOutput) -> bytes:
        response = self._get("/view", params={
            "filename": output.filename,
            "subfolder": output.subfolder,
            "type": output.type,
        })
        response.raise_for_status()
        return response.content

    def download_to(self, output: ComfyOutput, path: str) -> int:
        """Stream one output to disk; returns the byte count.

        Streaming keeps a large video or latent from becoming a second copy in
        process memory just before it is written.
        """
        try:
            with self._client.stream("GET", "/view", params={
                "filename": output.filename,
                "subfolder": output.subfolder,
                "type": output.type,
            }) as response:
                response.raise_for_status()
                with open(path, "wb") as handle:
                    for chunk in response.iter_bytes(1024 * 1024):
                        handle.write(chunk)
        except httpx.HTTPError as exc:
            raise ComfyUnavailable(f"Could not download {output.filename}: {exc}") from exc
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    # ── Control ────────────────────────────────────────────────
    def cancel(self, prompt_id: str) -> None:
        """Delete a queued prompt, or interrupt it when already running."""
        try:
            self._post("/queue", json={"delete": [prompt_id]})
        except ComfyError:
            logger.debug("Could not delete ComfyUI prompt %s", prompt_id, exc_info=True)
        position = self.queue_position(prompt_id)
        if position == 0:
            try:
                self._post("/interrupt", json={"prompt_id": prompt_id})
            except ComfyError:
                logger.debug("Could not interrupt ComfyUI prompt %s", prompt_id, exc_info=True)

    def free(self, unload_models: bool = True) -> Dict[str, Any]:
        """Ask ComfyUI to release cached models and memory."""
        response = self._post("/free", json={
            "unload_models": bool(unload_models),
            "free_memory": True,
        })
        response.raise_for_status()
        return {"ok": True, "status": response.status_code}

    # ── Async runner ────────────────────────────────────────────
    async def run(
        self,
        workflow: Dict[str, Any],
        *,
        on_progress: Optional[Callable[[Dict[str, Any]], Any]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        timeout: Optional[float] = None,
        estimate_seconds: float = 30.0,
        poll_interval: float = 1.0,
    ) -> tuple[str, Dict[str, Any]]:
        """Submit and wait. Progress comes from the websocket, else polling."""
        prompt = parse_workflow(workflow)
        client_id = uuid.uuid4().hex
        prompt_id = await asyncio.to_thread(self.submit, prompt, client_id)
        watcher = asyncio.create_task(
            self._watch_websocket(prompt_id, client_id, prompt, on_progress)
        )
        started = time.monotonic()
        deadline = started + float(timeout or config.COMFY_JOB_TIMEOUT_SEC)
        last_emit = 0.0
        try:
            while True:
                if is_cancelled and is_cancelled():
                    await asyncio.to_thread(self.cancel, prompt_id)
                    raise ComfyCancelled("ComfyUI workflow cancelled")

                entry = await asyncio.to_thread(self.history, prompt_id)
                if entry:
                    error = self.execution_error(entry)
                    if error:
                        raise ComfyError(f"ComfyUI failed: {error}")
                    if on_progress:
                        await _emit(on_progress, {
                            "progress": 99,
                            "stage": "saving",
                            "message": "ComfyUI finished; collecting outputs…",
                        })
                    return prompt_id, entry

                now = time.monotonic()
                if now > deadline:
                    await asyncio.to_thread(self.cancel, prompt_id)
                    raise ComfyError(
                        f"ComfyUI workflow timed out after {int(deadline - started)}s"
                    )
                # The websocket is the fast path; this heartbeat covers a
                # missing websockets package, a dropped socket, or a slow node
                # that emits no progress.
                if on_progress and now - last_emit >= max(2.0, poll_interval):
                    last_emit = now
                    position = await asyncio.to_thread(self.queue_position, prompt_id)
                    elapsed = now - started
                    if position and position > 0:
                        payload = {
                            "progress": min(9, 4 + position),
                            "stage": "waiting",
                            "message": f"Waiting in ComfyUI (#{position})",
                        }
                    else:
                        ratio = min(1.0, elapsed / max(5.0, estimate_seconds))
                        payload = {
                            "progress": int(10 + 80 * ratio),
                            "stage": "denoising",
                            "message": "ComfyUI is working…",
                        }
                    await _emit(on_progress, payload)
                await asyncio.sleep(poll_interval)
        finally:
            watcher.cancel()
            try:
                await watcher
            except (asyncio.CancelledError, Exception):
                pass

    async def _watch_websocket(
        self,
        prompt_id: str,
        client_id: str,
        workflow: Dict[str, Any],
        on_progress: Optional[Callable[[Dict[str, Any]], Any]],
    ) -> None:
        """Best-effort per-step progress. Polling still guarantees completion."""
        if not on_progress:
            return
        try:
            import websockets
        except ImportError:
            logger.debug("websockets is not installed; using polling progress")
            return
        url = self.base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        url = f"{url}/ws?clientId={client_id}"
        try:
            async with websockets.connect(url, open_timeout=2, max_size=None, ping_interval=20) as socket:
                async for raw in socket:
                    if isinstance(raw, bytes):
                        continue
                    try:
                        message = json.loads(raw)
                    except ValueError:
                        continue
                    payload = self._progress_from_message(prompt_id, workflow, message)
                    if payload:
                        await _emit(on_progress, payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("ComfyUI websocket ended: %s", exc)

    @staticmethod
    def _progress_from_message(
        prompt_id: str,
        workflow: Dict[str, Any],
        message: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        kind = message.get("type")
        data = message.get("data") or {}
        if kind in {
            "execution_start", "execution_cached", "execution_success",
            "executing", "progress", "progress_state", "executed",
            "execution_error", "execution_interrupted",
        }:
            if data.get("prompt_id") not in (None, prompt_id):
                return None
        if kind == "execution_start":
            return {"progress": 10, "stage": "starting", "message": "ComfyUI started"}
        if kind == "execution_cached":
            cached = len(data.get("nodes") or [])
            if cached:
                return {"stage": "executing", "message": f"ComfyUI reused {cached} cached node(s)"}
            return None
        if kind == "executing":
            node_id = data.get("node")
            if not node_id:
                return {"progress": 98, "stage": "saving", "message": "ComfyUI finishing…"}
            node = workflow.get(str(node_id)) or {}
            node_name = node.get("class_type") or f"node {node_id}"
            return {"stage": "executing", "message": f"ComfyUI: {node_name}"}
        if kind == "progress":
            value = int(data.get("value") or 0)
            maximum = max(1, int(data.get("max") or 1))
            return {
                "progress": min(95, 10 + int(85 * value / maximum)),
                "stage": "denoising",
                "step": value,
                "total": maximum,
                "message": f"ComfyUI step {value}/{maximum}",
            }
        if kind == "progress_state":
            # ComfyUI 0.4x sends an aggregate node-state map rather than a
            # single progress message. Prefer the running node, else the
            # furthest-complete one.
            nodes = data.get("nodes") or {}
            rows = []
            for node_id, row in nodes.items():
                if not isinstance(row, dict):
                    continue
                maximum = max(1.0, float(row.get("max") or 1.0))
                rows.append((row.get("state") == "running", float(row.get("value") or 0) / maximum, node_id, row))
            if not rows:
                return None
            running = [row for row in rows if row[0]]
            chosen = max(running or rows, key=lambda row: row[1])
            value, maximum = chosen[1], max(1.0, float(chosen[3].get("max") or 1.0))
            current = int(round(chosen[1] * maximum))
            return {
                "progress": min(95, 10 + int(85 * chosen[1])),
                "stage": "denoising",
                "step": current,
                "total": int(maximum),
                "message": f"ComfyUI step {current}/{int(maximum)}",
            }
        if kind == "executed":
            return {"progress": 97, "stage": "saving", "message": "ComfyUI wrote an output"}
        if kind == "execution_success":
            return {"progress": 99, "stage": "saving", "message": "ComfyUI finished"}
        if kind == "execution_interrupted":
            return {"stage": "cancelled", "message": "ComfyUI interrupted"}
        if kind == "execution_error":
            detail = data.get("exception_message") or "ComfyUI execution error"
            return {"stage": "failed", "message": str(detail)}
        return None


async def _emit(callback: Callable[[Dict[str, Any]], Any], payload: Dict[str, Any]) -> None:
    try:
        result = callback(payload)
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        logger.debug("ComfyUI progress callback failed", exc_info=True)


def client() -> ComfyClient:
    """Factory used by routes and tasks; tests can monkeypatch this."""
    return ComfyClient()
