import asyncio
import json
import logging
import time
from collections import deque
from typing import Any, Dict, List

from fastapi import Request
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger("events")

# ── Replay / backpressure ──────────────────────────────────────────
MAX_QUEUE_SIZE = 0            # unbounded: terminal events must never be dropped
MESSAGE_TTL_SECONDS = 120     # replay window
REPLAY_LIMIT = 1000
TERMINAL_EVENTS = {
    "complete", "error", "cancelled",
    "job.completed", "job.failed", "job.cancelled",
    "render_completed", "render_failed", "render_cancelled",
}


class EventManager:
    def __init__(self):
        self.clients: List[asyncio.Queue] = []
        self._seq = 0
        self._replay: deque[dict[str, Any]] = deque(maxlen=REPLAY_LIMIT)
        self._dropped = 0
        self._peak_pending = 0
        self._last_broadcast_ts = 0.0

    async def subscribe(self, request: Request, last_event_id: str | None = None):
        queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        try:
            last_seq = int(last_event_id) if last_event_id else 0
        except (TypeError, ValueError):
            last_seq = 0
        if last_seq:
            now = time.time()
            for message in list(self._replay):
                if int(message["id"]) > last_seq and now - message["_ts"] <= MESSAGE_TTL_SECONDS:
                    queue.put_nowait(message)
        self.clients.append(queue)
        logger.info(
            "SSE client connected (last_event_id=%s, replayed=%s, total=%s)",
            last_event_id, queue.qsize(), len(self.clients),
        )

        async def event_generator():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        data = await asyncio.wait_for(queue.get(), timeout=15.0)
                        yield {
                            "id": data["id"],
                            "event": data["event"],
                            "data": data["data"],
                        }
                    except asyncio.TimeoutError:
                        yield {"comment": "keepalive"}
            except asyncio.CancelledError:
                pass
            finally:
                if queue in self.clients:
                    self.clients.remove(queue)
                logger.info("SSE client disconnected. Remaining: %s", len(self.clients))

        return EventSourceResponse(event_generator())

    async def broadcast(self, event_type: str, data: Dict[str, Any]):
        """Send an event to every client and keep it for Last-Event-ID replay."""
        self._seq += 1
        message = {
            "id": str(self._seq),
            "event": event_type,
            "data": json.dumps(data, default=str),
            "_ts": time.time(),
        }
        self._replay.append(message)
        if event_type not in ("progress", "job.progress", "edit_progress"):
            logger.info("Broadcasting %s to %s clients (seq %s)", event_type, len(self.clients), self._seq)
        elif len(self.clients) > 0 and int(data.get("progress") or 0) % 10 == 0:
            logger.debug("Progress update: %s%%", data.get("progress"))

        self._last_broadcast_ts = time.time()
        stale: list[asyncio.Queue] = []
        pending = 0
        for queue in self.clients:
            try:
                queue.put_nowait(message)
                pending += queue.qsize()
            except Exception:
                self._dropped += 1
                stale.append(queue)
        self._peak_pending = max(self._peak_pending, pending)
        for queue in stale:
            if queue in self.clients:
                self.clients.remove(queue)
                logger.warning("Removed stale SSE client. Remaining: %s", len(self.clients))

    def stats(self) -> dict[str, Any]:
        return {
            "clients": len(self.clients),
            "sequence": self._seq,
            "replay_buffer": len(self._replay),
            "dropped": self._dropped,
            "peak_pending": self._peak_pending,
            "last_broadcast_age_s": (
                round(time.time() - self._last_broadcast_ts, 2)
                if self._last_broadcast_ts else None
            ),
        }


event_manager = EventManager()
