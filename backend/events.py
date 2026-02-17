import asyncio
import logging
import json
import time
from typing import List, Dict, Any
from fastapi import Request
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger("events")

# ── SSE Backpressure Settings ──────────────────────────────────────
MAX_QUEUE_SIZE = 100          # Max events per client queue
MESSAGE_TTL_SECONDS = 30      # Drop messages older than this


class EventManager:
    def __init__(self):
        self.clients: List[asyncio.Queue] = []

    async def subscribe(self, request: Request):
        queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self.clients.append(queue)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        async def event_generator():
            try:
                while True:
                    # Check for disconnection
                    if await request.is_disconnected():
                        break
                        
                    # Get event with timeout to periodically check disconnection
                    try:
                        data = await asyncio.wait_for(queue.get(), timeout=15.0)
                        
                        # TTL check: skip stale messages
                        ts = data.get("_timestamp", 0)
                        if ts and (time.time() - ts) > MESSAGE_TTL_SECONDS:
                            continue
                        
                        # Remove internal timestamp before yielding
                        data_clean = {k: v for k, v in data.items() if k != "_timestamp"}
                        yield data_clean
                        
                    except asyncio.TimeoutError:
                        # Send keepalive comment to detect disconnects
                        yield {"comment": "keepalive"}
                        
            except asyncio.CancelledError:
                pass
            finally:
                if queue in self.clients:
                    self.clients.remove(queue)
                logger.info(f"Client disconnected. Remaining: {len(self.clients)}")

        return EventSourceResponse(event_generator())

    async def broadcast(self, event_type: str, data: Dict[str, Any]):
        """
        Broadcasts an event to all connected clients with backpressure.
        If a client's queue is full, oldest messages are dropped.
        """
        message = {
            "event": event_type,
            "data": json.dumps(data),
            "_timestamp": time.time()
        }
        
        # Log non-progress events
        if event_type not in ["progress", "edit_progress"]:
            logger.info(f"Broadcasting {event_type} to {len(self.clients)} clients")
        elif len(self.clients) > 0 and data.get("progress", 0) % 10 == 0:
             logger.debug(f"Progress update: {data.get('progress')}%")

        stale_clients = []
        for queue in self.clients:
            try:
                if queue.full():
                    # Drop oldest message to make room (backpressure)
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                queue.put_nowait(message)
            except Exception:
                stale_clients.append(queue)
        
        # Remove stale/broken clients
        for stale in stale_clients:
            if stale in self.clients:
                self.clients.remove(stale)
                logger.warning(f"Removed stale SSE client. Remaining: {len(self.clients)}")

# Global Instance
event_manager = EventManager()
