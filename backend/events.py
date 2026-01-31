import asyncio
import logging
import json
from typing import List, Dict, Any
from fastapi import Request
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger("events")

class EventManager:
    def __init__(self):
        self.clients: List[asyncio.Queue] = []

    async def subscribe(self, request: Request):
        queue = asyncio.Queue()
        self.clients.append(queue)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        async def event_generator():
            try:
                while True:
                    # Check for disconnection
                    if await request.is_disconnected():
                        break
                        
                    # Get event
                    data = await queue.get()
                    yield data
            except asyncio.CancelledError:
                pass
            finally:
                self.clients.remove(queue)
                logger.info(f"Client disconnected. Remaining: {len(self.clients)}")

        return EventSourceResponse(event_generator())

    async def broadcast(self, event_type: str, data: Dict[str, Any]):
        """
        Broadcasts an event to all connected clients.
        Format: event: type \n data: json_string
        """
        message = {
            "event": event_type,
            "data": json.dumps(data)
        }
        
        # Log frequent events only periodically to avoid spam
        if event_type not in ["progress", "edit_progress"]:
            logger.info(f"Broadcasting {event_type} to {len(self.clients)} clients")
        elif len(self.clients) > 0 and data.get("progress", 0) % 10 == 0:
             logger.debug(f"Progress update: {data.get('progress')}%")

        # We need to dispatch to all queues
        
        # We need to dispatch to all queues
        # Use asyncio.gather to avoid blocking? 
        # Actually EventSourceResponse expects dict or specific format.
        # sse_starlette handles it if we yield dict.
        
        for queue in self.clients:
            await queue.put(message)

# Global Instance
event_manager = EventManager()
