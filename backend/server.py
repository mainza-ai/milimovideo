import sys
import os
import asyncio
import logging
import logging.handlers
import mimetypes
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import config

# ── Structured Logging Setup ──────────────────────────────────────
# File handler: JSON-structured for machine parsing
# Console handler: Human-readable for development

class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter for production log files."""
    def format(self, record):
        import json, time
        log_entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "logger": record.name,
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)

# File handler: structured JSON (rotating, max 10MB, keep 3 backups)
file_handler = logging.handlers.RotatingFileHandler(
    "server.log", maxBytes=10*1024*1024, backupCount=3
)
file_handler.setFormatter(StructuredFormatter())
file_handler.setLevel(logging.INFO)

# Console handler: human-readable  
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
console_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)
BACKGROUND_TASKS: set[asyncio.Task] = set()

# Ensure directories exist
os.makedirs(config.PROJECTS_DIR, exist_ok=True)
legacy_upload_dir = os.path.join(config.BACKEND_DIR, "uploads")
legacy_generated_dir = os.path.join(config.BACKEND_DIR, "generated")
os.makedirs(legacy_upload_dir, exist_ok=True)
os.makedirs(legacy_generated_dir, exist_ok=True)

# Imports after config setup
from database import init_db, engine, Job, QueuedJob
from sqlmodel import Session, select
from routes import router as api_router
from events import event_manager
from memory_manager import memory_manager
from services.memory import budget as memory_budget
from services.memory import sampler as memory_sampler

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Milimo Video Backend...")
    init_db()
    
    # Initialize Model Manager (Lazy load, but we can warm up if needed)
    # from model_engine import manager
    
    # Capture main loop for thread-safe operations in tasks
    import job_utils
    import asyncio
    job_utils.global_loop = asyncio.get_running_loop()

    # Cap this process so PyTorch fails before macOS thrashes, then start the
    # idle/pressure monitor. Both are best-effort on non-MPS hosts.
    memory_budget.set_mps_cap()
    memory_manager.start_monitor()
    
    # Startup Cleanup: Fail any jobs that were 'processing' when server died
    try:
        with Session(engine) as session:
            processing = session.exec(
                select(Job).where(Job.status == "processing")
            ).all()
            for job in processing:
                job.status = "failed"
                job.error_message = "Server restarted while the job was running"
                job.completed_at = datetime.utcnow()
                session.add(job)

            pending = session.exec(
                select(Job).where(Job.status.in_(["pending", "queued"]))
            ).all()
            for job in pending:
                durable = session.exec(
                    select(QueuedJob)
                    .where(QueuedJob.job_id == job.id)
                    .where(QueuedJob.status.in_(["queued", "running"]))
                ).first()
                if not durable:
                    job.status = "failed"
                    job.error_message = "Server restarted before dispatch"
                    job.completed_at = datetime.utcnow()
                session.add(job)
            session.commit()
            if processing or pending:
                logger.warning(
                    f"Recovered startup state: {len(processing)} interrupted, "
                    f"{len(pending)} pending/queued evaluated"
                )
        # Resume durable GPU dispatch records after the zombie pass. Keep a
        # strong reference so the task cannot be garbage-collected mid-flight.
        resume_task = asyncio.create_task(job_utils.resume_queued_jobs())
        BACKGROUND_TASKS.add(resume_task)
        resume_task.add_done_callback(BACKGROUND_TASKS.discard)
    except Exception as e:
        logger.error(f"Startup cleanup failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    memory_manager.stop_monitor()
    
    # Signal threads to stop
    count = job_utils.cancel_all_jobs()
    if count > 0:
        logger.info(f"Waiting for {count} jobs to cancel...")
        # Give threads a chance to hit the cancellation check
        await asyncio.sleep(2.0)

if config.API_HOST not in {"127.0.0.1", "localhost", "::1"} and not config.API_TOKEN:
    raise RuntimeError(
        "MILIMO_HOST is not loopback but MILIMO_API_TOKEN is empty. "
        "Refusing to expose an unauthenticated local studio to the network."
    )

app = FastAPI(lifespan=lifespan)

# CORS: local-only by default. Add origins through MILIMO_CORS_ORIGINS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    allow_headers=["Range", "Content-Type", "Authorization", "X-API-Token"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)

PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

def _request_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (
        request.headers.get("x-api-token", "").strip()
        or request.query_params.get("token", "").strip()
    )

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    """Guarantee CORS headers on allowed responses, including errors."""
    import time as time_module
    import uuid

    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    started = time_module.perf_counter()
    origin = request.headers.get("origin", "")
    origin_allowed = not origin or origin in config.CORS_ORIGINS

    if request.method == "OPTIONS":
        return Response(
            status_code=200 if origin_allowed else 403,
            headers={
                "Access-Control-Allow-Origin": origin if origin_allowed else "null",
                "Vary": "Origin",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, PATCH, OPTIONS, HEAD",
                "Access-Control-Allow-Headers": "Range, Content-Type, Authorization, X-API-Token",
                "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges, Content-Length",
                "Access-Control-Max-Age": "86400",
            },
        )

    if not origin_allowed:
        return Response(status_code=403, content="Origin not allowed")

    # Optional bearer-token gate. Defaults to off for loopback single-user use.
    path = request.url.path
    share_route = path.startswith("/share/")
    if config.API_TOKEN and path not in PUBLIC_PATHS and not share_route:
        import hmac
        if not hmac.compare_digest(_request_token(request), config.API_TOKEN):
            return Response(status_code=401, content="Unauthorized")

    response = await call_next(request)
    if origin_allowed and origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Expose-Headers"] = "Content-Range, Accept-Ranges, Content-Length"
    response.headers["X-Request-ID"] = request_id
    elapsed_ms = (time_module.perf_counter() - started) * 1000
    logger.info(
        "request %s %s %s -> %s %.1fms",
        request_id, request.method, request.url.path, response.status_code, elapsed_ms,
    )
    return response


@app.get("/health")
async def health():
    import shutil
    import torch
    from job_utils import is_gpu_busy, is_queue_paused
    from memory_manager import memory_manager

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        free_bytes = shutil.disk_usage(config.PROJECTS_DIR).free
    except OSError:
        free_bytes = None
    memory = memory_manager.status()
    return {
        "status": "ok",
        "device": device,
        "api_host": config.API_HOST,
        "auth_required": bool(config.API_TOKEN),
        "gpu": is_gpu_busy(),
        "queue_paused": is_queue_paused(),
        "slots": memory,
        "memory": memory,
        "sampler": memory_sampler.current(),
        "disk_free_bytes": free_bytes,
    }

# Include Routers (Must be before static mounts to avoid shadowing)
app.include_router(api_router)

# --- Safari-Compatible Media Serving (Range Requests) ---
# Safari REQUIRES HTTP Range requests (206 Partial Content) for <video>/<audio> playback.
# FastAPI's StaticFiles doesn't always handle Range headers properly for media.
# This endpoint intercepts media file requests and serves them with full Range support.
MEDIA_EXTENSIONS = {'.mp4', '.webm', '.mp3', '.wav', '.ogg', '.m4a', '.aac'}
MEDIA_DIRS = {
    "/projects": config.PROJECTS_DIR,
    "/uploads": legacy_upload_dir,
    "/generated": legacy_generated_dir,
}

CORS_HEADERS = {
    "Access-Control-Allow-Methods": "GET, OPTIONS, HEAD, PATCH",
    "Access-Control-Allow-Headers": "Range, Content-Type, Authorization, X-API-Token",
    "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges, Content-Length",
}

@app.get("/projects/{file_path:path}")
@app.get("/uploads/{file_path:path}")
@app.get("/generated/{file_path:path}")
async def serve_media(request: Request, file_path: str):
    """Serve media files with HTTP Range request support for Safari."""
    # Determine which base directory to use from the route
    route_prefix = request.url.path.split('/')[1]  # 'projects', 'uploads', or 'generated'
    base_dir = MEDIA_DIRS.get(f"/{route_prefix}")
    if not base_dir:
        return Response(status_code=404)

    base_dir = os.path.realpath(base_dir)
    full_path = os.path.realpath(os.path.join(base_dir, file_path))

    # Security: commonpath, not startswith (which accepts sibling prefixes).
    try:
        if os.path.commonpath([base_dir, full_path]) != base_dir:
            return Response(status_code=403)
    except ValueError:
        return Response(status_code=403)

    if not os.path.isfile(full_path):
        return Response(status_code=404)

    content_type, _ = mimetypes.guess_type(full_path)
    content_type = content_type or "application/octet-stream"
    file_size = os.path.getsize(full_path)
    range_header = request.headers.get("range")

    if range_header:
        try:
            range_spec = range_header.strip().lower()
            if not range_spec.startswith("bytes="):
                raise ValueError("unsupported range unit")
            start_s, _, end_s = range_spec[6:].partition("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else file_size - 1
            start = max(0, start)
            end = min(end, file_size - 1)
            if start > end:
                raise ValueError("invalid range")
        except (TypeError, ValueError):
            return Response(
                status_code=416,
                headers={**CORS_HEADERS, "Content-Range": f"bytes */{file_size}"},
            )

        content_length = end - start + 1

        def _range_stream(chunk_size: int = 1024 * 1024):
            with open(full_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk = f.read(min(chunk_size, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            _range_stream(),
            status_code=206,
            media_type=content_type,
            headers={
                **CORS_HEADERS,
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Cache-Control": "public, max-age=3600",
            },
        )

    return FileResponse(
        full_path,
        media_type=content_type,
        headers={
            **CORS_HEADERS,
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
        },
    )

# Static Mounts - Fallback for non-media files (HTML, images, etc.)
# Note: The media endpoint above takes priority for matching paths due to route precedence.
app.mount("/projects", StaticFiles(directory=config.PROJECTS_DIR), name="projects")
app.mount("/uploads", StaticFiles(directory=legacy_upload_dir), name="uploads")
app.mount("/generated", StaticFiles(directory=legacy_generated_dir), name="generated") # Legacy

# Event Manager Endpoint
@app.get("/events")
async def events(request: Request):
    last_event_id = request.headers.get("last-event-id") or request.query_params.get("last_event_id")
    return await event_manager.subscribe(request, last_event_id=last_event_id)

if __name__ == "__main__":
    # Reload is opt-in: the watcher walks the repo (including checkpoint
    # directories) and burns a core while a generation is running.
    reload_enabled = os.environ.get("MILIMO_RELOAD", "0") == "1"
    uvicorn.run(
        "server:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=reload_enabled,
        reload_includes=["*.py"] if reload_enabled else None,
        reload_excludes=[
            "models/**", "checkpoints/**", "projects/**", "outputs/**", "**/*.safetensors",
        ] if reload_enabled else None,
    )
