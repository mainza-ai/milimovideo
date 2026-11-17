import os
from sqlmodel import SQLModel, create_engine, Session, Field, Relationship
from typing import Optional, List
from sqlalchemy import Column, JSON, text
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

# Define Models directly here for simplicity in this phase

class Project(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    # Settings
    resolution_w: int = 768
    resolution_h: int = 512
    fps: int = 25
    seed: int = 42
    # Storyboard Support
    script_content: Optional[str] = None # Raw script text
    # Optimistic concurrency. Clients send the revision they read; a mismatch
    # returns 409 instead of silently overwriting a newer project.
    revision: int = Field(default=0)
    # Semantic board layout and ephemeral view are separate saves by design.
    board_layout: Optional[str] = None
    board_revision: int = Field(default=0)
    view_state: Optional[str] = None
    captions: Optional[str] = None  # JSON list of {id,start_frame,end_frame,text}

    shots: List["Shot"] = Relationship(back_populates="project", sa_relationship_kwargs={"cascade": "all, delete-orphan"})
    scenes: List["Scene"] = Relationship(back_populates="project", sa_relationship_kwargs={"cascade": "all, delete-orphan"})

class Scene(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    project_id: str = Field(index=True, foreign_key="project.id")
    index: int
    name: str # "Scene 1: The Chase"
    script_content: Optional[str] = None # The text segment for this scene
    
    project: Optional[Project] = Relationship(back_populates="scenes")
    shots: List["Shot"] = Relationship(back_populates="scene", sa_relationship_kwargs={"cascade": "all, delete-orphan"})

class Shot(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    scene_id: Optional[str] = Field(default=None, index=True, foreign_key="scene.id") # Link to Scene
    project_id: str = Field(index=True, foreign_key="project.id") # Redundant but useful for queries
    index: Optional[int] = None
    # Project-wide story order. Canvas x/y and created_at never define order.
    sequence_index: Optional[int] = Field(default=None, index=True)
    track_index: int = Field(default=0) # 0=Main, 1=Overlay, 2=Audio?
    start_frame: int = Field(default=0) # Absolute start time (frames) for Non-Linear tracks
    trim_in: int = Field(default=0) # Trim from start (frames)
    trim_out: int = Field(default=0) # Trim from end (frames)
    
    project: Optional[Project] = Relationship(back_populates="shots")
    scene: Optional[Scene] = Relationship(back_populates="shots")
    
    # Content
    action: Optional[str] = None # "Hero runs down the alley" (Storyboard context)
    prompt: Optional[str] = None # (Generator context)
    dialogue: Optional[str] = None
    character: Optional[str] = None
    matched_elements: Optional[str] = None  # JSON: [{element_id, element_name, element_type, image_url, confidence, ...}]
    
    # Generation Params (Persisted from ShotConfig)
    negative_prompt: Optional[str] = ""
    seed: int = 42
    width: int = 768
    height: int = 512
    num_frames: int = 121
    fps: int = 25
    cfg_scale: float = 1.0
    enhance_prompt: bool = True
    upscale: bool = False
    pipeline_override: str = "auto"
    auto_continue: bool = Field(default=False)
    shot_type: Optional[str] = None  # close_up, medium, wide, establishing, insert, tracking
    timeline: Optional[str] = None # JSON list of objects: [{path, frame_index, strength, type}]
    
    # Generation State
    prompt_enhanced: Optional[str] = None # Legacy field
    enhanced_prompt_result: Optional[str] = None # For frontend display
    status: str = "pending" # pending, generating, completed, failed
    last_job_id: Optional[str] = None
    selected_take_id: Optional[str] = None  # Job id of the selected immutable take
    audio_gain: float = 1.0
    audio_muted: bool = False

    # Results
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    duration: float = 4.0
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Element(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    project_id: str = Field(index=True)
    name: str          # e.g., "Hero", "Kitchen"
    trigger_word: str  # e.g., "@Hero"
    type: str          # "character", "location", "object"
    description: str   # "A tall woman with red hair"
    image_path: Optional[str] = None # Reference image path
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Asset(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    project_id: Optional[str] = Field(default=None, index=True) # Can be global if None
    type: str # 'image', 'video'
    path: str
    url: str
    filename: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # Metadata
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None
    meta_json: Optional[str] = None # JSON string for prompt, seed, refs, etc.

class QueuedJob(SQLModel, table=True):
    """Durable dispatch record for GPU work.

    The Job row is user-facing state; this row is the scheduler's ledger. It
    exists before a background task starts, so a restart can recover queued work.
    """
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    job_id: str = Field(index=True)
    kind: str = Field(index=True)  # video | image | inpaint | element
    params_json: str = "{}"
    status: str = Field(default="queued", index=True)  # queued | running | completed | failed
    attempts: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class Job(SQLModel, table=True):
    id: str = Field(primary_key=True) # Job ID from worker
    project_id: Optional[str] = Field(default=None, index=True)
    shot_id: Optional[str] = Field(default=None, index=True)
    type: str = "generation"
    status: str = "pending" # pending, processing, completed, failed, cancelled
    progress: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    # Result
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    # Config
    prompt: Optional[str] = None
    params_json: Optional[str] = None
    enhanced_prompt: Optional[str] = None
    status_message: Optional[str] = None
    actual_frames: Optional[int] = None
    thumbnail_path: Optional[str] = None
    # Recovery/idempotency metadata for the unified scheduler.
    resume_policy: str = Field(default="fail")  # fail | resume
    request_id: Optional[str] = Field(default=None, index=True)

class ShareLink(SQLModel, table=True):
    """Capability link for a reviewer.

    Only the hash is stored; the token is returned once when the link is
    created. Share routes are read-only plus comments/approvals.
    """
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    project_id: str = Field(index=True)
    token_hash: str = Field(index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    revoked: bool = False


class ReviewItem(SQLModel, table=True):
    """Frame-accurate review comments and per-shot approvals.

    Comments are immutable history unless resolved; approvals are upserted per
    project+shot so the latest decision is the only one that gates a cut.
    """
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    project_id: str = Field(index=True)
    shot_id: Optional[str] = Field(default=None, index=True)
    kind: str = Field(default="comment", index=True)  # comment | approval
    frame: int = Field(default=0)
    body: str = ""
    author: str = "local"
    status: str = Field(default="open", index=True)  # open | resolved | approved | changes_requested
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# Database Setup
from config import DATABASE_URL
# check_same_thread=False is required for SQLite when using FastAPI BackgroundTasks
# Increased pool size to prevent "QueuePool limit" errors during heavy polling
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, pool_size=20, max_overflow=40)

def _add_missing_columns() -> None:
    """Idempotent SQLite schema repair for installs created before a model field.

    `create_all()` never alters existing tables. This is deliberately small and
    explicit; new schema work should move to Alembic rather than growing this
    list forever.
    """
    migrations = {
        "project": {
            "revision": "INTEGER NOT NULL DEFAULT 0",
            "board_layout": "TEXT",
            "board_revision": "INTEGER NOT NULL DEFAULT 0",
            "view_state": "TEXT",
            "captions": "TEXT",
        },
        "shot": {
            "sequence_index": "INTEGER",
            "selected_take_id": "TEXT",
            "audio_gain": "FLOAT NOT NULL DEFAULT 1.0",
            "audio_muted": "BOOLEAN NOT NULL DEFAULT 0",
        },
        "job": {
            "resume_policy": "TEXT NOT NULL DEFAULT 'fail'",
            "request_id": "TEXT",
            "shot_id": "TEXT",
        },
    }
    with engine.begin() as conn:
        for table, columns in migrations.items():
            existing = {
                row[1]
                for row in conn.execute(text(f"PRAGMA table_info({table})"))
            }
            if not existing:
                continue  # fresh table will be created by create_all
            for name, ddl in columns.items():
                if name not in existing:
                    logger.info(f"Schema repair: adding {table}.{name}")
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))

        # Backfill story order for existing projects. NULL means "never ordered".
        for ddl in (
            "CREATE INDEX IF NOT EXISTS ix_shot_sequence_index ON shot(sequence_index)",
            "CREATE INDEX IF NOT EXISTS ix_job_shot_id ON job(shot_id)",
        ):
            try:
                conn.execute(text(ddl))
            except Exception:
                pass

        project_ids = [
            row[0]
            for row in conn.execute(text(
                "SELECT DISTINCT project_id FROM shot WHERE sequence_index IS NULL"
            ))
        ]
        for project_id in project_ids:
            shot_ids = [
                row[0]
                for row in conn.execute(text(
                    "SELECT id FROM shot WHERE project_id = :pid "
                    "ORDER BY created_at ASC, id ASC"
                ), {"pid": project_id})
            ]
            for order, shot_id in enumerate(shot_ids):
                conn.execute(text(
                    "UPDATE shot SET sequence_index = :idx WHERE id = :sid"
                ), {"idx": order, "sid": shot_id})
        if project_ids:
            logger.info(f"Schema repair: backfilled sequence_index for {len(project_ids)} project(s)")


def init_db():
    SQLModel.metadata.create_all(engine)
    _add_missing_columns()

def get_session():
    with Session(engine) as session:
        yield session
