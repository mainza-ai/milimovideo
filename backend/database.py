import os
from sqlmodel import SQLModel, create_engine, Session, Field, Relationship
from typing import Optional, List
from sqlalchemy import Column, JSON
from datetime import datetime
import uuid

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
    
    shots: List["Shot"] = Relationship(back_populates="project", sa_relationship_kwargs={"cascade": "all, delete-orphan"})
    scenes: List["Scene"] = Relationship(back_populates="project", sa_relationship_kwargs={"cascade": "all, delete-orphan"})

class Scene(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    project_id: str = Field(index=True, foreign_key="project.id")
    index: int
    name: str # "Scene 1: The Chase"
    script_content: Optional[str] = None # The text segment for this scene
    
    project: Optional[Project] = Relationship(back_populates="scenes")
    # shots: List["Shot"] = Relationship(back_populates="scene") # Future

class Shot(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    scene_id: Optional[str] = Field(default=None, index=True) # Optional link to Scene
    project_id: str = Field(index=True, foreign_key="project.id") # Redundant but useful for queries
    index: Optional[int] = None
    track_index: int = Field(default=0) # 0=Main, 1=Overlay, 2=Audio?
    start_frame: int = Field(default=0) # Absolute start time (frames) for Non-Linear tracks
    trim_in: int = Field(default=0) # Trim from start (frames)
    trim_out: int = Field(default=0) # Trim from end (frames)
    
    project: Optional[Project] = Relationship(back_populates="shots")
    
    # Content
    action: Optional[str] = None # "Hero runs down the alley" (Storyboard context)
    prompt: Optional[str] = None # (Generator context)
    dialogue: Optional[str] = None
    character: Optional[str] = None
    
    # Generation Params (Persisted from ShotConfig)
    negative_prompt: Optional[str] = ""
    seed: int = 42
    width: int = 768
    height: int = 512
    num_frames: int = 121
    fps: int = 25
    cfg_scale: float = 3.0
    enhance_prompt: bool = True
    upscale: bool = False
    pipeline_override: str = "auto"
    auto_continue: bool = Field(default=False)
    
    # Generation State
    prompt_enhanced: Optional[str] = None # Legacy field
    enhanced_prompt_result: Optional[str] = None # For frontend display
    status: str = "pending" # pending, generating, completed, failed
    last_job_id: Optional[str] = None
    
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

class Job(SQLModel, table=True):
    id: str = Field(primary_key=True) # Job ID from worker
    project_id: Optional[str] = Field(default=None, index=True)
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

# Database Setup
from config import DATABASE_URL
# check_same_thread=False is required for SQLite when using FastAPI BackgroundTasks
# Increased pool size to prevent "QueuePool limit" errors during heavy polling
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, pool_size=20, max_overflow=40)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
