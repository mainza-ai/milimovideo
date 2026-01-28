import os
from sqlmodel import SQLModel, create_engine, Session, Field
from typing import Optional, List
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

# Database Setup
DATABASE_URL = "sqlite:///./milimovideo.db"
# check_same_thread=False is required for SQLite when using FastAPI BackgroundTasks
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
