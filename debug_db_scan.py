from sqlmodel import Session, select, create_engine
from backend.database import Shot, Project
from backend.config import DATABASE_URL

# Force absolute path for DB if needed, but DATABASE_URL should work if running from root/backend
if "sqlite" in DATABASE_URL:
    db_path = DATABASE_URL.replace("sqlite:///", "")
    import os
    if not os.path.exists(db_path):
        # Try adjusting relative to backend
        db_path = os.path.join("backend", "milimovideo.db")
        DATABASE_URL = f"sqlite:///{db_path}"

engine = create_engine(DATABASE_URL)

def scan_shots():
    with Session(engine) as session:
        shots = session.exec(select(Shot)).all()
        print(f"Found {len(shots)} shots.")
        for shot in shots:
            if shot.video_url:
                print(f"Shot ID: {shot.id[:8]} | Project: {shot.project_id[:8]} | URL: {shot.video_url}")
            else:
                print(f"Shot ID: {shot.id[:8]} | No Video URL")

if __name__ == "__main__":
    scan_shots()
