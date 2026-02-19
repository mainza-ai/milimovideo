import sqlite3
import os

DB_PATH = "backend/milimovideo.db"

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Database {DB_PATH} not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if column exists
        cursor.execute("PRAGMA table_info(shot)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if "timeline" not in columns:
            print("Adding 'timeline' column to 'shot' table...")
            cursor.execute("ALTER TABLE shot ADD COLUMN timeline TEXT")
            conn.commit()
            print("Migration successful.")
        else:
            print("'timeline' column already exists.")
            
    except Exception as e:
        print(f"Migration failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
