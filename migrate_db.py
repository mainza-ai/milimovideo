import sqlite3
import os

# Path to database file
DB_PATH = "backend/milimovideo.db"

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Check Shot Table
    print("Checking 'shot' table...")
    cursor.execute("PRAGMA table_info(shot)")
    columns = [info[1] for info in cursor.fetchall()]
    print(f"Current columns: {columns}")

    # Columns to add to 'shot'
    new_columns = {
        "scene_id": "TEXT",
        "action": "TEXT",
        "dialogue": "TEXT",
        "character": "TEXT",
        "auto_continue": "BOOLEAN DEFAULT 0"
    }

    for col, dtype in new_columns.items():
        if col not in columns:
            print(f"Adding column '{col}' to 'shot' table...")
            try:
                cursor.execute(f"ALTER TABLE shot ADD COLUMN {col} {dtype}")
                print(f"  - Added {col}")
            except Exception as e:
                print(f"  - Failed to add {col}: {e}")
        else:
            print(f"  - {col} exists")

    # 2. Check Project Table
    print("\nChecking 'project' table...")
    cursor.execute("PRAGMA table_info(project)")
    p_columns = [info[1] for info in cursor.fetchall()]
    print(f"Current columns: {p_columns}")

    if "script_content" not in p_columns:
        print("Adding column 'script_content' to 'project' table...")
        try:
            cursor.execute("ALTER TABLE project ADD COLUMN script_content TEXT")
            print("  - Added script_content")
        except Exception as e:
            print(f"  - Failed to add script_content: {e}")
            
    # 3. Check Scene Table
    print("\nChecking 'scene' table...")
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='scene'")
        if not cursor.fetchone():
             print("Table 'scene' does not exist. It should be created by SQLModel startup. Skipping manual creation to avoid conflicts, or should we create it?")
             # SQLModel create_all usually handles new tables. If server restart runs create_all, it will be fine.
             # But let's verify if we need to do anything. The user restart likely creates it.
             pass
        else:
             print("Table 'scene' exists.")
    except Exception as e:
        print(e)
            
    conn.commit()
    conn.close()
    print("\nMigration complete.")

if __name__ == "__main__":
    migrate()
