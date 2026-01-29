import sqlite3
import os

DB_PATH = "milimovideo.db"

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"No database found at {DB_PATH}. It will be created by the server.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get existing columns
    cursor.execute("PRAGMA table_info(project)")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"Existing columns: {columns}")

    # Define new columns and their types/defaults
    new_columns = [
        ("resolution_w", "INTEGER DEFAULT 768"),
        ("resolution_h", "INTEGER DEFAULT 512"),
        ("fps", "INTEGER DEFAULT 25"),
        ("seed", "INTEGER DEFAULT 42"),
        ("shots", "JSON DEFAULT '[]'")
    ]

    for col_name, col_def in new_columns:
        if col_name not in columns:
            print(f"Adding column: {col_name}")
            try:
                cursor.execute(f"ALTER TABLE project ADD COLUMN {col_name} {col_def}")
                print(f"Successfully added {col_name}")
            except Exception as e:
                print(f"Failed to add {col_name}: {e}")
        else:
            print(f"Column {col_name} already exists.")

    # Job Table Migration
    cursor.execute("PRAGMA table_info(job)")
    job_columns = [row[1] for row in cursor.fetchall()]
    print(f"Existing Job columns: {job_columns}")

    if "actual_frames" not in job_columns:
        print("Adding actual_frames to job table")
        try:
             cursor.execute("ALTER TABLE job ADD COLUMN actual_frames INTEGER")
             print("Successfully added actual_frames")
        except Exception as e:
             print(f"Failed to add actual_frames: {e}")
    else:
        print("Column actual_frames already exists in job.")

    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
