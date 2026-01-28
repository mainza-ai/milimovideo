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

    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
