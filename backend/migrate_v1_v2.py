
import sqlite3
import os

# Database Path
DB_PATH = os.path.join(os.path.dirname(__file__), 'milimovideo.db')

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}. Nothing to migrate.")
        return

    print(f"Migrating database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Columns to add
    columns = [
        ("track_index", "INTEGER DEFAULT 0"),
        ("start_frame", "INTEGER DEFAULT 0"),
        ("trim_in", "INTEGER DEFAULT 0"),
        ("trim_out", "INTEGER DEFAULT 0")
    ]

    for col_name, col_type in columns:
        try:
            print(f"Adding column '{col_name}'...")
            cursor.execute(f"ALTER TABLE shot ADD COLUMN {col_name} {col_type}")
            print(f"Column '{col_name}' added successfully.")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print(f"Column '{col_name}' already exists. Skipping.")
            else:
                print(f"Error adding column '{col_name}': {e}")

    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
