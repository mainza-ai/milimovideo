# Milimo Video - Data Models

## 1. Entity-Relationship Diagram

```mermaid
erDiagram
    Project ||--o{ Scene : contains
    Project ||--o{ Shot : "flat list (legacy)"
    Scene ||--o{ Shot : contains
    Shot ||--o{ ConditioningItem : "has input"
    Shot }|..|{ Asset : "uses"
    
    Project {
        string id PK
        string name
        int fps
        int width
        int height
        int seed
    }

    Scene {
        string id PK
        string name
        int index
        string script_content
    }

    Shot {
        string id PK
        string prompt
        int duration_frames
        int start_frame
        int track_index
        string status
        string video_url
    }

    ConditioningItem {
        string id PK
        string type "image|video"
        string path
        float strength
    }

    Job {
        string id PK
        string shot_id FK
        string status
        float progress
        string result_path
    }

    Shot ||--o{ Job : "triggers"
```

## 2. Core Structures

### Project
The root container.
- **Structure**: Defined in `database.py` (SQL) and `types.ts` (Frontend).
- **Storage**: `projects/{id}/project.json` + SQLite metadata.

### Shot
The atomic unit of generation AND editing.
- **Dual Nature**:
    1.  **Generation Spec**: Contains `prompt`, `seed`, `cfg`, `model_params`.
    2.  **Timeline Clip**: Contains `trackIndex`, `startFrame`, `trimIn`, `trimOut`.
- **Properties**:
    - `timeline`: Array of `ConditioningItem` (images/videos used as inputs).
    - `main_asset`: The resulting video file.

### Asset (Story Element)
Persistent visual identities.
- **Types**: Character, Location, Object.
- **Usage**: Stored in `ElementSlice`. Used to inject consistent visual conditioning into shots via `triggerWord`.

## 3. Database Schema (SQLite)

The backend uses SQLModel (SQLAlchemy).

| Table | Primary Key | Description |
|---|---|---|
| `Project` | `id` (UUID) | Global project settings. |
| `Job` | `id` (UUID) | Async task tracking. Linked to shots. |
| `Shot` | `id` (UUID) | *[Planned]* Moving full shot state to DB (currently hybrid JSON/DB). |
