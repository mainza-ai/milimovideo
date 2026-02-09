# Milimo Video - Dependency Graph

## 1. Module Dependencies

This graph shows the high-level import relationships between key modules.

```mermaid
graph TD
    subgraph "Entry Points"
        Server[server.py]
        Worker[worker.py]
    end

    subgraph "Core Logic"
        Tasks[tasks/video.py]
        DB[database.py]
        Config[config.py]
        Utils[job_utils.py]
    end

    subgraph "AI Models"
        FluxW[models/flux_wrapper.py]
        LTX_Pipe[ltx_pipelines/ti2vid_two_stages.py]
        LTX_Core[ltx_core/*]
    end

    Server --> DB
    Server --> Tasks
    Server --> Config
    
    Worker --> Tasks
    Worker --> Utils
    
    Tasks --> FluxW
    Tasks --> LTX_Pipe
    Tasks --> Utils
    Tasks --> DB
    
    FluxW --> LTX_Core : "Reuses VAE/TextEnc"
    LTX_Pipe --> LTX_Core
```

## 2. Key File Analysis

### `server.py`
The API Gateway. Importing `tasks` implies it knows about the *definition* of tasks, but execution is often deferred.

### `tasks/video.py`
The heavy hitter. It imports everything:
- `job_utils`: For DB updates.
- `PIL`, `numpy`, `torch`: For data manipulation.
- `ltx_pipelines`: For the actual generation.

### `database.py`
The bedrock. Imported by almost everyone. Contains `SQLModel` definitions. Circular dependencies here are the most likely source of bugs (e.g., if `database.py` tried to import `tasks`).
