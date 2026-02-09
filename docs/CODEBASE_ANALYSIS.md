# Milimo Video - Codebase Analysis

**Date:** 2026-02-06
**Version:** 0.1.0-Analysis

## Executive Summary

This document serves as the entry point for the comprehensive architectural analysis of the Milimo Video codebase. The system is a hybrid local-first application combining a React/Zustand frontend with a Python/FastAPI backend, orchestrating complex AI pipelines (LTX-2, Flux, SAM).

## Documentation Artifacts

The analysis has been broken down into the following detailed documents:

| Document | Description |
|---|---|
| **[01_system_architecture.md](./01_system_architecture.md)** | High-level system context, container diagrams, and core subsystem interactions. |
| **[02_data_models.md](./02_data_models.md)** | Entity-Relationship diagrams for Projects, Scenes, Shots, and Assets. Database schema details. |
| **[03_ai_pipelines.md](./03_ai_pipelines.md)** | Deep dive into the `LTX-2` generation pipeline, conditioning mechanisms, and latent handoffs. |
| **[04_frontend_state.md](./04_frontend_state.md)** | Analysis of the `timelineStore` (Zustand), slices, and optimization strategies for 60fps playback. |
| **[05_execution_flow.md](./05_execution_flow.md)** | Sequence diagrams illustrating the end-to-end flow of a video generation task. |
| **[06_file_dependency.md](./06_file_dependency.md)** | Dependency graphs showing how core modules interact. |

## Key Findings & Observations

### 1. The "God Store" Pattern
The frontend relies heavily on a single `timelineStore`. While this simplifies state access, it poses performance risks. The use of `transientDuration` and `useShallow` indicates an awareness of this, but careful management of selectors is crucial as the app scales.

### 2. The Hybrid Job Queue
The backend uses a simple in-memory queue (via `worker.py` thread). This is sufficient for a single-user local app but is a potential bottleneck if multiple generations are queued. The data flow relies on `job_utils` to update SQLite, acting as a persistent state for jobs.

### 3. LTX-2 Integration complexity
The LTX-2 pipeline is not just a library call; it involves complex memory management (`ModelLedger`) to swap models in and out of VRAM. The custom `image_conditionings_by_replacing_latent` function is a critical piece of IP that enables the "Cinematic" control.

### 4. Data Model Evolution
The transition from a flat `shots` list to a hierarchical `Scene -> Shot` model is in progress. `database.py` reflects this, but the frontend `ProjectSlice` still largely operates on the flat mechanism. This will be a key area for refactoring.

## Recommendations

1.  **Strict Store Selectors**: Enforce `useShallow` for all timeline components to prevent wasted renders.
2.  **Job Persistence**: Ensure the worker thread can recover from a crash by checking the DB for `PENDING` jobs on startup.
3.  **Type Safety**: Share types between Backend (Pydantic) and Frontend (TypeScript) more strictly, possibly via code generation, to avoid drift.
