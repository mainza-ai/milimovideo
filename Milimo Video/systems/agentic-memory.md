---
type: systems
created: 2026-11-18
updated: 2026-11-18
tags: agentic-memory context retrieval sqlite fts5 security provenance project-memory
sources:
  - ../../backend/agent_memory.py
  - ../../backend/routes/agent_memory.py
  - ../../backend/alembic/versions/0012_agent_memory.py
  - ../../web-app/src/studio/components/AgentMemoryPanel.tsx
  - ../../tests/test_agent_memory.py
related:
  - ../roadmap/memory-management-plan-2026-11-18.md
  - memory-management-audit-2026-11-18.md
  - performance-memory-execution-2026-11-18.md
  - data-models.md
  - architecture.md
  - ../workflows/script-to-video.md
---

# Agentic Memory

Project-scoped, local-only memory for the studio agent. The store answers one
question well: *what does this project already know that should steer the next
prompt, plan, or review pass?* It is deliberately not a vector database.

## Why project-scoped

A studio is a sequence of decisions about one film: the courier wears a red
raincoat, take 4 was approved, the alley was rejected for camera shake, act two is
amber and teal. Those facts are small, structured, and auditable. A per-project
SQLite table gives scope isolation, exact deletion, and export without a hosted
service or an embedding model in the hot path.

## Schema

`agentmemory` (Alembic `0012_agent_memory`):

| Field | Meaning |
|---|---|
| `project_id` | Scope isolation; every read and write filters on it |
| `scope` | `project`, `shot`, `session`, or `agent` |
| `shot_id` | Set for shot-scoped notes; boosts same-shot retrieval |
| `kind` | `working`, `episodic`, `semantic`, `procedural` |
| `text` | Cleaned note, max 8,000 characters |
| `metadata_json` | Source (`review_comment`, `review_approval`, …) and free-form fields |
| `provenance` | `user`, `agent`, `imported`, `system` |
| `trust` | `user`, `verified`, `agent`, `imported` |
| `content_hash` | SHA-256 of the text; dedupes repeated writes in one project |
| `pinned` / `quarantined` | Retrieval priority / excluded until approved |
| `use_count`, `last_used_at` | Lifecycle and "recently useful" signals |

An FTS5 virtual table (`agentmemory_fts`) is maintained by insert/update/delete
triggers. Retrieval checks for the table and falls back to token matching when a
build lacks FTS5, so the feature never hard-depends on the extension.

## Write path

- Explicit tools: `remember`, `forget`, `search`, `update`, `approve`, `export`,
  `clear`.
- Duplicate text in the same project merges metadata and bumps the timestamp
  instead of creating a second row.
- Automatic extraction: review comments and approvals become `episodic` memory with
  `metadata.source` and `trust=user`, so the agent can recall why a shot changed.
- Writes are validated for length and control characters. Failures in extraction
  never fail the review request.

## Retrieval

`context_pack` is the agent-facing read:

1. Filter by project, optional kind/scope/shot, and quarantine state.
2. Tokenize the query, drop function words, and expand a small studio synonym map
   (`dislike → reject`, `rule → never|no more than`, …).
3. Score by lexical hits, FTS candidate set, pinning, trust, same-shot match, and
   30-day recency.
4. Sort and pack until the token budget (default 900, clamp 64–8000). Each entry
   carries a `why` list, and `dropped` reports what did not fit.
5. Increment `use_count`/`last_used_at` for packed entries.

`GET /projects/{id}/memory/context` exposes the same pack over HTTP. The planner
`ask` mode retrieves a 220-token pack and appends it as "Project memory" so answers
use approved facts, not guesses.

## Security (OWASP ASI06)

- Every entry carries provenance and trust; user-authored content is not treated
  the same as imported text.
- Agent/imported text matching instruction-injection signatures is quarantined
  (`ignore previous instructions`, `reveal the system prompt`, `print the API key`,
  …). It stays visible in the UI but is excluded from context packs.
- `POST /memory/{id}/approve` is the only path from quarantine to retrieval; it
  sets `trust=verified`.
- Delete/clear/export are project-scoped and tested; deleting a project's memory
  removes every row, and the export includes provenance for audit.

## UI and API

- Setup → **Agent memory**: search, add note (kind/pin), approve, pin/unpin, delete,
  export JSON, clear all, and a quarantine toggle.
- Routes: `GET/POST /projects/{id}/memory`, `GET .../context`, `GET .../export`,
  `PATCH/DELETE .../{memory_id}`, `POST .../{memory_id}/approve`,
  `DELETE /projects/{id}/memory`.
- Generated OpenAPI types are checked in `web-app/openapi.json` and
  `src/api/schema.d.ts`.

## Evaluation and tests

`backend/agent_memory_eval.py` runs 20 studio-style questions (approved take,
rejection reason, wardrobe, changed fact, lens rule, …) against a throwaway
in-memory database. Current result: **top-1/top-3 100%**, ~199 tokens per pack,
p50 ~0.8 ms, p95 ~1.6 ms. CI runs it with a 0.8 top-3 floor.

`tests/test_agent_memory.py` covers scope isolation, dedupe, token budgets,
quarantine/approve, project-scoped update/delete, export/clear, same-shot boost,
review extraction, planner RAG, and the migration/FTS presence.

## Open work

- Automatic extraction from takes and board notes (comments/approvals are done).
- Consolidation/summarization near context limits and TTL decay for working notes.
- Optional sqlite-vec hybrid retrieval; lexical ranking is the current contract.
- A larger 50-task evaluation set and a store-size quota.
- Context packs are consumed by the planner today; wiring them into every prompt
  enhancement call is the next integration point.
