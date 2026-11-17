---
type: systems
created: 2026-11-17
updated: 2026-11-17
tags: llm settings audit kubona ollama gemma providers roles security
sources:
  - https://github.com/mainza-ai/kubona
  - kubona@0dbf31c7a3d44ed7876640b7079fce7ebf752c52
  - ../backend/llm.py
  - ../backend/routes/projects.py
  - ../web-app/src/components/LLMSettings.tsx
related:
  - prompt-enhancement.md
  - architecture.md
  - ../roadmap/implementation-plan-2026-11.md
  - ../concepts/memory-slots.md
---

# LLM Configuration Audit & Implementation Plan — 2026-11-17

## Verdict

Milimo has a working **two-provider prompt-enhancement switch** (Gemma and
Ollama), but it does not have an LLM configuration plane:

- The only UI is the legacy `LLMSettings` popover. Studio Setup has no LLM panel.
- Settings are **process memory only**. `PATCH /settings/llm` changes
  `config.*`; a restart reverts to environment defaults.
- There is no typed settings schema, revision/CAS, provider catalog, API-key
  handling, local discovery, model capability data, verification, role routing, or
  health telemetry.
- Provider behavior is hardcoded at each call site. AI storyboard parsing always
  takes the LTX Gemma path, even if Ollama is selected.
- There are no tests for `/settings/llm`.

Kubona's LLM plane is a strong reference for the missing layer. It is Apache-2.0;
borrow the **architecture and interaction patterns**, not the quantum domain code.

## Kubona reference

Reviewed `mainza-ai/kubona` at commit
`0dbf31c7a3d44ed7876640b7079fce7ebf752c52` (2026-11-17).

| Kubona file | Pattern worth borrowing |
|---|---|
| `platform/core/pkg/llm/config.go` | Typed `Settings`, 14 provider slots, 5 wire protocols, role catalog, fallback chains, per-role sampling, schema version, revision CAS |
| `platform/core/pkg/llm/discover.go` | Parallel localhost discovery on 8787/11434/1234/47831/8080/8000/8899/8081; Ollama `/api/tags` + OpenAI-compatible `/v1/models`; context-window warnings |
| `platform/core/pkg/llm/endpoint.go` | No userinfo/query/fragment; local plaintext HTTP only; remote endpoints must be HTTPS |
| `platform/core/pkg/llm/secretbox.go` | AES-256-GCM sealing at rest, redacted public reads, key source/status |
| `platform/core/pkg/llm/budget.go` + `api/llm_budget_guard.go` | Context and spend bounds; unpriced/unmetered calls refuse |
| `platform/core/pkg/api/server.go` | Public redacted GET; bearer-gated writes; revision conflicts; air-gap PATCH; verify/model/test routes; audit events |
| `platform/console/src/components/surfaces/LLMSettings.tsx` | Provider cards, discovery strip, role routing matrix, fallback editor, profiles, save-state mapping, benchmark panel |
| `platform/console/e2e/llm-settings.spec.ts` | Discovery → attach, provider verify, role options, mono-model warning, axe |

Kubona's roles are quantum-research-specific (`ansatz-scout`,
`disconfirmation-officer`, `device-auditor`, etc.). Do not copy those. Reuse the
**shape**: a role catalog, ordered fallback, per-role sampling, and a safe default.

## Current Milimo audit

### Surface inventory

| Surface / path | Current behavior | Gap |
|---|---|---|
| Studio → Setup | Memory, disk, pipelines, integrity, error budget, models | No LLM panel; users cannot reach AI settings from Studio |
| Legacy `LLMSettings.tsx` | Popover with Gemma/Ollama, Ollama model list, keep-alive, URL | Unlabeled inputs, no provider verify, no role routing, no save conflict, no discovery |
| `GET /settings/llm` | Returns `provider`, `ollama_base_url`, `ollama_model`, `ollama_keep_alive` | Unversioned; no provider/role/model metadata; no health |
| `PATCH /settings/llm` | Mutates `config.*` in memory; accepts arbitrary dict; checks only provider name | No persistence, schema, revision, audit, URL validation, key handling |
| `GET /settings/llm/models` | Ollama `/api/tags` only; returns 503 if Ollama is down | No local discovery, OpenAI-compatible catalog, capabilities, or graceful refusal |
| `backend/llm.py` | Gemma or Ollama; hardcoded temperature/top_p/max_predict; sync `requests`; 30s timeout | No provider abstraction, retry, async client, capability checks, vision input, fallback, token budget |
| `tasks/video.py`, `tasks/chained.py` | Branch on `config.LLM_PROVIDER == "gemma"`; load text encoder for Gemma | Provider logic duplicated; no role routing; no provider/model provenance |
| `tasks/image.py` | Gemma selected → logs “cannot enhance” and skips | A configured provider silently does nothing for image prompts |
| `services/ai_storyboard.py` | Always Gemma → loads full `ti2vid` pipeline for its text encoder | Cannot use Ollama for script parsing; heavy model load for text-only work |
| `routes/planner.py` | Deterministic; no model | Future agent roles have no model router |
| Tests | No `/settings/llm` tests | Regressions invisible |

### Severity findings

**P0 — configuration is not durable**

`PATCH /settings/llm` changes module globals. After restart, env defaults return.
Users cannot trust the settings screen.

**P0 — Studio has no LLM settings**

The new shell is the current product. A legacy-only popover is not a surface.

**P0 — no typed contract or revision**

The endpoint accepts any dict and silently ignores unknown keys. Two clients can
overwrite each other. No schema version means a future provider/role migration is
undefined.

**P1 — provider model is too small**

Only `gemma` and `ollama` exist. There is no local OpenAI-compatible provider, no
LM Studio/vLLM/llama.cpp, no cloud opt-in, no API key posture, no model
capabilities, and no provider verification.

**P1 — role behavior is hardcoded**

Prompt enhancement, image prompt enhancement, script parsing, and AI storyboard
have different effective providers. AI parsing cannot use Ollama. There is no
per-role fallback or per-role sampling.

**P1 — blocking HTTP in async paths**

The Ollama call uses synchronous `requests` with a 30s timeout. Some call sites
are inside an executor; image/single-frame paths call before the executor and can
stall the API event loop.

**P1 — vision is advertised but not wired**

The legacy UI badges Ollama vision models, but `enhance_prompt_ollama` never sends
the reference image in the Ollama payload. Vision capability is cosmetic.

**P2 — no health, discovery, or verification**

No “test provider”, “scan local servers”, “model list with context/capabilities”,
or “last success”. The only signal is a 503 from Ollama `/api/tags`.

**P2 — no provenance or budget**

Jobs store `enhanced_prompt` but not role/provider/model/template/version. There is
no token count, latency, cost, or cloud spend guard.

**P2 — no cloud safety posture**

No key sealing, redacted snapshot, endpoint URL validation, explicit cloud opt-in,
or air-gap switch. Milimo should remain local-first by default and make cloud
egress a deliberate, visible exception.

## Recommended Milimo design

### Principles

1. **Local-first by default.** Gemma and discovered local servers require no
   network opt-in. Cloud is disabled unless explicitly enabled.
2. **One typed settings plane.** All LLM choices live in one persisted document,
   not in module globals.
3. **Roles, not one global model.** A film studio has different text jobs:
   video prompt, image prompt, script parse, agent plan, review critique.
4. **Show the plan before spending.** Model choice, fallback, estimated VRAM/time,
   and provider status must be visible.
5. **Fail closed and honestly.** Missing model/key/provider names the exact fix;
   no silent fallback to a different cloud.

### Provider catalog (initial)

| Provider | Wire | Local | Notes |
|---|---|---|---|
| `gemma` | `gemma_native` | yes | LTX text encoder; only available when video pipeline/text encoder is loaded; no Flux image prompting |
| `ollama` | `ollama_native` | yes | `/api/tags` + `/api/generate` |
| `lmstudio` | `openai_chat` | yes | OpenAI-compatible local server |
| `openai_compatible` | `openai_chat` | local/custom | vLLM, llama.cpp, custom local endpoint |
| `openai` | `openai_chat` | no | optional cloud, hidden behind opt-in |
| `anthropic` | `anthropic_messages` | no | optional cloud, hidden behind opt-in |

Do not ship 14 providers on day one. Start with four local slots and add cloud only
behind `MILIMO_LLM_CLOUD=1` + air-gap off.

### Settings schema (proposed)

```json
{
  "schema_version": 1,
  "revision": 0,
  "active_provider": "gemma",
  "active_model": "",
  "air_gapped": true,
  "temperature": 0.7,
  "max_tokens": 400,
  "providers": {
    "gemma": {
      "type": "gemma",
      "name": "Gemma 3 (LTX text encoder)",
      "wire": "gemma_native",
      "base_url": "",
      "default_model": "ltx-gemma3",
      "enabled": true,
      "is_local": true,
      "timeout_sec": 120
    },
    "ollama": {
      "type": "ollama",
      "name": "Ollama",
      "wire": "ollama_native",
      "base_url": "http://127.0.0.1:11434",
      "default_model": "qwen3:8b",
      "enabled": true,
      "is_local": true,
      "timeout_sec": 120,
      "keep_alive": "0"
    }
  },
  "roles": {
    "prompt_video": { "provider": "gemma", "model": "ltx-gemma3", "fallback": [] },
    "prompt_image": { "provider": "ollama", "model": "qwen3:8b", "fallback": [] },
    "script_parse": { "provider": "ollama", "model": "qwen3:8b", "fallback": [] },
    "agent_plan": { "provider": "ollama", "model": "qwen3:8b", "fallback": [] }
  }
}
```

Role seeds for Milimo:

| Role | Default | Why |
|---|---|---|
| `prompt_video` | Gemma | already loaded with LTX; cinematic prompt schema |
| `prompt_image` | Ollama/local | Flux has no LTX text encoder |
| `script_parse` | Ollama/local | text-only; avoids loading 19B video pipeline |
| `agent_plan` | Ollama/local | command dock/agent planning; deterministic fallback exists |
| `review_critic` | local future | frame/comments critique |
| `continuity` | local future | cross-shot consistency checks |

### API

| Endpoint | Purpose |
|---|---|
| `GET /api/settings/llm` | Redacted public snapshot + revision + provider status |
| `PUT /api/settings/llm` | Whole-document update with `expected_revision`; 409 on stale |
| `PATCH /api/settings/llm` | Field-level air-gap only |
| `GET /api/settings/llm/models?provider=` | Model catalog/capabilities |
| `POST /api/settings/llm/discover` | Local loopback scan with ports/warnings |
| `POST /api/settings/llm/providers/{id}/verify` | Live connectivity/model check |
| `POST /api/settings/llm/test` | Role or explicit provider/model test prompt + latency |
| `DELETE /api/settings/llm/providers/{id}/key` | Remove stored credential |
| `GET /api/settings/llm/health` | Last success/fault, active role resolution |

### Storage

Add `app_setting` (or dedicated `llm_settings`) table with:
- `key` primary key (`active_settings`)
- `value_json`
- `updated_at`
- revision is stored inside the JSON (as Kubona does) or a `revision` column; one
  source of truth only.

Secrets: prefer a separate `llm_credential` table with AES-GCM sealed values.
Reads return `has_key`, `secret_source`, `key_status`; never the key. If
`MILIMO_SECRET_KEY` is unset, allow plaintext local-only keys with an explicit
warning, or refuse cloud keys. Do not silently store plaintext credentials.

### Runtime

- New `backend/services/llm/` package:
  - `settings.py` — load/save/migrate/validate, snapshots, revision.
  - `providers.py` — provider registry, wire adapters, endpoint validation.
  - `runtime.py` — async `complete(role, messages, ...)`, fallback chain, retry.
  - `discovery.py` — local probes.
  - `credentials.py` — sealed key storage.
- `backend/llm.py` keeps its public `enhance_prompt()` signature as a compatibility
  facade that resolves `prompt_video`/`prompt_image`.
- Ollama HTTP uses `httpx.AsyncClient` or `asyncio.to_thread`; never block the
  event loop.
- `script_parse` uses the role router. AI parse no longer loads the 19B pipeline
  merely to use Gemma.
- Store `llm_provider`, `llm_model`, `llm_role`, `llm_template_version`,
  `llm_latency_ms` on the job/shot enhancement result.

### Studio UI

Add **Setup → AI & LLM** (or a collapsible panel, given Setup density):

1. Status strip: active provider/model, local/cloud, revision, save state.
2. Provider cards: name, local/cloud, base URL, wire, default model, enable,
   verify, last success, key posture.
3. **Scan local servers** with discovered cards and Attach.
4. Generation defaults: temperature, max tokens, keep-alive policy.
5. Role matrix grouped:
   - Prompt (video/image)
   - Story (script parse)
   - Agent (plan/review/future)
   Each row: provider, model, fallback chain, per-role temperature/top_p/seed.
6. **Test prompt** + latency/model reply.
7. Air-gap/cloud opt-in if cloud providers are enabled.
8. Save with unsaved state, revision conflict handling, and field-adjacent errors.

The legacy popover can remain until cutover, but should link to the Studio panel;
do not implement two independent editors.

## Implementation plan

### Phase L0 — Persist the contract (P0, 2–3 days)

| Task | Change | Acceptance |
|---|---|---|
| L0-01 | Add `llm_settings` table + Alembic `0005_llm_settings` | Restart preserves provider/model/roles; legacy env seeds only on first boot |
| L0-02 | Typed Pydantic settings, providers, roles, revision | Unknown keys/providers/models → 400; snapshot is deterministic |
| L0-03 | `GET` redacted + `PUT` revision CAS + `PATCH` air-gap | Stale write → 409 with current revision; no key leakage |
| L0-04 | Move routes to `routes/settings.py`; audit events | Routes in OpenAPI; write logs name provider/model/role |
| L0-05 | Tests: persistence, schema, redaction, conflict, legacy env | Backend suite green |

### Phase L1 — Provider runtime (P0/P1, 3–4 days)

| Task | Change | Acceptance |
|---|---|---|
| L1-01 | `services/llm` registry: gemma/ollama/lmstudio/openai-compatible | Each provider resolves or refuses with a named reason |
| L1-02 | Async Ollama/OpenAI clients + timeout/retry | Event loop stays responsive during a slow provider |
| L1-03 | Discovery + model catalog + capabilities | Scan finds local servers; model list includes context/vision flags |
| L1-04 | Verify/test endpoints | Provider test returns latency/model; failure is actionable |
| L1-05 | URL validation + cloud opt-in + air-gap | Non-local plaintext HTTP refused; cloud requires explicit opt-in |
| L1-06 | Credential sealing/redaction | GET never returns keys; key status/source visible |

### Phase L2 — Role routing & call sites (P1, 3–4 days)

| Task | Change | Acceptance |
|---|---|---|
| L2-01 | Role catalog: prompt_video, prompt_image, script_parse, agent_plan, review_critic, continuity | Role defaults are visible and stable |
| L2-02 | Resolve primary + fallback + per-role sampling | Fallback order is deterministic; bad role assignment refuses |
| L2-03 | Replace provider branches in video/chained/image/storyboard | No `if provider == "gemma"` outside the router |
| L2-04 | AI parse uses role router | Script parse can run entirely on Ollama without LTX pipeline |
| L2-05 | Store prompt provenance | Enhanced prompt records role/provider/model/template/latency |
| L2-06 | Tests: routing, fallback, image role, AI parse role | Fake provider tests; no real GPU |

### Phase L3 — Studio UI (P1, 2–3 days)

| Task | Change | Acceptance |
|---|---|---|
| L3-01 | Setup → AI & LLM panel | All settings reachable in Studio |
| L3-02 | Provider cards + discovery + verify + model select | Discovered server can be attached and saved |
| L3-03 | Role matrix + fallback editor + defaults | Role changes persist and resolve |
| L3-04 | Save state, revision conflict, adjacent errors, tests | Unsaved/stale/refused states are honest; axe green |
| L3-05 | Link/retire legacy popover | One source of truth |

### Phase L4 — Governance (P2, optional)

| Task | Change | Acceptance |
|---|---|---|
| L4-01 | Token/latency/cost telemetry | Health/metrics show per-role spend |
| L4-02 | Cloud budget guard | Unpriced/unmetered calls refuse before egress |
| L4-03 | Prompt quality/eval gate | A model change can require a recorded eval |
| L4-04 | Profiles | “Sovereign local”, “cost-aware local”, “hybrid” staged presets |

## Verification

- L0: restart/persistence test, 409 conflict, redaction test.
- L1: fake Ollama/OpenAI server tests, timeout/retry, discovery fixtures, URL
  refusal, sealed-key test.
- L2: role resolution/fallback tests, image/AI-parse fake providers.
- L3: Playwright discovery/verify/role-save, axe, reload persistence.
- L4: budget/eval gate tests.

## Decisions & risks

- **Do not copy Kubona's quantum roles or 14-provider surface.** Start with four
  local providers and two optional cloud providers.
- **Do not make Gemma look like a general provider.** It is a native LTX text
  encoder; it cannot serve Flux image prompts and has no independent HTTP endpoint.
- **Do not store keys in plaintext by default.** A local single-user app can use an
  env-provided key; if stored, seal it or refuse cloud keys.
- **Do not implement cloud before local discovery/roles.** The immediate user need
  is reliable local settings, not cloud breadth.
- **Air-gap must be real.** If cloud is added, a toggle must block egress in the
  runtime, not just hide UI.

## Related

- [[prompt-enhancement.md]] — current Gemma/Ollama dispatcher
- [[architecture.md]] — backend service map
- [[../concepts/memory-slots.md]] — Gemma/Ollama VRAM discipline
- [[../roadmap/implementation-plan-2026-11.md]] — phased work
- [Kubona LLM config](https://github.com/mainza-ai/kubona/blob/main/platform/core/pkg/llm/config.go)
