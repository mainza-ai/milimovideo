---
type: systems
created: 2026-11-17
updated: 2026-11-17
tags: llm local-first ollama omlx strata roles roadmap
sources:
  - llm-settings-audit-2026-11.md
  - ../../backend/services/llm/
  - ../../backend/tasks/video.py
  - ../../backend/memory_manager.py
related:
  - llm-settings-audit-2026-11.md
  - prompt-enhancement.md
  - architecture.md
  - ../concepts/memory-slots.md
  - ../roadmap/implementation-plan-2026-11.md
---

# LLM Local-First Next Steps — 2026-11-17

## Verdict

The configuration plane shipped in L0–L3 works end-to-end against this machine's
three real local engines — Ollama, oMLX, and Strata — including discovery,
verification, role routing, live fallback, and prompt provenance. That is the
foundation to build on.

The platform is not yet *complete* locally. The gaps are no longer "settings do
not persist"; they are **identity, capability truth, cold-start latency, output
contract, and evaluation**. Cloud governance stays deferred: cloud is an optional
layer on a working local platform, and the local platform is not finished.

## Measured evidence (this machine, 2026-11-17)

Apple Silicon, MPS. Calls went through the shipped runtime (`services/llm`), not
raw curl, except where noted.

| Call | Engine / model | Condition | Latency | Output |
|---|---|---|---|---|
| `prompt_video` enhance | Ollama `llama3.2:3b-instruct-fp16` | cold (`keep_alive: 0`) | 7.4 s | 71 tok |
| `prompt_video` enhance | Ollama `llama3.2:3b-instruct-fp16` | warm (`keep_alive: 5m`) | 2.2 s | 70 tok |
| `prompt_video` enhance | oMLX `Qwen3.8-27B-MLX-8bit` | warm | 23.3–24.8 s | 288 tok |
| `prompt_video` enhance | oMLX `Llama-3.2-3B-Instruct-bf16` | cold | 5.9 s | — |
| `prompt_video` enhance | Strata `unsloth-ud-iq4_xs-v31` | warm | 3.0 s | 83 tok |
| `script_parse` | Strata `unsloth-ud-iq4_xs-v31` | warm | 31.7 s | 2 scenes / 10 shots |
| `prompt_image` enhance | Ollama `granite4.1:3b-bf16` | cold / warm | 9.7 s / 3.5 s | 75–109 tok |
| First model load | oMLX 27B 8-bit | cold | 36.2 s | — |
| Model catalog | oMLX (`/v1/models`) | warm | 20 ms | 11 entries |
| Model catalog | Strata (`/v1/models`) | warm | 19 ms | 1 entry |
| Verify | Ollama | warm | 17–26 ms | 4 models |

Discovery found all three engines and their model lists. A live fallback test
(primary Ollama model deliberately missing → oMLX fallback) returned a completion
and recorded `last_fault` on Ollama and `last_success_at` on the fallback.

Transport reality:

- **Ollama** speaks native `/api/generate`; `keep_alive` controls load/unload and
  materially changes latency (7.4 s → 2.2 s).
- **oMLX** is OpenAI-compatible on `:8787`; `stream: true` returned **0 bytes for
  60 s** while the same non-stream request took 0.55 s. Streaming is effectively
  unsupported and must be negotiated, not assumed.
- **Strata** is OpenAI-compatible on `:47831` and streams SSE, but the first
  deltas are `reasoning_content`; a stream reader that only concatenates
  `delta.content` will look stalled.
- The Kubona production stack is on `:8080` in Docker. Discovery rejected it only
  because `/v1/models` returned 404 — port probing needs response-shape checks,
  not status checks.

## Findings

### F1 — Provider identity is too coarse (P0)

`providers` is keyed by provider *type*, and `validate_settings()` requires
`provider_id == provider.type`. oMLX and Strata are both discovered as
`openai_compatible`, so attaching one replaces the other. This is the single
biggest structural gap: a real local machine runs multiple OpenAI-compatible
engines at once.

Evidence: `POST /settings/llm/discover` returned three servers, two of them
`openai_compatible`; the second `PUT` moved the slot and orphaned the first.

### F2 — Image prompt enhancement can send JSON to Flux (P0)

`runtime.parse_enhancement()` accepts only `enhanced_prompt`. Ollama
`granite4.1:3b` answered with `{"prompt": "..."}`, and the raw JSON string became
the "enhanced prompt" for `prompt_image`. The old dispatcher had the same
fallback, so this is a pre-existing defect that the new panel makes visible.

### F3 — Defaults name a model the machine may not have (P0)

First boot seeds `llama3.1`. On this machine Ollama has `llama3.2:3b`,
`granite4.1:3b`, `qwen3.6`, and an embed model — so `script_parse` failed with
`HTTP 404: model 'llama3.1' not found` until reseeded. Defaults must be chosen
from what discovery actually finds, or be explicitly "unassigned".

### F4 — Capability truth is missing (P1)

- oMLX reports context as `max_model_len`; the catalog only reads
  `context_length`, so every oMLX model shows `context None`.
- Strata returns a `meta` block (label, aliases) that is discarded.
- Non-chat entries are offered as role targets: oMLX `MarkItDown` and
  `mlx-community--MiniMax-Music3-bf16`; Ollama `nomic-embed-text-v2-moe`.
- No vision flag survives for OpenAI-compatible engines, even though Strata runs
  `strata-vision` with an mmproj projector.
- `parameter_size`/quantization are null for OpenAI-compatible catalogs.

### F5 — Cold starts are invisible and unbounded (P1)

A 36.2 s oMLX 27B load happened inside a role call. There is no warm-up action,
no "loaded" state, and no per-provider `keep_alive` editor in Studio (only the
legacy popover touches Ollama's). A generation job can sit at "Enhancing
Prompt…" for half a minute with no explanation.

### F6 — Discovery can attach an unrelated service (P1)

`_probe()` accepts any response except 404/500, then lists models (possibly
empty). A service that answers 200 on `/v1/models` with HTML or an unrelated API
would appear as an attachable engine. It should require a JSON catalog shape and
report 401/403 as "needs key" rather than "not found".

### F7 — Long local calls have no progress (P1)

Non-stream calls block for 3–32 s. The settings test endpoint blocks the request;
job progress only says "Enhancing Prompt…". Streaming is uneven: Strata streams
(with reasoning deltas), oMLX does not. The runtime needs a negotiated streaming
path with a fallback and a visible elapsed/status line.

### F8 — No context preflight (P1)

A 388-token enhancement prompt is comfortable, but a long treatment plus element
manifest can exceed a small model's window. Nothing estimates prompt size against
`max_model_len`, so the failure arrives as a provider 400 after the wait.

### F9 — No local evaluation or model guidance (P1)

Nothing measures whether a 3B, a 27B, or Strata is actually better per role.
The old L4 "eval gate" was cloud-shaped (cost). Local-first needs fixture-based
role quality and latency, plus a recommended local plan.

### F10 — LLM engines sit outside the memory slots (P2)

`MemoryManager` only knows `video` and `image`. A 27B MLX model held warm by
`keep_alive`, or a Strata engine with a 64k context, competes with LTX-2 19B on a
32 GB machine. The enhancement step runs before denoising, but a pinned model
stays resident through it.

### F11 — Vision remains unwired (P2)

Strata is vision-capable here; Ollama has no vision model installed. The panel
lists vision models but no image bytes are sent on either HTTP wire. Either wire
image input or stop implying it works.

## Plan

Cloud work (spend telemetry, budget guard, paid-provider profiles) stays behind
all of this. Local-first means the local path is the default, complete path.

### L5 — Provider identity and capability truth (P0, 2–3 days)

| Task | Change | Acceptance |
|---|---|---|
| L5-01 | Provider **instances**: key `providers` by stable id (`ollama`, `openai_compatible@8787`, `openai_compatible@47831`), keep `type` separate; schema v2 + migration of existing docs | oMLX and Strata both attach, enable, verify, and serve different roles at once |
| L5-02 | Relax `validate_settings()` (`provider_id == provider.type`), role/fallback references resolve by instance id | Old documents migrate without loss; unknown ids still 400 |
| L5-03 | Discovery identity: match by normalized base URL, create/refresh an instance instead of overwriting a type slot | Re-scan is idempotent; two engines on one port range never collide |
| L5-04 | Probe on response shape: require JSON `data[]`/Ollama `models[]`; treat 401/403 as `needs_key`; ignore HTML/other APIs | Kubona on `:8080` is not offered; authenticated engine is offered with a key prompt |
| L5-05 | Capability catalog: oMLX `max_model_len`, Strata `meta`, Ollama `num_ctx` when present; parameter size, quantization, vision, embedding-only, non-chat | Every model row shows a context window or an explicit `unknown`; `MarkItDown`, music, and embed models are flagged `chat: false` |
| L5-06 | Capability gating: refuse embedding-only/non-chat models for text roles; warn when a role model is absent from the live catalog | Save returns 400 with the model named, or a soft warning chip in the panel |
| L5-07 | Tests: instance migration, two-instance routing, shape-based discovery, capability flags | Backend suite green; no live engine required |

### L6 — Local runtime quality and latency (P0, 2–3 days)

| Task | Change | Acceptance |
|---|---|---|
| L6-01 | `parse_enhancement()` accepts `enhanced_prompt`, `prompt`, `text`, `image_prompt`, strips fences, and returns the original prompt if the payload is still JSON | granite `{"prompt": …}` yields the inner sentence, never the wrapper |
| L6-02 | Seed role models from discovery on first boot; otherwise leave the role explicitly unassigned and say so | Fresh install on this machine never defaults to `llama3.1` |
| L6-03 | Warm-up: `POST /settings/llm/providers/{id}/warm` plus a **Warm** button; optional preflight before enhancement; expose per-provider `timeout_sec` and `keep_alive` in the panel | Cold model load is visibly separated from inference time |
| L6-04 | Streaming negotiation: try SSE, fall back to non-stream on first-byte timeout; accumulate `delta.content` and ignore `reasoning_content`; report elapsed/provider/model in job status | oMLX still works; Strata streams; no call hangs past the configured timeout |
| L6-05 | Context preflight: estimate prompt tokens (chars/4), compare to catalog context, trim the element manifest or refuse with the exact fix | Over-long prompt fails fast with a named limit |
| L6-06 | Health: per-provider last load ms, last tokens/s, last fault; surface in Setup | Panel shows cold/warm reality, not just "ok" |
| L6-07 | Tests: wrapper parsing, seeding fallback, SSE/non-stream fallback, context refusal, timeout propagation | Unit tests with fake transports |

### L7 — Local evaluation and presets (P1, 2 days)

| Task | Change | Acceptance |
|---|---|---|
| L7-01 | Fixture pack: 8 shot prompts, 3 scripts, 4 image prompts, 1 reference-sheet case | Deterministic fixtures, no network |
| L7-02 | `POST /settings/llm/eval` runs fixtures per role: valid JSON, no wrapper, length bounds, action verbs, reference-sheet handling, latency | Results per provider/model/role with pass rate and p50 |
| L7-03 | Panel eval table + **recommended local plan** ("apply best passing model per role") | One click assigns `prompt_video`, `prompt_image`, `script_parse`, `agent_plan` |
| L7-04 | Record the last eval in the settings doc (timestamp, fixture version, pass rates) and show a stale-eval warning | A model swap without new evidence is visible |

### L8 — Generation integration and memory discipline (P1, 2–3 days)

| Task | Change | Acceptance |
|---|---|---|
| L8-01 | Register HTTP LLM engines with `MemoryManager` as an `llm_http` slot (or a policy module): unload/pause before video/Flux load, restore after | 32 GB machine never holds a 27B LLM and LTX-2 19B at once |
| L8-02 | Surface take provenance in the Inspector ("enhanced by oMLX/Qwen3.8-27B") from `params_json.llm` | Visible per take, not only in the DB |
| L8-03 | Cancellation propagates into in-flight LLM calls; a cancelled job does not wait 30 s | Cancel returns promptly |
| L8-04 | Image-conditioned local vision: Ollama `images[]` and OpenAI `image_url`, or remove the vision badge until wired | A reference image actually reaches the model, or the UI stops claiming it |
| L8-05 | `verify_setup.py` reports discovered engines and role readiness (missing model, disabled provider) | First-run verification explains a broken role |

### L9 — Cloud on top (P2, only after L5–L8)

Token/spend telemetry, budget guard before egress, eval-gated model changes, and
staged profiles. Air-gap and `MILIMO_LLM_CLOUD` remain the floor; no cloud work
should land before the local path passes L5–L7 gates.

## Immediate three commits

1. **F2 + F3** — fix `parse_enhancement()` and stop seeding non-existent models;
   add regression tests with the exact granite payload. Small, high impact.
2. **F1 + F4 + F6** — provider instances, shape-based discovery, capability
   catalog, and gating. This unblocks using oMLX and Strata together, which is the
   local-first promise.
3. **F5 + F7** — warm-up/keep-alive/timeout controls, streaming negotiation, and
   progress/health. This makes 3–32 s local calls honest instead of mysterious.

## Gates

- **G-LLM-1**: Three local engines attached simultaneously; a different role on
  each; all verifiable from Studio.
- **G-LLM-2**: `prompt_video` enhancement ≤ 8 s warm on a 3B-class model and
  ≤ 30 s on a 27B-class model, with the model named in the job status.
- **G-LLM-3**: `prompt_image` output contains no JSON wrapper for granite, qwen,
  llama, and gemma fixtures.
- **G-LLM-4**: `script_parse` produces a valid scene list on Strata and on a 3B
  Ollama model, or fails with the exact reason and falls back to regex.
- **G-LLM-5**: A cold model load is visible before inference starts; no role call
  exceeds its configured timeout.
- **G-LLM-6**: On a 32 GB machine, enhancement plus generation never holds two
  heavy models at once.

## Do not do yet

- Do not add cloud spend/budget machinery before L5–L7 pass.
- Do not expand the provider list; three local wires plus two opt-in cloud slots
  already cover the machine.
- Do not promise vision until L8-04 sends image bytes.
- Do not copy Kubona's role catalog; these six film roles are the right ones.

## Related

- [[llm-settings-audit-2026-11.md]] — what shipped in L0–L3 and why
- [[prompt-enhancement.md]] — the runtime path these findings affect
- [[../concepts/memory-slots.md]] — why F10 matters
- [[../roadmap/implementation-plan-2026-11.md]] — status table
