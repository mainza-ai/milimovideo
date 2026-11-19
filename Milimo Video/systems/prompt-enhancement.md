---
type: systems
created: 2026-10-09
updated: 2026-11-17
tags: llm prompt-enhancement gemma ollama roles
sources:
  - docs/03_ai_pipelines.md
  - docs/ltx2-bible.md
related:
  - storyboard-engine.md
  - llm-settings-audit-2026-11.md
  - ../concepts/memory-slots.md
  - ../models/ltx-2.md
---

# Prompt Enhancement

## Purpose

Every generation prompt passes through `llm.py`'s `enhance_prompt()` before it
reaches a model — rewritten for cinematic quality by a local LLM. Provider and
model are chosen by the persisted **role plan**, not by a global config flag.

## Components

| Component | Role |
|---|---|
| `backend/llm.py` | Compatibility facade: `enhance_prompt(prompt, system_prompt, is_video, text_encoder)` maps `is_video` to `prompt_video`/`prompt_image` and delegates to the runtime |
| `services/llm/runtime.py` | Resolves a role to ordered provider/model hops, applies context preflight, budget checks, per-role sampling, retries, cancellation, and returns provenance |
| `services/llm/providers.py` | Wire adapters: `gemma_native`, `ollama_native`, `openai_chat`, `anthropic_messages`; sync for worker threads, async/streaming for routes, optional image payloads |
| `services/llm/store.py` | Persisted settings, revision CAS, redaction, provider/role resolution, air-gap + cloud checks |
| **Gemma provider** (in-process) | Reuses the LTX-2 pipeline's Gemma 3 text encoder via `enhance_t2v`/`enhance_i2v`; no HTTP endpoint |
| **Ollama / LM Studio / OpenAI-compatible** | Local HTTP engines discovered on loopback ports; `POST /api/generate` or `/chat/completions` |
| `services/llm/prompts.py` | `get_video_system_prompt()` / `IMAGE_SYSTEM_PROMPT` |
| `prepare_text_encoder(pipeline, is_video)` | Fetches the LTX encoder **only** when the resolved role's first hop is Gemma |

## Key Interactions

- **Callers:** `tasks/video.py` (ti2vid/ic_lora/keyframe/single-frame),
  `tasks/chained.py` (chunk 0 + narrative continuation), `tasks/image.py`,
  `storyboard/manager.py`, and `services/ai_storyboard.py` (`script_parse`, a
  different system prompt but the same runtime).
- **Role split:** `prompt_video` defaults to Gemma (already loaded with LTX);
  `prompt_image`, `script_parse`, `agent_plan`, `review_critic`, and `continuity`
  default to the local Ollama model, because Flux has no LTX encoder and script
  parsing should not load a 19B video pipeline.
- **Ordered fallback:** a role may list fallbacks; the runtime tries them in order
  and refuses with a named reason when a provider is disabled, missing a key, or
  hidden by air-gap.
- **Ollama VRAM discipline:** requests carry the provider's `keep_alive`
  (default `0`), then an unload is unnecessary because `keep_alive: 0` already
  releases the model. A 27B model (~49GB) must not linger into denoising. See
  [[../concepts/memory-slots.md]].
- **VLM structured extraction:** with an input image, the VLM returns JSON with
  `enhanced_prompt` and `is_reference_only`. If the image is a character sheet or
  turnaround, the dispatcher **drops image conditioning** and forces pure T2V.
- **Order:** on the single-shot video path, a local HTTP role rewrites the prompt
  **before** LTX/Flux load (`_preenhance_http`), then the model is released. Gemma
  stays in-pipeline because it needs the loaded text encoder. Chained chunk 0 is
  still enhanced after the load.
- **Provenance:** `job_utils.update_job_db(..., enhanced_prompt=...)` merges
  `llm.last_enhancement()` into `Job.params_json.llm` (role, provider, model,
  latency, token usage); the Inspector shows it per take.
- **Settings surface:** Studio Setup → **AI & LLM**; the legacy header popover is
  a two-provider quick switch that saves through the same `PUT`. API:
  `GET/PUT/PATCH /settings/llm`, `/models`, `/discover`, `/providers/{id}/verify`,
  `/test`, `/health`. Design and gaps: [[llm-settings-audit-2026-11]].

## Prompt rewrite cache (2026-11-18)

`backend/llm.py` keeps an LRU (128) of successful rewrites keyed by settings
revision, role, prompt, system prompt, seed, `has_input_image`, duration, and the
image file signature (path, size, mtime). A cache hit returns the same text and
marks the provenance `cached: true`; a failed rewrite is never cached. Any change
to the persisted LLM plan bumps the revision and invalidates every entry. This
removes the repeated LLM pass when a user retries a take with the same prompt and
seed, without hiding a plan change.

## Gotchas

- The Gemma provider is only usable when the LTX-2 pipeline is loaded. Assigning
  `prompt_image` to Gemma produces a warning and skips enhancement; assign a local
  HTTP provider instead.
- `is_reference_only` is a contract between the VLM output and the video task. If
  JSON parsing fails, the raw text is used and image conditioning is kept.
- Prompt enhancement runs before pipeline selection, so its SSE `progress` update
  (with `enhanced_prompt`) arrives at ~5% — before any denoising.
- A failed enhancement never fails the job: the original prompt is returned and is
  not cached.
- A cache hit skips the provider entirely, including VRAM residency; that is the
  intended behavior for a same-seed retry.
- Wrapper-safe parsing accepts `enhanced_prompt`, `prompt`, `text`,
  `image_prompt`, `result`, `output`, or the longest string value, and returns the
  original prompt if a bare JSON object cannot be unwrapped. granite's
  `{"prompt": …}` now yields plain prose.
- Vision input is sent on a local HTTP provider when the caller has an image:
  Ollama `images[]`, OpenAI `image_url`, Anthropic base64 blocks. Gemma still uses
  the LTX encoder path.
- Cloud providers stay unreachable unless `MILIMO_LLM_CLOUD=1` and air-gap is off.

## Related

- [[storyboard-engine.md]] — AI script parsing uses the `script_parse` role
- [[llm-settings-audit-2026-11.md]] — settings plane, security, and remaining work
- [[../concepts/memory-slots.md]] — why the LLM releases VRAM before denoising
- [[../models/ltx-2.md]] — the text encoder doing double duty
