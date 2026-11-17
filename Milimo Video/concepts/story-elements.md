---
type: concepts
created: 2026-10-09
updated: 2026-10-09
tags: story-elements trigger-words ip-adapter character-consistency element-matching
sources:
  - README.md
  - docs/02_data_models.md
  - docs/03_ai_pipelines.md
related:
  - ../systems/storyboard-engine.md
  - ../systems/data-models.md
  - ../models/flux-2.md
  - ../workflows/script-to-video.md
---

# Story Elements

## Definition

Reusable **Characters, Locations, and Objects** with trigger words (e.g. `@Hero`).
Define once with a reference image and description; the studio auto-detects the
trigger in any prompt and injects the right visual conditioning — the mechanism
behind cross-shot character consistency.

## Where It Shows Up

- **Data**: the `element` table — name, `trigger_word`, type
  (character\|location\|object), description, `image_path` (reference art lives in
  `assets/elements/`). Managed by `ElementManager` (CRUD + `generate_visual` →
  Flux 2).
- **Prompt injection**: `ElementManager.inject_elements_into_prompt()` scans for
  `@Trigger` words and returns the enriched prompt plus reference-image paths. In
  image generation, *implicit* detection also runs: every project element is
  scanned, and any trigger word found in the prompt auto-attaches its reference
  image and appends missing trigger words.
- **Visual injection**: reference images ride the IP-Adapter (CLIP ViT-L → 4 tokens)
  in Flux 2, or native-AE reference conditioning — with "Visuals Only" prompt
  isolation so trigger words don't pollute the text prompt (a real bug that was
  fixed: ghost conditioning from `@Leona` leaking into prompts).
- **ElementMatcher** (`services/element_matcher.py`): deterministic 8-signal scoring
  that links parsed shots to elements without LLM inference — runs after script
  parsing and before commit:

| Signal | Confidence |
|---|---|
| `character_field_exact` | 1.0 |
| `trigger_word_in_action` | 0.95 |
| `name_in_action` | 0.85 |
| `partial_name_in_character` | 0.8 |
| `name_in_scene` | 0.6 |
| `description_tokens` / `dialogue_mention` | 0.5 |
| `action_keywords` | 0.4 |

  Matches at ≥0.35 confidence land in `shot.matched_elements`; results carry
  `confidence` and `match_source` on the frontend `MatchedElement` type.

## Why It Matters

AI video's hardest product problem is consistency: the same character must look the
same in shot 1 and shot 40. Elements are Milimo's answer — persistent identity as a
first-class project object, not a prompt-engineering trick. The matcher exists so
filmmakers don't hand-tag every shot; the injection exists so the models actually
see the reference.

## Related

- [[../systems/storyboard-engine.md]] — where matching runs
- [[../systems/data-models.md]] — the `element` table and `MatchedElement` type
- [[../models/flux-2.md]] — IP-Adapter, the visual mechanism
- [[../workflows/script-to-video.md]] — elements in the end-to-end flow
