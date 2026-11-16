---
type: research
created: 2026-10-09
updated: 2026-10-09
tags: frontend ux react tauri agentic-workflows eval research
sources:
  - research survey 2026-10-09 (web)
related:
  - ../systems/nle-timeline.md
  - ../systems/frontend-state.md
  - ../systems/storyboard-engine.md
  - ../systems/prompt-enhancement.md
  - ../roadmap/v2-roadmap.md
---

# Research: Frontend, UX & Creative-AI Workflows (Oct 2026)

Survey of the 2026 frontend/UX landscape for the v2 interface and workflow redesign.
All post-Jan-2026 facts are web-sourced; unconfirmed items are marked.

## Landscape

**React 19.2.x is the settled line** (19.2.7 reported Aug 2026); the React Compiler
is stable (Rust port upstreamed), concurrent rendering is default, and
`useTransition`/`useDeferredValue` accept async callbacks (React 18 required sync).
**Vite remains the 2026 standard** for pure-React builds. Server Components are
production-ready but consensus says they matter for server-first apps, **not local
desktop apps**. **Zustand is still the dominant client-state library** (~6M npm
weekly downloads vs ~2M Jotai) — keep it, but isolate high-frequency timeline state
(playhead/clip updates) into a dedicated slice with strict selectors (or Jotai atoms
for that hot path only).

**Timeline rendering: canvas beats DOM for dense timelines.** 2026 production
evidence: Canvas 10,000 elements in 18ms @ 60fps vs SVG 210ms/12fps; a Sep 2026
Blockbench PR fixed 518k-keyframe timeline perf via virtualization + marker culling
(DOM nodes 213k→4.2k, RSS 4.5GB→1.5GB, playback hung→38fps). References:
**react-timeline-editor** (multi-track drag/trim/split, keyframes, JSON→Remotion),
**Twick** (open-source React video editor SDK, canvas timeline, WebGL effects),
**Vanta** (Remotion-based AI video engine). Milimo's CSS/DOM timeline is the
highest-risk perf item in v2: keep DOM for handles/selected clips, **render clip
blocks and waveforms on a canvas layer with window-based virtualization**.

**Desktop packaging: Tauri 2 wins on footprint** — ~5–30MB vs ~150–289MB bundles,
~320ms vs ~1800ms cold start, ~42MB vs ~187MB idle RAM. The **sidecar pattern** is
the 2026 norm for local-first AI apps: Tauri spawns the existing Python backend as
a child process, WebView loads the local frontend — zero product-code rewrite.
Caveat: **opencode left Tauri for Electron** over WebKit-vs-Chromium rendering
inconsistency — real cost for pixel-sensitive canvas/WebGL UI. Recommendation:
**Tauri 2 + sidecar**, keep privileged work behind a `shell-api` interface so the
Electron escape hatch stays cheap; budget a WebKit-baseline test pass on Linux.

**AI-copilot UX (2026 patterns worth stealing):** (1) chat beside the timeline on a
shared project document, with agent tools like `edit_project`, `preview_frame`,
`preview_strip` (12-frame labeled contact sheet) — *preview before render* as a
first-class agent loop; (2) **MCP as the universal agent entry point** (Descript,
Shotstack, Creatomate, Vidmoat all ship MCP servers Oct 2026) — any "agent that
edits" story should be an MCP story; (3) transcript-first editing (Descript) — edit
video by editing a document, natural fit for Milimo's script→scenes→shots engine;
(4) conversational editing inside the NLE (Premiere + Firefly 2026); (5) ambient
assistance (auto-clip extraction, virality scoring).

**Agentic video workflows.** **Google AI Video Co-Director (announced Sep 25, 2026)**
is the template: four open frameworks — Co-Director (hierarchical multi-agent
planner), **CANVAS** (persistent visual memory for characters/locations/objects —
anti-drift), **A²RD** (segment-by-segment generation), **VQQA** (VLM critique as a
"semantic gradient" for prompt refinement; +11.57% T2V-CompBench, +8.43% VBench2).
10-minute coherent output, no fine-tuning, orchestrates existing models. Also:
Mind-of-Director (Mar 2026, script→scene→blocking→camera agents, ~25 min on consumer
hardware), MovieAgent (director/screenwriter/storyboard chain), StoryMem
(shot-by-shot memory-to-video diffusion on Wan2.2). **LangGraph is the dominant OSS
orchestrator** (MIT, durable execution, HITL interrupt, MCP adapters); CrewAI for
fast prototypes; Microsoft Agent Framework 1.0 GA'd Apr 2026; Mastra.ai for
TypeScript-first stacks.

**Prompt enhancement with 2026 local LLMs.** Community tooling defaults to
`huihui_ai/qwen3.5-abliterated:9b` (~6GB) for prompt rewriting; **Qwen3-VL 8B**
(6.1GB, 256K context, native video understanding, Apache 2.0, ungated) is the 8GB-
class VLM pick for shot analysis/pre-viz critique. Practical tiers: Qwen3 4B (~2.5GB)
for constrained rewriting; Qwen3-VL 8B for the VQQA critique loop. Keep Ollama as
the runtime; prefer Apache 2.0 ungated checkpoints (Gemma 3's HF gating is friction).

**Eval harnesses.** **VBench remains the standard**; **VBench-2.0** adds faithfulness
dimensions (physics, camera motion, instance preservation). **AISBench integrated
VBench 1.0** for cheaper local GPU runs with mini subsets (May 2026). The pragmatic
2026 pattern: VBench subset for diagnostics + FFmpeg/ffprobe objective sanity
(duration, resolution, fps, corrupt frames, A/V drift) + **blind human A/B as the
decision method**.

## Comparative Assessment

| Decision | 2026 answer | Rationale |
|---|---|---|
| React version | 19.2+ + React Compiler | Concurrent rendering, async transitions; skip RSC |
| State | Keep Zustand; isolate hot path | Dominant lib; atomic selectors win for playhead |
| Timeline renderer | Canvas layer + virtualization; DOM for handles | Production evidence: DOM collapses on dense timelines |
| Packaging | Tauri 2 sidecar | 10× smaller/faster; Electron escape hatch via shell-api |
| Agent interface | MCP server over timeline/storyboard | 2026 industry standard (Descript, Shotstack…) |
| Agent pattern | Supervisor crew (Director→Cinematographer→Continuity→Reviewer) | Google Co-Director template; LangGraph |
| Prompt LLM | Qwen3.x 4B–9B class | Replaces/augments Gemma 3; Ollama |
| Critique VLM | Qwen3-VL 8B | Native video understanding; VQQA loop |
| Eval | VBench-2.0 subset + FFmpeg + blind A/B | The missing gate before any model swap |

## Relevance to Milimo Video

v2 frontend: React 19.2+ with Compiler, `useTransition` for playhead scrubbing, canvas
timeline layer + virtualization (study Twick/react-timeline-editor first), Tauri 2
sidecar packaging. v2 workflows: copilot side panel bound to timeline selection with
an **agent diff/approval step** (contact-sheet previews before committing),
transcript-first rough-cut mode, and a **supervisor agent crew** (Director →
Cinematographer → Continuity w/ CANVAS-like reference store → VQQA Reviewer) on
LangGraph with **MCP tools over the timeline/storyboard** — a natural bridge to
Mainza's Milimo Quantum multi-agent work. Stand up the eval harness (VBench-2.0
subset + blind A/B) **before** any model swap; it gates Phase 1.

## Sources

- React/frontend: https://merge.rocks/blog/what-is-the-best-front-end-framework-in-2025-expert-breakdown · https://theydidntask.com/blog/frontend-frameworks-summer-2026 · https://medium.com/@safdaralii/react-19-features-what-actually-changed-and-what-i-use-ad88509b3c70 · https://dev.to/locionic/zustand-vs-jotai-in-2026-the-practical-decision-tree-i-use-after-shipping-both-200e
- Timeline: https://github.com/ansoatc/vanta · https://github.com/mark134340/twick · https://www.webpronews.com/when-canvas-beats-the-dom-inside-the-tradeoffs-powering-google-docs-figma-and-next-gen-web-tools/
- Packaging: https://github.com/anonymousnomad/aide-sovereign-workbench/blob/HEAD/docs/LAUNCH-AUDIT-2026-08-25.md · https://github.com/inkstrata/marxy/blob/HEAD/docs/adr/0010-stack-tauri.md · https://github.com/oxwen11/pie/blob/HEAD/docs/2026-07-14-desktop-embedding-research.md
- Copilot UX: http://dev.to/fredabila/the-best-ai-video-editors-for-claude-and-cursor-agents-in-2026-what-to-look-for-from-someone-who-builds-one-p93 · https://www.trendhunter.com/protrends/creative-copilots
- Agentic: https://www.marktechpost.com/2026/09/27/google-research-introduces-an-ai-video-co-director-4-agentic-frameworks-for-coherent-minutes-long-video-generation/ · https://arxiv.org/pdf/2603.14790 · https://github.com/mhhamdan/agentic-ai-engineer/blob/HEAD/concepts/multi-agent/multi-agent-frameworks-deep-dive.md
- LLMs/VLMs: https://github.com/gunther-schulz/sd-webui-prompt-enhancer · https://intelligibberish.com/articles/best-local-vision-models-by-vram/
- Eval: https://github.com/zhongzhui-coder/benchmark/blob/HEAD/README_en.md · https://dev.to/aimodels-fyi/video-ai-reality-check-new-test-reveals-major-gaps-between-ai-videos-and-human-expectations-180f

Unconfirmed: React 19.2.7 date (secondary source); Zustand download counts
approximate; "abliterated" Qwen3.5 safety/licensing — check Milimo's policy before
defaulting; one pulse.bot URL truncated in source material.

## Related

- [[../systems/nle-timeline.md]] — the DOM timeline this redesigns
- [[../systems/frontend-state.md]] — the Zustand store this keeps
- [[../systems/storyboard-engine.md]] — the engine the agent crew drives
- [[../systems/prompt-enhancement.md]] — the llm.py dispatcher this upgrades
- [[../roadmap/v2-roadmap.md]] — the v2 plan
