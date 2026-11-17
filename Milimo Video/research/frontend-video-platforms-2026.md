---
type: research
created: 2026-11-16
updated: 2026-11-16
tags: frontend ux canvas timeline video-generation short-form agents research
sources:
  - https://runway.com/research/introducing-runway-gen-4
  - https://runway.com/product/aleph-2
  - https://runway.com/news/introducing-runway-agent
  - https://help.runwayml.com/hc/en-us/articles/51601639579667-Creating-with-Runway-Agent
  - https://support.google.com/flow/answer/16353334
  - https://support.google.com/labs/answer/16935718
  - https://blog.google/technology/ai/veo-updates-flow/
  - https://kling.ai/quickstart/klingai-video-3-model-user-guide
  - https://lumalabs.ai/app
  - https://lumalabs.ai/learning-center/articles/welcome-to-luma-agents
  - https://pika.art/
  - https://ltx.io/studio
  - https://ltx.studio/platform/ai-movie-maker
  - https://blog.adobe.com/en/publish/2026/09/08/generate-create-directly-in-your-timeline-with-new-ai-powered-innovations-in-premiere-after-effects
  - https://descript.com/features
  - https://www.capcut.com/resource/ai-generation-and-full-timeline-editing-in-capcut-desktop
  - https://www.opus.pro/tools/video-repurposing-tool
  - https://boords.com/docs/storyboard-views
  - https://shotdesigner.com
  - https://www.studiobinder.com/
  - https://help.frame.io/en/articles/9101068-versioning-in-frame-io
  - https://weave.figma.com/
  - https://www.figma.com/solutions/ai-video-compositing
  - https://krea.ai/docs/user-guide/features/nodes
  - https://krea.ai/docs/realtime
  - https://github.com/Comfy-Org/ComfyUI
  - https://tldraw.dev/features/out-of-the-box-whiteboard
  - https://milanote.com/
  - https://openai.com/index/sora-2/
  - https://github.com/mainza-ai/kubona
related:
  - ../roadmap/frontend-refactor-2026-11.md
  - ../systems/studio-shell.md
  - frontend-ux-workflows-2026.md
---

# Frontier Video & Canvas Platforms — November 2026 Survey

## Landscape

The front end of AI video is no longer a prompt box with a gallery. Across the
platforms surveyed in November 2026, four surfaces keep reappearing:

1. **A conversation/agent surface** — plan, direct, refine, and explain.
2. **A spatial surface** — boards, storyboards, references, takes, areas, links.
3. **A temporal surface** — a real timeline for trimming, arranging, audio, and
   finishing.
4. **A review surface** — versions, compare, comments, approvals, activity.

The strongest products connect them to **one project context**, so a reference
placed on a board becomes an ingredient in a prompt, a shot card becomes a clip,
and a comment becomes an agent action. The weakest products make the user
re-upload or re-describe context at every step.

> **Method note (2026-11-16):** this is a product/UI survey based on vendor
> documentation, help centers, and public product pages, not hands-on trials.
> Vendor UIs change quickly. Claims marked **unconfirmed** were not verified in
> first-party documentation. Sora 2 is included as an interaction reference only;
> OpenAI's consumer Sora product was discontinued on 2026-04-26.

## Executive Synthesis

1. **The winning loop is conversation → board → cut.** Runway Agent, Luma Agents,
   LTX Studio, Google Flow, and Adobe all converge on “describe/direct → plan →
   generate → assemble → refine,” with the agent handing control back to a manual
   editor rather than replacing it.
2. **Storyboard is the control surface for narrative.** LTX Studio, Boords, Sora 2
   Storyboard, Kling Multi-Shot, and Flow Frames all treat shot planning as the
   place where continuity is decided. The timeline is for finishing, not for
   discovering the story.
3. **Consistency is an asset problem, not a prompt problem.** Flow “Ingredients,”
   Kling “Elements,” LTX “Elements,” Luma “Master Reference Assets,” Runway
   “References/Characters,” and Pika “Character Studio” all converge on reusable
   identity objects. Milimo's `@trigger` Elements are directionally right; they
   need to become first-class, inspectable master references.
4. **Takes are expected to be non-destructive.** Runway Generations/Final Cut,
   Frame.io version stacks, Google Flow history/save-frame, and Krea
   downstream-only reruns all preserve the lineage of a shot. Regeneration that
   overwrites the only result is now a UX anti-pattern.
5. **Agents must show the plan before spending.** Runway Agent's “Ask before
   generating media” shows model, prompt, and estimated credit cost per step;
   Krea Node Agent shows a plan, validates the graph, and shows per-node cost.
   Approval is a product feature, not friction.
6. **Multi-shot is becoming native.** Kling 3.0 has Multi-Shot and Custom
   Multi-Shot; Pika's Video Studio advertises up to 30-second multi-shot output;
   LTX Studio builds shot-by-shot boards; Runway Agent assembles multi-shot
   sequences. The UI must plan at shot level even when a model can generate
   several shots in one call.
7. **Text/transcript is a first-class editing surface for dialogue and shorts.**
   Descript edits video by editing the transcript; CapCut auto-captions and scene
   panels; OpusClip finds moments from transcript/visual/sentiment cues. Milimo's
   script/storyboard engine is close; it needs a transcript-first rough-cut mode.
8. **Canvas only scales with LOD, culling, a minimap, and a list view.** tldraw
   ships viewport culling, batching, camera, minimap, focus mode; Kubona's
   Workbench adds LOD tiers, viewport culling, areas, pins, and an accessible list
   view. A infinite canvas without these is a demo.
9. **Review belongs inside the studio.** Frame.io's version stacks, comparison
   viewer, frame-accurate comments, and approvals; Boords' client review links and
   activity log; Runway's @Agent-in-comments. Review is not an export destination.
10. **Local-first is Milimo's unclaimed moat.** None of the surveyed cloud
    platforms offer a local, private, per-project ownership story. Milimo can win
    on privacy, predictable cost, model choice, and full NLE control — if the
    front end stops feeling like a demo harness.

## Platform Profiles

### Runway — Gen-4, Aleph 2.0, Agent

Runway's public story is consistency plus control. Gen-4 uses visual references
to keep characters, objects, and locations coherent across shots. Aleph 2.0
edits one frame and carries the change through a clip, preserving what wasn't
asked to change, with multi-shot support and up to 30 seconds at 1080p. Its Edit
Studio previews the edit as an image before committing to video. Runway Agent is
a chat-based creative partner that proposes a concept and story structure, then
builds multi-shot video; its built-in **Final Cut** timeline lets the user trim,
split, arrange, adjust volume, and add tracks. Agent settings include
**ask before generating media** (plan + model + prompt + estimated credit cost per
step) versus auto-generate, and an optimize preference for speed/cost/quality.
The **Generations** tab collects every output in a session, and @Agent can act on
a comment.

**Steal:** plan-before-spend, model/cost preview, Generations grid, Final Cut
handoff, comment-triggered agent actions, preview-before-generate.
**Avoid:** hiding the manual timeline behind an agent session; credit-priced
black boxes.

### Google Flow / Veo 3.1

Flow is organized around a prompt box with a model selector, **Ingredients**
(reference images for characters, objects, style, voice), **Frames** (start/end
image), **Extend**, **Insert/Remove**, and **Scenebuilder** (arrange, reorder,
trim, preview, download a sequence). The History panel preserves previous versions
and prompts; any frame can be saved as an ingredient or start/end frame. There is
an Agent toggle and @-references for characters/assets. Flow's support docs
describe multi-turn conversational edits and a “Save frame” action directly from
the viewer.

**Steal:** ingredients as first-class attachments, @-references, saved frames,
history/version panel, scenebuilder as a lightweight sequence strip.
**Avoid:** making the sequence builder a separate mode with no shot-level detail.

### Kling 3.0

Kling 3.0 (Feb 2026) adds native audio, 3–15 second flexible duration,
Multi-Shot and Custom Multi-Shot, enhanced Element consistency, multi-image/video
references, multi-character coreference, multilingual dialogue, and native text
rendering. Elements can be built from 2–4 images or a video, with an optional
bound voice; camera movement and subject identity stay locked across shots.
Custom Multi-Shot lets the user specify shot count/duration and describe each shot
in one prompt.

**Steal:** shot-level planning inside a single generation, element libraries with
voice, custom multi-shot UI, native audio controls.
**Avoid:** one giant prompt as the only way to express shot structure.

### Luma — Agents / Boards

Luma organizes work on **infinite, zoomable boards** with a rich UI and an agent
chat. Boards group references, scenes, assets, versions, and explorations
spatially; **Master Reference Assets** are placed in a dedicated frame as the
source of truth for characters/products/logos. Luma advertises semantic search,
annotations (notes, arrows, labels, drawings), frames, parallel generation across
modalities, real-time collaboration, and an agent that routes across models.
Brainstorm Mode explores without generating; Create Mode generates and executes.

**Steal:** board as project home, Master Reference area, semantic search,
annotations, brainstorm/create split, shared context across modalities.
**Avoid:** agent-only control; Luma still exposes granular tools in the board UI.

### Pika

Pika presents a suite of self-contained “apps” (Video Studio, Character Studio,
Extend, Color Grade, Soundtrack, Speech, SFX, Product Shot) over many models.
Video Studio advertises short clips to 30-second multi-shot output. Character
Studio provides reusable characters.

**Steal:** task-specific apps that hide model complexity; character studio;
audio/soundtrack as a first-class app.
**Avoid:** app sprawl without a shared project spine.

### LTX Studio

LTX Studio is the closest analogue to Milimo's ambitions: “Write it. Board it.
Cut it.” It offers script-to-storyboard, a dynamic storyboard, control over
camera/keyframes/references, style presets, Elements for consistency, a timeline
editor, sound design, real-time collaboration, Retake for individual shots, auto
pitch decks, and XML export. Its marketing explicitly positions one workspace from
concept to final cut.

**Steal:** the exact three-surface narrative (script → board → cut), Retake,
pitch-deck export, XML handoff, Elements.
**Avoid:** cloud-only assumptions and subscription compute.

### Adobe Premiere / Firefly + Frame.io

Adobe's September 2026 update puts a **Generative Media** tool directly in the
timeline: select a gap/range, describe what you need, and generate context-aware
video/audio using Firefly or partner models (Veo, Kling, Runway, Luma). It samples
reference frames from the surrounding edit. It also adds generated sound
effects/soundscape/music, AI audio cleanup/ducking, AI Assistant in Premiere and
After Effects, Color Mode, and Frame.io mounted storage. Frame.io contributes
version stacks, comparison viewer, frame-accurate comments, and approvals.

**Steal:** generate in place, reference frames from the cut, model choice,
timeline-native AI, AI assistant with project-wide context, version stacks.
**Avoid:** pro-app density and modal overload; Milimo should be simpler by default.

### Descript

Descript's core is text-based editing: deleting words deletes media; scenes and
layouts arrange what is on screen without touching a timeline; Underlord is an AI
co-editor; clips, captions, translation, and MCP-based agent control are built in.
Its docs describe the timeline as optional — “only there if you want it.”

**Steal:** transcript as an editing surface, scenes/layouts, “edit by text,”
AI co-editor with undoable changes.
**Avoid:** forcing every project into a document metaphor; visual cinema still
needs a timeline.

### CapCut

CapCut Desktop combines an AI video maker with a multitrack timeline: generate a
draft from a topic/script, then edit voice, subtitles, music, effects, keyframes,
graphs, captions, and transitions in the same app. The Scenes panel organizes
generation; the timeline handles fine control; auto captions and smart search
work across local and generated media.

**Steal:** low-friction draft → timeline flow, Scenes panel, auto captions,
keyframes/graphs, smart search.
**Avoid:** template-first UI that fights custom film work.

### OpusClip

OpusClip is the short-form repurposing benchmark: upload/paste a long video,
AI finds clip candidates, each gets a **Virality Score** (Hook, Flow, Trend),
**Auto Reframe** keeps speakers/objects in frame across 9:16, 1:1, 4:5, 16:9,
auto captions, brand kit, ClipAnything prompt-based moment search, Scheduler, and
XML export to Premiere/Resolve.

**Steal:** candidate grid with scores, reframe previews, caption/brand presets,
variant matrix, scheduler/export.
**Avoid:** presenting a proprietary “virality score” as truth; label it as an
estimate and keep it local/optional.

### Boords

Boords separates storyboard views by task: global storyboard grid, shot list
table, script view for bulk text edits, frame editor with draw/annotate/camera
overlays, and a client review/share view with comments and activity log.

**Steal:** multiple views over one board, shot list table, script view, camera
overlays, review links, activity log.
**Avoid:** hiding text edits in a separate document that drifts from the board.

### Shot Designer / StudioBinder

Shot Designer is a camera-blocking tool: a top-down camera diagram that
automatically moves cameras when characters move, an integrated shot list that
stays in sync with the diagram, animated characters/cameras, storyboard frames
hung next to cameras, a director's viewfinder, and scene freeze snapshots.
StudioBinder covers the production side: script, breakdown, mood boards, shot
lists, storyboards, stripboards, calendars, task boards, call sheets.

**Steal:** top-down blocking as a canvas lens, diagram↔shot-list sync, scene
freeze, production metadata, call-sheet/export thinking.
**Avoid:** production-management scope creep in v1.

### Figma Weave / Krea / ComfyUI — node canvases

Figma Weave is a node-based creative platform: every model, reference, mask,
composite, and edit is a visible node; workflows are reusable and can be
simplified into apps. Krea Nodes exposes typed inputs/outputs, sections, groups,
sticky notes, a Node App Builder (hide the graph, expose 5–7 inputs), and a Node
Agent that reads the canvas, proposes a plan, validates connections, shows
per-node cost, builds nodes incrementally, and reruns only downstream nodes.
ComfyUI adds subgraphs, templates, App Mode, queueing, partial graph
re-execution, and offline operation.

**Steal:** reproducible recipes, plan/validate/cost, app mode for repeated
workflows, downstream-only reruns, node groups/sticky notes.
**Avoid:** making the node graph the default surface for filmmakers. It is a
power-user lens, not the home screen.

### tldraw / Milanote / Kubona — canvas craft

tldraw's SDK documents the production requirements: viewport culling, efficient
shape batching, smooth interactions with thousands of shapes, automatic
undo/redo, conflict-free collaboration, incremental saves, camera controls,
minimap, focus mode, custom shapes, and a runtime editor API. Milanote proves the
freeform board as a creative project home: mixed notes/images/videos/links,
nested boards, comments, client sharing. Kubona's Workbench is a working React
implementation of the same ideas: a DOM-transform infinite canvas with camera,
zoom tiers/LOD, viewport culling, areas, typed links, pins, marquee selection,
align/distribute/arrange, minimap, follow-live, undo, conflict-aware layout
saves, an accessible list equivalent, a command palette, a resizable inspector,
a bottom command dock, and a copilot drawer. Kubona also ships a lazy-loaded 3D
“observatory” with orbit controls, camera persistence, depth-conditioned
snapshot/chisel, FPS/DPR adaptation, and fail-closed WebGL.

**Steal:** the entire canvas interaction vocabulary; split semantic layout save
from ephemeral view save; list view as a first-class peer; “Needs you” tray;
command palette; 3D as an alternate lens, lazy-loaded.
**Avoid:** quantum-domain-specific cards and terminology; copy patterns, not the
product.

### Sora 2 Storyboard — historical pattern

Sora 2 (Sep 2025) popularized a multi-frame Storyboard mode: up to five frame
prompts, drag-to-reorder, per-frame reference images, automatic transitions, and
a single generated sequence. It also showed the limits: 3–5× slower than a single
shot, character drift, and no native audio in Storyboard. OpenAI discontinued the
consumer Sora product on 2026-04-26; treat the storyboard interaction as a design
reference, not an integration target.

**Steal:** frame strip + per-frame prompt + drag reorder + reference per frame.
**Avoid:** platform dependency on a discontinued product.

## Comparative Matrix

Legend: ● strong / first-class, ◐ partial or workflow-dependent, ○ absent or
unclear. This is a product-surface comparison, not a model-quality ranking.

| Platform | Spatial board | Storyboard | Timeline | Agent | References/Elements | Takes/Versions | Review | Short-form | Local/offline |
|---|---|---|---|---|---|---|---|---|---|
| Runway | ○ | ◐ | ● (Agent Final Cut) | ● | ● | ● | ◐ comments | ◐ | ○ |
| Google Flow | ○ | ● Frames/Scenes | ● Scenebuilder | ● | ● Ingredients | ● History | ○ | ◐ | ○ |
| Kling 3.0 | ○ | ● Multi-Shot | ◐ editor | ◐ | ● Elements | ◐ | ○ | ◐ | ○ |
| Luma Agents | ● Boards | ● | ◐ | ● | ● Master Refs | ● | ◐ | ● | ○ |
| Pika | ○ | ◐ | ◐ | ◐ | ● Character Studio | ◐ | ○ | ● | ○ |
| LTX Studio | ◐ | ● Dynamic Storyboard | ● | ◐ | ● Elements | ● Retake | ◐ collab | ● | ○ |
| Adobe Premiere | ○ | ◐ | ● | ● AI Assistant | ● | ● Frame.io | ● Frame.io | ◐ | ◐ desktop |
| Descript | ○ | ◐ Scenes | ● optional | ● Underlord | ◐ | ◐ | ◐ | ● | ◐ desktop |
| CapCut | ○ | ◐ Scenes | ● | ◐ | ◐ | ◐ | ○ | ● | ◐ desktop |
| OpusClip | ○ | ○ | ◐ clip editor | ◐ | ◐ | ◐ | ○ | ● | ○ |
| Boords | ◐ board | ● | ○ | ◐ | ◐ | ◐ | ● | ○ | ○ |
| Shot Designer | ● blocking | ● | ○ | ○ | ○ | ● scene freeze | ○ | ○ | ● app |
| StudioBinder | ◐ mood boards | ● | ○ | ○ | ◐ | ○ | ◐ | ○ | ○ |
| Figma Weave | ● nodes | ◐ | ○ | ● Node Agent | ● | ◐ | ○ | ◐ | ○ |
| Krea | ● nodes/realtime | ◐ | ○ | ● Node Agent | ● | ◐ | ○ | ◐ | ○ |
| ComfyUI | ● nodes | ○ | ○ | ◐ | ● | ● workflows | ○ | ◐ | ● |
| tldraw SDK | ● infinite canvas | ○ | ○ | ◐ | ○ | ◐ | ◐ collab | ○ | ● SDK |
| Milanote | ● boards | ◐ | ○ | ○ | ◐ | ○ | ◐ | ○ | ○ |
| Kubona | ● Workbench | ◐ cards | ○ | ● Copilot | ◐ | ◐ | ◐ | ○ | ● local |
| Sora 2 (discontinued) | ○ | ● 5-frame | ○ | ○ | ◐ per-frame | ◐ | ○ | ◐ | ○ |

## Pattern Catalog

### P1 — The context spine

One canonical chain: **Project → Sequence → Scene → Shot → Take → Cut →
Deliverable**. Every surface reads the same IDs; no surface owns a private copy.
This is the antidote to Milimo's current split between `project.shots`,
`scene.shots`, board positions, timeline clips, and job state.

### P2 — Dual representation: board and timeline

A **spatial board** answers “what is the story and how does it relate?” A
**timeline** answers “when does it happen and how does it feel?” They are two
lenses over the same shots. Board position must never define sequence order; a
separate `sequence_index` does. Selecting a card highlights its clip and vice
versa.

### P3 — The shot card is the atomic object

The card carries thumbnail/video, scene/shot number, prompt, elements, status,
duration, takes, model/seed, review state, and actions. It has three densities:
summary, compact, detail. The same card can be placed on a board, in a shot list
row, or as a timeline clip inspector.

### P4 — Ingredients / master references / elements

A reusable identity object for character, location, object, style, voice, or
prop. It owns reference images/video, description, trigger, voice, consistency
strength, and usage. It appears as a pinned “Master References” area on the board
and as attachable “ingredients” in the prompt composer.

### P5 — Take stack / version stack

Every generation is an immutable take with model, seed, params, duration, audio,
cost/time, lineage, and review state. The shot points at a selected take. The
review viewer compares takes with wipe/side-by-side/stack. Never overwrite.

### P6 — Plan before spend

Agent/plan surfaces show: steps, models, prompts, files/objects affected,
estimated GPU time/VRAM or credit cost, and a dry-run/preview. The user can drop
steps, swap models, or approve. Local GPU time is still a cost; treat it as one.

### P7 — Context chips and attachments

The prompt/composer shows exactly what is in context: selected shot, scene,
board area, time range, references, elements, prior take. Attachments are
drag-and-drop ingredients, not hidden uploads.

### P8 — Shot-level multi-shot

Even if the model generates multiple shots natively, the UI should express shot
count, per-shot duration, per-shot prompt, camera, and references. Custom
Multi-Shot is a planning surface, not a magic prompt.

### P9 — Text-based editing

For dialogue, interview, and shorts, the transcript is an editing surface:
deleting a sentence deletes the media range; captions and scene layouts follow.
Milimo's script parser is the pre-production half; a transcript editor is the
post-production half.

### P10 — Canvas craft

Pan/zoom/orbit, marquee select, shift multi-select, snap/grid, align/distribute,
group into areas, collapse areas, pin cards, typed links, edge visibility modes,
minimap, fit/focus, LOD, follow-live, undo, keyboard nudging, list/table view,
and a command palette. Kubona and tldraw are the best references.

### P11 — Review as a first-class surface

Version stacks, compare viewer, comments anchored to frame/timecode, mentions,
resolve, approval states, share links, activity log, and an inbox (“Needs you”).
Review comments can trigger an agent action with explicit approval.

### P12 — Shorts repurposing

Source/transcript → candidate moments with hook/flow/trend scores → reframe
previews for 9:16, 1:1, 4:5, 16:9 → caption/brand presets → variant matrix →
export/schedule. The score is an estimate, not a guarantee.

### P13 — Reusable skills/recipes

Capture a successful workflow (prompt structure, references, model, edit steps)
as a named, re-runnable skill/recipe with typed inputs. Runway Skills, Krea Node
Apps, ComfyUI workflows, and Figma Weave all converge here.

### P14 — Progressive disclosure

Default to five or six controls. Put model choice, sampler, seed, guidance,
LoRA, and advanced audio behind an “Advanced” disclosure. Krea explicitly
recommends 5–7 inputs for a node app; the same discipline applies to a shot.

### P15 — Model routing with intent

Users should not need to know every model. Offer “best quality / fastest /
cheapest / local-only / specific model,” then show what was chosen and why.
Runway Agent exposes exactly this preference.

## Anti-Patterns Observed

- **Prompt-box-only.** No shot structure, no references, no takes, no review.
- **Chat sidebar disconnected from the document.** The agent cannot see the
  selection, board, or timeline; the user re-explains everything.
- **Canvas without an accessible list.** Keyboard/screen-reader users are locked
  out; large boards become unmanageable.
- **Spatial position as sequence.** Moving a card changes the film order by
  accident. Sequence must be explicit.
- **Destructive regeneration.** One click replaces the only take.
- **Hidden spend.** No model, duration, or cost/time estimate before generation.
- **Mode sprawl.** Separate apps for image, video, audio, and editing with no
  shared project context.
- **Fake progress.** A spinner with no step, ETA, model, or cancel path.
- **Glassmorphism as hierarchy.** Blur and glow cannot substitute for layout,
  contrast, and grouping.
- **Desktop-only with no review path.** Reviewers need a link and comments, not a
  full NLE install.

## Relevance to Milimo Video

Milimo already has the rare pieces the cloud platforms do not: local models,
Quantum Alignment, SAM 3, a real storyboard engine, Elements, and a timeline.
The front-end gap is not ambition; it is **coherence**.

The redesign should position Milimo as:

> **The local-first film studio where the board, the shot, and the cut are the
> same project — and the agent works for you, on your machine, with approval.**

Three product bets:

1. **Board + timeline as dual truth.** A spatial story board for planning and
   continuity, a professional timeline for finishing, one canonical shot graph.
2. **Master References + takes.** Identity and iteration are the two things AI
   filmmakers actually need; make them first-class, not prompt hacks.
3. **Agent with a plan.** A command dock that can see the board/timeline
   selection, propose a plan with time/VRAM/model, and apply approved changes as
   undoable edits.

Two workspace presets:

- **Film** — Write → Board → Generate → Cut → Review → Deliver.
- **Shorts** — Source/Transcript → Find moments → Reframe/Caption → Variants →
  Deliver.

The canvas should eventually support four lenses over the same project:
**Story Board** (cards/areas/links), **Pipeline** (node recipe), **Blocking**
(top-down camera diagram), and **Previz 3D**. Ship Story Board first; the other
lenses are progressive disclosure, not day-one requirements.

## Sources

All accessed 2026-11-16 unless noted.

- Runway, “Introducing Runway Gen-4,” <https://runway.com/research/introducing-runway-gen-4>
- Runway, “Aleph 2.0,” <https://runway.com/product/aleph-2>
- Runway, “Introducing Runway Agent,” 2026-05-13, <https://runway.com/news/introducing-runway-agent>
- Runway Help, “Creating with Runway Agent,” <https://help.runwayml.com/hc/en-us/articles/51601639579667-Creating-with-Runway-Agent>
- Google Flow Help, “Create videos in Google Flow,” <https://support.google.com/flow/answer/16353334>
- Google Flow Help, “Edit videos & build scenes in Google Flow,” <https://support.google.com/labs/answer/16935718>
- Google, “Introducing Veo 3.1 and advanced capabilities in Flow,” 2025-10-15, <https://blog.google/technology/ai/veo-updates-flow/>
- Kling AI, “Kling VIDEO 3.0 Model User Guide,” 2026-02-06, <https://kling.ai/quickstart/klingai-video-3-model-user-guide>
- Luma, “Creative agents for creative professionals,” <https://lumalabs.ai/app>
- Luma, “Welcome to Luma Agents,” 2026-03-09, <https://lumalabs.ai/learning-center/articles/welcome-to-luma-agents>
- Pika, product page, <https://pika.art/>
- LTX Studio, product page, <https://ltx.io/studio>
- LTX Studio, “AI Movie Maker,” <https://ltx.studio/platform/ai-movie-maker>
- Adobe, “Generate and create directly in your timeline…,” 2026-09-08, <https://blog.adobe.com/en/publish/2026/09/08/generate-create-directly-in-your-timeline-with-new-ai-powered-innovations-in-premiere-after-effects>
- Descript, “Features,” <https://descript.com/features>
- CapCut, “AI Video Generation and Timeline Editing in CapCut Desktop,” 2026-11-13, <https://www.capcut.com/resource/ai-generation-and-full-timeline-editing-in-capcut-desktop>
- OpusClip, “Video Repurposing Tool,” <https://www.opus.pro/tools/video-repurposing-tool>
- Boords, “Storyboard views,” <https://boords.com/docs/storyboard-views>
- Shot Designer, product page, <https://shotdesigner.com>
- StudioBinder, product page, <https://www.studiobinder.com/>
- Frame.io, “Versioning in Frame.io,” 2026-09-25, <https://help.frame.io/en/articles/9101068-versioning-in-frame-io>
- Figma Weave, product page, <https://weave.figma.com/>
- Figma, “AI video compositing,” <https://www.figma.com/solutions/ai-video-compositing>
- Krea, “Nodes workflows,” 2026-04-13, <https://krea.ai/docs/user-guide/features/nodes>
- Krea, “Krea Realtime,” <https://krea.ai/docs/realtime>
- ComfyUI, GitHub, <https://github.com/Comfy-Org/ComfyUI>
- tldraw, “Ready-to-use whiteboard,” <https://tldraw.dev/features/out-of-the-box-whiteboard>
- Milanote, product page, <https://milanote.com/>
- OpenAI, “Sora 2 is here,” 2025-09-30; discontinuation note 2026-04-26, <https://openai.com/index/sora-2/>
- Kubona, repository (Apache 2.0), <https://github.com/mainza-ai/kubona>

## Related

- [[../roadmap/frontend-refactor-2026-11.md]] — the redesign program
- [[../systems/studio-shell.md]] — target front-end architecture
- [[frontend-ux-workflows-2026.md]] — the October 2026 React/canvas/copilot survey
- [[../systems/engineering-audit-2026-11.md]] — current-state defects the redesign must not inherit
