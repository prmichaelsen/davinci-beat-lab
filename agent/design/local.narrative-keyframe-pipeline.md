# Narrative Keyframe Pipeline

**Concept**: YAML-driven keyframe generation and Veo transition pipeline with narrative/motif-aware candidate selection
**Created**: 2026-03-25
**Status**: Design Specification

---

## Overview

Replaces the automatic section-based rendering pipeline with a narrative-driven workflow where the user authors a YAML document specifying keyframe timestamps, source images, prompts, and musical context drawn from the section-map. Each keyframe generates 4 image candidates for selection. After keyframe selection, Veo generates transition videos between keyframes, which are time-remapped to fit the exact duration between bordering keyframe timestamps. Long gaps (>8s) are subdivided into slots, each generating its own candidate set.

The goal is tighter sync between visuals and the narrative/mood/musical hits in the piece, with human curation at every decision point via a grid selector UI.

---

## Problem Statement

- The current pipeline generates style prompts per-section automatically, but sections are coarse musical boundaries — they don't capture the precise narrative beats, mood shifts, or motif callbacks that make a music video feel intentional
- Thread-pool-based source image selection is disconnected from the specific visual narrative the artist has in mind
- No mechanism for the artist to specify "this keyframe should use this specific reference image" or "this transition should feel like X"
- Long transitions between keyframes need subdivision to avoid single-take Veo clips that drift or lose coherence

---

## Solution

A two-phase pipeline:

1. **Keyframe Generation**: Ingest a `narrative_keyframes.yaml` file. For each keyframe entry, generate 4 image candidates using the specified source image + prompt + musical context. Present candidates in a grid selector for user pick.

2. **Transition Generation**: For each pair of adjacent selected keyframes, generate Veo video transitions. Gaps >8 seconds are subdivided into slots (each slot gets its own candidate set). User can prefill transition action prompts or leave them null for the LLM to decide based on bordering keyframe context. Generated clips are time-remapped to fit exactly within the timestamp window.

---

## Implementation

### YAML Schema

```yaml
meta:
  title: "Singularity Debut"
  audio: ".beatlab_work/beyond_the_veil/audio.wav"
  fps: 30
  resolution: [1920, 1080]
  candidates_per_slot: 4
  transition_max_seconds: 8

keyframes:
  - id: kf_001
    timestamp: "0:00"
    section: "1A"
    source: "assets/stills/dark.png"
    prompt: "Ethereal void, soft bloom emerging from darkness"
    context:
      mood: "dreamy, serene"
      energy: low
      instruments:
        - "ethereal pads"
        - "soothing vocals"
      motifs:
        - "PAD-VERSE-1A"
      events: []
      visual_direction: "Slow, gentle, ethereal. Soft bloom/glow."
    candidates: []
    selected: null

  - id: kf_002
    timestamp: "0:52"
    section: "1B"
    source: "assets/stills/light.png"
    prompt: "Hard impact, bass weight landing, flash of white"
    context:
      mood: "heavy, impactful"
      energy: high
      instruments:
        - "heavy bass dubs"
        - "punchy kicks"
      motifs:
        - "DUB-1A"
      events:
        - { type: "drop", at: "0:52", intensity: 0.9 }
      visual_direction: "Hard flash/shake on each dub hit."
    candidates: []
    selected: null

transitions:
  - id: tr_001
    from: kf_001
    to: kf_002
    duration_seconds: 52.0
    slots: 7
    action: null
    candidates: []
    selected: []
    remap:
      method: "linear"
      target_duration: 52.0
```

### Schema Field Reference

#### `meta`

| Field | Type | Description |
|---|---|---|
| `title` | string | Project/piece title |
| `audio` | string | Path to source audio file |
| `fps` | int | Timeline frame rate |
| `resolution` | [int, int] | Output resolution [width, height] |
| `candidates_per_slot` | int | Base candidates per keyframe or transition slot (default: 4) |
| `transition_max_seconds` | int | Max seconds per transition slot before subdivision (default: 8) |

#### `keyframes[]`

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique keyframe identifier (e.g., `kf_001`) |
| `timestamp` | string | Time position in `M:SS` or `M:SS.mmm` format |
| `section` | string | Section ID from section-map (e.g., `1A`, `2D`, `3G`) |
| `source` | string | File path to reference image (e.g., `assets/stills/breathe.png`) |
| `prompt` | string | Image generation prompt for this keyframe |
| `context.mood` | string | Mood descriptor from section-map |
| `context.energy` | string | Energy level: `low`, `low-mid`, `mid`, `mid-high`, `high`, `peak` |
| `context.instruments` | list[string] | Active instruments at this timestamp |
| `context.motifs` | list[string] | Motif Registry labels (e.g., `PAD-VERSE-1A`, `DUB-1A`, `LEAD-MELODY-1A`) |
| `context.events` | list[object] | Musical events at/near this timestamp |
| `context.events[].type` | string | Event type: `drop`, `riser`, `stab`, `vocal_entry`, `silence`, `kick`, `vocal_sample` |
| `context.events[].at` | string | Timestamp of the event |
| `context.events[].intensity` | float | 0.0-1.0 intensity scale |
| `context.visual_direction` | string | Visual direction notes from section-map |
| `candidates` | list[string] | Filled after generation: paths to candidate images |
| `selected` | int or null | Index (0-based) of selected candidate, or null |

#### `transitions[]`

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique transition identifier (e.g., `tr_001`) |
| `from` | string | Source keyframe ID |
| `to` | string | Target keyframe ID |
| `duration_seconds` | float | Computed: `to.timestamp - from.timestamp` |
| `slots` | int | Computed: `ceil(duration_seconds / transition_max_seconds)` |
| `action` | string or null | Transition prompt. `null` = LLM decides from bordering keyframe context |
| `candidates` | list[list[string]] | Nested: `slots` lists of `candidates_per_slot` video paths |
| `selected` | list[int] | One selection index per slot |
| `remap.method` | string | Time remap curve: `linear`, `ease_in`, `ease_out`, `ease_in_out`, `speed_curve` |
| `remap.target_duration` | float | Must equal `duration_seconds` — the exact window to fill |

### Slot Subdivision Logic

```
duration = to.timestamp - from.timestamp
slots = ceil(duration / meta.transition_max_seconds)
candidates_in_grid = slots * meta.candidates_per_slot

Examples:
  6s gap  -> 1 slot  -> 4 candidates in grid  (pick 1)
  8s gap  -> 1 slot  -> 4 candidates in grid  (pick 1)
  14s gap -> 2 slots -> 8 candidates in grid  (pick 2)
  52s gap -> 7 slots -> 28 candidates in grid (pick 7)
```

### Pipeline Phases

```
Phase 1: Author YAML (manual — user writes keyframes from section-map)
    ↓
Phase 2: Generate Keyframe Candidates (parallel — 4 per keyframe)
    ↓
Phase 3: Select Keyframes (grid selector UI)
    ↓
Phase 4: Generate Transition Candidates (parallel — N slots × 4 per transition)
    ↓
Phase 5: Select Transitions (grid selector UI)
    ↓
Phase 6: Time Remap + Assemble (ffmpeg — fit each clip to its window)
    ↓
Phase 7: Final Render (concat all clips with audio)
```

### LLM Transition Action Generation

When `action` is null, the LLM constructs a transition prompt from:
- The `from` keyframe's prompt, mood, energy, motifs
- The `to` keyframe's prompt, mood, energy, motifs
- The duration and number of slots
- Any events occurring between the two keyframes (from section-map)
- Motif callbacks: if `to` references a motif that appeared earlier, the LLM should reference the visual treatment from that motif's first keyframe appearance

### Work Directory

```
.beatlab_work/{project}/
├── narrative_keyframes.yaml        # The authored input
├── keyframe_candidates/
│   ├── kf_001/
│   │   ├── candidate_0.png
│   │   ├── candidate_1.png
│   │   ├── candidate_2.png
│   │   └── candidate_3.png
│   └── kf_002/
│       └── ...
├── transition_candidates/
│   ├── tr_001/
│   │   ├── slot_0/
│   │   │   ├── candidate_0.mp4
│   │   │   ├── candidate_1.mp4
│   │   │   ├── candidate_2.mp4
│   │   │   └── candidate_3.mp4
│   │   ├── slot_1/
│   │   │   └── ...
│   │   └── ...
│   └── tr_002/
│       └── ...
├── selected/
│   ├── kf_001.png
│   ├── kf_002.png
│   ├── tr_001_slot_0.mp4
│   ├── tr_001_slot_1.mp4
│   └── ...
├── remapped/
│   ├── tr_001.mp4
│   └── ...
└── final.mp4
```

---

## Benefits

- **Narrative precision**: Every keyframe is hand-placed at a musically meaningful moment with context from the section-map
- **Motif continuity**: Motif Registry labels let the LLM create visual leitmotifs — recurring motifs get visual callbacks that escalate
- **Human curation**: 4 candidates per slot means the artist always picks; no fully-automatic output
- **Long-gap handling**: Slot subdivision prevents Veo from generating single clips that drift over long durations
- **Flexible transitions**: User can override any transition action or let the LLM decide — mix of control and automation

---

## Trade-offs

- **Manual authoring**: The YAML requires upfront work mapping keyframes to timestamps — mitigated by section-map as a reference and potential future UI for authoring
- **Candidate volume**: A 36-minute piece with 50 keyframes and many long gaps could generate hundreds of candidates — mitigated by parallelization and grid selector batching
- **Veo cost**: Each slot generates 4 video candidates — multiply by number of slots across all transitions — mitigated by preview mode at lower resolution
- **Time remap quality**: Speed-changing generated video can introduce artifacts — mitigated by keeping slots to 8s max so remap ratios stay reasonable

---

## Dependencies

- Veo API for transition video generation
- Image generation API for keyframe candidates (Imagen or similar)
- Grid selector UI (extends M10 Hit Marker Web UI pattern)
- Section-map.md with Motif Registry labels
- ffmpeg for time remapping and final assembly
- Remote GPU via Vast.ai for generation (per project constraint)

---

## Testing Strategy

- Unit: YAML schema validation (required fields, timestamp format, motif label format)
- Unit: Slot subdivision math (`ceil(duration / max_seconds)`)
- Integration: Single keyframe → 4 candidates generated and selectable
- Integration: Single transition with 2 slots → 8 candidates in grid
- Integration: Time remap of a slot clip to exact target duration
- E2E: 3-keyframe mini sequence through full pipeline to final.mp4

---

## Key Design Decisions

### Schema

| Decision | Choice | Rationale |
|---|---|---|
| Source per keyframe | Explicit file path, no thread pool | Artist controls exactly which reference image drives each keyframe |
| Motif format | Motif Registry labels (e.g., `DUB-1A`) | Matches section-map convention, enables visual leitmotif tracking |
| Candidates per slot | 4 (configurable in meta) | Enough variety without overwhelming the selector |
| Transition max seconds | 8 (configurable in meta) | Keeps Veo clips short enough for coherence |

### Pipeline

| Decision | Choice | Rationale |
|---|---|---|
| Action null = LLM decides | LLM reads bordering keyframe context | Reduces authoring burden while preserving narrative awareness |
| Slot subdivision | `ceil(duration / 8)` | Prevents long single-take drift; each slot is independently selectable |
| Time remap per slot | Individual remap then concat | Better quality than remapping one long clip |
| Parallelization | Same technique as Wan pipeline | Proven pattern, maximizes GPU utilization |

---

## Future Considerations

- YAML authoring UI (extend beatlab-synthesizer web app)
- Auto-suggest keyframe timestamps from beat analysis + section-map
- Motif visual memory: store selected keyframe images per motif label so future callbacks can reference them
- A/B comparison mode in grid selector (side-by-side two candidates)
- Transition style presets (hard cut, dissolve, morph, wipe) as shorthand for common actions

---

**Status**: Design Specification
**Recommendation**: Create milestone and tasks for implementation
**Related Documents**: [Wan2.1 Pipeline](local.wan21-film-pipeline.md), [AI Effect Director](local.ai-effect-director.md), [Section Map](../../assets/notes/singularity-debut/section-map.md), [Requirements](requirements.md)
