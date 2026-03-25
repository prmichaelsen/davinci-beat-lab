# Hit Marker Web UI

**Concept**: Browser-based waveform editor for manually placing, repositioning, and classifying beat hit markers with sensation labels and intensity
**Created**: 2026-03-25
**Status**: Design Specification

---

## Overview

A lightweight web UI served by the beatlab CLI that lets users place manual "hit" markers on an audio waveform during playback. Each marker captures a sensation (hit, drop, swell, etc.) and intensity level. The UI exports a `hits.json` file that the existing generator pipeline consumes as accent effects layered on top of the AI-generated plan.

This solves the problem of users wanting to trigger specific visuals at precise moments that automated beat detection and AI planning can't anticipate — dramatic pauses, vocal accents, transitions the user "feels" but algorithms miss.

---

## Architecture

### System Diagram

```
Browser (local machine)                    Remote Server (beatlab CLI)
┌─────────────────────────────┐           ┌──────────────────────────┐
│  wavesurfer.js waveform     │◄─────────►│  FastAPI/Flask server     │
│  - playback + scrubber      │   HTTP    │  - serves static HTML     │
│  - marker placement (space) │           │  - serves audio file      │
│  - drag to reposition       │           │  - serves beats.json      │
│  - sensation dropdown       │           │  - POST /save → hits.json │
│  - intensity slider         │           └──────────────────────────┘
│  - beat overlay from librosa│
│  - zoom/scroll              │
└─────────────────────────────┘
```

### CLI Integration

```bash
# Start the marker UI server
beatlab marker-ui audio.mp4 --fps 30 --port 8080

# With existing beat analysis
beatlab marker-ui audio.mp4 --beats beats.json --fps 30
```

The command:
1. Extracts audio from video (if video file given)
2. Runs beat analysis (or reads existing `beats.json`)
3. Starts a web server on the specified port
4. Opens or prints the URL for the user to visit
5. Serves the single-page HTML + audio file + beats data
6. Accepts POST requests to save `hits.json`

### Frontend Stack

- **Single HTML file** with inline CSS and JS (no build step)
- **wavesurfer.js** via CDN for waveform rendering, playback, zoom, scroll
- Fixed viewport that follows playback position
- Beat markers from librosa overlaid on waveform (visual reference, not interactive)

---

## Marker Interaction

### Tap Phase
- **M key** during playback places a marker at the current position (spacebar reserved for play/pause)
- Markers snap to the nearest frame boundary (based on `--fps`)
- Unlimited markers allowed

### Editing
- **Drag** markers on the waveform to reposition (still snaps to frame)
- **Click** a marker to select it → shows sensation dropdown + intensity slider
- **Delete key** or right-click to remove selected marker
- **Multi-select** (shift+click or drag-select) for batch delete/classify (if implementation is straightforward)
- **Undo/redo** (Ctrl+Z / Ctrl+Shift+Z) if implementation is straightforward

### Audio Preview
- Clicking a marker plays a **2-second window** before and after the marker timestamp (4s total)

---

## Sensation Model

Users classify markers with abstract **sensation labels**, not raw effect presets. The pipeline maps sensations to preset combinations internally.

### Sensation Enum

| Sensation | Description | Suggested Preset Mapping |
|---|---|---|
| `hit` | Hard rhythmic impact | flash + shake_x + shake_y |
| `drop` | Energy explosion / bass drop | hard_cut + zoom_bounce + shake_x + shake_y |
| `swell` | Rising energy, building tension | glow_swell + zoom_pulse |
| `punch` | Quick percussive accent | contrast_pop + zoom_pulse |
| `freeze` | Sudden stop / silence moment | hard_cut (inverted — dim) |
| `bloom` | Soft, expansive visual | glow_swell |
| `shake` | Camera shake only | shake_x + shake_y |

The sensation-to-preset mapping lives in Python (`SENSATION_MAP` in a new module or in `presets.py`) and can be refined independently of the UI.

### Intensity

Each marker has an intensity value (0.0–1.0) controlled by a slider in the UI. Intensity scales the preset peak values via the existing `apply_intensity()` function. Marker height on the waveform reflects intensity visually.

---

## Output Format

The UI saves a `hits.json` file:

```json
{
  "fps": 30,
  "hits": [
    {
      "time": 12.35,
      "frame": 371,
      "sensation": "hit",
      "intensity": 1.0
    },
    {
      "time": 15.80,
      "frame": 474,
      "sensation": "swell",
      "intensity": 0.7
    }
  ]
}
```

Notes:
- `time` is in seconds (float)
- `frame` is computed from time and fps, snapped to nearest integer
- `sensation` is one of the sensation enum values
- `intensity` is 0.0–1.0

---

## Pipeline Integration

### As Accents

Manual hits layer **additively** on top of the AI-generated plan. The generator:
1. Reads `plan.json` (AI section-level effects) as the base layer
2. Reads `hits.json` (manual accents) if present
3. For each hit, maps sensation → presets via `SENSATION_MAP`
4. Applies `apply_intensity()` with the hit's intensity value
5. Inserts keyframes at the hit's frame number
6. If a hit timestamp falls within a section that already has effects at that frame, the hit's effects are added (not replaced)

### Standalone Mode

The UI is usable without `--ai`. Users can place manual hits and generate effects purely from `hits.json` with no AI plan at all. The generator checks for `hits.json` and processes it regardless of whether `plan.json` exists.

### Workflow

The marker UI is standalone — the user runs it whenever they want:
- Before beat analysis (just to listen and mark)
- After beat analysis (with beat overlay for reference)
- After AI plan generation (to add manual accents)
- Iteratively (edit hits, re-render, repeat)

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Serves the single-page HTML |
| GET | `/audio` | Streams the audio file |
| GET | `/beats` | Returns `beats.json` (if available) |
| GET | `/hits` | Returns existing `hits.json` (if any) |
| POST | `/hits` | Saves `hits.json` to the work directory |

---

## Key Design Decisions

### Architecture

| Decision | Choice | Rationale |
|---|---|---|
| Server framework | FastAPI or Flask | Already Python-based project, minimal dependency |
| Frontend | Single HTML file, inline JS | No build step, simple to serve and maintain |
| Waveform library | wavesurfer.js (CDN) | Mature, handles playback/scrub/zoom/markers natively |
| Viewport | Fixed, follows playback | Better for tap-to-mark during playback |

### Interaction

| Decision | Choice | Rationale |
|---|---|---|
| Tap key | M key, then categorize (spacebar = play/pause) | Spacebar is universal play/pause; M for "marker" is intuitive |
| Frame snapping | Always snap to frame boundary | Ensures frame-accurate effect placement |
| Effect model | Sensation labels, not raw presets | User thinks in "hit/drop/swell", not "shake_x + flash" |
| Intensity | Per-marker slider, visualized by height | Gives fine control without complexity |
| Audio preview | 2s before + 2s after | Enough context to evaluate marker placement |

### Integration

| Decision | Choice | Rationale |
|---|---|---|
| Merge strategy | Additive accents on top of AI plan | Manual hits complement, not replace, automated effects |
| Standalone use | Yes, works without --ai | Flexibility — some users want fully manual control |
| Workflow | Runs whenever user wants | No forced ordering, supports iterative workflow |

---

## Future Considerations

- **Resolve marker import**: Read markers placed in DaVinci Resolve timeline as an alternative input
- **Sensation preview**: Show a visual preview of what the effect will look like at each marker
- **Waveform spectrogram**: Toggle between waveform and spectrogram view for frequency-aware marking
- **Keyboard shortcuts for sensations**: After tap, press 1-7 to quickly assign sensation without mouse
- **Collaborative mode**: Multiple users marking simultaneously (WebSocket sync)

---

**Status**: Design Specification
**Recommendation**: Create milestone and tasks, then implement
**Related Documents**: [Requirements](requirements.md), [AI Effect Director](local.ai-effect-director.md), [Clarification 3](../clarifications/clarification-3-manual-hit-marker-web-ui.md)
