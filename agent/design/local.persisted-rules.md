# Persisted Rules

**Concept**: Versioned rule persistence with per-section config — users generate, curate, and compose rulesets across iterations
**Created**: 2026-03-28
**Status**: Proposal

---

## Overview

Claude rule generation is non-deterministic. A ruleset the user loves can be lost on the next run. This design introduces persisted, versioned rule files that the user owns and curates. Rules become first-class project artifacts — versionable, editable, composable across sections and iterations.

---

## Problem Statement

- Claude generates different rules every time, even with identical input
- No way to preserve a good ruleset between runs
- Per-section settings (vocal bleed threshold, effect offsets) are global — can't tune a quiet section differently from an intense section
- Sections may be split or merged between iterations, invalidating previous rule files
- Users can't mix-and-match rules from different generations (e.g., section 1 rules from run A, section 5 rules from run B)

---

## Solution

### File Structure

```
rules/
  rules_001.yaml    # First Claude generation
  rules_002.yaml    # Second generation (different sections/thresholds)
  rules_003.yaml    # Manual tweaks of 002
  rules_004.yaml    # New generation after section split
rules.yaml          # Master — the active ruleset, applied during render
```

### Master rules.yaml Format

Each section is self-contained with its own rules, boundaries, thresholds, and offsets:

```yaml
# rules.yaml — master ruleset (applied during render)
version: 1
generated_from: rules/rules_004.yaml  # lineage tracking

sections:
  - name: "Gentle Morning: ambient"
    start: "0:00"
    end: "0:52"
    direction: "Dreamy ambient, piano and vocals. Glow only."
    vocal_bleed_threshold: 0.30
    effect_offsets:
      zoom_pulse: -100
      glow_swell: 0
    rules:
      - stem: piano/full
        band: full
        effect: glow_swell
        min_strength: 0.15
        max_strength: 1.0
        intensity_scale: 0.8
        duration: 0.6
        sustain_from_rms: true
      - stem: vocals/full
        band: full
        effect: glow_swell
        min_strength: 0.20
        max_strength: 1.0
        intensity_scale: 1.0
        duration: 0.5

  - name: "D&L Zone 6: peak"
    start: "17:37"
    end: "18:57"
    direction: "Peak energy. Maximum drop intensity."
    vocal_bleed_threshold: 0.10
    effect_offsets:
      zoom_bounce: -120
      shake_x: -60
      shake_y: -60
    rules:
      - stem: kick/full
        effect: zoom_bounce
        min_strength: 0.15
        intensity_scale: 1.5
        duration: 0.2
        layer_with: [shake_y]
        layer_threshold: 0.7
      - stem: snare/mid
        effect: shake_x
        min_strength: 0.20
        intensity_scale: 1.3
        duration: 0.15
      # ... more rules
```

### Key Design Properties

**Self-contained sections**: Each section carries its own time boundaries, thresholds, offsets, and rules. No global settings that bleed across sections.

**Composable**: The user can take section 1 from `rules_001.yaml` and section 5 from `rules_003.yaml` and compose them into `rules.yaml`. Sections don't reference external state.

**Editable**: YAML is human-readable. Users can hand-edit `min_strength`, `intensity_scale`, `effect_offsets` directly. The GUI provides sliders that write to the same YAML.

**Versionable**: Each generation gets an incrementing number. The master `rules.yaml` tracks which version it was derived from.

**Section-independent**: Different rule files may define different sections (different split points). The master `rules.yaml` defines the authoritative section boundaries. Re-generating rules doesn't change the section structure unless the user explicitly re-splits.

---

## Implementation

### CLI Commands

```bash
# Generate a new ruleset (saved to rules/rules_NNN.yaml)
beatlab generate-rules --config scenecraft.yaml

# Apply a specific ruleset as master
beatlab apply-rules rules/rules_003.yaml

# Apply master rules.yaml and render
beatlab effects video.mp4 --rules rules.yaml

# Re-apply master rules without re-generating (instant)
beatlab apply-rules rules.yaml --render --preview
```

### Server Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/projects/:name/rules` | List all rule files in rules/ |
| GET | `/api/projects/:name/rules/:file` | Get a specific rule file |
| POST | `/api/projects/:name/rules/generate` | Generate new ruleset (async job) |
| POST | `/api/projects/:name/rules/apply` | Set a rule file as master |
| PUT | `/api/projects/:name/rules/:file` | Update a rule file (GUI edits) |
| PUT | `/api/projects/:name/rules/master/sections/:idx` | Update a specific section in master |

### Render Flow

```
rules.yaml (master)
  │
  ├── Section 1: load rules + thresholds + offsets
  │     └── apply_rules_in_range(layer1, rules, start, end, threshold, offsets)
  │
  ├── Section 2: load rules + thresholds + offsets
  │     └── apply_rules_in_range(...)
  │
  └── Section N: ...
```

No Claude call during render. All rules are pre-baked in the YAML. Render is purely deterministic and instant (minus the video processing).

### GUI Integration

- **Rules panel**: List of sections, each expandable to show rules
- **Per-rule sliders**: min_strength, intensity_scale, duration
- **Per-section sliders**: vocal_bleed_threshold, effect_offsets
- **Toggle rules on/off**: Soft-disable without deleting
- **"Regenerate section" button**: Re-runs Claude for just that section, adds to rules/ as new version
- **"Import from" dropdown**: Pull a section's rules from a different version file
- **Live preview**: Changing a slider re-applies rules and shows preview segment instantly

---

## Benefits

- **Deterministic renders**: Same rules.yaml = same output every time
- **User ownership**: Rules are artifacts the user controls, not ephemeral Claude output
- **Per-section tuning**: Each section has independent thresholds and offsets
- **Franken-doc composition**: Mix rules from different generations
- **Instant re-renders**: No Claude call needed — just re-apply and render
- **Version history**: Never lose a good ruleset

---

## Trade-offs

- **More files**: rules/ directory accumulates versions. Mitigated by cleanup command or auto-pruning.
- **YAML complexity**: Per-section config is verbose. Mitigated by GUI — users rarely edit YAML directly.
- **Section boundary drift**: If user changes sections in scenecraft.yaml but not in rules.yaml, they diverge. Mitigated by validation on render — warn if section boundaries don't match.

---

## Dependencies

- Existing: `apply_rules()`, `apply_rules_in_range()`, `extract_layer3_rules()`
- New: YAML serialization of rules (currently JSON in audio_intelligence output)
- Server: Additional endpoints for rules CRUD

---

## Migration Path

1. **Phase 1**: Add `--rules` flag to `beatlab effects` that loads rules from YAML instead of regenerating
2. **Phase 2**: Add `beatlab generate-rules` command that saves to `rules/rules_NNN.yaml`
3. **Phase 3**: Add `rules.yaml` master concept with per-section config
4. **Phase 4**: Server endpoints for GUI integration
5. **Phase 5**: GUI rules panel with sliders and live preview

---

## Key Design Decisions

### Architecture

| Decision | Choice | Rationale |
|---|---|---|
| Storage format | YAML | Human-readable, editable, git-friendly |
| Per-section config | Everything per-section | Ambient and hard step need different thresholds — global doesn't work |
| Version numbering | Sequential (001, 002, ...) | Simple, no conflicts, easy to reference |
| Master file | Separate `rules.yaml` | Decouples "active rules" from "version history" |
| Section boundaries | In the rules file itself | Self-contained — no external dependencies |

### Workflow

| Decision | Choice | Rationale |
|---|---|---|
| Re-generation scope | Per-section, not whole file | User may only want to re-roll one section |
| Composition | Manual copy-paste between versions | Simple, flexible, no tooling needed initially |
| Render dependency | rules.yaml only, no Claude | Deterministic, instant, offline-capable |

---

## Future Considerations

- **A/B comparison**: Render two sections with different rule versions side-by-side
- **Rule templates**: Pre-built rulesets for common genres (EDM, ambient, cinematic)
- **Auto-merge**: Tooling to compose franken-docs from multiple versions
- **Rule diffing**: Show what changed between two rule versions
- **Collaborative editing**: Multiple users editing rules.yaml on shared project

---

**Status**: Proposal
**Recommendation**: Implement Phase 1 (`--rules` flag on effects command) immediately — minimal change, maximum value
**Related Documents**: [local.multi-model-stem-pipeline.md](local.multi-model-stem-pipeline.md), [local.beatlab-server.md](local.beatlab-server.md)
