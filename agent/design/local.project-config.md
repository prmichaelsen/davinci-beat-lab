# Project Configuration

**Concept**: Per-project YAML config that stores render flags, eliminating long CLI commands
**Created**: 2026-03-25
**Status**: Design Specification

---

## Overview

Each project's work directory stores a `config.yaml` with default render settings. CLI flags override the config. This eliminates 200-character command lines and ensures consistent settings across runs.

---

## Problem Statement

The render command has grown to ~15 flags. Users must remember and re-type the full command each run, risking inconsistency (forgetting `--vertex`, wrong motion prompt, etc.). The command is too long to read or share.

---

## Solution

### Config File Location

```
.beatlab_work/<video_name>/config.yaml
```

Auto-created on first render, updated when flags change.

### Config Schema

```yaml
# .beatlab_work/beyond_the_veil/config.yaml
engine: google
vertex: true
ai: true
describe: .beatlab_work/beyond_the_veil/descriptions.md
prompt: "journey through death to another dimension"
audio_prompt: true
motion: "continuous forward dolly through a void, eyes locked at vanishing point never getting closer, swirls counterclockwise"
candidates: 4
labels: false
model: null
style: "artistic stylized"
base_denoise: 0.3
beat_denoise: 0.5
sr: 22050
```

### CLI Behavior

1. On render, check for `config.yaml` in the work dir
2. Load config values as defaults
3. CLI flags override config values
4. `--save-config` flag writes current flags to config (explicit save)
5. First run auto-saves config if none exists

### Flag Precedence

```
CLI flag > config.yaml > built-in default
```

### Commands

```
# First run — long command, auto-saves config
beatlab render video.mp4 --engine google --vertex --ai --prompt "..." --motion "..." -o styled.mp4

# Subsequent runs — just the video and output
beatlab render video.mp4 -o styled.mp4

# Override one setting
beatlab render video.mp4 -o styled.mp4 --candidates 1

# Explicitly save config changes
beatlab render video.mp4 -o styled.mp4 --labels --save-config

# View current config
beatlab config beyond_the_veil

# Edit config directly
beatlab config beyond_the_veil --set engine=google --set vertex=true
```

---

## Implementation

### New Files

```
src/beatlab/render/config.py    # Config loading, saving, merging
```

### Config Module

```python
@dataclass
class ProjectConfig:
    engine: str = "google"
    vertex: bool = False
    ai: bool = False
    describe: str | None = None
    prompt: str | None = None
    audio_prompt: bool = False
    motion: str | None = None
    candidates: int = 4
    labels: bool = False
    model: str | None = None
    style: str = "artistic stylized"
    base_denoise: float = 0.3
    beat_denoise: float = 0.5
    sr: int = 22050

def load_config(work_dir: str) -> ProjectConfig: ...
def save_config(config: ProjectConfig, work_dir: str) -> None: ...
def merge_cli_flags(config: ProjectConfig, **cli_flags) -> ProjectConfig: ...
```

### CLI Integration

In the `render` command, before processing flags:

```python
# Load config if exists
config = load_config(str(work.root))

# Merge: CLI flags override config, config overrides defaults
# Only override if the CLI flag was explicitly passed (not default)
effective = merge_cli_flags(config, **{k: v for k, v in ctx.params.items() if v != default})
```

The tricky part: distinguishing "user passed `--no-vertex`" from "user didn't pass anything and it defaulted to False". Click's `ctx.get_parameter_source()` helps here — it tells you if a param came from the CLI or from the default.

---

## Benefits

- **Short commands**: `beatlab render video.mp4 -o styled.mp4` instead of 200 chars
- **Consistency**: Settings persist between runs automatically
- **Shareable**: Copy config.yaml to reproduce someone else's settings
- **Overridable**: Any flag still works on the CLI

---

## Trade-offs

- **Implicit behavior**: Config loaded silently could surprise users. Mitigate with "Using config from ..." log message.
- **Stale config**: If defaults change in code updates, old configs might have outdated values. Mitigate with version field in config.

---

## Dependencies

- PyYAML (already installed for metrics.py)

---

## Key Design Decisions

### Configuration

| Decision | Choice | Rationale |
|---|---|---|
| Config location | Inside work dir | Per-project, travels with the project |
| Auto-save on first run | Yes | Reduces friction, user doesn't need to know about config |
| CLI precedence | CLI > config > default | Explicit always wins |
| Format | YAML | Human-readable, already used for metrics |

---

## Future Considerations

- **Global config** at `~/.beatlab/config.yaml` for cross-project defaults (API keys, vertex preference)
- **Config templates**: `beatlab config --template edm` loads preset configurations
- **Config diff**: Show what changed between runs

---

**Status**: Design Specification
**Recommendation**: Implement as a small task — one module + CLI wiring
**Related Documents**: [Requirements](requirements.md)
