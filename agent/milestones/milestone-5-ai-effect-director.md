# Milestone 5: AI Effect Director

**Goal**: Add LLM-powered effect selection that analyzes sections and makes creative preset/parameter choices
**Duration**: 1-2 weeks
**Dependencies**: M3 - Effect Library & Intelligence
**Status**: Not Started

---

## Overview

This milestone adds the `--ai` flag to the beatlab CLI. When used, audio section data and spectral features are sent to Claude, which returns a structured JSON effect plan. The existing generator consumes this plan to produce section-aware, creatively varied Fusion compositions.

This is the key feature that differentiates beat-lab from a simple beat-to-keyframe converter — the LLM acts as a creative director, choosing effects, tuning parameters, layering presets, and varying repeated sections.

---

## Deliverables

### 1. Extended Spectral Analysis
- Per-section spectral summaries: centroid, RMS energy, rolloff, contrast
- Included in beat map JSON and section data sent to LLM

### 2. LLM Provider Abstraction
- `LLMProvider` ABC with `complete(system, user) -> str`
- `AnthropicProvider` concrete implementation using `anthropic` SDK
- Optional dependency — only required when `--ai` is used

### 3. Effect Plan System
- JSON schema for effect plans (section → presets + params + custom effects)
- Schema validation with clear error messages
- Support for preset references and inline custom effect definitions

### 4. AI Director
- System prompt with preset catalog, section data, spectral features, coherence instructions
- User prompt from `--prompt` flag
- Calls LLM, parses JSON response, validates against schema

### 5. CLI Integration
- `--ai` flag on `beatlab run` and `beatlab generate`
- `--prompt` flag for freeform creative direction
- Clear error on missing API key or API failure

---

## Success Criteria

- [ ] `beatlab run track.mp3 --ai -o comp.setting` produces section-varied effects
- [ ] `--prompt "cinematic with hard drops"` influences LLM choices
- [ ] Different sections get different presets/parameters
- [ ] Repeated sections have variation (not identical effects)
- [ ] Clear error message when ANTHROPIC_API_KEY is missing
- [ ] Clear error message on API failure
- [ ] Works with existing presets and custom LLM-defined effects
- [ ] All existing tests still pass (no regressions)

---

## Tasks

1. [Task 12: Spectral Features](../tasks/milestone-5-ai-effect-director/task-12-spectral-features.md) - Extend analyzer with per-section spectral summaries
2. [Task 13: LLM Provider & Effect Plan](../tasks/milestone-5-ai-effect-director/task-13-llm-provider.md) - Provider abstraction, schema, validation
3. [Task 14: AI Director & Prompt](../tasks/milestone-5-ai-effect-director/task-14-ai-director.md) - Prompt construction, LLM call, plan generation
4. [Task 15: Generator Integration & CLI](../tasks/milestone-5-ai-effect-director/task-15-generator-cli.md) - Wire plan into generator, --ai and --prompt flags

---

## Risks and Mitigation

| Risk | Impact | Probability | Mitigation Strategy |
|------|--------|-------------|---------------------|
| LLM returns invalid JSON | Medium | Medium | Schema validation with retry (1 attempt) and clear error |
| LLM references non-existent presets | Low | Medium | Validate preset names, fall back to closest match |
| Prompt too large for long tracks | Low | Low | Send section summaries not per-beat data (~2-3K tokens) |

---

**Next Milestone**: [M4 - Resolve Integration & Distribution](milestone-4-resolve-integration.md) (optional)
**Blockers**: None
**Notes**: anthropic SDK is an optional dependency. The tool works fine without --ai.
