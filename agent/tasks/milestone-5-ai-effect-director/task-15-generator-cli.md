# Task 15: Generator Integration & CLI

**Milestone**: [M5 - AI Effect Director](../../milestones/milestone-5-ai-effect-director.md)
**Design Reference**: [AI Effect Director](../../design/local.ai-effect-director.md)
**Estimated Time**: 2 hours
**Dependencies**: Task 14
**Status**: Not Started

---

## Objective

Wire the effect plan into the existing generator, add `--ai` and `--prompt` CLI flags, and handle error cases (missing API key, API failure).

---

## Steps

### 1. Extend Generator to Accept Effect Plans

In `generator.py`, update `generate_comp()`:

```python
def generate_comp(
    beat_map: dict,
    effect_plan: EffectPlan | None = None,  # NEW
    ...
) -> FusionComp:
```

When `effect_plan` is provided:
- For each section in the plan, find the corresponding beats (by section_index)
- Apply the plan's presets and parameters to those beats
- Support custom effects by constructing ad-hoc `EffectPreset` objects
- Support layering (multiple presets per section = multiple nodes)

### 2. Add CLI Flags

On both `run` and `generate` commands:
- `--ai / --no-ai` flag (default: off)
- `--prompt TEXT` optional freeform creative direction

### 3. Wire Up the Pipeline

In the `run` command:
```python
if ai:
    from beatlab.ai.provider import AnthropicProvider
    from beatlab.ai.director import create_effect_plan
    provider = AnthropicProvider()
    effect_plan = create_effect_plan(beat_map, provider, user_prompt=prompt)
    comp = generate_comp(beat_map, effect_plan=effect_plan)
else:
    comp = generate_comp(beat_map, ...)
```

### 4. Error Handling

- If `--ai` used but `anthropic` not installed:
  ```
  Error: The 'anthropic' package is required for --ai mode.
  Install with: pip install davinci-beat-lab[ai]
  ```
- If `--ai` used but `ANTHROPIC_API_KEY` not set:
  ```
  Error: ANTHROPIC_API_KEY environment variable is required for --ai mode.
  ```
- If API call fails: propagate the error with clear message

### 5. End-to-End Test

- Run `beatlab run assets/test.wav --ai -o ai_output.setting` (requires API key)
- Verify the .setting file has section-varied effects
- Run without --ai to verify no regression

### 6. Add Unit Tests

- Test generator with mock effect plan
- Test CLI raises error when --ai used without anthropic
- Test CLI raises error when --ai used without API key

---

## Verification

- [ ] `generate_comp()` accepts and uses effect_plan parameter
- [ ] Custom effects from plan create ad-hoc EffectPreset objects
- [ ] Layered presets produce multiple nodes per section
- [ ] `--ai` flag works on both `run` and `generate` commands
- [ ] `--prompt` text is passed to the director
- [ ] Clear error on missing anthropic package
- [ ] Clear error on missing API key
- [ ] Clear error on API failure
- [ ] Existing tests pass (no regressions)
- [ ] End-to-end test with real API (manual)

---

**Related Design Docs**: [AI Effect Director](../../design/local.ai-effect-director.md)
