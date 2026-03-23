# Task 14: AI Director & Prompt

**Milestone**: [M5 - AI Effect Director](../../milestones/milestone-5-ai-effect-director.md)
**Design Reference**: [AI Effect Director](../../design/local.ai-effect-director.md)
**Estimated Time**: 3 hours
**Dependencies**: Task 12, Task 13
**Status**: Not Started

---

## Objective

Build the AI director module that constructs the system/user prompts, calls the LLM via the provider, and returns a validated effect plan. This is the "brain" that makes creative decisions.

---

## Steps

### 1. Create Director Module

`src/beatlab/ai/director.py`:

```python
def create_effect_plan(
    beat_map: dict,
    provider: LLMProvider,
    user_prompt: str | None = None,
) -> EffectPlan:
    """Send section data to LLM and get back an effect plan."""
```

### 2. Build System Prompt

`src/beatlab/ai/prompt.py`:

The system prompt includes:
1. **Role**: "You are a visual effects director for music videos..."
2. **Preset catalog**: All registered presets with name, description, node_type, parameter, base/peak values, curve type, and guidance on when each looks good
3. **Output format**: JSON schema with examples, instruction to output ONLY valid JSON
4. **Creative guidelines**:
   - Maintain visual coherence across similar sections
   - Introduce variation on repeated sections (e.g. add layered effect on 2nd chorus)
   - Match effect intensity to section energy
   - Use subtle effects for quiet passages, intense layered effects for drops
   - Custom effects allowed when no preset fits

### 3. Build User Prompt

Constructed from:
1. **Section summary table**: section index, type, duration, beat count, avg intensity, spectral features
2. **Track metadata**: tempo, total duration, total beats
3. **User creative direction**: from `--prompt` flag (if provided)

### 4. Parse and Validate Response

- Extract JSON from LLM response (handle markdown code fences)
- Parse into EffectPlan via `parse_effect_plan()`
- Validate via `validate_effect_plan()`
- Log warnings for unknown presets (substitute closest match)
- If JSON parsing fails: raise clear error (no retry for P0)

### 5. Add Tests

- Test system prompt includes preset catalog
- Test user prompt includes section data
- Test user prompt includes --prompt text when provided
- Test director with mock provider returning known JSON
- Test director handles invalid JSON gracefully

---

## Verification

- [ ] System prompt includes full preset catalog with descriptions
- [ ] User prompt includes section summaries with spectral features
- [ ] User prompt includes --prompt text when provided
- [ ] Director calls provider and parses response
- [ ] Invalid JSON raises clear error
- [ ] Unknown presets logged as warnings
- [ ] Tests pass with mock provider

---

**Next Task**: [Task 15: Generator Integration & CLI](task-15-generator-cli.md)
