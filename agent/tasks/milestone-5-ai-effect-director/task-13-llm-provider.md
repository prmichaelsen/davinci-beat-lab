# Task 13: LLM Provider & Effect Plan

**Milestone**: [M5 - AI Effect Director](../../milestones/milestone-5-ai-effect-director.md)
**Design Reference**: [AI Effect Director](../../design/local.ai-effect-director.md)
**Estimated Time**: 3 hours
**Dependencies**: None
**Status**: Not Started

---

## Objective

Create the LLM provider abstraction (`LLMProvider` ABC + `AnthropicProvider`) and the effect plan schema with JSON validation. The `anthropic` SDK is an optional dependency.

---

## Steps

### 1. Create ai/ Module Structure

```
src/beatlab/ai/
├── __init__.py
├── provider.py     # LLMProvider ABC + AnthropicProvider
└── plan.py         # EffectPlan schema + validation
```

### 2. Implement Provider Abstraction

`provider.py`:
```python
class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str: ...

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-20250514"):
        ...
```

- `AnthropicProvider` uses the `anthropic` Python SDK
- API key from constructor arg or `ANTHROPIC_API_KEY` env var
- Clear error if `anthropic` package not installed: "Install with: pip install davinci-beat-lab[ai]"
- Clear error if API key missing

### 3. Add Optional Dependency

In `pyproject.toml`:
```toml
[project.optional-dependencies]
ai = ["anthropic>=0.30.0"]
```

### 4. Implement Effect Plan Schema

`plan.py`:
```python
@dataclass
class SectionPlan:
    section_index: int
    presets: list[str]
    custom_effects: list[dict]
    intensity_curve: str = "linear"
    attack_frames: int | None = None
    release_frames: int | None = None

@dataclass
class EffectPlan:
    sections: list[SectionPlan]

def parse_effect_plan(json_str: str) -> EffectPlan: ...
def validate_effect_plan(plan: EffectPlan, available_presets: list[str]) -> list[str]: ...
```

- `parse_effect_plan` parses JSON string from LLM response, extracts the JSON block
- `validate_effect_plan` checks preset names exist, custom effects have required fields
- Returns list of warnings (invalid presets are warned, not fatal)

### 5. Add Tests

- Test provider instantiation (mock anthropic)
- Test effect plan parsing from valid JSON
- Test effect plan parsing from JSON wrapped in markdown code fence
- Test validation catches unknown presets
- Test validation accepts custom effects

---

## Verification

- [ ] `LLMProvider` ABC defined with `complete()` method
- [ ] `AnthropicProvider` works with anthropic SDK
- [ ] Clear error when anthropic not installed
- [ ] Clear error when API key missing
- [ ] `parse_effect_plan()` handles valid JSON
- [ ] `parse_effect_plan()` handles JSON inside markdown code fences
- [ ] `validate_effect_plan()` warns on unknown presets
- [ ] Optional dependency in pyproject.toml

---

**Next Task**: [Task 14: AI Director & Prompt](task-14-ai-director.md)
