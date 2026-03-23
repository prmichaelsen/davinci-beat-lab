# Task 12: Spectral Features

**Milestone**: [M5 - AI Effect Director](../../milestones/milestone-5-ai-effect-director.md)
**Design Reference**: [AI Effect Director](../../design/local.ai-effect-director.md)
**Estimated Time**: 2 hours
**Dependencies**: Task 8 (section detection)
**Status**: Not Started

---

## Objective

Extend `detect_sections()` in `analyzer.py` to compute per-section spectral summaries (centroid, RMS energy, rolloff, contrast) and include them in the section data. These features give the LLM richer context about each section's sonic character.

---

## Steps

### 1. Add Spectral Feature Computation to detect_sections()

For each section segment, compute:
- `spectral_centroid` — average brightness/frequency (librosa.feature.spectral_centroid)
- `rms_energy` — average loudness (librosa.feature.rms)
- `spectral_rolloff` — frequency below which 85% of energy lies (librosa.feature.spectral_rolloff)
- `spectral_contrast` — difference between peaks and valleys in spectrum (librosa.feature.spectral_contrast)

Normalize all values to 0.0-1.0 range across the track for comparability.

### 2. Include in Section Output

Each section dict gets additional fields:
```python
{
    "start_time": 0.0,
    "end_time": 30.5,
    "type": "low_energy",
    "label": "verse",
    "spectral": {
        "centroid": 0.35,
        "rms_energy": 0.22,
        "rolloff": 0.41,
        "contrast": 0.28,
    }
}
```

### 3. Update Beat Map Schema

- Include spectral data in the sections array of the beat map JSON
- Bump version to 1.2 when spectral data is present

### 4. Add Tests

- Test that spectral features are computed and normalized
- Test that section output includes spectral dict
- Test beat map includes spectral data when sections enabled

---

## Verification

- [ ] `detect_sections()` returns spectral features per section
- [ ] All spectral values are normalized 0.0-1.0
- [ ] Beat map JSON includes spectral data in sections
- [ ] Existing tests still pass
- [ ] New tests for spectral feature computation

---

**Next Task**: [Task 13: LLM Provider & Effect Plan](task-13-llm-provider.md)
