# Task 23: Keyframe Selection

**Milestone**: [M8 - EbSynth Hybrid Temporal Coherence](../../milestones/milestone-8-ebsynth-coherence.md)
**Design Reference**: [Temporal Coherence](../../design/local.temporal-coherence.md)
**Estimated Time**: 2 hours
**Dependencies**: Task 19
**Status**: Not Started

---

## Objective

Build a keyframe selection module that picks which frames to render through SD based on beat positions, section boundaries, and a base interval. Each keyframe gets assigned its own denoising strength — higher for beat hits, standard for spacing frames.

---

## Steps

### 1. Create Keyframe Selector Module

```
src/beatlab/render/keyframes.py
```

### 2. Implement Selection Logic

```python
def select_keyframes(
    beat_map: dict,
    total_frames: int,
    fps: float,
    interval: int = 12,
    base_denoise: float = 0.4,
    beat_denoise: float = 0.6,
    section_denoise: float = 0.5,
) -> list[dict]:
    """Select keyframes for EbSynth hybrid rendering.

    Returns list of {frame, denoise, prompt, seed, type} dicts.
    type is "interval", "beat", or "section_boundary".
    """
```

Selection rules:
- Every `interval` frames: base keyframe (denoise=0.4)
- At each beat position: beat keyframe (denoise=0.6) — skip if within 3 frames of existing keyframe
- First frame of each section: boundary keyframe (denoise=0.5)
- Always include frame 1 and last frame
- Deduplicate: if a beat/section keyframe is near an interval keyframe, keep the one with higher denoise

### 3. Integrate with Frame Params

Each keyframe gets the same style prompt from its section (from the AI plan or default style).

### 4. Add Tests

- Test interval keyframes are evenly spaced
- Test beat keyframes are added at beat positions
- Test section boundaries are included
- Test deduplication within 3-frame window
- Test first and last frame always included
- Test keyframe count is ~5-10% of total frames

---

## Verification

- [ ] Keyframes evenly spaced at interval N
- [ ] Beat positions produce keyframes with higher denoise
- [ ] Section boundaries included
- [ ] No duplicate keyframes within 3 frames
- [ ] Frame 1 and last frame always included
- [ ] Total keyframes ~5-10% of total frames

---

**Next Task**: [Task 24: EbSynth Integration](task-24-ebsynth-integration.md)
