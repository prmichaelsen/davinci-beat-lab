# Task 24: EbSynth Integration

**Milestone**: [M8 - EbSynth Hybrid Temporal Coherence](../../milestones/milestone-8-ebsynth-coherence.md)
**Design Reference**: [Temporal Coherence](../../design/local.temporal-coherence.md)
**Estimated Time**: 3 hours
**Dependencies**: Task 23
**Status**: Not Started

---

## Objective

Integrate EbSynth CLI for style propagation between keyframes. Given styled keyframes and original source frames, propagate the style to all intermediate frames with overlap blending.

---

## Steps

### 1. Create EbSynth Module

```
src/beatlab/render/ebsynth.py
```

### 2. EbSynth Binary Management

```python
def ensure_ebsynth() -> str:
    """Find or download the EbSynth binary. Returns path to binary."""
```

- Check if `ebsynth` is on PATH
- Check `~/.beatlab/ebsynth` for cached binary
- If not found, download from GitHub releases for the current platform
- Support Linux and macOS

### 3. Implement Single Propagation

```python
def propagate_style(
    ebsynth_bin: str,
    source_frame: str,       # original frame at keyframe position
    styled_keyframe: str,    # SD-rendered keyframe
    target_source: str,      # original frame at target position
    output_path: str,        # where to write styled target
) -> None:
    """Propagate style from a keyframe to a target frame using EbSynth."""
```

EbSynth CLI: `ebsynth -style styled.png -guide source.png target.png -output output.png -weight 1.0`

### 4. Implement Full Propagation Pass

```python
def propagate_all(
    source_frames_dir: str,
    styled_keyframes: dict[int, str],  # frame_num → styled path
    output_dir: str,
    ebsynth_bin: str | None = None,
    progress_callback: Callable | None = None,
) -> None:
    """Propagate style from all keyframes to all intermediate frames."""
```

For each pair of consecutive keyframes (A, B):
- Forward propagate from A: frames A+1, A+2, ... to midpoint
- Backward propagate from B: frames B-1, B-2, ... to midpoint
- At midpoint: blend forward and backward results (cross-fade)
- Copy styled keyframes directly to output

### 5. Overlap Blending

At the midpoint between two keyframes, blend the forward-propagated and backward-propagated results:
- Simple linear blend: weight shifts from forward to backward across a 3-5 frame window
- This prevents visible seams where propagation directions meet

### 6. Add Tests

- Test EbSynth binary detection
- Test single propagation calls correct CLI command
- Test full propagation covers all frames
- Test midpoint blending produces smooth transition
- Test keyframe frames are copied directly (not propagated)

---

## Verification

- [ ] EbSynth binary found or downloaded
- [ ] Single frame propagation produces output
- [ ] Forward + backward propagation covers all frames
- [ ] Midpoint blending is smooth (no visible seam)
- [ ] Keyframe frames are exact copies of SD output
- [ ] Progress callback reports correctly
- [ ] Fallback error when EbSynth unavailable

---

**Next Task**: [Task 25: Pipeline Integration](task-25-pipeline-integration.md)
