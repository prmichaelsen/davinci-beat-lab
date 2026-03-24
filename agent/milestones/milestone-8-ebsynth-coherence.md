# Milestone 8: EbSynth Hybrid Temporal Coherence

**Goal**: Replace per-frame SD rendering with keyframe-only rendering + EbSynth optical flow propagation for smooth, flicker-free video stylization
**Duration**: 1 week
**Dependencies**: M7 - AI Video Stylization
**Status**: Not Started

---

## Overview

Current pipeline renders every frame through SD img2img independently, causing flicker. This milestone switches to a keyframe-based approach: only 5-10% of frames go through SD, then EbSynth propagates the style to all intermediate frames using optical flow. This is 10-20x faster and produces significantly smoother output.

Beat-sync is enhanced: keyframes at beat positions get higher denoising for dramatic effect, and the optical flow propagation creates a natural "pulse" radiating out from each beat keyframe.

---

## Deliverables

### 1. Smart Keyframe Selection
- Base keyframes: every Nth frame (configurable, default N=12)
- Beat keyframes: at each beat position from the beat map
- Section boundary keyframes: first frame of each new section
- Denoising assignment: base keyframes 0.4, beat keyframes 0.6
- Returns a sparse list of frames to render + their params

### 2. EbSynth Integration
- EbSynth CLI wrapper (download binary if not present)
- Propagation: given source frame, styled keyframe, target frame → styled target
- Forward and backward propagation from each keyframe
- Overlap blending at midpoints between keyframes
- Error handling for scene cuts / fast motion

### 3. Pipeline Integration
- Replace per-frame SD render with: select keyframes → render keyframes → propagate
- Remote script updated for keyframe-only rendering
- Download only keyframe results + run EbSynth locally (CPU, fast)
- Fallback to per-frame if EbSynth not available

---

## Success Criteria

- [ ] 400-frame clip renders in ~2 minutes (vs ~7 minutes per-frame)
- [ ] Output video is smooth with no visible flicker within sections
- [ ] Style changes at section boundaries transition smoothly
- [ ] Beat positions show visible style emphasis
- [ ] EbSynth auto-downloads if not present
- [ ] Fallback to per-frame rendering works when EbSynth unavailable
- [ ] All existing tests pass

---

## Tasks

1. [Task 23: Keyframe Selection](../tasks/milestone-8-ebsynth-coherence/task-23-keyframe-selection.md) - Smart keyframe picking from beat map
2. [Task 24: EbSynth Integration](../tasks/milestone-8-ebsynth-coherence/task-24-ebsynth-integration.md) - CLI wrapper, propagation, blending
3. [Task 25: Pipeline Integration](../tasks/milestone-8-ebsynth-coherence/task-25-pipeline-integration.md) - Wire into render command, remote script update

---

## Risks and Mitigation

| Risk | Impact | Probability | Mitigation Strategy |
|------|--------|-------------|---------------------|
| EbSynth artifacts on fast motion | Medium | Medium | Increase keyframe density in high-motion areas |
| EbSynth licensing for commercial use | Low | Low | Free for non-commercial; document licensing |
| EbSynth binary not available for all platforms | Medium | Low | Fallback to per-frame rendering |

---

**Blockers**: None
**Notes**: EbSynth runs on CPU — the propagation step doesn't need the GPU instance. Only keyframe rendering needs the GPU.
