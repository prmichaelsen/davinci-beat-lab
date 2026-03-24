# Temporal Coherence for AI Video Stylization

**Concept**: Approaches for smooth, flicker-free AI video stylization with per-section style changes
**Created**: 2026-03-24
**Status**: Design Specification

---

## Overview

Frame-by-frame SD img2img produces flicker because each frame is generated independently. This document evaluates approaches to achieve temporal coherence — smooth, consistent video output where the AI stylization flows naturally between frames while still allowing per-section style changes for beat-synced music videos.

---

## Problem Statement

Current pipeline runs SD img2img on every frame independently. Even with consistent seed, low denoising (0.3), and ControlNet, consecutive frames show visible flicker and inconsistency. This is especially noticeable on slow camera movements and static shots. The effect gets worse at higher denoising strengths, which is exactly where the most interesting stylization happens.

---

## Options

### Option A: EbSynth Hybrid (Recommended)

**How it works**: Stylize a sparse set of keyframes (every 10th-20th frame) through SD, then use EbSynth's optical-flow-based propagation to fill in the intermediate frames.

**Architecture**:
1. Select keyframes: every Nth frame + extra keyframes at beat positions
2. Run SD img2img on keyframes only (with per-section style prompts)
3. Run EbSynth to propagate style from keyframes to all frames
4. EbSynth blends naturally between differently-styled keyframes at section boundaries

**Pros**:
- **Best coherence** — optical flow preserves motion and structure, no flicker
- **10-20x faster** — only 5-10% of frames go through SD. 400 frames → ~20-40 SD renders
- **Natural section transitions** — EbSynth blends between differently-styled keyframes at boundaries
- **Beat-sync friendly** — add extra keyframes at beat hits with higher denoising for emphasis
- **Well-proven** — widely used in music video production

**Cons**:
- **EbSynth is an external tool** — not native to ComfyUI. Available as CLI binary, or via `ComfyUI-EbSynth` (less mature)
- **Artifacts at scene cuts** — EbSynth struggles with hard cuts or fast motion (optical flow breaks down)
- **Keyframe selection matters** — bad keyframe spacing creates visible "waves" of style propagation
- **Two-pass pipeline** — more complex than single-pass approaches

**Speed**: ~15-30 effective fps (only keyframes rendered through SD)

**Complexity**: Medium — requires EbSynth binary + keyframe selection logic

---

### Option B: AnimateDiff

**How it works**: Motion module plugs into SD 1.5 to generate temporally coherent batches of 16 frames at a time with overlapping windows.

**Architecture**:
1. Load AnimateDiff motion module alongside SD checkpoint
2. Process video in overlapping 16-frame batches
3. Blend overlapping regions between batches
4. Change prompts at section boundaries via context scheduling

**ComfyUI nodes**: `ComfyUI-AnimateDiff-Evolved` (Kosinkadink)
- `AnimateDiff Loader`, `AnimateDiff Sampler`, `AnimateDiff Combine`

**Pros**:
- **High coherence** — temporal attention across frames handles motion naturally
- **Fully ComfyUI-native** — no external tools needed
- **Prompt scheduling** — can change style mid-batch at specific frame numbers
- **Works with ControlNet** — structure preservation + temporal coherence

**Cons**:
- **Slow** — 16 frames takes 30-60s on A100. Effective throughput: 2-4 fps
- **SD 1.5 only** — SDXL AnimateDiff exists but is less stable
- **Batch boundary artifacts** — transitions between 16-frame batches can show seams
- **Memory hungry** — 16 frames at once requires significant VRAM (16GB+)
- **Limited to 16-frame windows** — long-range coherence degrades

**Speed**: 2-4 effective fps

**Complexity**: Medium — ComfyUI workflow changes, motion module download (~1.5GB)

---

### Option C: ControlNet Batch + Deflicker Post-Process

**How it works**: Process each frame independently through img2img with ControlNet, then apply temporal smoothing as a post-process.

**Architecture**:
1. Run SD img2img per-frame with ControlNet (canny/depth) + fixed seed + low denoising
2. Post-process with deflicker: color matching between consecutive frames
3. Optional: IP-Adapter to lock style from a reference image

**Coherence techniques**:
- Fixed seed across all frames
- Low denoising strength (0.2-0.4)
- ControlNet for structural anchoring
- IP-Adapter for style consistency
- Temporal color smoothing (ffmpeg `deflicker` or custom)

**Pros**:
- **Fastest SD generation** — 10-20 fps per frame, no batching overhead
- **Simple pipeline** — same as current architecture + post-process
- **Easy to implement** — minimal changes to existing code
- **Works with any model** — SD 1.5, SDXL, etc.

**Cons**:
- **Weakest coherence** — post-process deflicker reduces but doesn't eliminate flicker
- **Low denoising limits creativity** — strong stylization (high denoising) = more flicker
- **No true temporal awareness** — each frame is still independent

**Speed**: 10-20 fps generation + fast post-process

**Complexity**: Low — add deflicker ffmpeg filter, optionally add IP-Adapter

---

### Option D: Deforum / Frame-Chaining

**How it works**: Each frame is generated using the previous frame as the init image, creating a chain where each frame is derived from its predecessor.

**Architecture**:
1. Generate frame 1 from source frame 1
2. For frame N: blend (source_frame_N, styled_frame_N-1) as init image
3. Run SD img2img with this blended init
4. Repeat for all frames

**Pros**:
- **Good temporal continuity** — each frame is derived from the previous, natural flow
- **Dreamlike quality** — the iterative process creates evolving, organic visuals
- **Beat-sync natural** — increase denoising on beat frames for dramatic shifts
- **Simple to implement** — just change the init image source

**Cons**:
- **Style drift** — over many frames, the style can drift away from the prompt
- **Error accumulation** — artifacts compound frame over frame
- **Sequential only** — can't parallelize, each frame depends on the previous
- **Slow** — 1-3 fps (same as single-frame but strictly sequential)
- **Hard to change styles** — abrupt prompt changes cause jarring transitions

**Speed**: 1-3 fps (sequential, cannot parallelize)

**Complexity**: Low-Medium

---

### Option E: AnimateDiff + IP-Adapter (Maximum Quality)

**How it works**: Combine AnimateDiff (temporal coherence) with IP-Adapter (style locking from a reference image) for double consistency.

**Architecture**:
1. For each section, select a reference style image
2. Load AnimateDiff motion module + IP-Adapter
3. Process in 16-frame batches with IP-Adapter providing style guidance
4. Change IP-Adapter reference at section boundaries

**Pros**:
- **Highest quality** — temporal + style consistency
- **Reference-based styling** — show the model what you want, not just describe it
- **Consistent across sections** — IP-Adapter prevents style drift

**Cons**:
- **Slowest option** — 1-3 fps
- **Most VRAM** — AnimateDiff + IP-Adapter + ControlNet = 20GB+
- **Requires reference images** — user needs to provide or generate style references
- **Complex setup** — multiple models and extensions

**Speed**: 1-3 fps

**Complexity**: High

---

## Comparison Matrix

| | EbSynth Hybrid | AnimateDiff | ControlNet+Deflicker | Frame-Chaining | AnimateDiff+IP |
|---|---|---|---|---|---|
| **Coherence** | Very High | High | Low-Medium | Medium-High | Very High |
| **Speed (eff. fps)** | 15-30 | 2-4 | 10-20 | 1-3 | 1-3 |
| **Style control** | High | High | High | Medium | Very High |
| **Section transitions** | Smooth (blend) | Schedulable | Abrupt | Drifty | Smooth |
| **Beat-sync support** | Excellent | Good | Good | Natural | Good |
| **Implementation** | Medium | Medium | Low | Low-Medium | High |
| **External deps** | EbSynth binary | AnimateDiff model | None | None | IP-Adapter + AnimateDiff |
| **VRAM needed** | 8GB (keyframes only) | 16GB+ | 8GB | 8GB | 20GB+ |
| **Model support** | Any | SD 1.5 mainly | Any | Any | SD 1.5 mainly |

---

## Recommendation

**Option A (EbSynth Hybrid)** for the default pipeline:
- Best coherence-to-speed ratio
- Only 5-10% of frames need SD rendering → 10-20x faster
- Natural section transitions via optical flow blending
- Extra keyframes at beat positions for rhythmic emphasis
- Well-proven in music video production

**Fallback to Option C (ControlNet+Deflicker)** for quick previews or when EbSynth is unavailable.

**Option B (AnimateDiff)** as a future alternative for fully ComfyUI-native workflow.

---

## Implementation Plan (EbSynth Hybrid)

### Keyframe Selection
```
For each section:
  - Base keyframes: every Nth frame (N=10-15 for smooth, N=20 for fast)
  - Beat keyframes: at each beat position (with higher denoising)
  - Section boundary keyframes: first and last frame of each section
  - Minimum: at least 2 keyframes per section
```

### Pipeline
```
1. Extract frames (existing)
2. Select keyframes from frame list
3. Render keyframes through SD (existing ComfyUI pipeline)
4. Run EbSynth: for each pair of keyframes, propagate style to intermediate frames
5. Blend overlapping propagations at midpoints
6. Reassemble video (existing)
```

### EbSynth Integration
- EbSynth CLI binary (free for non-commercial, licensed for commercial)
- Input: source frame, stylized keyframe, target source frame
- Output: stylized target frame
- Runs locally (CPU-based, fast) — no GPU needed for propagation

### Beat-Sync Enhancement
- Normal keyframes: denoising 0.4-0.5
- Beat keyframes: denoising 0.6-0.7 (more dramatic transformation)
- The EbSynth propagation naturally creates a "pulse" effect as the stronger style radiates out from beat keyframes

---

## Future Considerations

- **AnimateDiff integration** as alternative backend (P2)
- **IP-Adapter** for reference-image-based styling (P2)
- **Real-time preview** via StreamDiffusion for low-latency monitoring (P3)
- **Multi-GPU rendering** for parallel keyframe generation (P2)
- **Adaptive keyframe density** based on motion complexity (more keyframes during fast motion)

---

**Status**: Design Specification
**Recommendation**: Implement EbSynth hybrid as default, ControlNet+Deflicker as fallback
**Related Documents**: [AI Effect Director](local.ai-effect-director.md), [Requirements](requirements.md)
