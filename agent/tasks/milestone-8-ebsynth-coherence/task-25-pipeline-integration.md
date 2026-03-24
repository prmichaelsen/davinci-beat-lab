# Task 25: Pipeline Integration

**Milestone**: [M8 - EbSynth Hybrid Temporal Coherence](../../milestones/milestone-8-ebsynth-coherence.md)
**Design Reference**: [Temporal Coherence](../../design/local.temporal-coherence.md)
**Estimated Time**: 3 hours
**Dependencies**: Task 24
**Status**: Not Started

---

## Objective

Wire the EbSynth hybrid pipeline into the `beatlab render` command. Replace per-frame rendering with: select keyframes → render keyframes on GPU → download → propagate locally with EbSynth → reassemble.

---

## Steps

### 1. Update Remote Script

The remote render script only processes keyframes now (much fewer frames). Update `remote_script.py` to accept a keyframe list instead of all frame params.

### 2. Update CLI Render Pipeline

New cloud render flow:
```
1. Select keyframes from beat map (local, instant)
2. Upload only source keyframes to remote (much fewer files)
3. Render keyframes on GPU via remote script
4. Download styled keyframes (small transfer)
5. Run EbSynth propagation locally (CPU, fast)
6. Reassemble video from propagated frames
```

### 3. Add CLI Flags

- `--keyframe-interval N` — base keyframe interval (default: 12)
- `--no-ebsynth` — force per-frame rendering (fallback)

### 4. Update Work Directory

Cache keyframe selection and styled keyframes:
```
.beatlab_work/<video>/
├── keyframes.json          # selected keyframe list
├── styled_keyframes/       # SD-rendered keyframes only
├── propagated/             # EbSynth output (all frames)
```

### 5. Fallback Handling

- If EbSynth not available: warn and fall back to per-frame rendering
- If `--no-ebsynth` flag: skip EbSynth, use per-frame
- If EbSynth fails on specific frames: copy nearest keyframe as fallback

### 6. Optimize Upload/Download

- Upload only keyframe source frames (not all 6656 frames)
- Download only styled keyframes (not all frames)
- For 400 frames with interval=12: upload ~35 frames, download ~35 frames

### 7. Add Tests

- Test full pipeline: keyframes → render → propagate → reassemble
- Test fallback to per-frame when EbSynth unavailable
- Test work dir caching for keyframes and propagated frames
- Test --keyframe-interval flag changes keyframe count

---

## Verification

- [ ] Render command uses keyframe pipeline by default
- [ ] Only keyframes uploaded to remote (not all frames)
- [ ] Only styled keyframes downloaded
- [ ] EbSynth propagation runs locally (no GPU needed)
- [ ] Propagated frames are smooth and flicker-free
- [ ] Work dir caches keyframes and propagated frames
- [ ] --no-ebsynth falls back to per-frame
- [ ] --keyframe-interval adjusts density
- [ ] 10-20x speedup over per-frame rendering

---

**Related Design Docs**: [Temporal Coherence](../../design/local.temporal-coherence.md)
