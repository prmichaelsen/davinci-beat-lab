# Multi-Model Stem Pipeline

**Concept**: Combine MDX23C-InstVoc, MDX23C-DrumSep, and Demucs htdemucs_6s for optimal per-instrument isolation and effect routing
**Created**: 2026-03-26
**Status**: Proposal

---

## Overview

Instead of relying on a single stem separation model, chain three specialized models to extract the cleanest possible per-instrument signals. Each model handles what it's best at: MDX23C-InstVoc for the cleanest vocal/instrumental split (2.1x less bleed than Demucs), MDX23C-DrumSep for individual drum components (kick/snare/hh/crash — eliminates frequency-band hacking), and Demucs htdemucs_6s for melodic instruments (bass, guitar, piano, other).

---

## Problem Statement

- Single-model separation (Demucs htdemucs) produces 4 coarse stems (vocals, drums, bass, other) with significant bleed between them
- Frequency-band separation on the drum stem is a workaround — it approximates kick/snare/hh by frequency range but can't distinguish instruments that overlap in frequency
- No existing single model provides both clean vocal isolation AND granular drum decomposition AND melodic instrument separation
- Benchmarking showed MDX23C-InstVoc has 2.1x less vocal bleed than Demucs, while Demucs htdemucs_6s uniquely separates guitar and piano from "other"

---

## Solution

Run three models and merge their outputs:

```
Full Mix Audio
  │
  └──[MDX23C-InstVoc-HQ]──→ vocals, instrumental
                               │
                               ├── vocals → vocal onset detection + presence regions + confidence ratio
                               │
                               └── instrumental
                                     │
                                     ├──[MDX23C-DrumSep]──→ kick, snare, toms, hh, ride, crash
                                     │                        │
                                     │                        └── per-drum onset detection
                                     │
                                     └──[Demucs htdemucs_6s]──→ vocals*, drums*, bass, guitar, piano, other
                                                                  │
                                                                  └── bass, guitar, piano, other
                                                                      → onset detection + sustained regions
                                                                      (vocals* and drums* discarded)
```

**Key pipeline ordering**: Both DrumSep AND Demucs 6s run on the **instrumental output** from MDX23C-InstVoc, NOT on the full mix. This eliminates vocal bleed from ALL downstream stems — vocals are stripped first by the best vocal separator (2.1x less bleed than Demucs), then the remaining models only decompose instruments. Discovered during benchmarking when models running on the full mix produced stems contaminated with vocal transients (each syllable triggered false onsets).

### Why This Combination

| Model | Best At | SDR | Stems Used |
|---|---|---|---|
| MDX23C-InstVoc-HQ | Vocal isolation | 10.6 vocal, 15.8 inst | vocals (cleanest available) |
| MDX23C-DrumSep | Drum decomposition | N/A | kick, snare, toms, hh, ride, crash |
| Demucs htdemucs_6s | Melodic instruments | 9.6 vocal, 10.1 bass | bass, guitar, piano, other |

### Effect Routing

| Stem | Source Model | Primary Effect | Secondary | Notes |
|---|---|---|---|---|
| kick | DrumSep | shake_y, zoom_bounce | — | Physical low-end impact, layer zoom_bounce on strong hits |
| snare | DrumSep | shake_x | contrast_pop on strong | Crisp horizontal hit |
| hh | DrumSep | contrast_pop (subtle) | — | Light rhythmic texture, NOT flash |
| ride | DrumSep | contrast_pop | — | Accent moments |
| crash | DrumSep | contrast_pop | zoom_bounce on strong | Cymbal crashes are impact moments |
| toms | DrumSep | shake_y (lighter) | — | Fills, less intense than kick |
| bass | Demucs 6s | zoom_pulse | zoom_bounce on drops, shake_y on strong | Sustain from RMS for held bass notes |
| guitar | Demucs 6s | zoom_pulse (low), contrast_pop (mid/high) | — | Behaves like bass/synth on lows |
| piano | Demucs 6s | glow_swell | — | Soft, sustained, atmospheric |
| vocals | MDX23C-InstVoc | glow_swell | zoom_pulse on strong | Suppress aggressive effects via confidence ratio |
| other (synths) | Demucs 6s | zoom_pulse, contrast_pop | zoom_bounce on stabs | Sustain from RMS for pads/stabs |

### Vocal Bleed Confidence Ratio

Applied to all non-vocal stems: if a stem's RMS energy at onset time is <15% of the vocal stem's (MDX23C) RMS, the onset is suppressed as likely bleed. Configurable threshold (default 0.15).

---

## Implementation

### Step 1: MDX23C-InstVoc first, then DrumSep + Demucs 6s in parallel on instrumental

Both DrumSep and Demucs 6s depend on InstVoc's instrumental output. They run in parallel with each other but sequentially after InstVoc.

```python
# Step 1a: InstVoc strips vocals — must run first
instvoc_stems = run_mdx23c_instvoc(audio_path)  # vocals, instrumental

# Step 1b: DrumSep + Demucs 6s both run on the instrumental — in parallel
with ThreadPoolExecutor(max_workers=2) as executor:
    fut_drumsep = executor.submit(run_mdx23c_drumsep, instvoc_stems["instrumental"])
    fut_demucs = executor.submit(run_demucs_6s, instvoc_stems["instrumental"])

    drumsep_stems = fut_drumsep.result()   # kick, snare, toms, hh, ride, crash
    demucs_stems = fut_demucs.result()     # vocals*, drums*, bass, guitar, piano, other
```

### Step 2: Merge stems — pick best source for each instrument

```python
merged_stems = {
    "vocals": instvoc_stems["vocals"],       # MDX23C — cleanest
    "kick": drumsep_stems["kick"],           # DrumSep — actual kick isolation
    "snare": drumsep_stems["snare"],         # DrumSep
    "hh": drumsep_stems["hh"],              # DrumSep
    "ride": drumsep_stems["ride"],           # DrumSep
    "crash": drumsep_stems["crash"],         # DrumSep
    "toms": drumsep_stems["toms"],           # DrumSep
    "bass": demucs_stems["bass"],            # Demucs 6s — unique to this model
    "guitar": demucs_stems["guitar"],        # Demucs 6s — unique
    "piano": demucs_stems["piano"],          # Demucs 6s — unique
    "other": demucs_stems["other"],          # Demucs 6s — synths/pads/fx
}
```

### Step 3: Per-stem onset detection + RMS envelopes

Run directly on each merged stem — no frequency-band separation needed for drums since DrumSep already gives us kick/snare/hh individually.

For bass, guitar, piano, other: onset detection + sustained region detection + RMS envelopes.

For vocals: presence region detection + RMS envelope (for confidence ratio).

### Step 4: Claude rules generation + apply

Same rules-based approach, but Claude now sees 11 named instrument stems instead of 4 stems × 4 frequency bands. The rules become more precise:

```json
{"stem": "kick", "effect": "shake_y", "min_strength": 0.2, ...}
{"stem": "snare", "effect": "shake_x", "min_strength": 0.15, ...}
{"stem": "piano", "effect": "glow_swell", "sustain_from_rms": true, ...}
```

---

## Benefits

- **No frequency-band hack** — DrumSep gives actual kick/snare/hh isolation, not approximations
- **Cleanest vocals** — MDX23C-InstVoc at 2.1x less bleed than Demucs, enabling reliable confidence ratio
- **Guitar and piano as separate stems** — Demucs 6s uniquely provides these, enabling instrument-specific effects (piano → glow, guitar → zoom)
- **Parallel execution** — all three models run simultaneously, total time ≈ slowest model
- **Precise rules** — Claude generates rules per named instrument instead of per stem/band, more intuitive and auditable

---

## Render Comparison (9-11m test clip)

| Version | File | Events | Pipeline | Verdict |
|---|---|---|---|---|
| final (fallback) | `effects_ai_9_11m_final.mp4` | ~1800 | Demucs 4-stem + freq bands + onset-based rules | Best pre-multi-model result |
| new_pipeline_v2 | `effects_ai_9_11m_new_pipeline_v2.mp4` | 2059 | Multi-model (InstVoc→DrumSep+Demucs6s) + onset-based rules + vocal bleed + percussion sustain suppression | "Clean as fuck" — best overall so far |
| drumsep (full mix) | `effects_ai_9_11m_drumsep.mp4` | 1911 | DrumSep on full mix (not instrumental) | Worse — vocal artifacts in drum stems |
| stats-only | `effects_ai_9_11m_stats.mp4` | 3806 | Multi-model + stats-only Claude prompt | "More precise but less dynamic" |
| hybrid v1 | `effects_ai_9_11m_hybrid.mp4` | 3067 | Multi-model + hybrid stats+curves, 21 rules | More dynamic than stats-only, zoom intensity lower than v2 |
| hybrid v4 | `effects_ai_9_11m_hybrid_v4.mp4` | 1258 | Multi-model + hybrid stats+curves, 12 rules (capped) | **Sweet spot** — dynamic, punchy, focused |
| chunked (full track) | `google_muxed_effects_final.mp4` | 8566 | 27 chunks, per-section rules | Very sparse, underwhelming — chunking kills intensity |

**Current best**: `hybrid_v4` — multi-model stems from InstVoc instrumental, hybrid stats+curve-simplified onsets, 12 rules (capped), vocal bleed suppression, percussion sustained-region suppression, no hard_cut, flash→contrast_pop.

## Fallback

If the multi-model pipeline does not improve results, the proven fallback is:

**`effects_ai_9_11m_final.mp4` configuration** — Demucs htdemucs 4-stem + frequency-band DSP + Claude rules + vocal bleed confidence ratio (0.15) + flash→contrast_pop + no hard_cut. This produced the best results prior to the multi-model pipeline work.

---

## Trade-offs

- **Three models = 3x disk space for model weights** — ~2-3GB total. Acceptable for cloud desktops, cacheable.
- **CPU processing time** — MDX23C models are slow on CPU (~35 min each for 2 min). Mitigated by GPU (DigitalOcean) or parallel execution.
- **Demucs vocals/drums discarded** — We run Demucs 6s but only use bass/guitar/piano/other. ~40% of its output is thrown away. Acceptable since the alternative (not having guitar/piano separation) is worse.
- **Model availability** — Depends on audio-separator package and model checkpoint downloads. Models cached after first download.
- **DrumSep on full mix produces vocal artifacts** — Confirmed: running DrumSep directly on the full mix contaminates drum stems with vocal transients. MUST run on InstVoc instrumental output instead.

---

## Dependencies

- audio-separator Python package (wraps all models with unified API)
- MDX23C-8KFFT-InstVoc_HQ.ckpt (vocal/instrumental separation)
- MDX23C-DrumSep-aufr33-jarredou.ckpt (drum decomposition)
- Demucs htdemucs_6s (melodic instrument separation)
- Existing: audio_intelligence.py (Layer 1 DSP, Layer 2 descriptions, Layer 3 Claude rules)

---

## Migration Path

1. **Phase 1**: Add MDX23C-InstVoc as vocal separator, keep Demucs for everything else. Swap vocal confidence ratio to use MDX23C vocals.
2. **Phase 2**: Add MDX23C-DrumSep, remove frequency-band hack for drums. Update rules prompt with named drum stems.
3. **Phase 3**: Switch Demucs to htdemucs_6s, expose guitar and piano as separate stems. Update effect routing.
4. **Phase 4**: Remove legacy 4-stem Demucs code path. Frequency-band analysis becomes optional (for the "other" stem only, if needed).

---

## Key Design Decisions

### Model Selection

| Decision | Choice | Rationale |
|---|---|---|
| Vocal separation | MDX23C-InstVoc-HQ | 2.1x less bleed than Demucs (benchmark verified) |
| Drum separation | MDX23C-DrumSep | 6 individual drum stems (kick/snare/toms/hh/ride/crash) eliminates frequency-band hack |
| Melodic separation | Demucs htdemucs_6s | Only model that separates guitar and piano from "other" |
| Why 3 models | Best-of-each approach | No single model excels at all three tasks |

### Effect Routing

| Decision | Choice | Rationale |
|---|---|---|
| Piano → glow_swell only | Soft atmospheric effect | Piano is sustained, melodic — aggressive effects would clash |
| Guitar → zoom_pulse/contrast_pop | Treat like bass/synth | Guitar fills similar sonic role to synth leads |
| hh → contrast_pop, NOT flash | Subtle texture | Flash on every hi-hat is blinding; contrast_pop is visible but not aggressive |
| kick → shake_y + zoom_bounce | Maximum physical impact | Kick is the primary rhythmic driver in EDM |
| Vocal bleed ratio | 0.15 threshold | Verified: drum bleed during vocals is typically <10% of vocal energy, real drum hits are >30% |

---

## Future Considerations

- **Ensemble separation** — audio-separator supports running multiple models and ensembling their outputs for even cleaner stems
- **Per-section model selection** — use different models for different parts of the track (e.g., DrumSep only during drum-heavy sections)
- **Custom model fine-tuning** — train separation models on the user's specific genre for better results
- **Real-time separation** — for live preview in the SceneCraft GUI, use smaller/faster models
- **Suno API integration** — if the source audio was generated by Suno, use their 12-stem API instead of local separation

---

**Status**: Proposal
**Recommendation**: Implement Phase 1 (MDX23C-InstVoc for vocals) immediately since it's the highest-impact change. Phase 2 (DrumSep) next. Phase 3 (Demucs 6s) last.
**Related Documents**: [local.multi-layer-audio-intelligence.md](local.multi-layer-audio-intelligence.md), [local.platform-architecture.md](local.platform-architecture.md)
