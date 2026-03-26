"""Multi-layer audio intelligence — DSP extraction, Gemini listening, Claude creative direction."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, sosfilt


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


# ─── Layer 1: DSP Signal Extraction ────────────────────────────────────────

FREQUENCY_BANDS = {
    "low": (20, 200),      # kicks, sub-bass
    "mid": (200, 2000),    # snares, vocals, melodic content
    "high": (2000, 10000), # hi-hats, cymbals, sibilance
}


def _bandpass_filter(y: np.ndarray, sr: int, low_hz: int, high_hz: int, order: int = 4) -> np.ndarray:
    """Apply a Butterworth bandpass filter."""
    nyquist = sr / 2
    low = max(low_hz / nyquist, 0.001)
    high = min(high_hz / nyquist, 0.999)
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfilt(sos, y)


def _detect_onsets(y: np.ndarray, sr: int, hop_length: int = 512) -> list[dict]:
    """Detect onsets and return list of {time, strength}."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, onset_envelope=onset_env, backtrack=True,
    )
    if len(onset_frames) == 0:
        return []
    max_idx = len(onset_env) - 1
    onset_frames = np.clip(onset_frames, 0, max_idx)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    strengths = onset_env[onset_frames]
    if strengths.max() > 0:
        strengths = strengths / strengths.max()
    return [{"time": float(t), "strength": float(s)} for t, s in zip(onset_times, strengths)]


def _compute_rms_envelope(y: np.ndarray, sr: int, hop_length: int = 512, window_sec: float = 0.05) -> list[dict]:
    """Compute RMS energy envelope over time. Returns [{time, energy}]."""
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if len(rms) == 0:
        return []
    if rms.max() > 0:
        rms = rms / rms.max()
    frames_per_sec = sr / hop_length
    # Downsample to ~20 points per second for manageable output
    step = max(1, int(frames_per_sec / 20))
    return [
        {"time": float(i / frames_per_sec), "energy": float(rms[i])}
        for i in range(0, len(rms), step)
    ]


def _compute_spectral_features(y: np.ndarray, sr: int, hop_length: int = 512) -> dict:
    """Compute summary spectral features for a signal."""
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    return {
        "centroid_mean": float(np.mean(centroid)),
        "centroid_std": float(np.std(centroid)),
        "flux_mean": float(np.mean(flux)),
        "flux_std": float(np.std(flux)),
        "rolloff_mean": float(np.mean(rolloff)),
    }


def _detect_sustained_regions(y: np.ndarray, sr: int, hop_length: int = 512,
                               min_duration: float = 0.3, threshold_ratio: float = 0.2) -> list[dict]:
    """Detect sustained energy regions (held notes, pad swells, sustained stabs).

    Returns [{start_time, end_time, peak_energy, duration}].
    """
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if len(rms) == 0:
        return []

    threshold = float(np.max(rms)) * threshold_ratio
    frames_per_sec = sr / hop_length

    regions = []
    in_region = False
    start_idx = 0
    peak_val = 0.0

    for i, val in enumerate(rms):
        if val >= threshold and not in_region:
            in_region = True
            start_idx = i
            peak_val = float(val)
        elif val >= threshold and in_region:
            peak_val = max(peak_val, float(val))
        elif val < threshold and in_region:
            in_region = False
            start_t = start_idx / frames_per_sec
            end_t = i / frames_per_sec
            dur = end_t - start_t
            if dur >= min_duration:
                regions.append({
                    "start_time": float(start_t),
                    "end_time": float(end_t),
                    "duration": float(dur),
                    "peak_energy": peak_val / float(np.max(rms)) if np.max(rms) > 0 else 0,
                })

    if in_region:
        end_t = len(rms) / frames_per_sec
        start_t = start_idx / frames_per_sec
        dur = end_t - start_t
        if dur >= min_duration:
            regions.append({
                "start_time": float(start_t),
                "end_time": float(end_t),
                "duration": float(dur),
                "peak_energy": peak_val / float(np.max(rms)) if np.max(rms) > 0 else 0,
            })

    return regions


def extract_layer1(stem_paths: dict[str, str], sr: int = 22050) -> dict:
    """Run full Layer 1 DSP extraction on all stems.

    Args:
        stem_paths: {"drums": path, "bass": path, "vocals": path, "other": path}
        sr: Sample rate for analysis.

    Returns:
        Nested dict: {stem_name: {band_name: {onsets, rms_envelope, sustained_regions, spectral}}}
    """
    results = {}

    for stem_name, path in stem_paths.items():
        if not Path(path).exists():
            _log(f"  Warning: stem {stem_name} not found at {path}, skipping")
            continue

        _log(f"  Layer 1: analyzing {stem_name}...")
        y, sr_out = librosa.load(path, sr=sr, mono=True)

        stem_data = {}

        # Full-band analysis
        _log(f"    full band...")
        stem_data["full"] = {
            "onsets": _detect_onsets(y, sr_out),
            "rms_envelope": _compute_rms_envelope(y, sr_out),
            "sustained_regions": _detect_sustained_regions(y, sr_out),
            "spectral": _compute_spectral_features(y, sr_out),
        }

        # Per-frequency-band analysis
        for band_name, (low_hz, high_hz) in FREQUENCY_BANDS.items():
            _log(f"    {band_name} band ({low_hz}-{high_hz}Hz)...")
            y_band = _bandpass_filter(y, sr_out, low_hz, high_hz)
            stem_data[band_name] = {
                "onsets": _detect_onsets(y_band, sr_out),
                "rms_envelope": _compute_rms_envelope(y_band, sr_out),
                "sustained_regions": _detect_sustained_regions(y_band, sr_out),
                "spectral": _compute_spectral_features(y_band, sr_out),
            }

        # Summary stats
        for band_name, band_data in stem_data.items():
            n_onsets = len(band_data["onsets"])
            n_sustained = len(band_data["sustained_regions"])
            _log(f"    {stem_name}/{band_name}: {n_onsets} onsets, {n_sustained} sustained regions")

        results[stem_name] = stem_data

    return results


# ─── Layer 2: Gemini Audio Listening ────────────────────────────────────────

def _chunk_audio_for_gemini(audio_path: str, chunk_duration: float = 30.0) -> list[dict]:
    """Split audio into chunks for Gemini analysis.

    Returns list of {start_time, end_time, path} dicts.
    """
    import subprocess
    import tempfile

    duration_result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", audio_path],
        capture_output=True, text=True,
    )
    total_duration = float(duration_result.stdout.strip())

    chunks = []
    chunk_dir = Path(audio_path).parent / "gemini_chunks"
    chunk_dir.mkdir(exist_ok=True)

    start = 0.0
    idx = 0
    while start < total_duration:
        end = min(start + chunk_duration, total_duration)
        chunk_path = str(chunk_dir / f"chunk_{idx:03d}.mp3")

        if not Path(chunk_path).exists():
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-ss", str(start), "-t", str(end - start),
                 "-acodec", "libmp3lame", "-b:a", "128k", chunk_path],
                capture_output=True, check=True,
            )

        chunks.append({"start_time": start, "end_time": end, "path": chunk_path, "index": idx})
        start = end
        idx += 1

    return chunks


def _gemini_describe_chunk(chunk_path: str, start_time: float, end_time: float,
                            stem_name: str = "full_mix") -> str:
    """Send an audio chunk to Gemini and get a musical description."""
    from google import genai
    from google.genai import types
    import os
    import time

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    with open(chunk_path, "rb") as f:
        audio_bytes = f.read()

    chunk_offset = start_time
    prompt = f"""You are a professional music producer with perfect pitch and rhythm. You are analyzing an audio stem for the purpose of syncing precise visual effects to every musical event.

This is the **{stem_name}** stem from {start_time:.0f}s to {end_time:.0f}s in the track. Timestamps in your response should be ABSOLUTE (relative to the full track, starting at {start_time:.0f}s), NOT relative to this chunk.

Your primary job is to produce a DETAILED TIMESTAMP LOG of every audible musical event. We need sub-second precision for beat-syncing visual effects to this audio.

## Required Output

### 1. EVENT LOG (most important — be exhaustive)

List EVERY distinct audible event with its timestamp. Use the format:
  [{start_time:.0f}s + offset] event_type: description

Event types: kick, snare, hi-hat, cymbal_crash, tom, percussion_other, bass_note, bass_drop, bass_sustain_start, bass_sustain_end, synth_stab, synth_pad_start, synth_pad_end, synth_lead, arpeggio, riser_start, riser_peak, drop, breakdown_start, buildup_start, vocal_start, vocal_end, vocal_chop, fx_sweep, fx_impact, silence_start, silence_end

For repeating patterns (e.g., hi-hats every 8th note), you may describe the pattern AND list the first few timestamps, then note the interval:
  "hi-hat pattern: every ~0.23s from {start_time:.0f}s+2.1 to {start_time:.0f}s+15.3"

For sustained sounds, give BOTH start and end timestamps and approximate duration:
  "{start_time:.0f}s+5.2 synth_pad_start: warm pad enters, sustained ~3.5s"
  "{start_time:.0f}s+8.7 synth_pad_end: pad fades out"

### 2. RHYTHM ANALYSIS
- BPM estimate for this section
- Time signature (4/4, 3/4, 6/8, etc.)
- Kick pattern description (four-on-the-floor, syncopated, etc.)
- Snare pattern description
- Hi-hat pattern description

### 3. ENERGY PROFILE
Rate intensity 1-10 at these checkpoints: start, 25%, 50%, 75%, end.
Note any sudden energy changes with timestamps.

### 4. SUSTAINED SOUNDS
List every sustained sound with start time, end time, and character:
- Pads, drones, held chords
- Reverb tails on impacts
- Risers and sweeps
- Sustained bass notes

### 5. KEY MOMENTS
The 3-5 most visually impactful moments in this chunk — the ones that should trigger the strongest visual effects. Give precise timestamps and describe why they're impactful.

Be EXHAUSTIVE with timestamps. More events = better visual sync. We will cross-reference your timestamps against DSP onset detection data, so precision matters."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(parts=[
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/mpeg"),
                types.Part.from_text(text=prompt),
            ]),
        ],
    )

    return response.text


def extract_layer2(audio_path: str, chunk_duration: float = 30.0,
                    descriptions_md: str | None = None) -> list[dict]:
    """Run Layer 2 — Gemini audio listening OR load from cached descriptions.md.

    Args:
        audio_path: Path to audio file (WAV or MP3).
        chunk_duration: Duration of each chunk sent to Gemini.
        descriptions_md: Path to existing descriptions.md file (fallback when Gemini unavailable).

    Returns:
        List of {start_time, end_time, description} dicts.
    """
    # If descriptions.md provided, parse it instead of calling Gemini
    if descriptions_md and Path(descriptions_md).exists():
        _log(f"  Layer 2: loading cached descriptions from {Path(descriptions_md).name}")
        return _parse_descriptions_md(descriptions_md)

    _log("  Layer 2: Gemini audio listening...")
    chunks = _chunk_audio_for_gemini(audio_path, chunk_duration)
    _log(f"    {len(chunks)} chunks ({chunk_duration:.0f}s each)")

    results = []
    for chunk in chunks:
        _log(f"    Chunk {chunk['index']}: {chunk['start_time']:.0f}s - {chunk['end_time']:.0f}s...")
        try:
            description = _gemini_describe_chunk(
                chunk["path"], chunk["start_time"], chunk["end_time"],
            )
            results.append({
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "description": description,
            })
        except Exception as e:
            _log(f"    Chunk {chunk['index']} failed: {e}")
            results.append({
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "description": f"(analysis failed: {e})",
            })

    return results


def _parse_descriptions_md(path: str) -> list[dict]:
    """Parse an existing descriptions.md into Layer 2 format.

    The file has sections like:
        ## Section N (type, energy)
        **Time**: start_s - end_s
        ... description text ...
    """
    import re

    with open(path) as f:
        content = f.read()

    results = []
    # Split on section headers
    sections = re.split(r"(?=^## Section \d+)", content, flags=re.MULTILINE)

    for section in sections:
        if not section.strip().startswith("## Section"):
            continue

        # Extract time range
        time_match = re.search(r"\*\*Time\*\*:\s*([\d.]+)s\s*-\s*([\d.]+)s", section)
        if not time_match:
            continue

        start_time = float(time_match.group(1))
        end_time = float(time_match.group(2))

        # Everything after the Time line is the description
        time_line_end = section.index(time_match.group(0)) + len(time_match.group(0))
        description = section[time_line_end:].strip()

        results.append({
            "start_time": start_time,
            "end_time": end_time,
            "description": description,
        })

    _log(f"    Parsed {len(results)} section descriptions")
    return results


# ─── Layer 3: Claude Creative Direction ─────────────────────────────────────

def _format_layer1_for_claude(layer1_data: dict, time_offset: float = 0.0,
                                time_limit: float | None = None) -> str:
    """Format Layer 1 DSP data into a compact text summary for Claude.

    Filters to the given time window and summarizes rather than dumping raw arrays.
    """
    lines = []

    for stem_name, bands in layer1_data.items():
        lines.append(f"\n## Stem: {stem_name}")

        for band_name, data in bands.items():
            # Filter onsets to time window
            onsets = data.get("onsets", [])
            if time_limit is not None:
                onsets = [o for o in onsets if time_offset <= o["time"] < time_offset + time_limit]

            # Top onsets by strength (max 80 per band — Claude needs dense data for aggressive coverage)
            top_onsets = sorted(onsets, key=lambda o: o["strength"], reverse=True)[:80]
            top_onsets.sort(key=lambda o: o["time"])

            # Sustained regions in window
            sustained = data.get("sustained_regions", [])
            if time_limit is not None:
                sustained = [s for s in sustained
                             if s["start_time"] < time_offset + time_limit and s["end_time"] > time_offset]

            lines.append(f"\n### {stem_name}/{band_name}")
            lines.append(f"Total onsets: {len(onsets)}")

            if top_onsets:
                lines.append(f"Top {len(top_onsets)} onsets (time → strength):")
                for o in top_onsets:
                    lines.append(f"  {o['time']:.3f}s → {o['strength']:.3f}")

            if sustained:
                lines.append(f"Sustained regions ({len(sustained)}):")
                for s in sustained:
                    lines.append(f"  {s['start_time']:.2f}s - {s['end_time']:.2f}s ({s['duration']:.2f}s, peak={s['peak_energy']:.2f})")

            spectral = data.get("spectral", {})
            if spectral:
                lines.append(f"Spectral: centroid={spectral.get('centroid_mean', 0):.0f}Hz, "
                             f"flux={spectral.get('flux_mean', 0):.2f}")

    return "\n".join(lines)


def _format_layer2_for_claude(layer2_data: list[dict]) -> str:
    """Format Layer 2 Gemini descriptions for Claude."""
    lines = ["\n## Gemini Audio Descriptions\n"]
    for chunk in layer2_data:
        lines.append(f"### {chunk['start_time']:.0f}s - {chunk['end_time']:.0f}s")
        lines.append(chunk["description"])
        lines.append("")
    return "\n".join(lines)


DEFAULT_SENSITIVITY = {
    "zoom_pulse": 0.5,
    "zoom_bounce": 0.5,
    "shake_x": 0.5,
    "shake_y": 0.5,
    "flash": 0.5,
    "hard_cut": 0.5,
    "contrast_pop": 0.5,
    "glow_swell": 0.5,
}


def extract_layer3(
    layer1_data: dict,
    layer2_data: list[dict],
    time_offset: float = 0.0,
    time_limit: float | None = None,
    creative_direction: str | None = None,
    fps: float = 24.0,
    sensitivity: dict[str, float] | None = None,
) -> list[dict]:
    """Run Layer 3 Claude creative direction.

    Takes Layer 1 DSP data + Layer 2 Gemini descriptions, asks Claude to produce
    frame-accurate effect assignments.

    Returns list of effect events:
        [{time, duration, effect, intensity, stem_source, rationale}]
    """
    import anthropic
    import os

    _log("  Layer 3: Claude creative direction...")

    dsp_summary = _format_layer1_for_claude(layer1_data, time_offset, time_limit)
    gemini_summary = _format_layer2_for_claude(layer2_data)

    system_prompt = """You are a visual effects director for music videos. You receive detailed audio analysis data from two sources:

1. **DSP Data (Layer 1)**: Precise onset timestamps, energy envelopes, sustained regions, and spectral features — extracted per stem (drums, bass, vocals, other) per frequency band (low/mid/high). These timestamps are exact to the millisecond.

2. **Gemini Descriptions (Layer 2)**: Musical context from an AI that listened to the actual audio — instrument identification, rhythm patterns, energy dynamics, sustained sounds, key moments.

Your job: synthesize both sources to produce **frame-accurate effect assignments**. Each effect event specifies exactly when an effect should fire, how long it should last, what type of effect, and how intense.

## Available Effects

- **zoom_pulse**: Gentle zoom in/out. Good for melodic hits, subtle rhythmic elements.
- **zoom_bounce**: Aggressive zoom. Good for bass drops, heavy kicks, impacts.
- **shake_x**: Horizontal camera shake. Good for snare hits, percussive impacts.
- **shake_y**: Vertical camera shake. Good for kick drums, sub-bass hits.
- **flash**: Brightness flash. Good for hi-hats, cymbals, crisp transients.
- **hard_cut**: Extreme brightness spike. Good for massive drops, climactic moments.
- **contrast_pop**: Contrast boost. Good for synth stabs, melodic accents.
- **glow_swell**: Soft glow bloom. Good for sustained pads, ambient textures, vocal sections.

## Effect Properties

Each effect event must have:
- **time**: Exact timestamp in seconds (use DSP onset times for precision)
- **duration**: How long the effect lasts. Transients: 0.1-0.3s. Sustained: match the RMS envelope duration.
- **effect**: One of the effect names above
- **intensity**: 0.0-1.0 scale
- **sustain**: Optional. If the sound is sustained (held synth stab, pad swell), set this to the sustain duration. The effect will hold at intensity during sustain before releasing.
- **stem_source**: Which stem/band this was derived from (e.g., "drums/low", "other/mid")
- **rationale**: Brief explanation of why this effect at this moment

## Rules

1. **Use DSP timestamps** for exact timing — never invent timestamps that aren't in the DSP data.
2. **Use Gemini descriptions** to understand WHAT each onset is — a kick, snare, synth stab, etc.
3. **Sustained sounds**: When Gemini describes a sustained sound AND the DSP shows a sustained_region, create an effect with matching sustain duration. Sustained synth stabs should get a sustained glow or zoom that HOLDS for the full duration of the sustain.
4. **Suppress during vocals**: When vocals are present (check vocals stem), reduce intensity of aggressive effects by 30% — but still apply effects, just softer.
5. **Layer effects**: Big moments can have multiple simultaneous effects (e.g., bass drop = zoom_bounce + shake_y + hard_cut). Layer generously — real music videos have multiple visual events per beat.
6. **BE AGGRESSIVE WITH COVERAGE**: Assign effects to MOST onsets, not just the strongest ones. A music video should have visible effects on nearly every beat. For a 120-second clip at 130 BPM, you should produce 100-300+ effect events. Every kick should get at least a subtle shake. Every snare should get at least a flash. Every bass note should get at least a zoom pulse. Only truly inaudible ghost notes (strength < 0.02) should be skipped.
7. **Match the music's energy arc**: Buildups = gradually increasing intensity. Drops = hit hard with max layering. Breakdowns = reduce but don't stop — keep subtle glow/zoom pulsing.
8. **Vary intensity, not presence**: Instead of skipping quiet onsets, assign them lower intensity (0.2-0.4). The video should always be breathing with the music. Silence = no effects. Sound = some effect, even if subtle.

## Output Format

Respond with ONLY a JSON array of effect events. No markdown, no explanation outside the JSON.

```json
[
  {
    "time": 1.234,
    "duration": 0.2,
    "effect": "shake_y",
    "intensity": 0.8,
    "sustain": null,
    "stem_source": "drums/low",
    "rationale": "strong kick hit"
  },
  {
    "time": 5.678,
    "duration": 1.5,
    "effect": "glow_swell",
    "intensity": 0.6,
    "sustain": 1.2,
    "stem_source": "other/mid",
    "rationale": "sustained synth pad, Gemini describes 'warm chord held for ~2 seconds'"
  }
]
```"""

    # Merge sensitivity defaults with overrides
    sens = dict(DEFAULT_SENSITIVITY)
    if sensitivity:
        sens.update(sensitivity)

    sensitivity_text = "\n".join(
        f"- **{effect}**: {level:.1f}" + (
            " (very aggressive — trigger on nearly every relevant onset, high intensity)" if level >= 0.8
            else " (aggressive — trigger frequently, moderate-high intensity)" if level >= 0.6
            else " (moderate — trigger on clear, distinct onsets)" if level >= 0.4
            else " (conservative — only trigger on strong, obvious moments)" if level >= 0.2
            else " (minimal — rarely trigger, only on the most dramatic moments)"
        )
        for effect, level in sens.items()
    )

    user_prompt = f"""# Audio Analysis Data

{dsp_summary}

{gemini_summary}

## Effect Sensitivity Settings (0.0 = never trigger, 1.0 = trigger on everything)

These sensitivity levels are directives from the user controlling how aggressively each effect should be used. Higher = more frequent triggers, lower thresholds, more events. Lower = fewer triggers, only on the most prominent moments.

{sensitivity_text}

"""
    if creative_direction:
        user_prompt += f"\n## Creative Direction\n{creative_direction}\n"

    user_prompt += "\nGenerate the effect events JSON for this audio segment."

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    _log("    Sending to Claude (streaming)...")
    text = ""
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=32768,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        for chunk in stream.text_stream:
            text += chunk

    text = text.strip()

    # Parse JSON — handle markdown code blocks
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    if text.endswith("```"):
        text = text[:-3]

    try:
        events = json.loads(text.strip())
    except json.JSONDecodeError as e:
        _log(f"    Failed to parse Claude response: {e}")
        _log(f"    Response: {text[:500]}")
        events = []

    _log(f"    Claude produced {len(events)} effect events")
    return events


# ─── Full Pipeline ──────────────────────────────────────────────────────────

def run_audio_intelligence(
    stem_paths: dict[str, str],
    audio_path: str,
    output_path: str | None = None,
    sr: int = 22050,
    chunk_duration: float = 30.0,
    creative_direction: str | None = None,
    fps: float = 24.0,
    descriptions_md: str | None = None,
    sensitivity: dict[str, float] | None = None,
) -> dict:
    """Run the full 3-layer audio intelligence pipeline.

    Args:
        stem_paths: {"drums": path, "bass": path, "vocals": path, "other": path}
        audio_path: Path to the full mix audio (WAV) for Gemini.
        output_path: Optional path to save results JSON.
        sr: Sample rate for DSP analysis.
        chunk_duration: Duration of chunks for Gemini.
        creative_direction: Optional creative direction for Claude.
        fps: Video frame rate.
        descriptions_md: Path to existing descriptions.md (fallback for Gemini).

    Returns:
        Dict with layer1, layer2, layer3 results.
    """
    _log("=== Multi-Layer Audio Intelligence Pipeline ===")

    # Layer 1: DSP
    layer1 = extract_layer1(stem_paths, sr=sr)

    # Layer 2: Gemini (or cached descriptions)
    layer2 = extract_layer2(audio_path, chunk_duration=chunk_duration, descriptions_md=descriptions_md)

    # Layer 3: Claude
    duration = librosa.get_duration(path=audio_path, sr=sr)
    layer3 = extract_layer3(
        layer1, layer2,
        time_offset=0.0,
        time_limit=duration,
        creative_direction=creative_direction,
        fps=fps,
        sensitivity=sensitivity,
    )

    result = {
        "layer1_summary": {
            stem: {
                band: {
                    "onset_count": len(data.get("onsets", [])),
                    "sustained_count": len(data.get("sustained_regions", [])),
                }
                for band, data in bands.items()
            }
            for stem, bands in layer1.items()
        },
        "layer2_chunks": len(layer2),
        "layer3_events": layer3,
        "layer1": layer1,
        "layer2": layer2,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        _log(f"  Results saved to {output_path}")

    _log(f"=== Pipeline complete: {len(layer3)} effect events ===")
    return result
