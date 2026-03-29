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
    if len(strengths) > 0 and strengths.max() > 0:
        # Adaptive percentile normalization — p10 = noise floor, p95 = full intensity
        p10 = np.percentile(strengths, 10)
        p95 = np.percentile(strengths, 95)
        rng = p95 - p10
        if rng > 0:
            strengths = np.clip((strengths - p10) / rng, 0.0, 1.0)
        else:
            strengths = np.ones_like(strengths) * 0.5
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
                               min_duration: float = 0.3, threshold_ratio: float = 0.1,
                               merge_gap: float = 2.0) -> list[dict]:
    """Detect sustained energy regions (held notes, pad swells, sustained stabs).

    Args:
        threshold_ratio: RMS threshold as fraction of max (0.1 = 10% of peak).
        merge_gap: Merge sustained regions separated by less than this (seconds).
            Catches strings/pads that briefly dip below threshold.

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

    # Merge nearby sustained regions — bridges brief dips in strings/pads
    if merge_gap > 0 and len(regions) >= 2:
        merged = [regions[0]]
        for r in regions[1:]:
            gap = r["start_time"] - merged[-1]["end_time"]
            if gap <= merge_gap:
                # Merge: extend previous region to cover the gap
                merged[-1]["end_time"] = r["end_time"]
                merged[-1]["duration"] = merged[-1]["end_time"] - merged[-1]["start_time"]
                merged[-1]["peak_energy"] = max(merged[-1]["peak_energy"], r["peak_energy"])
            else:
                merged.append(r)
        regions = merged

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


def extract_layer1_multimodel(
    vocals_path: str,
    drumsep_paths: dict[str, str],
    melodic_paths: dict[str, str],
    sr: int = 22050,
) -> dict:
    """Run Layer 1 DSP extraction using multi-model stems.

    DrumSep stems (kick/snare/hh/etc) get full-band analysis only — no frequency
    band splitting since they're already instrument-isolated.
    Melodic stems (bass/guitar/piano/other) get full-band + sustained region detection.
    Vocals get full-band + presence detection.

    Args:
        vocals_path: Path to MDX23C-InstVoc vocals stem.
        drumsep_paths: {"kick": path, "snare": path, "hh": path, "ride": path, "crash": path, "toms": path}
        melodic_paths: {"bass": path, "guitar": path, "piano": path, "other": path}
        sr: Sample rate.

    Returns:
        Dict keyed by stem name: {stem: {"full": {onsets, rms_envelope, sustained_regions, spectral}}}
    """
    results = {}

    # Vocals — full band only, focus on presence
    if vocals_path and Path(vocals_path).exists():
        _log(f"  Layer 1: analyzing vocals...")
        y, sr_out = librosa.load(vocals_path, sr=sr, mono=True)
        results["vocals"] = {
            "full": {
                "onsets": _detect_onsets(y, sr_out),
                "rms_envelope": _compute_rms_envelope(y, sr_out),
                "sustained_regions": _detect_sustained_regions(y, sr_out, min_duration=0.5),
                "spectral": _compute_spectral_features(y, sr_out),
            }
        }
        n = len(results["vocals"]["full"]["onsets"])
        _log(f"    vocals/full: {n} onsets, {len(results['vocals']['full']['sustained_regions'])} sustained")

    # DrumSep stems — full band only, no frequency splitting needed
    for stem_name, path in drumsep_paths.items():
        if not Path(path).exists():
            _log(f"  Warning: drum stem {stem_name} not found at {path}, skipping")
            continue
        _log(f"  Layer 1: analyzing {stem_name} (drum)...")
        y, sr_out = librosa.load(path, sr=sr, mono=True)
        results[stem_name] = {
            "full": {
                "onsets": _detect_onsets(y, sr_out),
                "rms_envelope": _compute_rms_envelope(y, sr_out),
                "sustained_regions": _detect_sustained_regions(y, sr_out, min_duration=0.1),
                "spectral": _compute_spectral_features(y, sr_out),
            }
        }
        n = len(results[stem_name]["full"]["onsets"])
        ns = len(results[stem_name]["full"]["sustained_regions"])
        _log(f"    {stem_name}/full: {n} onsets, {ns} sustained")

    # Melodic stems — full band + sustained regions
    for stem_name, path in melodic_paths.items():
        if not Path(path).exists():
            _log(f"  Warning: melodic stem {stem_name} not found at {path}, skipping")
            continue
        _log(f"  Layer 1: analyzing {stem_name} (melodic)...")
        y, sr_out = librosa.load(path, sr=sr, mono=True)
        results[stem_name] = {
            "full": {
                "onsets": _detect_onsets(y, sr_out),
                "rms_envelope": _compute_rms_envelope(y, sr_out),
                "sustained_regions": _detect_sustained_regions(y, sr_out, min_duration=0.3),
                "spectral": _compute_spectral_features(y, sr_out),
            }
        }
        n = len(results[stem_name]["full"]["onsets"])
        ns = len(results[stem_name]["full"]["sustained_regions"])
        _log(f"    {stem_name}/full: {n} onsets, {ns} sustained")

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

            # Representative sample of onsets — evenly spaced across the strength range
            # so Claude sees strong, medium, and weak onsets for each stem/band
            if len(onsets) <= 30:
                top_onsets = onsets
            else:
                sorted_by_strength = sorted(onsets, key=lambda o: o["strength"])
                step = max(1, len(sorted_by_strength) // 30)
                top_onsets = sorted_by_strength[::step][:30]
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


def _simplify_curve(points: list[dict], time_key: str = "time", value_key: str = "strength",
                     epsilon: float = 0.05) -> list[dict]:
    """Ramer-Douglas-Peucker simplification on onset strength-over-time.

    Removes points that are within epsilon of the line between their neighbors.
    Keeps only shape-defining points — attacks, drops, accents.
    """
    if len(points) <= 2:
        return points

    # Find the point farthest from the line between first and last
    first = points[0]
    last = points[-1]

    t0, v0 = first[time_key], first[value_key]
    t1, v1 = last[time_key], last[value_key]

    max_dist = 0
    max_idx = 0
    dt = t1 - t0 if t1 != t0 else 1e-9

    for i in range(1, len(points) - 1):
        t = points[i][time_key]
        v = points[i][value_key]
        # Distance from point to line between first and last
        interpolated = v0 + (v1 - v0) * (t - t0) / dt
        dist = abs(v - interpolated)
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    if max_dist > epsilon:
        # Recurse on both halves
        left = _simplify_curve(points[:max_idx + 1], time_key, value_key, epsilon)
        right = _simplify_curve(points[max_idx:], time_key, value_key, epsilon)
        return left[:-1] + right
    else:
        # All points between first and last are redundant
        return [first, last]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate — ~4 chars per token for English text."""
    return len(text) // 4


def _format_hybrid_for_claude(layer1_data: dict, stats: dict, layer2_data: list[dict],
                                time_offset: float = 0.0, time_limit: float | None = None,
                                max_tokens: int = 100000) -> str:
    """Format stats + simplified onset curves, fitting within token budget.

    Strategy:
    1. Always include stats summary (cheap, ~2k tokens)
    2. Always include descriptions (cheap, ~3k tokens)
    3. Fill remaining budget with simplified onset curves
    4. Adjust epsilon (simplification tolerance) to fit
    """
    # Fixed sections
    stats_text = _format_stats_for_claude(stats)
    descriptions_text = _format_layer2_for_claude(layer2_data)

    fixed_tokens = _estimate_tokens(stats_text) + _estimate_tokens(descriptions_text) + 5000  # system prompt overhead
    remaining_tokens = max_tokens - fixed_tokens

    # Collect all onset series across stems/bands
    all_series = []
    for stem_name, bands in layer1_data.items():
        for band_name, data in bands.items():
            onsets = data.get("onsets", [])
            if time_limit is not None:
                onsets = [o for o in onsets if time_offset <= o["time"] < time_offset + time_limit]
            if len(onsets) < 2:
                continue
            all_series.append((stem_name, band_name, onsets))

    if not all_series:
        return f"# Audio Analysis\n\n{stats_text}\n\n{descriptions_text}"

    # Binary search for epsilon that fits the budget
    epsilon_low = 0.001
    epsilon_high = 0.5
    best_text = ""

    for _ in range(15):  # converge in ~15 iterations
        epsilon = (epsilon_low + epsilon_high) / 2

        lines = ["\n## Onset Shape Data (curve-simplified, shape-defining points only)\n"]
        for stem_name, band_name, onsets in all_series:
            simplified = _simplify_curve(onsets, epsilon=epsilon)
            if len(simplified) < 2:
                continue
            lines.append(f"### {stem_name}/{band_name} ({len(simplified)} points from {len(onsets)} onsets)")
            for o in simplified:
                lines.append(f"  {o['time']:.3f}s → {o['strength']:.3f}")

        onset_text = "\n".join(lines)
        total_tokens = _estimate_tokens(onset_text)

        if total_tokens <= remaining_tokens:
            best_text = onset_text
            epsilon_high = epsilon  # try to fit more
        else:
            epsilon_low = epsilon  # too many, simplify more

    total_points = best_text.count("→")
    _log(f"    Hybrid prompt: stats + {total_points} shape-defining onset points (epsilon={epsilon_high:.4f})")

    return f"# Audio Analysis\n\n## Statistical Summary\n{stats_text}\n\n## Onset Shapes\n{best_text}\n\n{descriptions_text}"


def _compute_stem_stats(layer1_data: dict, time_offset: float = 0.0,
                         time_limit: float | None = None) -> dict:
    """Compute statistical summary per stem/band for compact Claude prompts."""
    import librosa

    stats = {}
    for stem_name, bands in layer1_data.items():
        stem_stats = {}
        for band_name, data in bands.items():
            onsets = data.get("onsets", [])
            if time_limit is not None:
                onsets = [o for o in onsets if time_offset <= o["time"] < time_offset + time_limit]

            sustained = data.get("sustained_regions", [])
            if time_limit is not None:
                sustained = [s for s in sustained
                             if s["start_time"] < time_offset + time_limit and s["end_time"] > time_offset]

            rms_env = data.get("rms_envelope", [])
            if time_limit is not None:
                rms_env = [r for r in rms_env if time_offset <= r["time"] < time_offset + time_limit]

            spectral = data.get("spectral", {})

            s = {}

            # Onset stats
            strengths = np.array([o["strength"] for o in onsets]) if onsets else np.array([])
            s["onset_count"] = len(onsets)

            if len(strengths) > 0:
                s["strength_min"] = float(np.min(strengths))
                s["strength_max"] = float(np.max(strengths))
                s["strength_mean"] = float(np.mean(strengths))
                s["strength_median"] = float(np.median(strengths))
                s["strength_std"] = float(np.std(strengths))
                for p in [10, 25, 75, 90, 95]:
                    s[f"strength_p{p}"] = float(np.percentile(strengths, p))

                # Density
                duration = time_limit or (onsets[-1]["time"] - onsets[0]["time"]) if len(onsets) > 1 else 1.0
                s["density_per_sec"] = len(onsets) / max(duration, 1.0)

                # Interval stats (temporal regularity)
                if len(onsets) >= 2:
                    times = np.array([o["time"] for o in onsets])
                    intervals = np.diff(times)
                    s["interval_mean_ms"] = float(np.mean(intervals) * 1000)
                    s["interval_median_ms"] = float(np.median(intervals) * 1000)
                    s["interval_min_ms"] = float(np.min(intervals) * 1000)
                    s["interval_max_ms"] = float(np.max(intervals) * 1000)
                    s["interval_std_ms"] = float(np.std(intervals) * 1000)
                    s["regularity"] = 1.0 - min(1.0, float(np.std(intervals) / np.mean(intervals))) if np.mean(intervals) > 0 else 0.0
            else:
                s["strength_min"] = s["strength_max"] = s["strength_mean"] = s["strength_median"] = s["strength_std"] = 0.0
                s["density_per_sec"] = 0.0

            # Sustained stats
            s["sustained_count"] = len(sustained)
            if sustained:
                durations = [r["duration"] for r in sustained]
                s["sustained_avg_duration"] = float(np.mean(durations))
                s["sustained_max_duration"] = float(np.max(durations))
                total_sustained = sum(durations)
                track_dur = time_limit or 120.0
                s["sustained_pct"] = min(100.0, total_sustained / track_dur * 100)
            else:
                s["sustained_avg_duration"] = 0.0
                s["sustained_max_duration"] = 0.0
                s["sustained_pct"] = 0.0

            # RMS / loudness stats
            if rms_env:
                energies = np.array([r["energy"] for r in rms_env])
                s["rms_mean"] = float(np.mean(energies))
                s["rms_max"] = float(np.max(energies))
                s["rms_min"] = float(np.min(energies))
                s["dynamic_range"] = float(np.max(energies) - np.min(energies))
            else:
                s["rms_mean"] = s["rms_max"] = s["rms_min"] = s["dynamic_range"] = 0.0

            # Spectral
            s["spectral_centroid"] = spectral.get("centroid_mean", 0.0)
            s["spectral_flux"] = spectral.get("flux_mean", 0.0)
            s["spectral_rolloff"] = spectral.get("rolloff_mean", 0.0)

            stem_stats[band_name] = s
        stats[stem_name] = stem_stats
    return stats


def _format_stats_for_claude(stats: dict) -> str:
    """Format statistical summary for Claude prompt — compact and information-dense."""
    lines = []

    for stem_name, bands in stats.items():
        lines.append(f"\n## Stem: {stem_name}")

        for band_name, s in bands.items():
            if s["onset_count"] == 0 and s["sustained_count"] == 0:
                continue

            lines.append(f"\n### {stem_name}/{band_name}")
            lines.append(f"Onsets: {s['onset_count']} | Density: {s['density_per_sec']:.2f}/sec")

            if s["onset_count"] > 0:
                lines.append(
                    f"Strength: min={s['strength_min']:.2f} p10={s.get('strength_p10', 0):.2f} "
                    f"p25={s.get('strength_p25', 0):.2f} median={s['strength_median']:.2f} "
                    f"p75={s.get('strength_p75', 0):.2f} p90={s.get('strength_p90', 0):.2f} "
                    f"p95={s.get('strength_p95', 0):.2f} max={s['strength_max']:.2f} std={s['strength_std']:.2f}"
                )

                if "interval_mean_ms" in s:
                    lines.append(
                        f"Intervals: mean={s['interval_mean_ms']:.0f}ms median={s['interval_median_ms']:.0f}ms "
                        f"min={s['interval_min_ms']:.0f}ms max={s['interval_max_ms']:.0f}ms | "
                        f"Regularity: {s.get('regularity', 0):.2f}"
                    )

            if s["sustained_count"] > 0:
                lines.append(
                    f"Sustained: {s['sustained_count']} regions, avg={s['sustained_avg_duration']:.1f}s "
                    f"max={s['sustained_max_duration']:.1f}s ({s['sustained_pct']:.0f}% of track)"
                )

            lines.append(
                f"Energy: mean={s['rms_mean']:.3f} max={s['rms_max']:.3f} range={s['dynamic_range']:.3f} | "
                f"Spectral: centroid={s['spectral_centroid']:.0f}Hz flux={s['spectral_flux']:.2f}"
            )

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

- **zoom_pulse**: Gentle zoom in/out. The workhorse effect — good for melodic hits, bass notes, rhythmic elements.
- **zoom_bounce**: Aggressive zoom punch. THE go-to effect for bass drops, heavy kicks, big impacts. Drops should BOUNCE the visuals, not blind the viewer.
- **shake_x**: Horizontal camera shake. Good for snare hits, percussive impacts.
- **shake_y**: Vertical camera shake. Good for kick drums, sub-bass hits.
- **flash**: Brightness flash. Use SPARINGLY — only for the crispest hi-hat accents or cymbal crashes. Too much flash is blinding and cheap-looking. Prefer zoom_pulse or contrast_pop for most rhythmic elements.
- **hard_cut**: Extreme brightness spike. AVOID for drops — use zoom_bounce + shake instead. Only use hard_cut for rare, singular climactic moments (1-2 per minute MAX). Drops should bounce, not blind.
- **contrast_pop**: Contrast boost. Good for synth stabs, melodic accents. A subtler alternative to flash.
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
9. **PATTERN RECOGNITION IS CRITICAL**: Music is repetitive. If a 4-bar phrase has a kick-snare-kick-snare pattern, and that phrase repeats 8 times, you MUST apply effects to ALL 8 repetitions — not just the first 1-2. Scan the DSP onset data for repeating rhythmic patterns (regular intervals, consistent strength profiles) and ensure every instance of the pattern gets effects. A listener will notice if beats 1-8 have effects but beats 9-32 of the same pattern are dead. Consistency across repeated patterns is more important than variety.
10. **FILL THE GAPS**: After generating your events, mentally scan through the timeline second by second. If there's a gap longer than 1-2 seconds where no effect fires, find an onset in the DSP data to fill it. The video should NEVER feel like effects stopped — there should be continuous visual motion synchronized to the audio.

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
            " (MAXIMUM — trigger on EVERY SINGLE relevant onset without exception. Every kick, every snare, every hi-hat, every bass note, every synth hit. The result should be overwhelming, relentless, nauseating visual intensity. If you can hear it, it gets an effect. Multiple layered effects per onset. No gaps. No mercy.)" if level >= 0.95
            else " (very aggressive — trigger on nearly every relevant onset, high intensity, multiple layers on strong hits)" if level >= 0.8
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


# ─── Layer 3 Rules Mode ─────────────────────────────────────────────────────

def extract_layer3_rules(
    layer1_data: dict,
    layer2_data: list[dict],
    time_offset: float = 0.0,
    time_limit: float | None = None,
    creative_direction: str | None = None,
    sensitivity: dict[str, float] | None = None,
    stats_mode: bool = False,
) -> list[dict]:
    """Ask Claude to generate effect RULES instead of individual events.

    Claude produces a small set of rules (~15-25) that map stem/band/strength
    ranges to effects. We then apply those rules programmatically to every onset.

    If stats_mode=True, sends statistical summaries instead of individual onset
    timestamps — much more compact, fits full 35m tracks in one call.
    """
    import anthropic
    import os

    _log(f"  Layer 3 (rules mode, {'stats' if stats_mode else 'onsets'}): Claude creative direction...")

    if stats_mode:
        stem_stats = _compute_stem_stats(layer1_data, time_offset, time_limit)
        # Hybrid: stats + curve-simplified onset shapes, budget-aware
        hybrid_prompt = _format_hybrid_for_claude(
            layer1_data, stem_stats, layer2_data,
            time_offset=time_offset, time_limit=time_limit,
        )
        dsp_summary = hybrid_prompt
        gemini_summary = ""  # already included in hybrid
    else:
        dsp_summary = _format_layer1_for_claude(layer1_data, time_offset, time_limit)
    if not stats_mode:
        gemini_summary = _format_layer2_for_claude(layer2_data)

    sens = dict(DEFAULT_SENSITIVITY)
    if sensitivity:
        sens.update(sensitivity)

    sensitivity_text = "\n".join(
        f"- **{effect}**: {level:.1f}" + (
            " (MAXIMUM — match the widest possible range of onsets, low thresholds, high intensity)" if level >= 0.95
            else " (very aggressive — wide matching, high intensity)" if level >= 0.8
            else " (aggressive — generous matching)" if level >= 0.6
            else " (moderate — match clear onsets)" if level >= 0.4
            else " (conservative — only strong onsets)" if level >= 0.2
            else " (minimal — only the most dramatic moments)"
        )
        for effect, level in sens.items()
    )

    system_prompt = """You are a visual effects director for music videos. You receive detailed audio analysis data and must produce EFFECT RULES — not individual events.

Your rules will be applied programmatically to EVERY onset in the DSP data. This guarantees complete coverage across all pattern repetitions with zero gaps.

## Available Effects

- **zoom_pulse**: Gentle zoom in/out. The workhorse effect — good for melodic hits, bass notes, rhythmic elements.
- **zoom_bounce**: Aggressive zoom punch. THE go-to effect for bass drops, heavy kicks, big impacts. Drops should BOUNCE the visuals, not blind the viewer.
- **shake_x**: Horizontal camera shake. Good for snare hits.
- **shake_y**: Vertical camera shake. Good for kick drums, sub-bass.
- **flash**: Brightness flash. Use SPARINGLY — only for the crispest hi-hat accents or cymbal crashes. Too much flash is blinding and cheap-looking. Prefer zoom_pulse or contrast_pop for most rhythmic elements.
- **hard_cut**: Extreme brightness spike. AVOID for drops — use zoom_bounce + shake instead. Only use hard_cut for rare, singular climactic moments (1-2 per minute MAX). Drops should bounce, not blind.
- **contrast_pop**: Contrast boost. Good for synth stabs, melodic accents. A subtler alternative to flash.
- **glow_swell**: Soft glow bloom. Good for sustained pads, ambient textures, vocal sections.

## Rule Schema

Each rule specifies: which onsets to match → what effect to apply.

```json
{
  "stem": "drums",
  "band": "low",
  "min_strength": 0.1,
  "max_strength": 1.0,
  "effect": "shake_y",
  "intensity_scale": 0.8,
  "duration": 0.2,
  "sustain_from_rms": false,
  "layer_with": [],
  "layer_threshold": 0.7,
  "rationale": "every kick drum hit gets vertical shake"
}
```

### Rule fields:

- **stem**: "drums", "bass", "vocals", "other"
- **band**: "low", "mid", "high", "full" (which frequency band's onsets to match)
- **min_strength**: Minimum onset strength to trigger (0.0-1.0). Lower = more triggers.
- **max_strength**: Maximum onset strength (for targeting specific ranges, e.g. just ghost notes)
- **effect**: The primary effect to apply
- **intensity_scale**: Multiply the onset's strength by this to get effect intensity. 1.0 = onset strength IS effect intensity. 1.5 = amplify weak onsets.
- **duration**: Base duration in seconds. Transients: 0.1-0.3s. Sustained: 0.5-2.0s.
- **sustain_from_rms**: If true, when the onset falls within a sustained_region, extend duration to match the region's duration. Great for held chords, pad swells.
- **layer_with**: Array of additional effects to fire simultaneously when this rule triggers. E.g., ["zoom_bounce"] means the primary effect + zoom_bounce both fire.
- **layer_threshold**: Only apply layer_with effects when onset strength exceeds this. Prevents layering on weak hits.
- **rationale**: Why this rule exists.

## Guidelines

1. **Cover every audible element**: Create rules for kicks, snares, hi-hats, bass notes, synth stabs, pads, vocals — everything that makes sound should have at least one rule.
2. **Use frequency bands for instrument separation**: drums/low = kicks, drums/mid = snares/toms, drums/high = hi-hats/cymbals. bass/low = sub-bass. other/mid = synth stabs. other/high = arpeggios/leads.
3. **Layer on big hits**: Use layer_with + layer_threshold to stack effects on strong onsets without cluttering weak ones.
4. **Sustained sounds**: Use sustain_from_rms=true for pads, held synths, vocal notes. This makes glow/zoom hold for the natural duration.
5. **Sensitivity settings control how aggressive each effect is**: Higher sensitivity = lower min_strength thresholds, higher intensity_scale.
6. **Aim for 12-18 rules MAX**. Fewer rules = more dynamic contrast. Each rule should target a DISTINCT musical element. Do NOT create separate rules for the same instrument across multiple bands unless they serve genuinely different purposes. One strong rule per instrument is better than three weak ones. The silence between effects is as important as the effects themselves.
7. **CRITICAL — min_strength thresholds**: Set min_strength at or ABOVE the p25 (25th percentile) of that stem's strength distribution. NEVER set min_strength to 0.00 — this captures noise and micro-transients that produce tiny barely-visible effects, diluting the impact of real hits. Silence between beats is essential for dynamics — the contrast between "no effect" and "big effect" is what makes beats feel punchy. If the stats show p25=0.22, set min_strength to 0.20-0.25.
8. **CRITICAL — intensity_scale**: For percussion (kick, snare, toms), intensity_scale should be >= 1.0. For melodic/sustained content (bass, other, piano), intensity_scale can be 0.6-1.0. Percussion should always feel impactful.

## Output Format

Respond with ONLY a JSON array of rules. No markdown, no explanation outside the JSON."""

    user_prompt = f"""# Audio Analysis Data

{dsp_summary}

{gemini_summary}

## Effect Sensitivity Settings

{sensitivity_text}

"""
    if creative_direction:
        user_prompt += f"\n## Creative Direction\n{creative_direction}\n"

    user_prompt += "\nGenerate the effect rules JSON."

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    _log("    Sending to Claude (streaming)...")
    text = ""
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        for chunk in stream.text_stream:
            text += chunk

    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    if text.endswith("```"):
        text = text[:-3]

    try:
        rules = json.loads(text.strip())
    except json.JSONDecodeError as e:
        _log(f"    Failed to parse Claude response: {e}")
        _log(f"    Response: {text[:500]}")
        rules = []

    _log(f"    Claude produced {len(rules)} effect rules")
    return rules


def _build_rms_lookup(rms_envelope: list[dict]) -> callable:
    """Build a fast RMS lookup function from an envelope.

    Returns a function f(t) -> rms_energy at time t (interpolated).
    """
    if not rms_envelope:
        return lambda t: 0.0

    times = np.array([p["time"] for p in rms_envelope])
    energies = np.array([p["energy"] for p in rms_envelope])

    def lookup(t):
        idx = np.searchsorted(times, t, side="right") - 1
        if idx < 0:
            return float(energies[0])
        if idx >= len(energies) - 1:
            return float(energies[-1])
        return float(energies[idx])

    return lookup


def apply_rules(layer1_data: dict, rules: list[dict],
                vocal_bleed_threshold: float = 0.25,
                reverb_dedup: bool = False,
                bleed_exempt_stems: set[str] | None = None) -> list[dict]:
    """Apply effect rules to all DSP onsets, producing frame-accurate events.

    This is the deterministic step — every onset that matches a rule gets an effect.
    Guarantees complete coverage across all pattern repetitions.

    Args:
        layer1_data: Layer 1 DSP data.
        rules: Effect rules from Claude.
        vocal_bleed_threshold: Confidence ratio threshold. When a non-vocal stem's
            RMS energy at onset time is less than this fraction of the vocal stem's
            RMS, the onset is suppressed as likely bleed. Set to 0.0 to disable.
            Default: 0.15 (suppress when stem is <15% of vocal energy).
        bleed_exempt_stems: Stem names exempt from bleed check. Stems derived from
            the instrumental output are already vocal-free by construction.
    """
    _log("  Applying rules to DSP data...")

    # Build vocal RMS lookup for confidence ratio
    vocal_rms_env = layer1_data.get("vocals", {}).get("full", {}).get("rms_envelope", [])
    vocal_rms = _build_rms_lookup(vocal_rms_env)
    bleed_enabled = vocal_bleed_threshold > 0 and len(vocal_rms_env) > 0
    if bleed_enabled:
        _log(f"    Vocal bleed suppression enabled (threshold={vocal_bleed_threshold})")

    exempt = bleed_exempt_stems or set()
    events = []
    suppressed_total = 0

    for rule in rules:
        stem = rule.get("stem", "drums")
        band = rule.get("band", "full")
        min_str = rule.get("min_strength", 0.0)
        max_str = rule.get("max_strength", 1.0)
        effect = rule.get("effect", "shake_y")
        intensity_scale = rule.get("intensity_scale", 1.0)
        duration = rule.get("duration", 0.2)
        sustain_from_rms = rule.get("sustain_from_rms", False)
        layer_with = rule.get("layer_with", [])
        layer_threshold = rule.get("layer_threshold", 0.7)

        # Build this stem's RMS lookup for confidence ratio
        stem_rms_env = layer1_data.get(stem, {}).get(band, {}).get("rms_envelope", [])
        stem_rms = _build_rms_lookup(stem_rms_env)

        stem_data = layer1_data.get(stem, {})
        band_data = stem_data.get(band, {})
        onsets = band_data.get("onsets", [])
        sustained_regions = band_data.get("sustained_regions", [])

        matched = 0
        suppressed = 0
        for onset in onsets:
            strength = onset.get("strength", 0)
            if strength < min_str or strength > max_str:
                continue

            t = onset["time"]

            # Percussion sustained-bleed suppression: if a percussion stem has
            # a sustained region at this time, the "onset" is actually bleed
            # from a sustained instrument (pad, bass), not a real drum hit.
            # Real percussion is transient by nature — never sustained.
            # NOTE: Skip for bleed-exempt stems (from instrumental) — their
            # sustained regions are reverb tails, not instrument bleed.
            PERCUSSION_STEMS = {"kick", "snare", "hh", "toms", "ride", "crash", "drums"}
            if stem in PERCUSSION_STEMS and stem not in exempt:
                full_sustained = layer1_data.get(stem, {}).get("full", {}).get("sustained_regions", [])
                in_sustained = any(
                    r["start_time"] <= t <= r["end_time"]
                    for r in full_sustained
                )
                if in_sustained:
                    suppressed += 1
                    continue

            # Confidence ratio: suppress bleed from non-vocal stems
            # Skip bleed check for stems derived from instrumental (already vocal-free)
            if bleed_enabled and stem != "vocals" and stem not in exempt:
                v_energy = vocal_rms(t)
                if v_energy > 0.01:
                    s_energy = stem_rms(t)
                    ratio = s_energy / v_energy if v_energy > 0 else 999
                    if ratio < vocal_bleed_threshold:
                        suppressed += 1
                        continue

            # Percussion cross-stem bleed: DrumSep doesn't know about non-drum
            # instruments (piano, bass, guitar, etc). Check percussion onsets against
            # ALL melodic/harmonic stems. If any melodic stem dominates at this
            # timestamp, the percussion onset is likely bleed from that instrument.
            PERC_STEMS = {"kick", "snare", "hh", "toms", "ride", "crash", "drums"}
            MELODIC_STEMS = {"vocals", "bass", "piano", "guitar", "other"}
            if bleed_enabled and stem in PERC_STEMS:
                s_energy = stem_rms(t)
                is_bleed = False
                for mel_stem in MELODIC_STEMS:
                    mel_rms_env = layer1_data.get(mel_stem, {}).get("full", {}).get("rms_envelope", [])
                    if not mel_rms_env:
                        continue
                    mel_lookup = _build_rms_lookup(mel_rms_env)
                    mel_energy = mel_lookup(t)
                    if mel_energy > 0.01 and s_energy > 0:
                        ratio = s_energy / mel_energy
                        if ratio < vocal_bleed_threshold:
                            is_bleed = True
                            break
                if is_bleed:
                    suppressed += 1
                    continue

            intensity = min(1.0, strength * intensity_scale)
            evt_duration = duration
            sustain = None

            # Check for sustained region overlap — only for non-transient effects
            # Shake effects should NEVER sustain — sustained shaking is just wrong
            TRANSIENT_EFFECTS = {"shake_x", "shake_y", "flash", "hard_cut"}
            if sustain_from_rms and effect not in TRANSIENT_EFFECTS:
                for region in sustained_regions:
                    if region["start_time"] <= t <= region["end_time"]:
                        sustain = region["duration"]
                        evt_duration = max(duration, region["duration"])
                        break

            events.append({
                "time": t,
                "duration": evt_duration,
                "effect": effect,
                "intensity": intensity,
                "sustain": sustain,
                "stem_source": f"{stem}/{band}",
                "rationale": rule.get("rationale", ""),
            })
            matched += 1

            # Layered effects on strong hits
            if layer_with and strength >= layer_threshold:
                for layer_effect in layer_with:
                    # Transient layered effects get transient duration, not sustained
                    TRANSIENT_LAYER = {"shake_x", "shake_y", "flash", "hard_cut"}
                    layer_dur = duration if layer_effect in TRANSIENT_LAYER else evt_duration
                    layer_sustain = None if layer_effect in TRANSIENT_LAYER else sustain
                    events.append({
                        "time": t,
                        "duration": layer_dur,
                        "effect": layer_effect,
                        "intensity": min(1.0, intensity * 0.8),
                        "sustain": layer_sustain,
                        "stem_source": f"{stem}/{band}",
                        "rationale": f"layered with {effect} on strong hit",
                    })

        suppressed_total += suppressed
        suppressed_str = f" ({suppressed} suppressed)" if suppressed > 0 else ""
        _log(f"    Rule '{effect}' on {stem}/{band} [{min_str:.2f}-{max_str:.2f}]: {matched} events{suppressed_str}")

    # Sort by time
    events.sort(key=lambda e: e["time"])

    # Reverb deduplication (optional): for percussion stems only, if two events
    # from the same stem_source are within 200ms, keep only the stronger one.
    if reverb_dedup:
        PERCUSSION_DEDUP_STEMS = {"kick", "snare", "hh", "toms", "ride", "crash", "drums"}
        before_dedup = len(events)
        deduped = []
        last_by_source = {}

        for event in events:
            src = event["stem_source"]
            stem_name = src.split("/")[0] if "/" in src else src
            t = event["time"]
            intensity = event["intensity"]

            if stem_name in PERCUSSION_DEDUP_STEMS and src in last_by_source:
                prev_t, prev_int, prev_idx = last_by_source[src]
                if t - prev_t < 0.2:
                    if intensity > prev_int:
                        deduped[prev_idx] = event
                        last_by_source[src] = (t, intensity, prev_idx)
                    continue

            last_by_source[src] = (t, intensity, len(deduped))
            deduped.append(event)

        reverb_removed = before_dedup - len(deduped)
        _log(f"  Total effect events: {len(deduped)} ({suppressed_total} bleed suppressed, {reverb_removed} reverb ghosts removed)")
        return deduped

    _log(f"  Total effect events: {len(events)} ({suppressed_total} bleed suppressed)")
    return events


def apply_rules_in_range(layer1_data: dict, rules: list[dict],
                          start_time: float, end_time: float,
                          vocal_bleed_threshold: float = 0.25) -> list[dict]:
    """Apply effect rules only to onsets within a time range."""
    vocal_rms_env = layer1_data.get("vocals", {}).get("full", {}).get("rms_envelope", [])
    vocal_rms = _build_rms_lookup(vocal_rms_env)
    bleed_enabled = vocal_bleed_threshold > 0 and len(vocal_rms_env) > 0

    events = []

    for rule in rules:
        stem = rule.get("stem", "drums")
        band = rule.get("band", "full")
        min_str = rule.get("min_strength", 0.0)
        max_str = rule.get("max_strength", 1.0)
        effect = rule.get("effect", "shake_y")
        intensity_scale = rule.get("intensity_scale", 1.0)
        duration = rule.get("duration", 0.2)
        sustain_from_rms = rule.get("sustain_from_rms", False)
        layer_with = rule.get("layer_with", [])
        layer_threshold = rule.get("layer_threshold", 0.7)

        stem_rms_env = layer1_data.get(stem, {}).get(band, {}).get("rms_envelope", [])
        stem_rms = _build_rms_lookup(stem_rms_env)

        stem_data = layer1_data.get(stem, {})
        band_data = stem_data.get(band, {})
        onsets = band_data.get("onsets", [])
        sustained_regions = band_data.get("sustained_regions", [])

        for onset in onsets:
            t = onset["time"]
            if t < start_time or t >= end_time:
                continue
            strength = onset.get("strength", 0)
            if strength < min_str or strength > max_str:
                continue

            if bleed_enabled and stem != "vocals":
                v_energy = vocal_rms(t)
                if v_energy > 0.01:
                    s_energy = stem_rms(t)
                    ratio = s_energy / v_energy if v_energy > 0 else 999
                    if ratio < vocal_bleed_threshold:
                        continue

            # Percussion cross-stem bleed (same as apply_rules)
            PERC_STEMS = {"kick", "snare", "hh", "toms", "ride", "crash", "drums"}
            MELODIC_STEMS = {"vocals", "bass", "piano", "guitar", "other"}
            if bleed_enabled and stem in PERC_STEMS:
                s_energy = stem_rms(t)
                is_bleed = False
                for mel_stem in MELODIC_STEMS:
                    mel_rms_env = layer1_data.get(mel_stem, {}).get("full", {}).get("rms_envelope", [])
                    if not mel_rms_env:
                        continue
                    mel_lookup = _build_rms_lookup(mel_rms_env)
                    mel_energy = mel_lookup(t)
                    if mel_energy > 0.01 and s_energy > 0:
                        ratio = s_energy / mel_energy
                        if ratio < vocal_bleed_threshold:
                            is_bleed = True
                            break
                if is_bleed:
                    continue

            intensity = min(1.0, strength * intensity_scale)
            evt_duration = duration
            sustain = None

            TRANSIENT_EFFECTS = {"shake_x", "shake_y", "flash", "hard_cut"}
            if sustain_from_rms and effect not in TRANSIENT_EFFECTS:
                for region in sustained_regions:
                    if region["start_time"] <= t <= region["end_time"]:
                        sustain = region["duration"]
                        evt_duration = max(duration, region["duration"])
                        break

            events.append({
                "time": t,
                "duration": evt_duration,
                "effect": effect,
                "intensity": intensity,
                "sustain": sustain,
                "stem_source": f"{stem}/{band}",
                "rationale": rule.get("rationale", ""),
            })

            if layer_with and strength >= layer_threshold:
                for layer_effect in layer_with:
                    # Transient layered effects get transient duration, not sustained
                    TRANSIENT_LAYER = {"shake_x", "shake_y", "flash", "hard_cut"}
                    layer_dur = duration if layer_effect in TRANSIENT_LAYER else evt_duration
                    layer_sustain = None if layer_effect in TRANSIENT_LAYER else sustain
                    events.append({
                        "time": t,
                        "duration": layer_dur,
                        "effect": layer_effect,
                        "intensity": min(1.0, intensity * 0.8),
                        "sustain": layer_sustain,
                        "stem_source": f"{stem}/{band}",
                        "rationale": f"layered with {effect} on strong hit",
                    })

    return events


def _group_sections_into_chunks(layer2_data: list[dict], max_gap: float = 30.0) -> list[dict]:
    """Group Layer 2 description sections into energy-coherent chunks.

    Adjacent sections with the same energy classification get merged.
    Returns [{start_time, end_time, energy, description_summary}].
    """
    import re

    chunks = []
    current = None

    for section in sorted(layer2_data, key=lambda s: s["start_time"]):
        # Extract energy from section header in description (e.g., "high-energy", "low energy")
        desc = section.get("description", "")
        header_match = re.search(r"(low|mid|high)[_\s-]?energy", desc, re.IGNORECASE)
        energy = header_match.group(0).lower().replace(" ", "_").replace("-", "_") if header_match else "mid_energy"

        # Simplify to low/mid/high
        if "low" in energy:
            energy = "low"
        elif "high" in energy:
            energy = "high"
        else:
            energy = "mid"

        if current is None:
            current = {
                "start_time": section["start_time"],
                "end_time": section["end_time"],
                "energy": energy,
                "descriptions": [desc],
            }
        elif energy == current["energy"] and section["start_time"] - current["end_time"] < max_gap:
            current["end_time"] = section["end_time"]
            current["descriptions"].append(desc)
        else:
            current["description_summary"] = current["descriptions"][0][:200] + (
                f" ... (+{len(current['descriptions'])-1} more sections)" if len(current["descriptions"]) > 1 else ""
            )
            del current["descriptions"]
            chunks.append(current)
            current = {
                "start_time": section["start_time"],
                "end_time": section["end_time"],
                "energy": energy,
                "descriptions": [desc],
            }

    if current:
        current["description_summary"] = current["descriptions"][0][:200] + (
            f" ... (+{len(current['descriptions'])-1} more sections)" if len(current["descriptions"]) > 1 else ""
        )
        del current["descriptions"]
        chunks.append(current)

    return chunks


def extract_layer3_rules_chunked(
    layer1_data: dict,
    layer2_data: list[dict],
    creative_direction: str | None = None,
    sensitivity: dict[str, float] | None = None,
    vocal_bleed_threshold: float = 0.25,
) -> tuple[list[dict], list[dict]]:
    """Generate per-section rules by chunking the track into energy-coherent regions.

    Each chunk gets its own Claude call for tailored rules. Returns (all_rules_with_ranges, all_events).
    """
    chunks = _group_sections_into_chunks(layer2_data)
    _log(f"  Layer 3 (chunked rules mode): {len(chunks)} energy chunks")
    for c in chunks:
        _log(f"    {c['start_time']:.0f}s - {c['end_time']:.0f}s: {c['energy']}")

    all_rules = []
    all_events = []

    for i, chunk in enumerate(chunks):
        energy = chunk["energy"]
        start = chunk["start_time"]
        end = chunk["end_time"]
        dur = end - start

        # Build chunk-specific creative direction
        energy_guidance = {
            "low": "This is a LOW energy section — ambient, dreamy, meditative. Use gentle effects: glow_swell, subtle zoom_pulse. Minimize shake and flash. Let the music breathe.",
            "mid": "This is a MID energy section — building tension, melodic, driving but not peak. Use moderate zoom_pulse, contrast_pop, some shake on strong beats. Hold back on aggressive effects.",
            "high": "This is a HIGH energy section — peak intensity, drops, aggressive beats. Use zoom_bounce, shake_x, shake_y aggressively. Layer effects on strong hits. This should feel powerful and punchy.",
        }

        chunk_direction = energy_guidance.get(energy, energy_guidance["mid"])
        if creative_direction:
            chunk_direction = f"{creative_direction}\n\n{chunk_direction}"

        chunk_direction += f"\n\nThis chunk covers {start:.0f}s to {end:.0f}s ({dur:.0f}s)."
        chunk_direction += f"\n\nAudio context: {chunk['description_summary']}"

        _log(f"  Chunk {i+1}/{len(chunks)}: {start:.0f}s-{end:.0f}s ({energy}, {dur:.0f}s)")

        # Get rules for this chunk
        rules = extract_layer3_rules(
            layer1_data, layer2_data,
            time_offset=start,
            time_limit=dur,
            creative_direction=chunk_direction,
            sensitivity=sensitivity,
        )

        # Tag rules with their time range
        for rule in rules:
            rule["_chunk_start"] = start
            rule["_chunk_end"] = end
            rule["_chunk_energy"] = energy

        # Apply rules only to onsets in this time range
        chunk_events = apply_rules_in_range(layer1_data, rules, start, end,
                                            vocal_bleed_threshold=vocal_bleed_threshold)
        _log(f"    → {len(rules)} rules, {len(chunk_events)} events")

        all_rules.extend(rules)
        all_events.extend(chunk_events)

    all_events.sort(key=lambda e: e["time"])
    _log(f"  Total: {len(all_rules)} rules, {len(all_events)} events across {len(chunks)} chunks")
    return all_rules, all_events


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
    rules_mode: bool = False,
    chunked: bool = False,
    vocal_bleed_threshold: float = 0.25,
    stats_mode: bool = False,
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

    if rules_mode and chunked:
        rules, layer3 = extract_layer3_rules_chunked(
            layer1, layer2,
            creative_direction=creative_direction,
            sensitivity=sensitivity,
            vocal_bleed_threshold=vocal_bleed_threshold,
        )
    elif rules_mode:
        rules = extract_layer3_rules(
            layer1, layer2,
            time_offset=0.0,
            time_limit=duration,
            creative_direction=creative_direction,
            sensitivity=sensitivity,
            stats_mode=stats_mode,
        )
        layer3 = apply_rules(layer1, rules, vocal_bleed_threshold=vocal_bleed_threshold)
    else:
        rules = None
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
        **({"layer3_rules": rules} if rules else {}),
        "layer1": layer1,
        "layer2": layer2,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        _log(f"  Results saved to {output_path}")

    _log(f"=== Pipeline complete: {len(layer3)} effect events ===")
    return result


def run_audio_intelligence_multimodel(
    vocals_path: str,
    drumsep_paths: dict[str, str],
    melodic_paths: dict[str, str],
    audio_path: str,
    output_path: str | None = None,
    sr: int = 22050,
    descriptions_md: str | None = None,
    creative_direction: str | None = None,
    sensitivity: dict[str, float] | None = None,
    vocal_bleed_threshold: float = 0.25,
    fps: float = 24.0,
    stats_mode: bool = False,
) -> dict:
    """Run the 3-layer pipeline using multi-model stems.

    Uses MDX23C-InstVoc vocals + DrumSep kick/snare/hh/etc + Demucs 6s bass/guitar/piano/other.

    Args:
        vocals_path: Path to MDX23C-InstVoc vocals stem.
        drumsep_paths: {"kick": path, "snare": path, "hh": path, ...}
        melodic_paths: {"bass": path, "guitar": path, "piano": path, "other": path}
        audio_path: Path to full mix audio for Gemini/descriptions.
        output_path: Optional path to save results JSON.
        sr: Sample rate.
        descriptions_md: Path to descriptions.md fallback.
        creative_direction: Creative direction for Claude.
        sensitivity: Per-effect sensitivity overrides.
        vocal_bleed_threshold: Confidence ratio threshold.
        fps: Video frame rate.
        stats_mode: Use stats-only Claude prompt.
    """
    _log("=== Multi-Model Audio Intelligence Pipeline ===")

    # Layer 1: DSP extraction on multi-model stems
    layer1 = extract_layer1_multimodel(vocals_path, drumsep_paths, melodic_paths, sr=sr)

    # Layer 2: Gemini (or cached descriptions)
    layer2 = extract_layer2(audio_path, descriptions_md=descriptions_md)

    # Layer 3: Claude rules + apply
    duration = librosa.get_duration(path=audio_path, sr=sr)
    rules = extract_layer3_rules(
        layer1, layer2,
        time_offset=0.0,
        time_limit=duration,
        creative_direction=creative_direction,
        sensitivity=sensitivity,
        stats_mode=stats_mode,
    )
    # All drum + melodic stems are derived from the instrumental — exempt from vocal bleed check
    exempt = set(drumsep_paths.keys()) | set(melodic_paths.keys())
    _log(f"  Bleed-exempt stems (from instrumental): {exempt}")
    layer3 = apply_rules(layer1, rules, vocal_bleed_threshold=vocal_bleed_threshold, bleed_exempt_stems=exempt)

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
        "layer3_rules": rules,
        "layer1": layer1,
        "layer2": layer2,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        _log(f"  Results saved to {output_path}")

    _log(f"=== Pipeline complete: {len(layer3)} effect events ===")
    return result
