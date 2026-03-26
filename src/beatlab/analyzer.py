"""Audio analysis module — beat tracking, onset detection, section detection, and feature extraction."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def load_audio(path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load an audio file and return (samples, sample_rate).

    Args:
        path: Path to audio file (WAV, MP3, FLAC, OGG, M4A).
        sr: Target sample rate. Defaults to 22050.

    Returns:
        Tuple of (audio time series as numpy array, sample rate).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{p.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    y, sr_out = librosa.load(str(p), sr=sr, mono=True)
    return y, sr_out


def _compute_spectral_features(
    y: np.ndarray, sr: int, start_sample: int, end_sample: int,
) -> dict[str, float]:
    """Compute spectral features for a segment of audio."""
    segment = y[start_sample:end_sample]
    if len(segment) == 0:
        return {"centroid": 0.0, "rms_energy": 0.0, "rolloff": 0.0, "contrast": 0.0}

    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
    rms = librosa.feature.rms(y=segment)
    rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=segment, sr=sr)

    return {
        "centroid": float(np.mean(centroid)),
        "rms_energy": float(np.mean(rms)),
        "rolloff": float(np.mean(rolloff)),
        "contrast": float(np.mean(contrast)),
    }


def _normalize_spectral(sections: list[dict]) -> None:
    """Normalize spectral features across all sections to 0.0-1.0 range in-place."""
    if not sections or "spectral" not in sections[0]:
        return

    for key in ("centroid", "rms_energy", "rolloff", "contrast"):
        values = [s["spectral"][key] for s in sections]
        vmin, vmax = min(values), max(values)
        rng = vmax - vmin
        for s in sections:
            s["spectral"][key] = (s["spectral"][key] - vmin) / rng if rng > 0 else 0.0


def detect_sections(
    y: np.ndarray, sr: int, hop_length: int = 512, segment_duration: float = 4.0,
) -> list[dict]:
    """Detect musical sections using RMS energy thresholding with spectral features.

    Divides audio into fixed-length segments, computes average RMS energy and spectral
    features, classifies each as low_energy, mid_energy, or high_energy, and merges
    consecutive segments with the same classification.

    Returns list of dicts with: start_time, end_time, type, label, spectral
    """
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    duration = librosa.get_duration(y=y, sr=sr)

    # Number of frames per segment
    frames_per_sec = sr / hop_length
    frames_per_segment = int(frames_per_sec * segment_duration)

    if frames_per_segment < 1:
        frames_per_segment = 1

    segments = []
    for start_idx in range(0, len(rms), frames_per_segment):
        end_idx = min(start_idx + frames_per_segment, len(rms))
        seg_rms = rms[start_idx:end_idx]
        avg_energy = float(np.mean(seg_rms))
        start_time = start_idx / frames_per_sec
        end_time = min(end_idx / frames_per_sec, duration)

        # Compute spectral features for this segment
        start_sample = int(start_time * sr)
        end_sample = min(int(end_time * sr), len(y))
        spectral = _compute_spectral_features(y, sr, start_sample, end_sample)

        segments.append({
            "start_time": start_time,
            "end_time": end_time,
            "energy": avg_energy,
            "spectral": spectral,
        })

    if not segments:
        return []

    # Compute thresholds using percentiles of segment energies
    energies = [s["energy"] for s in segments]
    p33 = float(np.percentile(energies, 33))
    p66 = float(np.percentile(energies, 66))

    # Merge consecutive segments with the same classification
    LABELS = {
        "low_energy": "verse",
        "mid_energy": "bridge",
        "high_energy": "chorus",
    }

    classified = []
    for seg in segments:
        if seg["energy"] <= p33:
            stype = "low_energy"
        elif seg["energy"] <= p66:
            stype = "mid_energy"
        else:
            stype = "high_energy"

        if classified and classified[-1]["type"] == stype:
            # Extend previous section — average the spectral features
            prev = classified[-1]
            prev["end_time"] = seg["end_time"]
            prev["_seg_count"] += 1
            n = prev["_seg_count"]
            for key in ("centroid", "rms_energy", "rolloff", "contrast"):
                prev["spectral"][key] = (
                    prev["spectral"][key] * (n - 1) + seg["spectral"][key]
                ) / n
        else:
            classified.append({
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "type": stype,
                "label": LABELS[stype],
                "spectral": dict(seg["spectral"]),
                "_seg_count": 1,
            })

    # Clean up internal field and normalize spectral features
    for s in classified:
        del s["_seg_count"]

    _normalize_spectral(classified)

    return classified


def analyze_audio(path: str, sr: int = 22050, detect_sections_flag: bool = False) -> dict:
    """Analyze an audio file and return beat data.

    Returns a dict with:
        tempo: float — estimated BPM
        duration: float — audio duration in seconds
        sample_rate: int
        beats: list of {time: float, intensity: float}
        onsets: list of {time: float, strength: float}
        sections: list of {start_time, end_time, type, label, spectral} (if detect_sections_flag)
    """
    y, sr_out = load_audio(path, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr_out)

    # Beat tracking — beat_this (SOTA neural beat/downbeat tracker)
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr_out, hop_length=hop_length)

    try:
        from beat_this.inference import File2Beats
        f2b = File2Beats(device='cpu')
        beat_times_arr, downbeat_times_arr = f2b(path)

        # Estimate tempo from beat intervals
        if len(beat_times_arr) >= 2:
            intervals = np.diff(beat_times_arr)
            median_interval = float(np.median(intervals))
            tempo_val = 60.0 / median_interval if median_interval > 0 else 120.0
        else:
            tempo_val = 120.0

        # Build beat list with intensity from onset envelope + downbeat flag
        # Downbeats get full intensity (1.0), regular beats get scaled by onset strength
        max_env_idx = len(onset_env) - 1
        downbeat_set = set(float(round(t, 3)) for t in downbeat_times_arr)

        beat_env_frames = np.clip(
            librosa.time_to_frames(beat_times_arr, sr=sr_out, hop_length=hop_length),
            0, max_env_idx,
        )
        beat_strengths = onset_env[beat_env_frames]

        if len(beat_strengths) > 0 and beat_strengths.max() > 0:
            normalized = (beat_strengths - beat_strengths.min()) / (
                beat_strengths.max() - beat_strengths.min()
            )
            beat_intensities = 0.2 + 0.8 * normalized
        else:
            beat_intensities = np.full_like(beat_times_arr, 0.5)

        beats = []
        for t, intensity in zip(beat_times_arr, beat_intensities):
            is_downbeat = round(float(t), 3) in downbeat_set
            # Boost downbeats — these are the "1" of each measure
            final_intensity = min(1.0, float(intensity) * 1.3) if is_downbeat else float(intensity)
            beats.append({
                "time": float(t),
                "intensity": final_intensity,
                "downbeat": is_downbeat,
            })

    except Exception as e:
        # Fallback to librosa if beat_this fails
        import sys
        print(f"[beat_this failed: {e}, falling back to librosa]", file=sys.stderr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr_out, hop_length=hop_length)
        tempo_val = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
        beat_times_arr = librosa.frames_to_time(beat_frames, sr=sr_out, hop_length=hop_length)
        beats = [{"time": float(t), "intensity": 0.5} for t in beat_times_arr]

    # Onset detection (for onset data in beat map)
    max_env_idx = len(onset_env) - 1
    onset_frames_arr = librosa.onset.onset_detect(
        y=y, sr=sr_out, hop_length=hop_length, onset_envelope=onset_env, backtrack=True,
    )
    onset_frames_arr = np.clip(onset_frames_arr, 0, max_env_idx) if len(onset_frames_arr) > 0 else onset_frames_arr
    onset_times = librosa.frames_to_time(onset_frames_arr, sr=sr_out, hop_length=hop_length)
    onset_strengths = onset_env[onset_frames_arr] if len(onset_frames_arr) > 0 else np.array([])
    if len(onset_strengths) > 0 and onset_strengths.max() > 0:
        onset_strengths_norm = onset_strengths / onset_strengths.max()
    else:
        onset_strengths_norm = onset_strengths

    onsets = [
        {"time": float(t), "strength": float(s)}
        for t, s in zip(onset_times, onset_strengths_norm)
    ]

    result = {
        "tempo": tempo_val,
        "duration": float(duration),
        "sample_rate": sr_out,
        "beats": beats,
        "onsets": onsets,
    }

    # Section detection
    if detect_sections_flag:
        result["sections"] = detect_sections(y, sr_out)

    return result


def detect_drops(y: np.ndarray, sr: int, hop_length: int = 512, threshold: float = 3.0) -> list[dict]:
    """Detect sudden energy jumps (bass drops, impacts).

    Finds frames where RMS energy jumps by more than `threshold` standard deviations
    above the local mean.

    Returns list of {"time": float, "intensity": float}.
    """
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if len(rms) < 10:
        return []

    # Compute frame-to-frame energy delta
    delta = np.diff(rms)
    if len(delta) == 0:
        return []

    mean_delta = np.mean(delta)
    std_delta = np.std(delta)
    if std_delta == 0:
        return []

    drops = []
    for i, d in enumerate(delta):
        if d > mean_delta + threshold * std_delta:
            t = librosa.frames_to_time(i + 1, sr=sr, hop_length=hop_length)
            intensity = min(1.0, float((d - mean_delta) / (std_delta * threshold * 2)))
            drops.append({"time": float(t), "intensity": intensity})

    return drops


def detect_presence(y: np.ndarray, sr: int, hop_length: int = 512, threshold_ratio: float = 0.15) -> list[dict]:
    """Detect regions where audio is present (above RMS threshold).

    Useful for finding vocal regions in a vocals-only stem.

    Returns list of {"start_time": float, "end_time": float}.
    """
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if len(rms) == 0:
        return []

    threshold = float(np.max(rms)) * threshold_ratio
    frames_per_sec = sr / hop_length

    regions = []
    in_region = False
    start_idx = 0

    for i, val in enumerate(rms):
        if val >= threshold and not in_region:
            in_region = True
            start_idx = i
        elif val < threshold and in_region:
            in_region = False
            start_t = start_idx / frames_per_sec
            end_t = i / frames_per_sec
            if end_t - start_t >= 0.5:  # minimum 0.5s to count
                regions.append({"start_time": float(start_t), "end_time": float(end_t)})

    # Close final region
    if in_region:
        end_t = len(rms) / frames_per_sec
        start_t = start_idx / frames_per_sec
        if end_t - start_t >= 0.5:
            regions.append({"start_time": float(start_t), "end_time": float(end_t)})

    return regions
