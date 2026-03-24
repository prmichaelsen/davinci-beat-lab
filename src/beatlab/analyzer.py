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

    # Beat tracking — BPM grid anchored to percussion
    hop_length = 512

    # Separate percussive component for beat tracking
    # This isolates kicks, snares, hi-hats from melodic content
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Use percussive onset envelope for beat detection and intensity
    perc_env = librosa.onset.onset_strength(y=y_percussive, sr=sr_out, hop_length=hop_length)
    # Full onset envelope still used for general onset detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr_out, hop_length=hop_length)

    # Estimate tempo from percussive signal (more accurate for rhythmic music)
    tempo, beat_frames = librosa.beat.beat_track(
        y=y_percussive, sr=sr_out, hop_length=hop_length, onset_envelope=perc_env,
    )
    tempo_val = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])

    # Generate a perfect metronome grid from the detected BPM
    beat_interval = 60.0 / tempo_val  # seconds per beat

    # Find the best grid anchor: strongest PERCUSSIVE onset in the first few beats
    search_window = beat_interval * 4
    perc_onset_frames = librosa.onset.onset_detect(
        y=y_percussive, sr=sr_out, hop_length=hop_length, onset_envelope=perc_env,
    )
    perc_onset_times = librosa.frames_to_time(perc_onset_frames, sr=sr_out, hop_length=hop_length)

    early_mask = perc_onset_times < search_window
    if np.any(early_mask):
        early_onsets = perc_onset_frames[early_mask]
        early_strengths = perc_env[np.clip(early_onsets, 0, len(perc_env) - 1)]
        strongest_idx = np.argmax(early_strengths)
        first_beat_time = float(perc_onset_times[early_mask][strongest_idx])
    elif len(beat_frames) > 0:
        first_beat_time = float(librosa.frames_to_time(beat_frames[0], sr=sr_out, hop_length=hop_length))
    else:
        first_beat_time = 0.0

    # Build grid: extend backward from first beat to time 0, then forward to end
    grid_times = []
    t = first_beat_time
    while t >= 0:
        grid_times.insert(0, t)
        t -= beat_interval
    t = first_beat_time + beat_interval
    while t < duration:
        grid_times.append(t)
        t += beat_interval
    beat_times = np.array(grid_times)

    # Sample PERCUSSIVE onset envelope at grid positions for intensity
    # (how hard the drums hit at each beat position, not vocals/synths)
    max_perc_idx = len(perc_env) - 1
    max_env_idx = len(onset_env) - 1
    beat_env_frames = np.clip(
        librosa.time_to_frames(beat_times, sr=sr_out, hop_length=hop_length),
        0, max_perc_idx,
    )
    beat_strengths = perc_env[beat_env_frames]

    if len(beat_strengths) > 0 and beat_strengths.max() > 0:
        normalized = (beat_strengths - beat_strengths.min()) / (
            beat_strengths.max() - beat_strengths.min()
        )
        beat_intensities = 0.2 + 0.8 * normalized  # range: 0.2 to 1.0
    else:
        beat_intensities = np.zeros_like(beat_times)

    beats = [
        {"time": float(t), "intensity": float(i)}
        for t, i in zip(beat_times, beat_intensities)
    ]

    # Onset detection (for onset data in beat map, not used for beat timing)
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
