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


def detect_sections(
    y: np.ndarray, sr: int, hop_length: int = 512, segment_duration: float = 4.0,
) -> list[dict]:
    """Detect musical sections using RMS energy thresholding.

    Divides audio into fixed-length segments, computes average RMS energy,
    and classifies each as low_energy, mid_energy, or high_energy.

    Returns list of dicts: {start_time, end_time, type, label}
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
        segments.append({"start_time": start_time, "end_time": end_time, "energy": avg_energy})

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
            # Extend previous section
            classified[-1]["end_time"] = seg["end_time"]
        else:
            classified.append({
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "type": stype,
                "label": LABELS[stype],
            })

    return classified


def analyze_audio(path: str, sr: int = 22050, detect_sections_flag: bool = False) -> dict:
    """Analyze an audio file and return beat data.

    Returns a dict with:
        tempo: float — estimated BPM
        duration: float — audio duration in seconds
        sample_rate: int
        beats: list of {time: float, intensity: float}
        onsets: list of {time: float, strength: float}
        sections: list of {start_time, end_time, type, label} (if detect_sections_flag)
    """
    y, sr_out = load_audio(path, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr_out)

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr_out)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr_out)

    # Onset envelope for beat intensity
    onset_env = librosa.onset.onset_strength(y=y, sr=sr_out)

    # Sample onset envelope at beat positions and normalize
    beat_strengths = onset_env[beat_frames] if len(beat_frames) > 0 else np.array([])
    if len(beat_strengths) > 0 and beat_strengths.max() > 0:
        beat_intensities = (beat_strengths - beat_strengths.min()) / (
            beat_strengths.max() - beat_strengths.min()
        )
    else:
        beat_intensities = beat_strengths

    beats = [
        {"time": float(t), "intensity": float(i)}
        for t, i in zip(beat_times, beat_intensities)
    ]

    # Onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr_out)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr_out)
    onset_strengths = onset_env[onset_frames] if len(onset_frames) > 0 else np.array([])
    if len(onset_strengths) > 0 and onset_strengths.max() > 0:
        onset_strengths_norm = onset_strengths / onset_strengths.max()
    else:
        onset_strengths_norm = onset_strengths

    onsets = [
        {"time": float(t), "strength": float(s)}
        for t, s in zip(onset_times, onset_strengths_norm)
    ]

    # tempo may be an ndarray with one element in newer librosa
    tempo_val = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])

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
