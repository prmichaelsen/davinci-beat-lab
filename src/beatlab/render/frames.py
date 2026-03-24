"""Frame extraction, reassembly, and per-frame parameter generation."""

from __future__ import annotations

import json
import subprocess
import shutil
from pathlib import Path


def detect_fps(video_path: str) -> float:
    """Detect the frame rate of a video using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "csv=p=0",
            video_path,
        ],
        capture_output=True, text=True,
    )
    # Output is like "30/1" or "30000/1001"
    rate_str = result.stdout.strip()
    if "/" in rate_str:
        num, den = rate_str.split("/")
        return float(num) / float(den)
    return float(rate_str)


def extract_audio(video_path: str, output_path: str, sr: int = 22050) -> None:
    """Extract audio from video as WAV for analysis."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", str(sr),
            output_path,
        ],
        capture_output=True, check=True,
    )


def extract_frames(
    video_path: str, output_dir: str, fps: float | None = None,
) -> tuple[int, float]:
    """Extract frames from video using ffmpeg.

    Args:
        video_path: Path to input video.
        output_dir: Directory to write frame PNGs.
        fps: Target frame rate. None = use source fps.

    Returns:
        (frame_count, fps) tuple.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    detected_fps = detect_fps(video_path)
    target_fps = fps or detected_fps

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
    ]
    if fps:
        cmd += ["-vf", f"fps={fps}"]
    cmd += [
        "-qscale:v", "2",
        f"{output_dir}/frame_%06d.png",
    ]

    subprocess.run(cmd, capture_output=True, check=True)

    frame_count = len(list(Path(output_dir).glob("frame_*.png")))
    return frame_count, target_fps


def reassemble_video(
    frames_dir: str,
    output_path: str,
    fps: float,
    audio_source: str | None = None,
) -> None:
    """Reassemble frames into video, optionally with original audio.

    Args:
        frames_dir: Directory containing frame_NNNNNN.png files.
        output_path: Output video file path.
        fps: Frame rate for output video.
        audio_source: Path to audio file or original video (for audio track).
    """
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", f"{frames_dir}/frame_%06d.png",
    ]

    if audio_source:
        cmd += ["-i", audio_source, "-c:a", "aac", "-shortest"]

    cmd += [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_path,
    ]

    subprocess.run(cmd, capture_output=True, check=True)


def generate_frame_params(
    beat_map: dict,
    total_frames: int,
    fps: float,
    base_denoise: float = 0.3,
    beat_denoise: float = 0.5,
    section_styles: dict[int, str] | None = None,
    default_style: str = "artistic stylized",
    seed: int = 42,
) -> list[dict]:
    """Generate per-frame SD parameters from beat map.

    Args:
        beat_map: Parsed beat map dict.
        total_frames: Total number of extracted frames.
        fps: Frame rate.
        base_denoise: Denoising strength between beats.
        beat_denoise: Peak denoising on beat frames.
        section_styles: Map of section_index → SD style prompt.
        default_style: Fallback style prompt.
        seed: Random seed for SD (consistent = temporal coherence).

    Returns:
        List of dicts: {frame, denoise, prompt, seed}
    """
    beats = beat_map.get("beats", [])
    sections = beat_map.get("sections", [])

    # Build a frame → beat intensity lookup
    beat_frames: dict[int, float] = {}
    for b in beats:
        frame = b.get("frame", round(b["time"] * fps))
        intensity = b.get("intensity", 1.0)
        beat_frames[frame] = intensity

    # Build a frame → section index lookup
    def _section_for_frame(f: int) -> int | None:
        t = f / fps
        for i, sec in enumerate(sections):
            if sec.get("start_time", 0) <= t < sec.get("end_time", 0):
                return i
        return None

    params = []
    for f in range(1, total_frames + 1):  # frames are 1-indexed from ffmpeg
        # Denoising: pulse on beats
        denoise = base_denoise
        # Check nearby frames for beat (within 1 frame tolerance)
        for offset in range(-1, 2):
            if (f + offset) in beat_frames:
                intensity = beat_frames[f + offset]
                beat_d = base_denoise + (beat_denoise - base_denoise) * intensity
                denoise = max(denoise, beat_d)
                break

        # Style prompt from section
        sec_idx = _section_for_frame(f)
        if section_styles and sec_idx is not None and sec_idx in section_styles:
            prompt = section_styles[sec_idx]
        else:
            prompt = default_style

        params.append({
            "frame": f,
            "denoise": round(denoise, 3),
            "prompt": prompt,
            "seed": seed,
        })

    return params


def save_frame_params(params: list[dict], output_path: str) -> None:
    """Save frame parameters to JSON for the render pipeline."""
    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)


def load_frame_params(path: str) -> list[dict]:
    """Load frame parameters from JSON."""
    with open(path) as f:
        return json.load(f)
