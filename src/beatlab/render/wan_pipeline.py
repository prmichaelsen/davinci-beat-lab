"""Wan2.1 + FILM end-to-end pipeline: section chunking → Wan2.1 v2v → FILM transitions → reassembly."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

from beatlab.render.film import FILMInterpolator, generate_transition
from beatlab.render.wan import Wan21Client, chunk_section_frames, frames_to_clip


# Default denoise mapping by section energy type
DEFAULT_DENOISE = {
    "low_energy": 0.35,
    "mid_energy": 0.45,
    "high_energy": 0.6,
}

DEFAULT_TRANSITION_FRAMES = 8


def render_wan_pipeline(
    video_file: str,
    beat_map: dict,
    effect_plan: object | None,
    work_dir: str,
    comfyui_host: str = "127.0.0.1",
    comfyui_port: int = 8188,
    fps: float | None = None,
    preview: bool = False,
    model: str = "wan2.1_v2v_720p.safetensors",
    default_style: str = "artistic stylized",
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> str:
    """Run the full Wan2.1 + FILM pipeline.

    Args:
        video_file: Source video path.
        beat_map: Parsed beat map dict with sections.
        effect_plan: EffectPlan from AI director (optional).
        work_dir: Work directory root for caching.
        comfyui_host: ComfyUI host.
        comfyui_port: ComfyUI port.
        fps: Frame rate (auto-detected if None).
        preview: If True, render at 512x512.
        model: Wan2.1 model checkpoint name.
        default_style: Fallback style prompt.
        progress_callback: Called with (stage, completed, total).

    Returns:
        Path to final assembled video.
    """
    work = Path(work_dir)
    frames_dir = work / "frames"
    wan_clips_dir = work / "wan_clips"
    transitions_dir = work / "transitions"
    styled_dir = work / "styled"
    output_path = work / "wan_output.mp4"

    wan_clips_dir.mkdir(parents=True, exist_ok=True)
    transitions_dir.mkdir(parents=True, exist_ok=True)
    styled_dir.mkdir(parents=True, exist_ok=True)

    sections = beat_map.get("sections", [])
    if not sections:
        raise ValueError("Beat map has no sections — Wan2.1 engine requires sections")

    video_fps = fps or beat_map.get("fps", 30.0)
    resolution = (512, 512) if preview else (1280, 720)

    client = Wan21Client(host=comfyui_host, port=comfyui_port)
    interpolator = FILMInterpolator()

    # Build plan map: section_index → plan data
    plan_map: dict[int, object] = {}
    if effect_plan is not None:
        for sp in effect_plan.sections:
            plan_map[sp.section_index] = sp

    # ── Phase 1: Chunk sections and render through Wan2.1 ──
    all_section_outputs: list[list[str]] = []  # per-section list of output clip paths
    total_chunks = 0

    section_chunks: list[list[tuple[int, int]]] = []
    for i, sec in enumerate(sections):
        start_frame = sec.get("start_frame", int(sec["start_time"] * video_fps))
        end_frame = sec.get("end_frame", int(sec["end_time"] * video_fps))
        chunks = chunk_section_frames(str(frames_dir), start_frame, end_frame, video_fps)
        section_chunks.append(chunks)
        total_chunks += len(chunks)

    rendered = 0
    for i, (sec, chunks) in enumerate(zip(sections, section_chunks)):
        sp = plan_map.get(i)
        style = (sp.style_prompt if sp and sp.style_prompt else default_style)
        denoise = (sp.wan_denoise if sp and sp.wan_denoise else DEFAULT_DENOISE.get(sec.get("type", "mid_energy"), 0.45))

        section_clip_paths: list[str] = []

        for ci, (chunk_start, chunk_end) in enumerate(chunks):
            clip_name = f"section_{i:03d}_chunk_{ci:03d}.mp4"
            output_clip = str(wan_clips_dir / clip_name)

            # Resume: skip if already rendered
            if Path(output_clip).exists():
                section_clip_paths.append(output_clip)
                rendered += 1
                if progress_callback:
                    progress_callback("wan", rendered, total_chunks)
                continue

            # Create input clip from frames
            input_clip = str(wan_clips_dir / f"input_{i:03d}_{ci:03d}.mp4")
            frames_to_clip(str(frames_dir), chunk_start, chunk_end, video_fps, input_clip)

            # Render through Wan2.1
            output_info = client.render_clip(
                video_path=input_clip,
                style_prompt=style,
                denoise=denoise,
                resolution=resolution,
                model=model,
            )

            # Download result
            client.download_output_video(
                filename=output_info["filename"],
                subfolder=output_info.get("subfolder", ""),
                output_path=output_clip,
            )

            # Clean up input clip
            Path(input_clip).unlink(missing_ok=True)

            section_clip_paths.append(output_clip)
            rendered += 1
            if progress_callback:
                progress_callback("wan", rendered, total_chunks)

        all_section_outputs.append(section_clip_paths)

    # ── Phase 2: FILM transitions ──
    # First, extract frames from each Wan2.1 clip for FILM processing
    all_section_frames: list[list[str]] = []

    for i, clip_paths in enumerate(all_section_outputs):
        section_frame_dir = str(styled_dir / f"section_{i:03d}")
        Path(section_frame_dir).mkdir(parents=True, exist_ok=True)

        section_frames: list[str] = []
        frame_offset = 0

        for clip_path in clip_paths:
            # Extract frames from Wan2.1 output clip
            clip_frame_dir = f"{section_frame_dir}/clip_frames"
            Path(clip_frame_dir).mkdir(parents=True, exist_ok=True)

            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", clip_path,
                    f"{clip_frame_dir}/frame_%06d.png",
                ],
                check=True, capture_output=True,
            )

            clip_frames = sorted(Path(clip_frame_dir).glob("frame_*.png"))
            # Rename to global section frame numbering
            for j, cf in enumerate(clip_frames):
                dst = f"{section_frame_dir}/frame_{frame_offset + j:06d}.png"
                cf.rename(dst)
                section_frames.append(dst)
            frame_offset += len(clip_frames)

            # Clean up temp clip frames dir
            Path(clip_frame_dir).rmdir()

        all_section_frames.append(section_frames)

    # Build transition frame counts per boundary
    transition_counts: list[int] = []
    for i in range(len(sections) - 1):
        sp = plan_map.get(i + 1)  # transition INTO next section
        trans = (sp.transition_frames if sp and sp.transition_frames else DEFAULT_TRANSITION_FRAMES)
        transition_counts.append(trans)

    if progress_callback:
        progress_callback("film", 0, len(transition_counts))

    # Generate FILM transitions between sections
    from beatlab.render.film import assemble_with_transitions
    final_frames = assemble_with_transitions(
        section_clips=all_section_frames,
        transition_frames_per_boundary=transition_counts,
        interpolator=interpolator,
        work_dir=str(transitions_dir),
        window_size=3,
    )

    if progress_callback:
        progress_callback("film", len(transition_counts), len(transition_counts))

    # ── Phase 3: Reassemble final video ──
    final_frames_dir = str(styled_dir / "final")
    Path(final_frames_dir).mkdir(parents=True, exist_ok=True)

    # Copy/rename all final frames sequentially
    for i, frame_path in enumerate(final_frames):
        dst = f"{final_frames_dir}/frame_{i:06d}.png"
        if frame_path != dst:
            import shutil
            shutil.copy2(frame_path, dst)

    # Reassemble with audio from original
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(video_fps),
            "-i", f"{final_frames_dir}/frame_%06d.png",
            "-i", video_file,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            str(output_path),
        ],
        check=True, capture_output=True,
    )

    return str(output_path)
