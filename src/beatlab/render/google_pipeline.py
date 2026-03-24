"""Google AI render pipeline — Nano Banana + Veo. No GPU instance needed."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

from beatlab.render.google_video import GoogleVideoClient


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


# Default denoise/style mapping by section energy type
DEFAULT_DENOISE = {
    "low_energy": 0.35,
    "mid_energy": 0.45,
    "high_energy": 0.6,
}

DEFAULT_TRANSITION_FRAMES = 8
SECTION_CLIP_DURATION = 8  # Veo max clip length


def render_google_pipeline(
    video_file: str,
    beat_map: dict,
    effect_plan: object | None,
    work_dir: str,
    fps: float | None = None,
    default_style: str = "artistic stylized",
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> str:
    """Run the full Nano Banana + Veo pipeline.

    Phase 1: Extract keyframes (one per section) from source video
    Phase 2: Nano Banana stylizes each keyframe
    Phase 3: Veo generates video clips from styled keyframes
    Phase 4: Veo generates transitions between sections (first/last frame)
    Phase 5: Concatenate clips + transitions, mux audio

    Args:
        video_file: Source video path.
        beat_map: Parsed beat map dict with sections.
        effect_plan: EffectPlan from AI director (optional).
        work_dir: Work directory root for caching.
        fps: Frame rate.
        default_style: Fallback style prompt.
        progress_callback: Called with (stage, completed, total).

    Returns:
        Path to final assembled video.
    """
    work = Path(work_dir)
    frames_dir = work / "frames"
    styled_dir = work / "google_styled"
    clips_dir = work / "google_clips"
    transitions_dir = work / "google_transitions"
    output_path = work / "google_output.mp4"

    styled_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    transitions_dir.mkdir(parents=True, exist_ok=True)

    sections = beat_map.get("sections", [])
    if not sections:
        raise ValueError("Beat map has no sections — Google pipeline requires sections")

    video_fps = fps or beat_map.get("fps", 30.0)

    client = GoogleVideoClient()

    # Build plan map
    plan_map: dict[int, object] = {}
    if effect_plan is not None:
        for sp in effect_plan.sections:
            plan_map[sp.section_index] = sp

    total_sections = len(sections)

    # ── Phase 1: Pick a keyframe per section ──
    _log(f"Phase 1: Selecting {total_sections} keyframes...")
    keyframe_paths: list[str] = []
    for i, sec in enumerate(sections):
        start_frame = sec.get("start_frame", int(sec["start_time"] * video_fps))
        end_frame = sec.get("end_frame", int(sec["end_time"] * video_fps))
        # Pick frame 1/3 into the section (avoids transition edges)
        mid_frame = start_frame + (end_frame - start_frame) // 3
        kf_path = str(frames_dir / f"frame_{mid_frame:06d}.png")
        if not Path(kf_path).exists():
            # Fall back to first frame
            kf_path = str(frames_dir / f"frame_{start_frame:06d}.png")
        keyframe_paths.append(kf_path)

    # ── Phase 2: Nano Banana stylization ──
    _log(f"Phase 2: Stylizing {total_sections} keyframes with Nano Banana...")
    styled_paths: list[str] = []
    for i, (sec, kf_path) in enumerate(zip(sections, keyframe_paths)):
        sp = plan_map.get(i)
        style = (sp.style_prompt if sp and sp.style_prompt else default_style)

        styled_path = str(styled_dir / f"styled_{i:03d}.png")

        if Path(styled_path).exists():
            _log(f"  [{i+1}/{total_sections}] Section {i} (cached)")
            styled_paths.append(styled_path)
            continue

        _log(f"  [{i+1}/{total_sections}] Section {i}: {style[:50]}...")
        try:
            client.stylize_image(kf_path, style, styled_path)
        except Exception as e:
            _log(f"  [{i+1}/{total_sections}] FAILED: {e}")
            raise

        styled_paths.append(styled_path)

        if progress_callback:
            progress_callback("stylize", i + 1, total_sections)

    # ── Phase 3: Veo video generation per section ──
    _log(f"Phase 3: Generating {total_sections} video clips with Veo...")
    clip_paths: list[str] = []
    for i, (sec, styled_path) in enumerate(zip(sections, styled_paths)):
        sp = plan_map.get(i)
        style = (sp.style_prompt if sp and sp.style_prompt else default_style)

        sec_duration = sec.get("end_time", 0) - sec.get("start_time", 0)
        clip_duration = min(SECTION_CLIP_DURATION, max(4, int(sec_duration)))  # Veo min 4s, max 8s

        clip_path = str(clips_dir / f"clip_{i:03d}.mp4")

        if Path(clip_path).exists():
            _log(f"  [{i+1}/{total_sections}] Section {i} (cached)")
            clip_paths.append(clip_path)
            continue

        prompt = f"Smooth cinematic video in this visual style: {style}. Maintain the composition and mood."

        _log(f"  [{i+1}/{total_sections}] Section {i} ({clip_duration}s)...")
        try:
            client.generate_video_from_image(styled_path, prompt, clip_path, duration_seconds=clip_duration)
        except Exception as e:
            _log(f"  [{i+1}/{total_sections}] FAILED: {e}")
            raise

        clip_paths.append(clip_path)

        if progress_callback:
            progress_callback("veo", i + 1, total_sections)

    # ── Phase 4: Veo transitions between sections ──
    num_transitions = total_sections - 1
    _log(f"Phase 4: Generating {num_transitions} transitions with Veo...")
    transition_paths: list[str | None] = []

    for i in range(num_transitions):
        sp_next = plan_map.get(i + 1)
        trans_frames = (sp_next.transition_frames if sp_next and sp_next.transition_frames else DEFAULT_TRANSITION_FRAMES)

        # Convert frames to seconds (Veo min 4s)
        trans_seconds = max(4, int(trans_frames / video_fps + 0.5))

        trans_path = str(transitions_dir / f"trans_{i:03d}_{i+1:03d}.mp4")

        if Path(trans_path).exists():
            _log(f"  [{i+1}/{num_transitions}] Transition {i}→{i+1} (cached)")
            transition_paths.append(trans_path)
            continue

        # Extract last frame of clip A and first frame of clip B
        last_frame_a = str(transitions_dir / f"last_{i:03d}.png")
        first_frame_b = str(transitions_dir / f"first_{i+1:03d}.png")

        _extract_frame(clip_paths[i], last_frame_a, position="last")
        _extract_frame(clip_paths[i + 1], first_frame_b, position="first")

        style_a = plan_map.get(i)
        style_b = plan_map.get(i + 1)
        prompt_a = (style_a.style_prompt if style_a and style_a.style_prompt else default_style)
        prompt_b = (style_b.style_prompt if style_b and style_b.style_prompt else default_style)
        trans_prompt = f"Smooth cinematic transition morphing from '{prompt_a}' style into '{prompt_b}' style."

        _log(f"  [{i+1}/{num_transitions}] Transition {i}→{i+1} ({trans_seconds}s)...")
        try:
            client.generate_video_transition(
                last_frame_a, first_frame_b, trans_prompt, trans_path,
                duration_seconds=trans_seconds,
            )
        except Exception as e:
            _log(f"  [{i+1}/{num_transitions}] FAILED: {e}")
            raise

        transition_paths.append(trans_path)

        if progress_callback:
            progress_callback("transitions", i + 1, num_transitions)

    # ── Phase 5: Concatenate and mux audio ──
    _log("Phase 5: Assembling final video...")

    # Build concat list: clip0, trans0, clip1, trans1, clip2, ...
    concat_list = str(work / "google_concat.txt")
    with open(concat_list, "w") as f:
        for i, clip_path in enumerate(clip_paths):
            f.write(f"file '{Path(clip_path).resolve()}'\n")
            if i < len(transition_paths) and transition_paths[i]:
                f.write(f"file '{Path(transition_paths[i]).resolve()}'\n")

    # Concatenate
    concat_output = str(work / "google_concat.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", concat_output],
        check=True, capture_output=True,
    )

    # Mux audio from original video
    subprocess.run(
        ["ffmpeg", "-y",
         "-i", concat_output,
         "-i", video_file,
         "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", "-shortest",
         str(output_path)],
        check=True, capture_output=True,
    )

    _log(f"Done! Output: {output_path}")
    return str(output_path)


def _extract_frame(video_path: str, output_path: str, position: str = "first") -> None:
    """Extract first or last frame from a video clip."""
    if Path(output_path).exists():
        return

    if position == "last":
        # Get frame count, then seek to last frame
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-count_frames", "-show_entries",
             "stream=nb_read_frames", "-of", "csv=p=0", video_path],
            capture_output=True, text=True,
        )
        try:
            total_frames = int(probe.stdout.strip())
        except ValueError:
            total_frames = 1
        # Extract last frame using select filter
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-vf", f"select=eq(n\\,{max(0, total_frames - 1)})",
             "-vframes", "1", output_path],
            capture_output=True, check=True,
        )
    else:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vframes", "1", output_path],
            capture_output=True, check=True,
        )
