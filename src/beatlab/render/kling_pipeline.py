"""Kling 3.0 render pipeline — Nano Banana (stylize) + Kling (video between stills). No GPU needed."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

from beatlab.render.google_video import GoogleVideoClient
from beatlab.render.kling_video import KlingClient


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def render_kling_pipeline(
    video_file: str,
    beat_map: dict,
    effect_plan: object | None,
    work_dir: str,
    fps: float | None = None,
    default_style: str = "artistic stylized",
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> str:
    """Run the full Nano Banana + Kling 3.0 pipeline.

    Phase 1: Extract keyframes (one per section) from source video
    Phase 2: Nano Banana stylizes each keyframe (via Google API)
    Phase 3: Kling generates video segments between consecutive styled keyframes
    Phase 4: Concatenate all segments, mux audio

    Returns:
        Path to final assembled video.
    """
    work = Path(work_dir)
    frames_dir = work / "frames"
    styled_dir = work / "google_styled"  # Reuse Nano Banana cache from google engine
    segments_dir = work / "kling_segments"
    output_path = work / "kling_output.mp4"

    styled_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)

    sections = beat_map.get("sections", [])
    if not sections:
        raise ValueError("Beat map has no sections — Kling pipeline requires sections")

    video_fps = fps or beat_map.get("fps", 30.0)

    google_client = GoogleVideoClient()
    kling_client = KlingClient()

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
        mid_frame = start_frame + (end_frame - start_frame) // 3
        kf_path = str(frames_dir / f"frame_{mid_frame:06d}.png")
        if not Path(kf_path).exists():
            kf_path = str(frames_dir / f"frame_{start_frame:06d}.png")
        keyframe_paths.append(kf_path)

    # ── Phase 2: Nano Banana stylization (reuses google_styled cache) ──
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

        full_style = style
        if default_style and default_style != "artistic stylized":
            full_style = f"{default_style}. {style}"

        _log(f"  [{i+1}/{total_sections}] Section {i}: {full_style[:60]}...")
        try:
            google_client.stylize_image(kf_path, full_style, styled_path)
        except Exception as e:
            _log(f"  [{i+1}/{total_sections}] FAILED: {e}")
            raise

        styled_paths.append(styled_path)

        if progress_callback:
            progress_callback("stylize", i + 1, total_sections)

    # ── Phase 3: Kling segments between consecutive styled keyframes ──
    num_segments = total_sections - 1
    _log(f"Phase 3: Generating {num_segments} video segments with Kling 3.0 (still→still)...")
    segment_paths: list[str] = []

    for i in range(num_segments):
        seg_path = str(segments_dir / f"segment_{i:03d}_{i+1:03d}.mp4")

        if Path(seg_path).exists():
            _log(f"  [{i+1}/{num_segments}] Segment {i}→{i+1} (cached)")
            segment_paths.append(seg_path)
            continue

        sp_a = plan_map.get(i)
        sp_b = plan_map.get(i + 1)
        style_a = (sp_a.style_prompt if sp_a and sp_a.style_prompt else default_style)
        style_b = (sp_b.style_prompt if sp_b and sp_b.style_prompt else default_style)

        sec_a = sections[i]
        sec_b = sections[i + 1]
        label_a = sec_a.get("label", "")
        label_b = sec_b.get("label", "")

        prompt_parts = []
        if default_style and default_style != "artistic stylized":
            prompt_parts.append(f"{default_style}.")
        prompt_parts.append(f"Cinematic video transitioning from {style_a} ({label_a}) into {style_b} ({label_b}).")
        prompt_parts.append("Smooth, flowing motion. The visual style gradually transforms.")
        prompt = " ".join(prompt_parts)

        # Match duration to section length (Kling supports 5 or 10)
        sec_duration = sec_b.get("start_time", 0) - sec_a.get("start_time", 0)
        duration = 5 if sec_duration <= 7 else 10

        _log(f"  [{i+1}/{num_segments}] Segment {i}→{i+1}: {label_a}→{label_b} ({duration}s, span={sec_duration:.1f}s)...")
        try:
            kling_client.generate_segment(
                styled_paths[i], styled_paths[i + 1], prompt, seg_path,
                duration=duration,
            )
        except Exception as e:
            _log(f"  [{i+1}/{num_segments}] FAILED: {e}")
            raise

        segment_paths.append(seg_path)

        if progress_callback:
            progress_callback("kling", i + 1, num_segments)

    # ── Phase 4: Concatenate and mux audio ──
    _log("Phase 4: Assembling final video...")

    concat_list = str(work / "kling_concat.txt")
    with open(concat_list, "w") as f:
        for seg_path in segment_paths:
            f.write(f"file '{Path(seg_path).resolve()}'\n")

    concat_output = str(work / "kling_concat.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", concat_output],
        check=True, capture_output=True,
    )

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
