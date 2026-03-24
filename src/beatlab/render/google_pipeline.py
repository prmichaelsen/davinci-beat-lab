"""Google AI render pipeline — Nano Banana (stylize) + Veo (video between stills). No GPU needed."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

from beatlab.render.google_video import GoogleVideoClient


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def render_google_pipeline(
    video_file: str,
    beat_map: dict,
    effect_plan: object | None,
    work_dir: str,
    fps: float | None = None,
    default_style: str = "artistic stylized",
    progress_callback: Callable[[str, int, int], None] | None = None,
    vertex: bool = False,
    audio_descriptions: list[str] | None = None,
    motion_prompt: str | None = None,
) -> str:
    """Run the full Nano Banana + Veo pipeline.

    Phase 1: Extract keyframes (one per section) from source video
    Phase 2: Nano Banana stylizes each keyframe
    Phase 3: Veo generates video transitions between consecutive styled keyframes
    Phase 4: Concatenate all transition clips, mux audio

    Each Veo clip morphs from styled_keyframe[i] → styled_keyframe[i+1],
    so every clip boundary is seamless — no stitching mismatch.

    Returns:
        Path to final assembled video.
    """
    work = Path(work_dir)
    frames_dir = work / "frames"
    styled_dir = work / "google_styled"
    segments_dir = work / "google_segments"
    output_path = work / "google_output.mp4"

    styled_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)

    sections = beat_map.get("sections", [])
    if not sections:
        raise ValueError("Beat map has no sections — Google pipeline requires sections")

    video_fps = fps or beat_map.get("fps", 30.0)

    client = GoogleVideoClient(vertex=vertex)

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
        # Pick frame 1/3 into the section
        mid_frame = start_frame + (end_frame - start_frame) // 3
        kf_path = str(frames_dir / f"frame_{mid_frame:06d}.png")
        if not Path(kf_path).exists():
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

        _log(f"  [{i+1}/{total_sections}] Section {i}: {style[:60]}...")
        try:
            client.stylize_image(kf_path, style, styled_path)
        except Exception as e:
            _log(f"  [{i+1}/{total_sections}] FAILED: {e}")
            raise

        styled_paths.append(styled_path)

        if progress_callback:
            progress_callback("stylize", i + 1, total_sections)

    # ── Phase 3: Veo segments between consecutive styled keyframes ──
    num_segments = total_sections - 1
    _log(f"Phase 3: Generating {num_segments} video segments with Veo (still→still)...")
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

        # Build prompt describing the visual journey between sections
        prompt_parts = [f"Cinematic video transitioning from {style_a} ({label_a}) into {style_b} ({label_b})."]
        if motion_prompt:
            prompt_parts.append(f"Camera and motion: {motion_prompt}.")
        else:
            prompt_parts.append("Smooth, flowing motion. The visual style gradually transforms.")

        if audio_descriptions:
            desc_a = audio_descriptions[i] if i < len(audio_descriptions) else ""
            desc_b = audio_descriptions[i + 1] if i + 1 < len(audio_descriptions) else ""
            if desc_a:
                prompt_parts.append(f"The music starts with: {desc_a[:200]}")
            if desc_b:
                prompt_parts.append(f"And transitions into: {desc_b[:200]}")

        prompt = " ".join(prompt_parts)

        _log(f"  [{i+1}/{num_segments}] Segment {i}→{i+1}: {label_a}→{label_b} (8s)...")
        try:
            client.generate_video_transition(
                styled_paths[i], styled_paths[i + 1], prompt, seg_path,
                duration_seconds=8,
            )
        except Exception as e:
            _log(f"  [{i+1}/{num_segments}] FAILED: {e}")
            raise

        segment_paths.append(seg_path)

        if progress_callback:
            progress_callback("veo", i + 1, num_segments)

    # ── Phase 4: Concatenate with crossfade and mux audio ──
    _log("Phase 4: Assembling with 8-frame crossfades...")

    concat_output = str(work / "google_concat.mp4")
    from beatlab.render.crossfade import concat_with_crossfade
    concat_with_crossfade(segment_paths, concat_output, crossfade_frames=8, fps=video_fps)

    # Mux audio from original video
    muxed_output = str(work / "google_muxed.mp4")
    subprocess.run(
        ["ffmpeg", "-y",
         "-i", concat_output,
         "-i", video_file,
         "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", "-shortest",
         muxed_output],
        check=True, capture_output=True,
    )

    # ── Phase 5: Apply beat-synced effects ──
    _log("Phase 5: Applying beat-synced effects (zoom, shake, flash, color)...")
    from beatlab.render.effects import apply_effects

    apply_effects(
        video_path=muxed_output,
        output_path=str(output_path),
        beat_map=beat_map,
        effect_plan=effect_plan,
        fps=video_fps,
    )

    _log(f"Done! Output: {output_path}")
    return str(output_path)
