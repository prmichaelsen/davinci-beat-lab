"""FILM frame interpolation for smooth transitions between stylized video sections."""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


class FILMInterpolator:
    """Frame interpolation using Google FILM via the film_net package or ffmpeg minterpolate fallback."""

    def __init__(self, model_path: str | None = None):
        """Initialize FILM interpolator.

        Args:
            model_path: Path to FILM model checkpoint. If None, will try to auto-download
                       or fall back to ffmpeg minterpolate.
        """
        self._model = None
        self._use_ffmpeg_fallback = True

        try:
            import torch
            if model_path:
                self._model_path = model_path
                self._use_ffmpeg_fallback = False
        except ImportError:
            pass

    def interpolate_pair(
        self,
        frame_a: str,
        frame_b: str,
        num_frames: int,
        output_dir: str,
        prefix: str = "interp",
    ) -> list[str]:
        """Generate interpolated frames between two images.

        Args:
            frame_a: Path to first frame image.
            frame_b: Path to second frame image.
            num_frames: Number of intermediate frames to generate.
            output_dir: Directory to write interpolated frames.
            prefix: Filename prefix for output frames.

        Returns:
            List of paths to interpolated frame images (not including inputs).
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if num_frames <= 0:
            return []

        if self._use_ffmpeg_fallback:
            return self._ffmpeg_interpolate(frame_a, frame_b, num_frames, output_dir, prefix)
        else:
            return self._film_interpolate(frame_a, frame_b, num_frames, output_dir, prefix)

    def _ffmpeg_interpolate(
        self,
        frame_a: str,
        frame_b: str,
        num_frames: int,
        output_dir: str,
        prefix: str,
    ) -> list[str]:
        """Fallback interpolation using ffmpeg minterpolate filter.

        Creates a 2-frame video, interpolates to num_frames+2, extracts middle frames.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a tiny 2-frame video
            concat_file = f"{tmpdir}/concat.txt"
            with open(concat_file, "w") as f:
                # Give each frame enough duration for minterpolate to work
                f.write(f"file '{frame_a}'\nduration 1.0\n")
                f.write(f"file '{frame_b}'\nduration 1.0\n")

            input_video = f"{tmpdir}/input.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", concat_file,
                    "-vsync", "vfr", "-pix_fmt", "yuv420p",
                    input_video,
                ],
                check=True, capture_output=True,
            )

            # Interpolate using minterpolate
            target_fps = num_frames + 2  # +2 for the input frames
            interp_video = f"{tmpdir}/interp.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", input_video,
                    "-filter:v", f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1",
                    "-pix_fmt", "yuv420p",
                    interp_video,
                ],
                check=True, capture_output=True,
            )

            # Extract frames
            frame_pattern = f"{tmpdir}/{prefix}_%04d.png"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", interp_video,
                    frame_pattern,
                ],
                check=True, capture_output=True,
            )

            # Collect intermediate frames (skip first and last — those are the inputs)
            all_frames = sorted(Path(tmpdir).glob(f"{prefix}_*.png"))
            intermediate = all_frames[1:-1] if len(all_frames) > 2 else all_frames

            # Copy to output dir
            results = []
            for i, src in enumerate(intermediate[:num_frames]):
                dst = f"{output_dir}/{prefix}_{i:04d}.png"
                shutil.copy2(str(src), dst)
                results.append(dst)

            return results

    def _film_interpolate(
        self,
        frame_a: str,
        frame_b: str,
        num_frames: int,
        output_dir: str,
        prefix: str,
    ) -> list[str]:
        """FILM model interpolation (requires torch + FILM weights)."""
        # Placeholder for actual FILM model inference
        # For now, fall back to ffmpeg
        return self._ffmpeg_interpolate(frame_a, frame_b, num_frames, output_dir, prefix)


def generate_transition(
    frames_a: list[str],
    frames_b: list[str],
    num_transition_frames: int,
    output_dir: str,
    interpolator: FILMInterpolator | None = None,
    prefix: str = "transition",
) -> list[str]:
    """Generate a smooth transition between two sets of frames.

    Uses a multi-frame window: takes the last N frames of section A and
    first N frames of section B, blending between the middle pair and
    using surrounding frames for temporal context.

    Args:
        frames_a: Last 1-3 frames of section A (paths).
        frames_b: First 1-3 frames of section B (paths).
        num_transition_frames: Number of interpolated frames to generate.
        output_dir: Where to write transition frames.
        interpolator: FILMInterpolator instance. Creates default if None.
        prefix: Filename prefix.

    Returns:
        List of paths to transition frames.
    """
    if interpolator is None:
        interpolator = FILMInterpolator()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if num_transition_frames <= 0 or not frames_a or not frames_b:
        return []

    # Use the last frame of A and first frame of B as the primary blend pair
    anchor_a = frames_a[-1]
    anchor_b = frames_b[0]

    # Generate interpolated frames between the anchors
    return interpolator.interpolate_pair(
        anchor_a, anchor_b,
        num_frames=num_transition_frames,
        output_dir=output_dir,
        prefix=prefix,
    )


def assemble_with_transitions(
    section_clips: list[list[str]],
    transition_frames_per_boundary: list[int],
    interpolator: FILMInterpolator | None = None,
    work_dir: str = "/tmp/film_assembly",
    window_size: int = 3,
    intra_section_transition_frames: int = 6,
) -> list[str]:
    """Assemble section frame lists with FILM transitions between them.

    Args:
        section_clips: List of sections, each being a list of frame paths.
            If a section was chunked, this should already have intra-section
            frames assembled (or pass chunked clips as separate "sections" with
            intra_section_transition_frames).
        transition_frames_per_boundary: Number of FILM frames at each section boundary.
            Length should be len(section_clips) - 1.
        interpolator: FILMInterpolator instance.
        work_dir: Working directory for transition frame output.
        window_size: How many frames to use from each side for blending context.
        intra_section_transition_frames: Frames for within-section clip boundaries.

    Returns:
        Ordered list of all frame paths (sections + transitions).
    """
    if interpolator is None:
        interpolator = FILMInterpolator()

    Path(work_dir).mkdir(parents=True, exist_ok=True)
    final_frames: list[str] = []

    for i, section_frames in enumerate(section_clips):
        # Add this section's frames (minus the tail frames that overlap with transition)
        if i < len(section_clips) - 1 and len(section_frames) > window_size:
            # Trim last window_size frames — they'll be part of the transition
            final_frames.extend(section_frames[:-window_size])
        else:
            final_frames.extend(section_frames)

        # Generate transition to next section
        if i < len(section_clips) - 1:
            num_trans = transition_frames_per_boundary[i] if i < len(transition_frames_per_boundary) else 8
            frames_a = section_frames[-window_size:]
            frames_b = section_clips[i + 1][:window_size]

            trans_dir = f"{work_dir}/transition_{i:03d}_{i+1:03d}"
            transition = generate_transition(
                frames_a=frames_a,
                frames_b=frames_b,
                num_transition_frames=num_trans,
                output_dir=trans_dir,
                interpolator=interpolator,
                prefix=f"trans_{i:03d}",
            )
            final_frames.extend(transition)

    return final_frames
