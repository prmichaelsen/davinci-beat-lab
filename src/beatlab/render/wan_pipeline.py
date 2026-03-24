"""Wan2.1 + FILM pipeline orchestrator — prepares clips locally, runs rendering remotely via SSH."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

from beatlab.render.wan import chunk_section_frames, frames_to_clip


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


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
    fps: float | None = None,
    preview: bool = False,
    model: str = "v1-5-pruned-emaonly-fp16.safetensors",
    default_style: str = "artistic stylized",
    progress_callback: Callable[[str, int, int], None] | None = None,
    local_comfyui: str | None = None,
) -> str:
    """Run the full Wan2.1 + FILM pipeline.

    Phase 1 (local): Chunk sections into video clips, build render plan
    Phase 2 (remote): Upload clips + plan → SSH run remote_wan_script.py → download result
    Phase 3 (local): Move output to final location

    Args:
        video_file: Source video path.
        beat_map: Parsed beat map dict with sections.
        effect_plan: EffectPlan from AI director (optional).
        work_dir: Work directory root for caching.
        fps: Frame rate (auto-detected if None).
        preview: If True, render at 512x512.
        model: Model checkpoint name.
        default_style: Fallback style prompt.
        progress_callback: Called with (stage, completed, total).
        local_comfyui: If set, run locally against this ComfyUI URL instead of remote.

    Returns:
        Path to final assembled video.
    """
    work = Path(work_dir)
    frames_dir = work / "frames"
    wan_input_dir = work / "wan_input"
    wan_clips_dir = wan_input_dir / "clips"
    wan_output_dir = work / "wan_output"
    output_path = work / "wan_output.mp4"

    wan_input_dir.mkdir(parents=True, exist_ok=True)
    wan_clips_dir.mkdir(parents=True, exist_ok=True)
    wan_output_dir.mkdir(parents=True, exist_ok=True)

    sections = beat_map.get("sections", [])
    if not sections:
        raise ValueError("Beat map has no sections — Wan2.1 engine requires sections")

    video_fps = fps or beat_map.get("fps", 30.0)
    resolution = (512, 512) if preview else (1280, 720)

    # Build plan map: section_index → plan data
    plan_map: dict[int, object] = {}
    if effect_plan is not None:
        for sp in effect_plan.sections:
            plan_map[sp.section_index] = sp

    # ── Phase 1: Create input clips and render plan ──
    if progress_callback:
        progress_callback("prepare", 0, len(sections))

    clip_entries = []
    transition_entries = []

    for i, sec in enumerate(sections):
        start_frame = sec.get("start_frame", int(sec["start_time"] * video_fps))
        end_frame = sec.get("end_frame", int(sec["end_time"] * video_fps))
        chunks = chunk_section_frames(str(frames_dir), start_frame, end_frame, video_fps)

        sp = plan_map.get(i)
        style = (sp.style_prompt if sp and sp.style_prompt else default_style)
        denoise = (sp.wan_denoise if sp and sp.wan_denoise else DEFAULT_DENOISE.get(sec.get("type", "mid_energy"), 0.45))

        for ci, (chunk_start, chunk_end) in enumerate(chunks):
            clip_name = f"section_{i:03d}_chunk_{ci:03d}.mp4"
            clip_path = str(wan_clips_dir / clip_name)

            # Create input clip if not cached
            if not Path(clip_path).exists():
                frames_to_clip(str(frames_dir), chunk_start, chunk_end, video_fps, clip_path)

            clip_entries.append({
                "clip": clip_name,
                "section_index": i,
                "style_prompt": style,
                "denoise": denoise,
                "seed": random.randint(0, 2**32 - 1),
                "width": resolution[0],
                "height": resolution[1],
            })

        # Transition into this section (skip first)
        if i > 0:
            trans = (sp.transition_frames if sp and sp.transition_frames else DEFAULT_TRANSITION_FRAMES)
            transition_entries.append({"from_section": i - 1, "to_section": i, "frames": trans})

        if progress_callback:
            progress_callback("prepare", i + 1, len(sections))

    # Extract audio for remote muxing
    audio_path = str(wan_input_dir / "audio.wav")
    if not Path(audio_path).exists():
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_file, "-vn", "-acodec", "pcm_s16le",
             "-ar", "44100", audio_path],
            capture_output=True,
        )

    # Write render plan
    plan = {
        "clips": clip_entries,
        "transitions": transition_entries,
        "fps": video_fps,
        "model": model,
        "preview": preview,
    }
    plan_path = str(wan_input_dir / "plan.json")
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)

    if progress_callback:
        progress_callback("prepare", len(sections), len(sections))

    # ── Phase 2: Remote execution ──
    if local_comfyui:
        # Local mode — run the remote script directly as a subprocess
        import beatlab.render.remote_wan_script as rws
        script_path = rws.__file__
        _run_local(script_path, str(wan_input_dir), str(wan_output_dir), model, progress_callback)
    else:
        # Cloud mode — upload to Vast.ai, run via SSH, download results
        _run_remote(str(wan_input_dir), str(wan_output_dir), model, progress_callback)

    # ── Phase 3: Grab final output ──
    remote_output = wan_output_dir / "output.mp4"
    if remote_output.exists():
        shutil.move(str(remote_output), str(output_path))
    else:
        raise RuntimeError(f"Remote render did not produce output.mp4 in {wan_output_dir}")

    return str(output_path)


def _run_local(script_path: str, input_dir: str, output_dir: str, model: str,
               progress_callback: Callable | None) -> None:
    """Run the wan render script locally as a subprocess."""
    if progress_callback:
        progress_callback("render", 0, 1)

    proc = subprocess.Popen(
        ["python3", script_path, input_dir, output_dir, model],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in proc.stdout:
        _log(f"  [local] {line.rstrip()}")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Local wan render failed with exit code {proc.returncode}")

    if progress_callback:
        progress_callback("render", 1, 1)


def _run_remote(input_dir: str, output_dir: str, model: str,
                progress_callback: Callable | None) -> None:
    """Upload to Vast.ai instance, run via SSH, download results."""
    from beatlab.render.cloud import VastAIManager

    vast = VastAIManager()

    _log("[wan] Looking for GPU instance...")
    instance_id, reused = vast.get_or_create_instance()

    if reused:
        _log(f"[wan] Reusing running instance {instance_id}")
    else:
        _log(f"[wan] Created new instance {instance_id}, waiting for it to start...")
        vast.wait_until_ready(instance_id)

    try:
        # Get SSH info
        host, port = vast.get_ssh_info(instance_id)
        key = vast._ssh_key_arg()
        key_opt = f"-i {key} " if key else ""
        ssh_base = f"ssh {key_opt}-o StrictHostKeyChecking=no -p {port} root@{host}"

        # Upload remote script
        import beatlab.render.remote_wan_script as rws
        script_path = rws.__file__
        _log("[wan] Uploading render script...")
        vast.ssh_run(instance_id, "mkdir -p /workspace")
        subprocess.run(
            f'scp {key_opt}-o StrictHostKeyChecking=no -P {port} {script_path} root@{host}:/workspace/render_wan.py',
            shell=True, check=True, capture_output=True,
        )

        # Upload input clips + plan
        _log("[wan] Uploading clips and plan...")
        vast.ssh_run(instance_id, "rm -rf /workspace/wan_input /workspace/wan_output && mkdir -p /workspace/wan_output")
        vast.upload_files(instance_id, input_dir, "/workspace/wan_input")

        # Run remote script with streaming output
        _log("[wan] Running remote render (Wan2.1 + FILM)...")
        if progress_callback:
            progress_callback("render", 0, 1)

        ssh_cmd = f'{ssh_base} "python3 /workspace/render_wan.py /workspace/wan_input /workspace/wan_output {model}"'
        proc = subprocess.Popen(
            ssh_cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

        # Stream output and sync styled clips after each completion
        styled_local = Path(output_dir) / "styled_clips"
        styled_local.mkdir(parents=True, exist_ok=True)
        sync_cmd = f'rsync -az --ignore-existing -e "ssh {key_opt}-o StrictHostKeyChecking=no -p {port}" root@{host}:/workspace/wan_output/styled_clips/ {styled_local}/'

        for line in proc.stdout:
            _log(f"  [remote] {line.rstrip()}")
            # When a clip finishes, sync it down immediately
            if " done " in line or "(cached)" in line:
                subprocess.Popen(sync_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Remote wan render failed with exit code {proc.returncode}")

        # Final sync to get everything (transitions, final output)
        _log("[wan] Downloading output...")
        vast.download_files(instance_id, "/workspace/wan_output", output_dir)

        if progress_callback:
            progress_callback("render", 1, 1)

    except Exception as e:
        _log(f"[wan] Error: {e}")
        _log(f"[wan] Instance {instance_id} kept alive — fix and retry.")
        raise
