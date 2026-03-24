"""CLI interface for beatlab."""

from __future__ import annotations

import json
import sys
from datetime import datetime

import click


def _log(msg: str, **kwargs) -> None:
    """Print a timestamped log line to stderr."""
    ts = datetime.now().strftime("%H:%M:%S")
    click.echo(f"[{ts}] {msg}", err=True, **kwargs)


EFFECT_CHOICES = click.Choice(["zoom", "flash", "glow", "all"])
CURVE_CHOICES = click.Choice(["linear", "exponential", "logarithmic"])


@click.group()
@click.version_option(package_name="davinci-beat-lab")
def main():
    """beatlab — AI-powered beat detection and visual effects for DaVinci Resolve."""
    pass


@main.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--fps", default=30.0, type=float, help="Timeline frame rate (default: 30)")
@click.option("--output", "-o", default=None, type=click.Path(), help="Output JSON file (default: stdout)")
@click.option("--sr", default=22050, type=int, help="Sample rate for analysis (default: 22050)")
@click.option("--sections/--no-sections", default=False, help="Detect musical sections (verse/chorus/drop)")
def analyze(audio_file: str, fps: float, output: str | None, sr: int, sections: bool):
    """Analyze an audio file and produce a beat map JSON."""
    from beatlab.analyzer import analyze_audio
    from beatlab.beat_map import create_beat_map, save_beat_map

    _log(f"Analyzing: {audio_file}")
    analysis = analyze_audio(audio_file, sr=sr, detect_sections_flag=sections)
    _log(
        f"  Tempo: {analysis['tempo']:.1f} BPM | "
        f"Beats: {len(analysis['beats'])} | "
        f"Onsets: {len(analysis['onsets'])} | "
        f"Duration: {analysis['duration']:.1f}s"
    )
    if sections and "sections" in analysis:
        _log(f"  Sections: {len(analysis['sections'])} detected")

    beat_map = create_beat_map(analysis, fps=fps, source_file=audio_file)

    if output:
        save_beat_map(beat_map, output)
        _log(f"  Beat map written to: {output}")
    else:
        json.dump(beat_map, sys.stdout, indent=2)
        sys.stdout.write("\n")


@main.command(name="presets")
def list_presets():
    """List available effect presets."""
    from beatlab.presets import list_presets as _list

    click.echo("Available presets:\n")
    for p in _list():
        click.echo(f"  {p['name']:20s} {p['description']}")
        click.echo(f"  {'':20s} node={p['node']}.{p['parameter']}  curve={p['curve']}")
        click.echo()


@main.command()
@click.argument("beats_json", type=click.Path(exists=True))
@click.option("--output", "-o", default="output.setting", type=click.Path(), help="Output .setting file")
@click.option("--effect", default=None, type=EFFECT_CHOICES, help="Legacy effect type")
@click.option("--preset", default=None, type=str, help="Preset name(s), comma-separated")
@click.option("--attack", default=None, type=int, help="Override attack frames")
@click.option("--release", default=None, type=int, help="Override release frames")
@click.option("--intensity-curve", default="linear", type=CURVE_CHOICES, help="Intensity mapping curve")
@click.option("--section-mode/--no-section-mode", default=False, help="Vary effects by musical section")
@click.option("--overshoot/--no-overshoot", default=False, help="Add overshoot bounce to zoom effects")
@click.option("--ai/--no-ai", default=False, help="Use AI to select effects per section (requires ANTHROPIC_API_KEY)")
@click.option("--prompt", default=None, type=str, help="Creative direction for AI mode (e.g. 'cinematic with hard drops')")
@click.option("--describe", default=None, is_flag=False, flag_value="generate", help="Describe sections with Gemini. Pass a .md file to reuse existing descriptions.")
def generate(
    beats_json: str, output: str, effect: str | None, preset: str | None,
    attack: int | None, release: int | None, intensity_curve: str,
    section_mode: bool, overshoot: bool, ai: bool, prompt: str | None,
    describe: str | None,
):
    """Generate a Fusion .setting file from a beat map JSON."""
    from beatlab.beat_map import load_beat_map
    from beatlab.generator import generate_comp

    beat_map = load_beat_map(beats_json)
    plan = None

    if ai:
        plan = _get_ai_plan(beat_map, prompt)

    if plan:
        _log("Generating Fusion comp from AI effect plan")
        comp = generate_comp(beat_map, effect_plan=plan)
    else:
        preset_names = [p.strip() for p in preset.split(",")] if preset else None
        label = preset or effect or "zoom_pulse"
        _log(f"Generating Fusion comp: {label}")
        comp = generate_comp(
            beat_map, effect=effect, preset_names=preset_names,
            attack_frames=attack, release_frames=release,
            intensity_curve=intensity_curve,
            section_mode=section_mode, overshoot=overshoot,
        )

    comp.save(output)
    _log(f"  Written to: {output}")


@main.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--fps", default=30.0, type=float, help="Timeline frame rate (default: 30)")
@click.option("--output", "-o", default="output.setting", type=click.Path(), help="Output .setting file")
@click.option("--effect", default=None, type=EFFECT_CHOICES, help="Legacy effect type")
@click.option("--preset", default=None, type=str, help="Preset name(s), comma-separated")
@click.option("--attack", default=None, type=int, help="Override attack frames")
@click.option("--release", default=None, type=int, help="Override release frames")
@click.option("--sr", default=22050, type=int, help="Sample rate for analysis (default: 22050)")
@click.option("--beats-out", default=None, type=click.Path(), help="Also save beat map JSON")
@click.option("--intensity-curve", default="linear", type=CURVE_CHOICES, help="Intensity mapping curve")
@click.option("--section-mode/--no-section-mode", default=False, help="Vary effects by musical section")
@click.option("--overshoot/--no-overshoot", default=False, help="Add overshoot bounce to zoom effects")
@click.option("--ai/--no-ai", default=False, help="Use AI to select effects per section (requires ANTHROPIC_API_KEY)")
@click.option("--prompt", default=None, type=str, help="Creative direction for AI mode")
@click.option("--describe", default=None, is_flag=False, flag_value="generate", help="Describe sections with Gemini. Pass a .md file to reuse existing descriptions.")
def run(
    audio_file: str, fps: float, output: str, effect: str | None,
    preset: str | None, attack: int | None, release: int | None,
    sr: int, beats_out: str | None, intensity_curve: str,
    section_mode: bool, overshoot: bool, ai: bool, prompt: str | None,
    describe: str | None,
):
    """Full pipeline: audio file → beat analysis → Fusion .setting file."""
    from beatlab.analyzer import analyze_audio
    from beatlab.beat_map import create_beat_map, save_beat_map
    from beatlab.generator import generate_comp

    # AI or describe mode always needs sections
    detect_sections = section_mode or ai or (describe is not None)
    _log(f"Analyzing: {audio_file}")
    analysis = analyze_audio(audio_file, sr=sr, detect_sections_flag=detect_sections)
    _log(
        f"  Tempo: {analysis['tempo']:.1f} BPM | "
        f"Beats: {len(analysis['beats'])} | "
        f"Duration: {analysis['duration']:.1f}s"
        )
    if detect_sections and "sections" in analysis:
        _log(f"  Sections: {len(analysis['sections'])} detected")

    beat_map = create_beat_map(analysis, fps=fps, source_file=audio_file)

    if beats_out:
        save_beat_map(beat_map, beats_out)
        _log(f"  Beat map saved to: {beats_out}")

    # Audio descriptions — generate fresh or load from file
    audio_descriptions = None
    if describe and "sections" in analysis:
        if describe != "generate" and describe.endswith(".md"):
            audio_descriptions = _load_descriptions(describe, len(analysis["sections"]))
        else:
            audio_descriptions = _describe_sections(audio_file, sr, analysis["sections"])

    plan = None
    if ai:
        plan = _get_ai_plan(beat_map, prompt, audio_descriptions=audio_descriptions)

    if plan:
        _log("Generating Fusion comp from AI effect plan")
        comp = generate_comp(beat_map, effect_plan=plan)
    else:
        preset_names = [p.strip() for p in preset.split(",")] if preset else None
        label = preset or effect or "zoom_pulse"
        _log(f"Generating Fusion comp: {label}")
        comp = generate_comp(
            beat_map, effect=effect, preset_names=preset_names,
            attack_frames=attack, release_frames=release,
            intensity_curve=intensity_curve,
            section_mode=section_mode, overshoot=overshoot,
        )

    comp.save(output)
    _log(f"  Fusion comp written to: {output}")
    _log("Done! Import the .setting file into Resolve's Fusion page.")


@main.command()
@click.argument("video_file", type=click.Path(exists=True))
@click.option("--beats", default=None, type=click.Path(), help="Beat map JSON (skip audio analysis)")
@click.option("--fps", default=None, type=float, help="Override frame rate")
@click.option("--style", default="artistic stylized", type=str, help="Default SD style prompt")
@click.option("--ai/--no-ai", default=False, help="Use AI to pick styles per section")
@click.option("--prompt", default=None, type=str, help="Creative direction for AI")
@click.option("--output", "-o", default="output_styled.mp4", type=click.Path(), help="Output video file")
@click.option("--base-denoise", default=0.3, type=float, help="Base denoising strength (default: 0.3)")
@click.option("--beat-denoise", default=0.5, type=float, help="Beat denoising strength (default: 0.5)")
@click.option("--model", default="sd_xl_base_1.0.safetensors", type=str, help="SD model name")
@click.option("--local-comfyui", default=None, type=str, help="Local ComfyUI URL (e.g. http://localhost:8188)")
@click.option("--sr", default=22050, type=int, help="Sample rate for analysis")
@click.option("--dry-run/--no-dry-run", default=False, help="Show cost estimate without rendering")
@click.option("--destroy/--keep-alive", default=False, help="Destroy instance after render (default: keep alive)")
@click.option("--fresh/--resume", default=False, help="Wipe work dir and start fresh (default: resume)")
@click.option("--work-dir", default=".beatlab_work", type=str, help="Work directory for caching (default: .beatlab_work)")
@click.option("--engine", default="ebsynth", type=click.Choice(["ebsynth", "wan"]), help="Render engine (default: ebsynth)")
@click.option("--preview/--no-preview", default=False, help="Render at 512x512 for fast preview (Wan2.1 only)")
def render(
    video_file: str, beats: str | None, fps: float | None, style: str,
    ai: bool, prompt: str | None, output: str, base_denoise: float,
    beat_denoise: float, model: str, local_comfyui: str | None,
    sr: int, dry_run: bool, destroy: bool, fresh: bool, work_dir: str,
    engine: str, preview: bool,
):
    """Render AI-stylized video: extract frames → SD img2img → reassemble.

    Caches intermediate results in .beatlab_work/ for resume on failure.
    Re-run the same command to pick up where it left off.
    Use --fresh to start over.
    """
    import subprocess
    from beatlab.render.frames import (
        detect_fps, extract_audio, extract_frames,
        generate_frame_params, reassemble_video, save_frame_params,
    )
    from beatlab.render.cloud import estimate_cost
    from beatlab.render.workdir import WorkDir

    # Set up persistent work directory
    work = WorkDir(video_file, base_dir=work_dir)
    if fresh:
        work.clean()
        _log("  Cleaned work directory.")

    _log(f"Video: {video_file}")

    # Show cached state if resuming
    if not fresh and (work.has_beats() or work.has_frames()):
        _log(f"  Resuming from cached state:\n  {work.summary()}")

    # ── Step 1: Detect FPS ──
    video_fps = fps or detect_fps(video_file)
    _log(f"  FPS: {video_fps:.2f}")

    # ── Step 2: Audio analysis → beat map ──
    if beats:
        from beatlab.beat_map import load_beat_map
        beat_map = load_beat_map(beats)
        work.save_beats(beat_map)
    elif work.has_beats():
        _log("  Beats: using cached")
        beat_map = work.load_beats()
    else:
        if work.has_audio():
            _log("  Audio: using cached")
            audio_path = str(work.audio_path)
        else:
            _log("  Extracting audio...")
            extract_audio(video_file, str(work.audio_path), sr=sr)
            audio_path = str(work.audio_path)

        from beatlab.analyzer import analyze_audio
        from beatlab.beat_map import create_beat_map
        _log("  Analyzing beats and sections...")
        analysis = analyze_audio(audio_path, sr=sr, detect_sections_flag=True)
        beat_map = create_beat_map(analysis, fps=video_fps, source_file=video_file)
        work.save_beats(beat_map)

    _log(
        f"  Tempo: {beat_map['tempo']:.1f} BPM | "
        f"Beats: {len(beat_map['beats'])} | "
        f"Sections: {len(beat_map.get('sections', []))}"
        )

    # ── Step 3: AI effect plan ──
    section_styles: dict[int, str] = {}
    if ai:
        if work.has_plan() and not fresh:
            _log("  AI plan: using cached")
            plan_data = work.load_plan()
            from beatlab.ai.plan import parse_effect_plan
            plan = parse_effect_plan(json.dumps(plan_data))
        else:
            plan = _get_ai_plan(beat_map, prompt)
            if plan:
                # Cache the plan
                plan_dict = {
                    "sections": [
                        {
                            "section_index": sp.section_index,
                            "presets": sp.presets,
                            "style_prompt": sp.style_prompt,
                            "intensity_curve": sp.intensity_curve,
                            "sustained_effects": sp.sustained_effects,
                        }
                        for sp in plan.sections
                    ]
                }
                work.save_plan(plan_dict)

        if plan:
            for sp in plan.sections:
                if sp.style_prompt:
                    section_styles[sp.section_index] = sp.style_prompt

    if not section_styles:
        # Use --prompt as SD style if provided, otherwise fall back to --style
        fallback_style = prompt or style
        for i in range(len(beat_map.get("sections", []))):
            section_styles[i] = fallback_style

    # ── Step 4: Extract frames ──
    frames_dir = work.ensure_frames_dir()
    if work.has_frames():
        frame_count = work.frame_count()
        _log(f"  Frames: using {frame_count} cached")
        actual_fps = video_fps
    else:
        _log("  Extracting frames...")
        frame_count, actual_fps = extract_frames(video_file, frames_dir, fps=fps)
        _log(f"  Extracted {frame_count} frames")

    # ── Wan2.1 engine branch ──
    if engine == "wan":
        from beatlab.render.wan_pipeline import render_wan_pipeline

        comfyui_url = local_comfyui or "127.0.0.1:8188"
        host, port = comfyui_url.replace("http://", "").split(":")

        def _wan_progress(stage, done, total):
            _log(f"  [{stage}] {done}/{total}")

        _log(f"  Wan2.1 engine: {'preview 512x512' if preview else 'full 1280x720'}")
        result = render_wan_pipeline(
            video_file=video_file,
            beat_map=beat_map,
            effect_plan=plan if ai else None,
            work_dir=str(work.root),
            comfyui_host=host,
            comfyui_port=int(port),
            fps=actual_fps,
            preview=preview,
            model=model,
            default_style=prompt or style,
            progress_callback=_wan_progress,
        )

        # Move to final output
        import shutil
        shutil.move(result, output)
        work.save_status("complete", {"output": output, "engine": "wan"})
        _log(f"Done! Output: {output}")
        return

    # ── Step 5: Select keyframes (EbSynth path) ──
    from beatlab.render.keyframe_selector import select_keyframes
    keyframes_path = work.root / "keyframes.json"

    if keyframes_path.exists() and not fresh:
        _log("  Keyframes: using cached")
        with open(keyframes_path) as f:
            keyframe_list = json.load(f)
    else:
        keyframe_list = select_keyframes(
            beat_map, frame_count, actual_fps,
            base_denoise=base_denoise, beat_denoise=beat_denoise,
            section_styles=section_styles,
            default_style=prompt or style,
        )
        with open(str(keyframes_path), "w") as f:
            json.dump(keyframe_list, f, indent=2)

    _log(
        f"  Keyframes: {len(keyframe_list)} selected "
        f"({len(keyframe_list) * 100 // max(1, frame_count)}% of {frame_count} frames)"
        )

    # ── Step 6: Cost estimate (keyframes only) ──
    already_styled = work.styled_count()
    remaining_kf = max(0, len(keyframe_list) - already_styled)
    cost = estimate_cost(remaining_kf)
    _log(
        f"  SD render: {remaining_kf} keyframes"
        + f", ~{cost['estimated_hours']:.2f}h, ~${cost['estimated_cost_usd']:.2f}"
        + " + EbSynth propagation (fast, CPU)"
        )

    if dry_run:
        _log("\n  Dry run — no rendering performed.")
        return

    # ── Step 7: Render ──
    styled_dir = work.ensure_styled_dir()

    if local_comfyui:
        from beatlab.render.comfyui import ComfyUIClient
        host, port = local_comfyui.replace("http://", "").split(":")
        client = ComfyUIClient(host=host, port=int(port))

        _log(f"  Rendering via {local_comfyui}...")
        with click.progressbar(length=frame_count, label="  Rendering", file=sys.stderr) as bar:
            client.render_batch(
                frame_params, frames_dir, styled_dir,
                model=model,
                progress_callback=lambda done, total: bar.update(1),
            )
    else:
        from beatlab.render.cloud import VastAIManager
        vast = VastAIManager()

        _log("  Looking for GPU instance...")
        instance_id, reused = vast.get_or_create_instance()

        if reused:
            _log(f"  Reusing running instance {instance_id}")
        else:
            _log(f"  Created new instance {instance_id}, waiting for it to start...")
            vast.wait_until_ready(instance_id)

        try:
            # Copy keyframes.json into frames dir for upload
            import shutil
            shutil.copy2(str(keyframes_path), f"{frames_dir}/keyframes.json")

            # Upload v2 render script
            import beatlab.render.remote_script_v2 as rs
            script_path = rs.__file__
            _log("  Uploading render script (v2 — keyframe + EbSynth)...")
            vast.ssh_run(instance_id, "mkdir -p /workspace")
            host, port = vast.get_ssh_info(instance_id)
            ssh_opts = vast._ssh_opts(port)
            key = vast._ssh_key_arg()
            key_opt = f"-i {key} " if key else ""
            subprocess.run(
                f'scp {key_opt}-o StrictHostKeyChecking=no -P {port} {script_path} root@{host}:/workspace/render_v2.py',
                shell=True, check=True,
            )

            # Clean remote output from previous runs and upload frames + keyframes
            vast.ssh_run(instance_id, "rm -rf /workspace/output && mkdir -p /workspace/output")
            _log(f"  Uploading {frame_count} frames + keyframes.json...")
            vast.upload_files(instance_id, frames_dir, "/workspace/input")

            # Run v2 render script (keyframe SD + EbSynth propagation)
            _log(
                f"  Phase 1: Rendering {len(keyframe_list)} keyframes on GPU...\n"
                f"  Phase 2: EbSynth propagation to {frame_count} frames (CPU)..."
                )

            ssh_cmd = (
                f'{ssh_opts} root@{host} '
                f'"python3 /workspace/render_v2.py /workspace/input /workspace/output {model}"'
            )
            proc = subprocess.Popen(
                ssh_cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                _log(f"    {line.rstrip()}")
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"Remote render failed with exit code {proc.returncode}")

            # Download results to work dir
            _log("  Downloading styled frames...")
            vast.download_files(instance_id, "/workspace/output", styled_dir)

            if destroy:
                _log("  Destroying cloud instance...")
                vast.destroy_instance(instance_id)
            else:
                _log(
                    f"  Instance {instance_id} kept alive. "
                    f"Run 'beatlab destroy-gpu' to stop it."
                    )
        except Exception as e:
            _log(
                f"\n  Error: {e}\n"
                f"  Instance {instance_id} kept alive — fix the issue and retry.\n"
                f"  Run 'beatlab destroy-gpu' to stop it when done."
                )
            raise

    # ── Step 8: Reassemble ──
    _log("  Reassembling video...")
    reassemble_video(styled_dir, output, actual_fps, audio_source=video_file)
    work.save_status("complete", {"output": output})

    _log(f"Done! Output: {output}")
    _log(f"  Work dir cached at: {work.root} (use --fresh to redo)")


@main.command(name="destroy-gpu")
def destroy_gpu():
    """Destroy the kept-alive Vast.ai GPU instance."""
    from beatlab.render.cloud import _load_instance_state, VastAIManager

    state = _load_instance_state()
    if not state:
        _log("No kept-alive instance found.")
        return

    instance_id = state["instance_id"]
    _log(f"Destroying instance {instance_id} ({state.get('gpu_name', '?')})...")
    try:
        vast = VastAIManager()
        vast.destroy_instance(instance_id)
        _log("  Instance destroyed.")
    except Exception as e:
        _log(f"  Failed: {e}")
        from beatlab.render.cloud import _clear_instance_state
        _clear_instance_state()


def _load_descriptions(md_path: str, num_sections: int) -> list[str]:
    """Load audio descriptions from a previously generated markdown file."""
    import re

    _log(f"  Loading descriptions from: {md_path}")
    with open(md_path) as f:
        content = f.read()

    # Parse sections: ## Section N ... or ## Sections N-M ...
    # Everything between section headers is the description
    parts = re.split(r"^## ", content, flags=re.MULTILINE)
    descriptions_by_index: dict[int, str] = {}

    for part in parts[1:]:  # skip content before first ##
        lines = part.strip().split("\n")
        header = lines[0]

        # Extract section indices from header
        range_match = re.match(r"Sections? (\d+)(?:-(\d+))?", header)
        if not range_match:
            continue

        start_idx = int(range_match.group(1))
        end_idx = int(range_match.group(2)) if range_match.group(2) else start_idx

        # Description is everything after the **Time** line
        desc_lines = []
        past_time = False
        for line in lines[1:]:
            if line.startswith("**Time**"):
                past_time = True
                continue
            if past_time and line.strip():
                desc_lines.append(line.strip())
        desc = "\n".join(desc_lines)

        for i in range(start_idx, end_idx + 1):
            descriptions_by_index[i] = desc

    # Build ordered list, filling gaps
    descriptions = []
    last_desc = ""
    for i in range(num_sections):
        if i in descriptions_by_index:
            last_desc = descriptions_by_index[i]
        descriptions.append(last_desc)

    _log(f"  Loaded {len(descriptions_by_index)} unique descriptions for {num_sections} sections")
    return descriptions


def _describe_sections(audio_file: str, sr: int, sections: list[dict]):
    """Describe each section's audio content using Gemini Flash."""
    from pathlib import Path

    try:
        from beatlab.ai.audio_describer import GeminiAudioDescriber, describe_sections
        from beatlab.analyzer import load_audio
    except ImportError as e:
        raise click.ClickException(str(e))

    _log("  Connecting to Gemini Flash for audio descriptions...")
    try:
        describer = GeminiAudioDescriber()
    except ValueError as e:
        raise click.ClickException(str(e))

    y, sr_out = load_audio(audio_file, sr=sr)

    # Build markdown report as we go
    source_name = Path(audio_file).stem
    md_path = f"{source_name}_descriptions.md"
    md_lines = [
        f"# Audio Descriptions: {Path(audio_file).name}\n",
        f"Generated by beatlab using Gemini Flash\n",
        f"---\n",
    ]

    bar = None

    def on_progress(completed, total, group_indices, desc):
        nonlocal bar
        if bar is None:
            bar = click.progressbar(length=total, label="  Describing sections", file=sys.stderr)
            bar.__enter__()
        bar.update(1)

        # Append to markdown
        sec = sections[group_indices[0]]
        start = sec.get("start_time", 0)
        end = sections[group_indices[-1]].get("end_time", 0)
        sec_type = sec.get("type", "unknown")
        label = sec.get("label", "")
        section_range = f"Sections {group_indices[0]}-{group_indices[-1]}" if len(group_indices) > 1 else f"Section {group_indices[0]}"
        md_lines.append(f"## {section_range} ({label}, {sec_type})\n")
        md_lines.append(f"**Time**: {start:.1f}s - {end:.1f}s\n")
        md_lines.append(f"{desc}\n")
        md_lines.append("")

    descriptions = describe_sections(describer, y, sr_out, sections, on_progress=on_progress)

    if bar is not None:
        bar.__exit__(None, None, None)

    # Write markdown report
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    _log(f"  Descriptions saved to: {md_path}")

    return descriptions


def _get_ai_plan(beat_map: dict, user_prompt: str | None, audio_descriptions: list[str] | None = None):
    """Get an AI effect plan, handling errors."""
    try:
        from beatlab.ai.provider import AnthropicProvider
        from beatlab.ai.director import create_effect_plan
    except ImportError as e:
        raise click.ClickException(str(e))

    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise click.ClickException(
            "ANTHROPIC_API_KEY environment variable is required for --ai mode.\n"
            "Set it with: export ANTHROPIC_API_KEY=your_key_here"
        )

    _log("  Asking AI for effect plan...")
    try:
        provider = AnthropicProvider()
        plan = create_effect_plan(beat_map, provider, user_prompt=user_prompt, audio_descriptions=audio_descriptions)
        _log(f"  AI plan: {len(plan.sections)} section(s) configured")
        return plan
    except Exception as e:
        raise click.ClickException(f"AI effect plan failed: {e}")
