"""CLI interface for beatlab."""

from __future__ import annotations

import json
import sys

import click


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

    click.echo(f"Analyzing: {audio_file}", err=True)
    analysis = analyze_audio(audio_file, sr=sr, detect_sections_flag=sections)
    click.echo(
        f"  Tempo: {analysis['tempo']:.1f} BPM | "
        f"Beats: {len(analysis['beats'])} | "
        f"Onsets: {len(analysis['onsets'])} | "
        f"Duration: {analysis['duration']:.1f}s",
        err=True,
    )
    if sections and "sections" in analysis:
        click.echo(f"  Sections: {len(analysis['sections'])} detected", err=True)

    beat_map = create_beat_map(analysis, fps=fps, source_file=audio_file)

    if output:
        save_beat_map(beat_map, output)
        click.echo(f"  Beat map written to: {output}", err=True)
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
        click.echo("Generating Fusion comp from AI effect plan", err=True)
        comp = generate_comp(beat_map, effect_plan=plan)
    else:
        preset_names = [p.strip() for p in preset.split(",")] if preset else None
        label = preset or effect or "zoom_pulse"
        click.echo(f"Generating Fusion comp: {label}", err=True)
        comp = generate_comp(
            beat_map, effect=effect, preset_names=preset_names,
            attack_frames=attack, release_frames=release,
            intensity_curve=intensity_curve,
            section_mode=section_mode, overshoot=overshoot,
        )

    comp.save(output)
    click.echo(f"  Written to: {output}", err=True)


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
    click.echo(f"Analyzing: {audio_file}", err=True)
    analysis = analyze_audio(audio_file, sr=sr, detect_sections_flag=detect_sections)
    click.echo(
        f"  Tempo: {analysis['tempo']:.1f} BPM | "
        f"Beats: {len(analysis['beats'])} | "
        f"Duration: {analysis['duration']:.1f}s",
        err=True,
    )
    if detect_sections and "sections" in analysis:
        click.echo(f"  Sections: {len(analysis['sections'])} detected", err=True)

    beat_map = create_beat_map(analysis, fps=fps, source_file=audio_file)

    if beats_out:
        save_beat_map(beat_map, beats_out)
        click.echo(f"  Beat map saved to: {beats_out}", err=True)

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
        click.echo("Generating Fusion comp from AI effect plan", err=True)
        comp = generate_comp(beat_map, effect_plan=plan)
    else:
        preset_names = [p.strip() for p in preset.split(",")] if preset else None
        label = preset or effect or "zoom_pulse"
        click.echo(f"Generating Fusion comp: {label}", err=True)
        comp = generate_comp(
            beat_map, effect=effect, preset_names=preset_names,
            attack_frames=attack, release_frames=release,
            intensity_curve=intensity_curve,
            section_mode=section_mode, overshoot=overshoot,
        )

    comp.save(output)
    click.echo(f"  Fusion comp written to: {output}", err=True)
    click.echo("Done! Import the .setting file into Resolve's Fusion page.", err=True)


@main.command()
@click.argument("video_file", type=click.Path(exists=True))
@click.option("--beats", default=None, type=click.Path(), help="Beat map JSON (analyzes video audio if not provided)")
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
@click.option("--destroy/--keep-alive", default=False, help="Destroy instance after render (default: keep alive for reuse)")
def render(
    video_file: str, beats: str | None, fps: float | None, style: str,
    ai: bool, prompt: str | None, output: str, base_denoise: float,
    beat_denoise: float, model: str, local_comfyui: str | None,
    sr: int, dry_run: bool, destroy: bool,
):
    """Render AI-stylized video: extract frames → SD img2img → reassemble."""
    import tempfile
    from beatlab.render.frames import (
        detect_fps, extract_audio, extract_frames,
        generate_frame_params, reassemble_video, save_frame_params,
    )
    from beatlab.render.cloud import estimate_cost

    # 1. Detect FPS
    video_fps = fps or detect_fps(video_file)
    click.echo(f"Video: {video_file} ({video_fps:.2f} fps)", err=True)

    # 2. Analyze audio if no beat map provided
    if beats:
        from beatlab.beat_map import load_beat_map
        beat_map = load_beat_map(beats)
    else:
        click.echo("  Extracting audio for analysis...", err=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name
        extract_audio(video_file, audio_path, sr=sr)

        from beatlab.analyzer import analyze_audio
        from beatlab.beat_map import create_beat_map
        analysis = analyze_audio(audio_path, sr=sr, detect_sections_flag=True)
        beat_map = create_beat_map(analysis, fps=video_fps, source_file=video_file)
        click.echo(
            f"  Tempo: {beat_map['tempo']:.1f} BPM | "
            f"Beats: {len(beat_map['beats'])} | "
            f"Sections: {len(beat_map.get('sections', []))}",
            err=True,
        )

    # 3. Get AI plan with style_prompts if --ai
    section_styles: dict[int, str] = {}
    if ai:
        plan = _get_ai_plan(beat_map, prompt)
        if plan:
            for sp in plan.sections:
                if sp.style_prompt:
                    section_styles[sp.section_index] = sp.style_prompt

    if not section_styles:
        # Use default style for all sections
        for i in range(len(beat_map.get("sections", []))):
            section_styles[i] = style

    # 4. Extract frames
    with tempfile.TemporaryDirectory(prefix="beatlab_frames_") as frames_dir:
        click.echo("  Extracting frames...", err=True)
        frame_count, actual_fps = extract_frames(video_file, frames_dir, fps=fps)
        click.echo(f"  Extracted {frame_count} frames", err=True)

        # 5. Generate per-frame params
        frame_params = generate_frame_params(
            beat_map, frame_count, actual_fps,
            base_denoise=base_denoise, beat_denoise=beat_denoise,
            section_styles=section_styles, default_style=style,
        )

        # 6. Cost estimate
        cost = estimate_cost(frame_count)
        click.echo(
            f"  Render estimate: {cost['frames']} frames, "
            f"~{cost['estimated_hours']:.1f}h, ~${cost['estimated_cost_usd']:.2f}",
            err=True,
        )

        if dry_run:
            click.echo("\n  Dry run — no rendering performed.", err=True)
            return

        # 7. Render
        with tempfile.TemporaryDirectory(prefix="beatlab_styled_") as styled_dir:
            if local_comfyui:
                # Local ComfyUI render
                from beatlab.render.comfyui import ComfyUIClient
                host, port = local_comfyui.replace("http://", "").split(":")
                client = ComfyUIClient(host=host, port=int(port))

                click.echo(f"  Rendering {frame_count} frames via {local_comfyui}...", err=True)
                with click.progressbar(length=frame_count, label="  Rendering", file=sys.stderr) as bar:
                    client.render_batch(
                        frame_params, frames_dir, styled_dir,
                        model=model,
                        progress_callback=lambda done, total: bar.update(1),
                    )
            else:
                # Cloud GPU render
                from beatlab.render.cloud import VastAIManager
                vast = VastAIManager()

                click.echo("  Looking for GPU instance...", err=True)
                instance_id, reused = vast.get_or_create_instance()

                if reused:
                    click.echo(f"  Reusing running instance {instance_id}", err=True)
                else:
                    click.echo(f"  Created new instance {instance_id}, waiting for it to start...", err=True)
                    vast.wait_until_ready(instance_id)

                try:
                    comfyui_url = vast.get_comfyui_url(instance_id)
                    click.echo(f"  ComfyUI at: {comfyui_url}", err=True)

                    # Save params for remote use
                    params_path = f"{frames_dir}/frame_params.json"
                    save_frame_params(frame_params, params_path)

                    click.echo("  Uploading frames...", err=True)
                    vast.upload_files(instance_id, frames_dir, "/workspace/input")

                    click.echo(f"  Rendering {frame_count} frames on cloud GPU...", err=True)
                    host, port = comfyui_url.replace("http://", "").split(":")
                    from beatlab.render.comfyui import ComfyUIClient
                    client = ComfyUIClient(host=host, port=int(port))
                    client.render_batch(
                        frame_params, "/workspace/input", "/workspace/output",
                        model=model,
                    )

                    click.echo("  Downloading results...", err=True)
                    vast.download_files(instance_id, "/workspace/output", styled_dir)

                    if destroy:
                        click.echo("  Destroying cloud instance...", err=True)
                        vast.destroy_instance(instance_id)
                    else:
                        click.echo(
                            f"  Instance {instance_id} kept alive for reuse. "
                            f"Run 'beatlab destroy-gpu' to stop it.",
                            err=True,
                        )
                except Exception as e:
                    # Don't destroy on failure — instance is expensive to recreate
                    click.echo(
                        f"\n  Error: {e}\n"
                        f"  Instance {instance_id} kept alive — fix the issue and retry.\n"
                        f"  Run 'beatlab destroy-gpu' to stop it when done.",
                        err=True,
                    )
                    raise

            # 8. Reassemble
            click.echo("  Reassembling video...", err=True)
            reassemble_video(styled_dir, output, actual_fps, audio_source=video_file)

    click.echo(f"Done! Output: {output}", err=True)


@main.command(name="destroy-gpu")
def destroy_gpu():
    """Destroy the kept-alive Vast.ai GPU instance."""
    from beatlab.render.cloud import _load_instance_state, VastAIManager

    state = _load_instance_state()
    if not state:
        click.echo("No kept-alive instance found.", err=True)
        return

    instance_id = state["instance_id"]
    click.echo(f"Destroying instance {instance_id} ({state.get('gpu_name', '?')})...", err=True)
    try:
        vast = VastAIManager()
        vast.destroy_instance(instance_id)
        click.echo("  Instance destroyed.", err=True)
    except Exception as e:
        click.echo(f"  Failed: {e}", err=True)
        from beatlab.render.cloud import _clear_instance_state
        _clear_instance_state()


def _load_descriptions(md_path: str, num_sections: int) -> list[str]:
    """Load audio descriptions from a previously generated markdown file."""
    import re

    click.echo(f"  Loading descriptions from: {md_path}", err=True)
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

    click.echo(f"  Loaded {len(descriptions_by_index)} unique descriptions for {num_sections} sections", err=True)
    return descriptions


def _describe_sections(audio_file: str, sr: int, sections: list[dict]):
    """Describe each section's audio content using Gemini Flash."""
    from pathlib import Path

    try:
        from beatlab.ai.audio_describer import GeminiAudioDescriber, describe_sections
        from beatlab.analyzer import load_audio
    except ImportError as e:
        raise click.ClickException(str(e))

    click.echo("  Connecting to Gemini Flash for audio descriptions...", err=True)
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
    click.echo(f"  Descriptions saved to: {md_path}", err=True)

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

    click.echo("  Asking AI for effect plan...", err=True)
    try:
        provider = AnthropicProvider()
        plan = create_effect_plan(beat_map, provider, user_prompt=user_prompt, audio_descriptions=audio_descriptions)
        click.echo(f"  AI plan: {len(plan.sections)} section(s) configured", err=True)
        return plan
    except Exception as e:
        raise click.ClickException(f"AI effect plan failed: {e}")
