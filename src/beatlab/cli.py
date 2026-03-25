"""CLI interface for beatlab."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

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
@click.argument("video_file", type=click.Path(exists=True))
@click.option("--fps", default=None, type=float, help="Timeline frame rate (default: auto-detect from video)")
@click.option("--output", "-o", default=None, type=click.Path(), help="Output JSON file (default: work dir beats.json)")
@click.option("--sr", default=22050, type=int, help="Sample rate for analysis (default: 22050)")
@click.option("--sections/--no-sections", default=True, help="Detect musical sections (default: on)")
@click.option("--stems/--no-stems", default=False, help="Separate audio into stems via Demucs on Vast.ai for per-instrument analysis")
@click.option("--stems-local/--no-stems-local", default=False, help="Run Demucs locally on CPU instead of Vast.ai (slow)")
@click.option("--work-dir", default=".beatlab_work", type=str, help="Work directory for caching (default: .beatlab_work)")
@click.option("--fresh/--resume", default=False, help="Re-analyze from scratch (default: resume/use cache)")
def analyze(video_file: str, fps: float | None, output: str | None, sr: int, sections: bool,
            stems: bool, stems_local: bool, work_dir: str, fresh: bool):
    """Analyze audio from a video file — beat detection, sections, and optional stem separation.

    Caches audio, stems, and beats.json in .beatlab_work/<video_name>/.
    """
    from beatlab.render.frames import detect_fps, extract_audio
    from beatlab.render.workdir import WorkDir
    from beatlab.analyzer import analyze_audio
    from beatlab.beat_map import create_beat_map, save_beat_map

    work = WorkDir(video_file, base_dir=work_dir)

    # FPS
    video_fps = fps or detect_fps(video_file)
    _log(f"Analyzing: {video_file} ({video_fps:.2f} fps)")

    # Extract audio if needed
    if work.has_audio() and not fresh:
        _log("  Audio: using cached")
        audio_path = str(work.audio_path)
    else:
        _log("  Extracting audio...")
        extract_audio(video_file, str(work.audio_path), sr=sr)
        audio_path = str(work.audio_path)

    # Beat analysis
    analysis = analyze_audio(audio_path, sr=sr, detect_sections_flag=sections)
    _log(
        f"  Tempo: {analysis['tempo']:.1f} BPM | "
        f"Beats: {len(analysis['beats'])} | "
        f"Onsets: {len(analysis['onsets'])} | "
        f"Duration: {analysis['duration']:.1f}s"
    )
    if sections and "sections" in analysis:
        _log(f"  Sections: {len(analysis['sections'])} detected")

    # Stem separation + per-stem analysis
    stem_analyses = None
    if stems:
        from beatlab.stems import separate_stems_remote, separate_stems_local, analyze_all_stems

        if work.has_stems() and not fresh:
            _log("  Stems: using cached")
            stem_paths = work.stem_paths()
        elif stems_local:
            stem_paths = separate_stems_local(audio_path, str(work.stems_dir))
        else:
            from beatlab.render.cloud import VastAIManager
            vast = VastAIManager()
            stem_paths = separate_stems_remote(audio_path, str(work.stems_dir), vast)

        _log("  Analyzing stems locally...")
        stem_analyses = analyze_all_stems(stem_paths, sr=sr)

    beat_map = create_beat_map(analysis, fps=video_fps, source_file=video_file, stem_analyses=stem_analyses)

    # Save to work dir (and optionally to custom output)
    work.save_beats(beat_map)
    _log(f"  Beat map written to: {work.beats_path}")

    if output:
        save_beat_map(beat_map, output)
        _log(f"  Also written to: {output}")


@main.command(name="presets")
def list_presets():
    """List available effect presets."""
    from beatlab.presets import list_presets as _list

    click.echo("Available presets:\n")
    for p in _list():
        click.echo(f"  {p['name']:20s} {p['description']}")
        click.echo(f"  {'':20s} node={p['node']}.{p['parameter']}  curve={p['curve']}")
        click.echo()


@main.command(name="marker-ui")
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--beats", default=None, type=click.Path(exists=True), help="Path to beats.json from analysis")
@click.option("--hits", default="hits.json", type=click.Path(), help="Path to save/load hits.json (default: hits.json)")
@click.option("--fps", default=30.0, type=float, help="Timeline frame rate (default: 30)")
@click.option("--port", default=8080, type=int, help="Server port (default: 8080)")
def marker_ui(audio_file: str, beats: str | None, hits: str, fps: float, port: int):
    """Launch the hit marker web UI for manual effect placement."""
    from beatlab.marker_server import start_server

    start_server(
        audio_path=audio_file,
        beats_path=beats,
        hits_path=hits,
        fps=fps,
        port=port,
    )


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
@click.option("--hits", default=None, type=click.Path(exists=True), help="Path to hits.json for manual accent effects")
def generate(
    beats_json: str, output: str, effect: str | None, preset: str | None,
    attack: int | None, release: int | None, intensity_curve: str,
    section_mode: bool, overshoot: bool, ai: bool, prompt: str | None,
    hits: str | None,
):
    """Generate a Fusion .setting file from a beat map JSON."""
    from beatlab.beat_map import load_beat_map
    from beatlab.generator import generate_comp, load_hits, _apply_hits
    from beatlab.fusion.nodes import make_media_out

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

    # Layer manual hit accents on top
    if hits:
        hit_data = load_hits(hits)
        if hit_data:
            _log(f"  Layering {len(hit_data)} manual hit accents from {hits}")
            media_out = comp.nodes.pop()
            last_node = comp.nodes[-1].name if comp.nodes else None
            pos_x = comp.nodes[-1].pos_x + 110 if comp.nodes else 0
            last_name = _apply_hits(comp, hit_data, last_node, pos_x)
            media_out.inputs["MainInput"] = last_name
            media_out.pos_x = (comp.nodes[-1].pos_x + 110) if comp.nodes else 110
            comp.add_node(media_out)
            comp.active_tool = last_name

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


# ── Resolve Commands ────────────────────────────────────────────────────────


@main.group()
def resolve():
    """DaVinci Resolve headless integration commands."""
    pass


@resolve.command(name="status")
def resolve_status():
    """Show Resolve connection status and timeline info."""
    from beatlab.resolve import connect

    try:
        session = connect(timeout=10)
    except (RuntimeError, FileNotFoundError) as e:
        _log(f"Resolve not available: {e}")
        return

    _log(f"Resolve {session.get_version()}")
    _log(f"Project: {session.get_project_name()}")
    info = session.get_timeline_info()
    if "error" in info:
        _log(f"Timeline: {info['error']}")
    else:
        _log(f"Timeline: {info['name']}")
        _log(f"  FPS: {info['fps']} | Frames: {info['duration_frames']} | Duration: {info['duration_sec']}s")
        _log(f"  Video tracks: {info['track_count_video']} | Audio tracks: {info['track_count_audio']}")


@resolve.command(name="inject")
@click.argument("setting_file", type=click.Path(exists=True))
@click.option("--track", default=1, type=int, help="Video track number (1-based, default: 1)")
@click.option("--item", default=0, type=int, help="Item index on track (0-based, default: 0)")
def resolve_inject(setting_file: str, track: int, item: int):
    """Import a Fusion .setting file directly into the current timeline item."""
    from beatlab.resolve import connect

    session = connect()
    _log(f"Importing {setting_file} into timeline...")
    success = session.import_fusion_comp(setting_file, track_index=track, item_index=item)
    if success:
        _log("Fusion comp injected successfully.")
    else:
        _log("Failed to inject Fusion comp.")
        sys.exit(1)


@resolve.command(name="render")
@click.option("--output-dir", "-o", required=True, type=click.Path(), help="Output directory for rendered file")
@click.option("--filename", default="render", type=str, help="Output filename (default: render)")
@click.option("--wait/--no-wait", default=True, help="Wait for render to complete (default: yes)")
def resolve_render(output_dir: str, filename: str, wait: bool):
    """Add a render job and optionally wait for completion."""
    from beatlab.resolve import connect

    session = connect()
    job_id = session.add_render_job(output_dir, filename=filename)
    if not job_id:
        _log("Failed to add render job.")
        sys.exit(1)

    _log(f"Starting render...")
    started = session.start_render([job_id])
    if not started:
        _log("Failed to start render.")
        sys.exit(1)

    if wait:
        status = session.wait_for_render(job_id)
        if status.get("JobStatus") == "Complete":
            _log(f"Render complete: {output_dir}/{filename}")
        else:
            _log(f"Render failed: {status}")
            sys.exit(1)
    else:
        _log(f"Render queued: job {job_id}")


@resolve.command(name="pipeline")
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--output", "-o", default="output.setting", type=click.Path(), help="Output .setting file")
@click.option("--ai/--no-ai", default=False, help="Use AI to select effects per section")
@click.option("--prompt", default=None, type=str, help="Creative direction for AI mode")
@click.option("--hits", default=None, type=click.Path(exists=True), help="Path to hits.json for manual accents")
@click.option("--track", default=1, type=int, help="Video track to inject into (default: 1)")
@click.option("--item", default=0, type=int, help="Item index on track (default: 0)")
@click.option("--render-dir", default=None, type=click.Path(), help="If set, also queue and run a render to this dir")
def resolve_pipeline(audio_file: str, output: str, ai: bool, prompt: str | None, hits: str | None, track: int, item: int, render_dir: str | None):
    """Full pipeline: analyze → generate → inject into Resolve → optional render."""
    from beatlab.analyzer import analyze_audio
    from beatlab.beat_map import create_beat_map
    from beatlab.generator import generate_comp, load_hits, _apply_hits
    from beatlab.resolve import connect

    # Connect to Resolve first to get FPS
    session = connect()
    info = session.get_timeline_info()
    if "error" in info:
        _log(f"No timeline: {info['error']}")
        sys.exit(1)

    fps = info["fps"]
    _log(f"Timeline: {info['name']} @ {fps} fps")

    # Analyze
    _log(f"Analyzing: {audio_file}")
    analysis = analyze_audio(audio_file, detect_sections_flag=True)
    _log(f"  Tempo: {analysis['tempo']:.1f} BPM | Beats: {len(analysis['beats'])}")
    beat_map = create_beat_map(analysis, fps=fps, source_file=audio_file)

    # Generate comp
    plan = None
    if ai:
        plan = _get_ai_plan(beat_map, prompt)

    if plan:
        _log("Generating Fusion comp from AI effect plan")
        comp = generate_comp(beat_map, effect_plan=plan)
    else:
        _log("Generating Fusion comp with default presets")
        comp = generate_comp(beat_map)

    # Layer hits
    if hits:
        from beatlab.fusion.nodes import make_media_out
        hit_data = load_hits(hits)
        if hit_data:
            _log(f"  Layering {len(hit_data)} manual hit accents")
            media_out = comp.nodes.pop()
            last_node = comp.nodes[-1].name if comp.nodes else None
            pos_x = comp.nodes[-1].pos_x + 110 if comp.nodes else 0
            last_name = _apply_hits(comp, hit_data, last_node, pos_x)
            media_out.inputs["MainInput"] = last_name
            media_out.pos_x = (comp.nodes[-1].pos_x + 110) if comp.nodes else 110
            comp.add_node(media_out)
            comp.active_tool = last_name

    comp.save(output)
    _log(f"  Fusion comp written to: {output}")

    # Inject into Resolve
    _log("Injecting into Resolve timeline...")
    from pathlib import Path
    success = session.import_fusion_comp(str(Path(output).resolve()), track_index=track, item_index=item)
    if not success:
        _log("Failed to inject comp.")
        sys.exit(1)
    _log("Fusion comp injected.")

    # Optional render
    if render_dir:
        _log("Queueing render...")
        job_id = session.add_render_job(render_dir)
        if job_id:
            session.start_render([job_id])
            status = session.wait_for_render(job_id)
            if status.get("JobStatus") == "Complete":
                _log(f"Render complete: {render_dir}")
            else:
                _log(f"Render failed: {status}")
        else:
            _log("Failed to queue render.")


def _parse_segment_filter(spec: str) -> set[int]:
    """Parse segment filter spec like '1,2,20-25,30' into a set of indices."""
    indices = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            indices.update(range(int(start), int(end) + 1))
        else:
            indices.add(int(part))
    return indices


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
@click.option("--engine", default="ebsynth", type=click.Choice(["ebsynth", "wan", "google", "kling"]), help="Render engine: ebsynth, wan, google (Nano Banana+Veo), kling (Nano Banana+Kling 3.0)")
@click.option("--preview/--no-preview", default=False, help="Render at 512x512 for fast preview (Wan2.1 only)")
@click.option("--describe", default=None, is_flag=False, flag_value="generate", help="Describe sections with Gemini. Pass a .md file to reuse existing descriptions.")
@click.option("--vertex/--no-vertex", default=False, help="Use Vertex AI instead of AI Studio (higher rate limits, requires GCP project)")
@click.option("--audio-prompt/--no-audio-prompt", default=False, help="Include audio descriptions in Veo video generation prompts")
@click.option("--motion", default=None, type=str, help="Camera/motion direction for Veo (e.g. 'forward dolly through void, warp speed')")
@click.option("--plan-patch", default=None, type=click.Path(exists=True), help="Patch JSON to merge into cached plan — only re-renders changed sections")
@click.option("--labels/--no-labels", default=False, help="Burn section numbers into bottom-right of video for review")
@click.option("--segments", default=None, type=str, help="Only process specific segments: e.g. 1,2,20-25,30")
@click.option("--intra-transition-prompt", default=None, type=str, help="Override the default smooth transition prompt for intra-section sub-section transitions")
@click.option("--ai-transitions/--no-ai-transitions", default=True, help="Use Claude to describe intra-section transitions based on actual images (default: on)")
@click.option("--candidates", default=4, type=int, help="Number of styled image candidates per section (default: 4, 0 or 1 to disable)")
@click.option("--backfill-candidates/--no-backfill-candidates", default=False, help="Generate candidates for sections that already have styled images (promotes existing to v1)")
@click.option("--stems/--no-stems", default=False, help="Separate audio into stems (drums/bass/vocals/other) via Demucs on Vast.ai for per-instrument beat analysis")
def render(
    video_file: str, beats: str | None, fps: float | None, style: str,
    ai: bool, prompt: str | None, output: str, base_denoise: float,
    beat_denoise: float, model: str, local_comfyui: str | None,
    sr: int, dry_run: bool, destroy: bool, fresh: bool, work_dir: str,
    engine: str, preview: bool, describe: str | None, vertex: bool,
    audio_prompt: bool, motion: str | None, plan_patch: str | None,
    labels: bool, candidates: int, backfill_candidates: bool, segments: str | None,
    intra_transition_prompt: str | None, ai_transitions: bool, stems: bool,
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

        # Stem separation (optional)
        stem_analyses = None
        if stems:
            from beatlab.stems import separate_stems_remote, analyze_all_stems
            if work.has_stems():
                _log("  Stems: using cached")
                stem_paths = work.stem_paths()
            else:
                from beatlab.render.cloud import VastAIManager
                vast = VastAIManager()
                stem_paths = separate_stems_remote(audio_path, str(work.stems_dir), vast)
            _log("  Analyzing stems locally...")
            stem_analyses = analyze_all_stems(stem_paths, sr=sr)

        beat_map = create_beat_map(analysis, fps=video_fps, source_file=video_file, stem_analyses=stem_analyses)
        work.save_beats(beat_map)

    _log(
        f"  Tempo: {beat_map['tempo']:.1f} BPM | "
        f"Beats: {len(beat_map['beats'])} | "
        f"Sections: {len(beat_map.get('sections', []))}"
        )

    # ── Step 2.5: Audio descriptions (optional, cached in work dir) ──
    audio_descriptions = None
    descriptions_cache = str(work.root / "descriptions.md")
    if describe and beat_map.get("sections"):
        if describe != "generate" and describe.endswith(".md"):
            audio_descriptions = _load_descriptions(describe, len(beat_map["sections"]))
        elif Path(descriptions_cache).exists() and not fresh:
            _log("  Descriptions: using cached")
            audio_descriptions = _load_descriptions(descriptions_cache, len(beat_map["sections"]))
        else:
            if not work.has_audio():
                _log("  Extracting audio for descriptions...")
                extract_audio(video_file, str(work.audio_path), sr=sr)
            audio_descriptions = _describe_sections(str(work.audio_path), sr, beat_map["sections"], output_path=descriptions_cache)

    # ── Step 3: AI effect plan ──
    section_styles: dict[int, str] = {}
    plan = None
    if ai:
        if work.has_plan() and not fresh:
            _log("  AI plan: using cached")
            plan_data = work.load_plan()
            from beatlab.ai.plan import parse_effect_plan
            plan = parse_effect_plan(json.dumps(plan_data))
        else:
            plan = _get_ai_plan(beat_map, prompt, audio_descriptions=audio_descriptions)
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
                            "wan_denoise": sp.wan_denoise,
                            "transition_frames": sp.transition_frames,
                            "transition_action": sp.transition_action,
                        }
                        for sp in plan.sections
                    ]
                }
                work.save_plan(plan_dict)

        if plan:
            for sp in plan.sections:
                if sp.style_prompt:
                    section_styles[sp.section_index] = sp.style_prompt

    # ── Step 3.5: Apply plan patch if provided ──
    changed_indices: list[int] = []
    if plan_patch:
        from beatlab.render.patcher import load_patch, merge_plan, detect_stale_outputs, save_plan

        _log(f"  Applying plan patch: {plan_patch}")
        patch = load_patch(plan_patch)

        # Load current cached plan
        if work.has_plan():
            base_plan = work.load_plan()
        else:
            _log("  ERROR: No cached plan to patch. Run without --plan-patch first.")
            return

        merged, changed_indices = merge_plan(base_plan, patch)
        _log(f"  Patched {len(changed_indices)} sections: {changed_indices}")

        # Save merged plan
        save_plan(merged, str(work.root / "plan.json"))

        # Delete stale outputs for changed sections
        stale = detect_stale_outputs(str(work.root), changed_indices)
        if stale:
            _log(f"  Deleting {len(stale)} stale outputs...")
            for f in stale:
                Path(f).unlink(missing_ok=True)

        # Auto-generate candidates for sections that request them
        candidate_sections = [
            s for s in patch.get("sections", [])
            if s.get("candidates")
        ]
        if candidate_sections:
            from beatlab.render.candidates import generate_image_candidates, make_contact_sheet
            from beatlab.render.google_video import GoogleVideoClient

            _log(f"  Generating candidates for {len(candidate_sections)} sections...")
            cand_client = GoogleVideoClient(vertex=vertex)

            def _stylize(source_path, style_prompt, output_path):
                return cand_client.stylize_image(source_path, style_prompt, output_path)

            # Build plan lookup for style prompts
            merged_plan_by_idx = {s["section_index"]: s for s in merged.get("sections", [])}
            beats_data = work.load_beats() if work.has_beats() else beat_map
            bsections = beats_data.get("sections", [])
            bfps = beats_data.get("fps", 24)
            frames_dir = work.ensure_frames_dir()

            for cs in candidate_sections:
                idx = cs["section_index"]
                count = cs["candidates"]
                plan_entry = merged_plan_by_idx.get(idx, {})
                style = plan_entry.get("style_prompt", cs.get("style_prompt", "artistic stylized"))

                # Find source image
                source_img = str(work.root / "google_styled" / f"styled_{idx:03d}.png")
                if not Path(source_img).exists():
                    # Extract from source video frames
                    if idx < len(bsections):
                        t = bsections[idx].get("start_time", 0)
                        frame_num = round(t * bfps)
                        source_img = str(Path(frames_dir) / f"frame_{frame_num:06d}.png")

                if not Path(source_img).exists():
                    _log(f"    Section {idx}: no source image, skipping candidates")
                    continue

                _log(f"    Section {idx}: generating {count} candidates...")
                paths = generate_image_candidates(
                    section_idx=idx,
                    source_image_path=source_img,
                    style_prompt=style,
                    count=count,
                    work_dir=str(work.root),
                    stylize_fn=_stylize,
                )

                grid_path = str(work.root / "candidates" / f"section_{idx:03d}_grid.png")
                make_contact_sheet(paths, grid_path, idx)
                _log(f"    Section {idx}: contact sheet → candidates/section_{idx:03d}_grid.png")

            _log(f"  Review contact sheets, then run: beatlab select {Path(work.root).name} <idx>:<variant> ...")
            _log(f"  Then re-run render to apply selections.")

        # Re-parse the merged plan
        from beatlab.ai.plan import parse_effect_plan
        plan = parse_effect_plan(json.dumps(merged))

        # Rebuild section styles from patched plan
        section_styles = {}
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

        def _wan_progress(stage, done, total):
            _log(f"  [{stage}] {done}/{total}")

        mode = "local" if local_comfyui else "cloud (Vast.ai)"
        _log(f"  Wan2.1 engine: {mode}, {'preview 512x512' if preview else 'full 1280x720'}")
        result = render_wan_pipeline(
            video_file=video_file,
            beat_map=beat_map,
            effect_plan=plan if ai else None,
            work_dir=str(work.root),
            fps=actual_fps,
            preview=preview,
            model=model,
            default_style=prompt or style,
            progress_callback=_wan_progress,
            local_comfyui=local_comfyui,
        )

        # Move to final output
        import shutil
        shutil.move(result, output)
        work.save_status("complete", {"output": output, "engine": "wan"})
        _log(f"Done! Output: {output}")
        return

    # ── Google engine branch (Nano Banana + Veo) ──
    if engine == "google":
        from beatlab.render.google_pipeline import render_google_pipeline

        def _google_progress(stage, done, total):
            _log(f"  [{stage}] {done}/{total}")

        _log(f"  Google engine: Nano Banana + Veo ({'Vertex AI' if vertex else 'AI Studio'})")
        result = render_google_pipeline(
            video_file=video_file,
            beat_map=beat_map,
            effect_plan=plan if ai else None,
            work_dir=str(work.root),
            fps=actual_fps,
            default_style=prompt or style,
            progress_callback=_google_progress,
            vertex=vertex,
            audio_descriptions=audio_descriptions if audio_prompt else None,
            motion_prompt=motion,
            labels=labels,
            candidates=candidates,
            backfill_candidates=backfill_candidates,
            segment_filter=_parse_segment_filter(segments) if segments else None,
            intra_transition_prompt=intra_transition_prompt,
            ai_transitions=ai_transitions,
        )

        import shutil
        shutil.move(result, output)
        work.save_status("complete", {"output": output, "engine": "google"})
        _log(f"Done! Output: {output}")
        return

    # ── Kling engine branch (Nano Banana + Kling 3.0) ──
    if engine == "kling":
        from beatlab.render.kling_pipeline import render_kling_pipeline

        def _kling_progress(stage, done, total):
            _log(f"  [{stage}] {done}/{total}")

        _log("  Kling engine: Nano Banana + Kling 3.0 (Replicate API)")
        result = render_kling_pipeline(
            video_file=video_file,
            beat_map=beat_map,
            effect_plan=plan if ai else None,
            work_dir=str(work.root),
            fps=actual_fps,
            default_style=prompt or style,
            progress_callback=_kling_progress,
        )

        import shutil
        shutil.move(result, output)
        work.save_status("complete", {"output": output, "engine": "kling"})
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
                keyframe_list, frames_dir, styled_dir,
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


@main.command(name="make-patch")
@click.argument("video_name", type=str)
@click.argument("patch_file", type=click.Path())
@click.option("--work-dir", default=".beatlab_work", type=str, help="Work directory")
@click.option("--sections", "-s", type=str, help="Comma-separated section indices to include in patch")
def make_patch(video_name: str, patch_file: str, work_dir: str, sections: str | None):
    """Extract sections from a cached plan into a patch file for editing.

    Example: beatlab make-patch beyond_the_veil patch_001.json -s 88,89,90,91
    """
    plan_path = Path(work_dir) / video_name / "plan.json"
    if not plan_path.exists():
        _log(f"No cached plan found at {plan_path}")
        return

    with open(plan_path) as f:
        plan = json.load(f)

    if sections:
        indices = set(int(s.strip()) for s in sections.split(","))
        patch_sections = [s for s in plan["sections"] if s["section_index"] in indices]
    else:
        patch_sections = plan["sections"]

    patch = {"sections": patch_sections}
    with open(patch_file, "w") as f:
        json.dump(patch, f, indent=2)

    _log(f"Extracted {len(patch_sections)} sections to {patch_file}")
    _log(f"Edit the file, then run: beatlab render <video> --plan-patch {patch_file}")


@main.command(name="candidates")
@click.argument("video_name", type=str)
@click.option("--sections", "-s", required=True, type=str, help="Comma-separated section indices")
@click.option("--count", "-n", default=4, type=int, help="Number of candidates per section (default: 4)")
@click.option("--work-dir", default=".beatlab_work", type=str, help="Work directory")
@click.option("--vertex/--no-vertex", default=False, help="Use Vertex AI")
def candidates_cmd(video_name: str, sections: str, count: int, work_dir: str, vertex: bool):
    """Generate candidate styled images for sections to choose from.

    Example: beatlab candidates beyond_the_veil -s 88,92 -n 4
    """
    from beatlab.render.candidates import generate_image_candidates, make_contact_sheet

    work = Path(work_dir) / video_name
    plan_path = work / "plan.json"
    frames_dir = work / "frames"

    if not plan_path.exists():
        _log(f"No plan found at {plan_path}")
        return

    with open(plan_path) as f:
        plan = json.load(f)

    # Build section index → plan entry lookup
    plan_by_idx = {s["section_index"]: s for s in plan.get("sections", [])}

    # Set up stylize function
    from beatlab.render.google_video import GoogleVideoClient
    client = GoogleVideoClient(vertex=vertex)

    def stylize_fn(source_path, style_prompt, output_path):
        return client.stylize_image(source_path, style_prompt, output_path)

    indices = [int(s.strip()) for s in sections.split(",")]

    for idx in indices:
        plan_entry = plan_by_idx.get(idx, {})
        style = plan_entry.get("style_prompt", "artistic stylized")

        # Source image: extract from video at section start time
        source_img = str(work / "google_styled" / f"styled_{idx:03d}.png")
        if not Path(source_img).exists():
            # Use the original keyframe from frames dir
            beats_path = work / "beats.json"
            if beats_path.exists():
                with open(beats_path) as f:
                    beats = json.load(f)
                secs = beats.get("sections", [])
                if idx < len(secs):
                    t = secs[idx].get("start_time", 0)
                    fps = beats.get("fps", 24)
                    frame_num = round(t * fps)
                    source_img = str(frames_dir / f"frame_{frame_num:06d}.png")

        if not Path(source_img).exists():
            _log(f"  Section {idx}: no source image found, skipping")
            continue

        _log(f"  Section {idx}: generating {count} candidates with style: {style[:60]}...")
        paths = generate_image_candidates(
            section_idx=idx,
            source_image_path=source_img,
            style_prompt=style,
            count=count,
            work_dir=str(work),
            stylize_fn=stylize_fn,
        )

        # Make contact sheet
        grid_path = str(work / "candidates" / f"section_{idx:03d}_grid.png")
        make_contact_sheet(paths, grid_path, idx)
        _log(f"  Section {idx}: contact sheet → {grid_path}")

    _log(f"\nReview contact sheets in {work}/candidates/")
    _log(f"Then run: beatlab select {video_name} <idx>:<variant> ...")


@main.command(name="select")
@click.argument("video_name", type=str)
@click.argument("selections", nargs=-1, type=str)
@click.option("--work-dir", default=".beatlab_work", type=str, help="Work directory")
def select_cmd(video_name: str, selections: tuple[str], work_dir: str):
    """Apply candidate selections — copies chosen variant to styled image.

    Examples:
      beatlab select beyond_the_veil 88:v2 92:v4
      beatlab select beyond_the_veil 016_001:v2
      beatlab select beyond_the_veil 016_002:016_001/v3  (cross-section)
      beatlab select beyond_the_veil 016_001:v2+v4       (sequence: two clips filling the slot)
      beatlab select beyond_the_veil 016_002:016_001/v3+v1  (cross + sequence)
    """
    from beatlab.render.candidates import apply_selection, apply_cross_selection, apply_sequence_selection

    work = str(Path(work_dir) / video_name)

    for sel in selections:
        parts = sel.split(":")
        if len(parts) != 2:
            _log(f"  Invalid selection format: {sel}")
            continue

        target_str = parts[0]
        if "_" in target_str:
            target = target_str  # file key like "016_001"
        else:
            try:
                target = int(target_str)
            except ValueError:
                target = target_str

        source_str = parts[1]

        # Sequence selection: v2+v4 or 016_001/v3+v1
        if "+" in source_str:
            sequence_parts = source_str.split("+")
            sequence = []
            for sp in sequence_parts:
                if "/" in sp:
                    src_sec, var = sp.split("/")
                    if "_" not in src_sec:
                        try:
                            src_sec = int(src_sec)
                        except ValueError:
                            pass
                    sequence.append({"source": src_sec, "variant": int(var.replace("v", ""))})
                else:
                    sequence.append({"source": target, "variant": int(sp.replace("v", ""))})

            _log(f"  Section {target}: sequence of {len(sequence)} images")
            stale = apply_sequence_selection(target, sequence, work)
            if stale:
                _log(f"    Deleted {len(stale)} stale files")
            continue

        # Cross-section selection: 016_002:016_001/3
        if "/" in source_str:
            source_parts = source_str.split("/")
            source_section = source_parts[0]
            if "_" not in source_section:
                try:
                    source_section = int(source_section)
                except ValueError:
                    pass
            variant = int(source_parts[1].replace("v", ""))
            _log(f"  Section {target}: applying v{variant} from section {source_section}")
            stale = apply_cross_selection(target, source_section, variant, work)
        else:
            # Normal selection
            variant = int(source_str.replace("v", ""))
            _log(f"  Section {target}: applying variant v{variant}")
            stale = apply_selection(target, variant, work)

        if stale:
            _log(f"    Deleted {len(stale)} stale files")

    _log(f"\nSelections applied. Re-run render to generate new transitions for selected sections.")


@main.command(name="split-sections")
@click.argument("video_name", type=str)
@click.option("--max-duration", default=8.0, type=float, help="Max section duration in seconds (default: 8)")
@click.option("--work-dir", default=".beatlab_work", type=str, help="Work directory")
@click.option("--dry-run/--no-dry-run", default=False, help="Show what would be split without modifying anything")
@click.option("--clean/--no-clean", default=False, help="Delete stale files for split sections")
def split_sections(video_name: str, max_duration: float, work_dir: str, dry_run: bool, clean: bool):
    """Analyze cached plan for long sections and generate splits.json.

    Example: beatlab split-sections beyond_the_veil --max-duration 8
    """
    from beatlab.render.section_splitter import (
        generate_splits, save_splits, load_splits, find_long_sections, get_stale_files, get_keyframe_timestamps,
    )

    work = Path(work_dir) / video_name
    plan_path = work / "plan.json"
    beats_path = work / "beats.json"
    splits_path = work / "splits.json"

    if not plan_path.exists():
        _log(f"No plan found at {plan_path}")
        return
    if not beats_path.exists():
        _log(f"No beats found at {beats_path}")
        return

    with open(plan_path) as f:
        plan = json.load(f)
    with open(beats_path) as f:
        beats = json.load(f)

    sections = beats.get("sections", [])
    long = find_long_sections(plan, sections, max_duration)

    if not long:
        _log(f"No sections exceed {max_duration}s — nothing to split.")
        return

    _log(f"Found {len(long)} sections exceeding {max_duration}s:")
    total_new_clips = 0
    for ls in long:
        _log(f"  Section {ls['section_index']}: {ls['duration']:.1f}s → {ls['num_splits']} sub-sections")
        total_new_clips += ls['num_splits'] - 1  # -1 because original segment covers 1

    _log(f"  Total new clips needed: ~{total_new_clips}")

    if dry_run:
        _log("\nDry run — no changes made.")
        return

    # Load existing splits if present (for re-splitting)
    existing = None
    if splits_path.exists():
        existing = load_splits(str(splits_path))
        _log(f"  Found existing splits — will merge and further split if needed")

    # Generate splits (merges with existing if present)
    splits = generate_splits(plan, sections, max_duration, existing_splits=existing)
    save_splits(splits, str(splits_path))
    _log(f"\nSaved splits to: {splits_path}")

    # Show keyframe extraction needed
    kf_timestamps = get_keyframe_timestamps(splits, beats.get("fps", 24))
    _log(f"New keyframe images needed: {len(kf_timestamps)}")

    if clean:
        stale = get_stale_files(str(work), splits)
        if stale:
            _log(f"Deleting {len(stale)} stale files...")
            for f in stale:
                Path(f).unlink(missing_ok=True)
                _log(f"  Deleted: {Path(f).name}")
        else:
            _log("No stale files to clean.")

    _log(f"\nNext: re-run render to generate new styled images + transitions for split sections.")
    _log(f"  beatlab render <video> --engine google --vertex -o output.mp4")


@main.command(name="destroy-gpu")
@click.option("--all", "destroy_all", is_flag=True, default=False, help="Destroy all instances (default + stems)")
def destroy_gpu(destroy_all: bool):
    """Destroy the kept-alive Vast.ai GPU instance."""
    from beatlab.render.cloud import _load_instance_state, _clear_instance_state, VastAIManager

    keys = ["default", "stems"] if destroy_all else ["default", "stems"]
    destroyed_any = False

    for key in keys:
        state = _load_instance_state(key)
        if not state:
            continue

        instance_id = state["instance_id"]
        label = f" ({key})" if key != "default" else ""
        _log(f"Destroying instance {instance_id}{label} ({state.get('gpu_name', '?')})...")
        try:
            vast = VastAIManager()
            vast.destroy_instance(instance_id, instance_key=key)
            _log("  Instance destroyed.")
            destroyed_any = True
        except Exception as e:
            _log(f"  Failed: {e}")
            _clear_instance_state(key)
            destroyed_any = True

    if not destroyed_any:
        _log("No kept-alive instances found.")


@main.command()
@click.argument("file_path", type=click.Path())
@click.option("--work-dir", default=".beatlab_work", type=str, help="Work directory (default: .beatlab_work)")
def delete(file_path: str, work_dir: str):
    """Delete a cached file and cascade-delete all downstream artifacts.

    Examples:
        beatlab delete google_styled/styled_042.png
        beatlab delete google_segments/segment_042_043.mp4
        beatlab delete google_remapped/remapped_042.mp4

    Cascading logic:
        styled_NNN.png → deletes segments using that section, remapped, concat, muxed, output
        segment_NNN_MMM.mp4 → deletes remapped, concat, muxed, output
        remapped_NNN.mp4 → deletes concat, muxed, output
    """
    import glob
    import re

    # Find the work dir — could be multiple video subdirs
    work_base = Path(work_dir)
    if not work_base.exists():
        _log(f"Work dir not found: {work_dir}")
        return

    # Search all video work dirs for the file
    deleted = []
    for video_dir in work_base.iterdir():
        if not video_dir.is_dir():
            continue

        target = video_dir / file_path
        if not target.exists():
            continue

        _log(f"Found: {target}")

        # Determine what to cascade based on the file
        name = target.name
        cascade = []

        # styled_NNN.png → segments + remapped + assembly
        m = re.match(r"styled_(\d+)\.png", name)
        if m:
            idx = int(m.group(1))
            # Segments that use this section as start or end
            seg_dir = video_dir / "google_segments"
            if seg_dir.exists():
                for seg in seg_dir.glob(f"segment_{idx:03d}_*.mp4"):
                    cascade.append(seg)
                if idx > 0:
                    for seg in seg_dir.glob(f"segment_*_{idx:03d}.mp4"):
                        cascade.append(seg)
            # Corresponding remapped
            remap_dir = video_dir / "google_remapped"
            if remap_dir.exists():
                for r in remap_dir.glob(f"remapped_{idx:03d}.mp4"):
                    cascade.append(r)
                if idx > 0:
                    for r in remap_dir.glob(f"remapped_{idx-1:03d}.mp4"):
                        cascade.append(r)

        # segment_NNN_MMM.mp4 → remapped + assembly
        m = re.match(r"segment_(\d+)_(\d+)\.mp4", name)
        if m:
            idx = int(m.group(1))
            remap_dir = video_dir / "google_remapped"
            if remap_dir.exists():
                for r in remap_dir.glob(f"remapped_{idx:03d}.mp4"):
                    cascade.append(r)

        # Always delete assembly artifacts
        for assembly_file in [
            "google_concat.mp4", "google_muxed.mp4", "google_output.mp4",
            "kling_concat.mp4", "kling_muxed.mp4", "kling_output.mp4",
        ]:
            af = video_dir / assembly_file
            if af.exists():
                cascade.append(af)

        # Delete target
        target.unlink()
        deleted.append(str(target))
        _log(f"  Deleted: {target}")

        # Delete cascade
        for cf in cascade:
            if cf.exists():
                cf.unlink()
                deleted.append(str(cf))
                _log(f"  Cascade: {cf}")

    if not deleted:
        _log(f"File not found in any work dir: {file_path}")
    else:
        _log(f"Deleted {len(deleted)} files total. Re-run render to regenerate.")


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


def _describe_sections(audio_file: str, sr: int, sections: list[dict], output_path: str | None = None):
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
    md_path = output_path or f"{source_name}_descriptions.md"
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


# ── Narrative Keyframe Pipeline ────────────────────────────────────


@main.group()
def narrative():
    """Narrative keyframe pipeline — YAML-driven keyframe + Veo transition generation."""
    pass


@narrative.command()
@click.argument("yaml_path", type=click.Path(exists=True))
@click.option("--vertex/--no-vertex", default=False, help="Use Vertex AI vs AI Studio")
@click.option("--segments", default=None, help="Filter keyframes: kf_001,kf_005-kf_010")
@click.option("--candidates", "n_candidates", default=None, type=int, help="Override candidates per slot")
@click.option("--dry-run", is_flag=True, help="Validate YAML and print stats only")
@click.option("--replicate", "use_replicate", is_flag=True, help="Use Replicate API instead of Google AI Studio/Vertex")
def keyframes(yaml_path, vertex, segments, n_candidates, dry_run, use_replicate):
    """Generate keyframe candidates from narrative YAML."""
    from beatlab.render.narrative import load_narrative, narrative_stats, generate_keyframe_candidates

    data = load_narrative(yaml_path)
    stats = narrative_stats(data)
    click.echo(f"Narrative: {data['meta']['title']}")
    click.echo(f"  Keyframes: {stats['keyframes']} ({stats['keyframes_with_candidates']} with candidates, {stats['keyframes_selected']} selected)")
    click.echo(f"  Transitions: {stats['transitions']} ({stats['total_slots']} total slots)")
    click.echo(f"  Multi-slot transitions: {stats['multi_slot_transitions']} ({stats['intermediate_keyframes_needed']} intermediate keyframes needed)")

    if dry_run:
        return

    seg_filter = _parse_kf_filter(segments) if segments else None
    generate_keyframe_candidates(yaml_path, vertex=vertex, candidates_per_slot=n_candidates, segment_filter=seg_filter, use_replicate=use_replicate)


@narrative.command(name="select-keyframes")
@click.argument("yaml_path", type=click.Path(exists=True))
@click.argument("selections", nargs=-1, required=True)
def select_keyframes_cmd(yaml_path, selections):
    """Apply keyframe selections. E.g.: kf_001:v2 kf_005:v3"""
    from beatlab.render.narrative import apply_keyframe_selection

    parsed = {}
    for sel in selections:
        parts = sel.split(":")
        if len(parts) != 2:
            raise click.ClickException(f"Invalid selection format: {sel} (expected kf_id:vN)")
        kf_id = parts[0]
        variant = int(parts[1].lstrip("v"))
        parsed[kf_id] = variant

    apply_keyframe_selection(yaml_path, parsed)


@narrative.command()
@click.argument("yaml_path", type=click.Path(exists=True))
def actions(yaml_path):
    """Generate LLM transition actions for empty transitions."""
    from beatlab.render.narrative import generate_transition_actions
    generate_transition_actions(yaml_path)


@narrative.command(name="slot-keyframes")
@click.argument("yaml_path", type=click.Path(exists=True))
@click.option("--vertex/--no-vertex", default=False)
@click.option("--candidates", "n_candidates", default=None, type=int)
def slot_keyframes_cmd(yaml_path, vertex, n_candidates):
    """Generate intermediate keyframe candidates for multi-slot transitions."""
    from beatlab.render.narrative import generate_slot_keyframe_candidates
    generate_slot_keyframe_candidates(yaml_path, vertex=vertex, candidates_per_slot=n_candidates)


@narrative.command(name="select-slot-keyframes")
@click.argument("yaml_path", type=click.Path(exists=True))
@click.argument("selections", nargs=-1, required=True)
def select_slot_keyframes_cmd(yaml_path, selections):
    """Apply slot keyframe selections. E.g.: tr_041_slot_0:v2"""
    from beatlab.render.narrative import apply_slot_keyframe_selection

    parsed = {}
    for sel in selections:
        parts = sel.split(":")
        if len(parts) != 2:
            raise click.ClickException(f"Invalid selection format: {sel}")
        slot_key = parts[0]
        variant = int(parts[1].lstrip("v"))
        parsed[slot_key] = variant

    apply_slot_keyframe_selection(yaml_path, parsed)


@narrative.command()
@click.argument("yaml_path", type=click.Path(exists=True))
@click.option("--vertex/--no-vertex", default=False)
@click.option("--segments", default=None, help="Filter transitions: tr_001,tr_005-tr_010")
@click.option("--candidates", "n_candidates", default=None, type=int)
def transitions(yaml_path, vertex, segments, n_candidates):
    """Generate Veo transition video candidates."""
    from beatlab.render.narrative import generate_transition_candidates

    seg_filter = _parse_kf_filter(segments) if segments else None
    generate_transition_candidates(yaml_path, vertex=vertex, candidates_per_slot=n_candidates, segment_filter=seg_filter)


@narrative.command(name="select-transitions")
@click.argument("yaml_path", type=click.Path(exists=True))
@click.argument("selections", nargs=-1, required=True)
def select_transitions_cmd(yaml_path, selections):
    """Apply transition selections. E.g.: tr_001:v2 tr_005_slot_0:v3"""
    from beatlab.render.narrative import apply_transition_selection

    parsed = {}
    for sel in selections:
        parts = sel.split(":")
        if len(parts) != 2:
            raise click.ClickException(f"Invalid selection format: {sel}")
        key = parts[0]
        variant = int(parts[1].lstrip("v"))
        parsed[key] = variant

    apply_transition_selection(yaml_path, parsed)


@narrative.command()
@click.argument("yaml_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="narrative_output.mp4", help="Output video path")
def assemble(yaml_path, output):
    """Time-remap, concatenate, and mux audio into final video."""
    from beatlab.render.narrative import assemble_final
    assemble_final(yaml_path, output)


def _parse_kf_filter(spec: str) -> set[str]:
    """Parse a segment filter spec like 'kf_001,kf_005-kf_010' into a set of IDs."""
    result = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part and not part.startswith("-"):
            # Range: kf_005-kf_010
            start, end = part.split("-", 1)
            # Extract numeric suffix
            start_num = int(re.search(r"\d+$", start).group())
            end_num = int(re.search(r"\d+$", end).group())
            prefix = re.sub(r"\d+$", "", start)
            for i in range(start_num, end_num + 1):
                result.add(f"{prefix}{i:03d}")
        else:
            result.add(part)
    return result
