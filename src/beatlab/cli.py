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
def generate(
    beats_json: str, output: str, effect: str | None, preset: str | None,
    attack: int | None, release: int | None, intensity_curve: str,
    section_mode: bool, overshoot: bool, ai: bool, prompt: str | None,
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
def run(
    audio_file: str, fps: float, output: str, effect: str | None,
    preset: str | None, attack: int | None, release: int | None,
    sr: int, beats_out: str | None, intensity_curve: str,
    section_mode: bool, overshoot: bool, ai: bool, prompt: str | None,
):
    """Full pipeline: audio file → beat analysis → Fusion .setting file."""
    from beatlab.analyzer import analyze_audio
    from beatlab.beat_map import create_beat_map, save_beat_map
    from beatlab.generator import generate_comp

    # AI mode always needs sections
    detect_sections = section_mode or ai
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
    click.echo(f"  Fusion comp written to: {output}", err=True)
    click.echo("Done! Import the .setting file into Resolve's Fusion page.", err=True)


def _get_ai_plan(beat_map: dict, user_prompt: str | None):
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
        plan = create_effect_plan(beat_map, provider, user_prompt=user_prompt)
        click.echo(f"  AI plan: {len(plan.sections)} section(s) configured", err=True)
        return plan
    except Exception as e:
        raise click.ClickException(f"AI effect plan failed: {e}")
