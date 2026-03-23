"""CLI interface for beatlab."""

from __future__ import annotations

import json
import sys

import click


EFFECT_CHOICES = click.Choice(["zoom", "flash", "glow", "all"])


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
def analyze(audio_file: str, fps: float, output: str | None, sr: int):
    """Analyze an audio file and produce a beat map JSON."""
    from beatlab.analyzer import analyze_audio
    from beatlab.beat_map import create_beat_map, save_beat_map

    click.echo(f"Analyzing: {audio_file}", err=True)
    analysis = analyze_audio(audio_file, sr=sr)
    click.echo(
        f"  Tempo: {analysis['tempo']:.1f} BPM | "
        f"Beats: {len(analysis['beats'])} | "
        f"Onsets: {len(analysis['onsets'])} | "
        f"Duration: {analysis['duration']:.1f}s",
        err=True,
    )

    beat_map = create_beat_map(analysis, fps=fps, source_file=audio_file)

    if output:
        save_beat_map(beat_map, output)
        click.echo(f"  Beat map written to: {output}", err=True)
    else:
        json.dump(beat_map, sys.stdout, indent=2)
        sys.stdout.write("\n")


@main.command()
@click.argument("beats_json", type=click.Path(exists=True))
@click.option("--output", "-o", default="output.setting", type=click.Path(), help="Output .setting file")
@click.option("--effect", default="zoom", type=EFFECT_CHOICES, help="Effect type (default: zoom)")
@click.option("--attack", default=2, type=int, help="Attack frames (default: 2)")
@click.option("--release", default=4, type=int, help="Release frames (default: 4)")
def generate(beats_json: str, output: str, effect: str, attack: int, release: int):
    """Generate a Fusion .setting file from a beat map JSON."""
    from beatlab.generator import generate_from_file

    click.echo(f"Generating Fusion comp: {effect} effect", err=True)
    generate_from_file(
        beats_json, output,
        effect=effect, attack_frames=attack, release_frames=release,
    )
    click.echo(f"  Written to: {output}", err=True)


@main.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--fps", default=30.0, type=float, help="Timeline frame rate (default: 30)")
@click.option("--output", "-o", default="output.setting", type=click.Path(), help="Output .setting file")
@click.option("--effect", default="zoom", type=EFFECT_CHOICES, help="Effect type (default: zoom)")
@click.option("--attack", default=2, type=int, help="Attack frames (default: 2)")
@click.option("--release", default=4, type=int, help="Release frames (default: 4)")
@click.option("--sr", default=22050, type=int, help="Sample rate for analysis (default: 22050)")
@click.option("--beats-out", default=None, type=click.Path(), help="Also save beat map JSON")
def run(
    audio_file: str, fps: float, output: str, effect: str,
    attack: int, release: int, sr: int, beats_out: str | None,
):
    """Full pipeline: audio file → beat analysis → Fusion .setting file."""
    from beatlab.analyzer import analyze_audio
    from beatlab.beat_map import create_beat_map, save_beat_map
    from beatlab.generator import generate_comp

    click.echo(f"Analyzing: {audio_file}", err=True)
    analysis = analyze_audio(audio_file, sr=sr)
    click.echo(
        f"  Tempo: {analysis['tempo']:.1f} BPM | "
        f"Beats: {len(analysis['beats'])} | "
        f"Duration: {analysis['duration']:.1f}s",
        err=True,
    )

    beat_map = create_beat_map(analysis, fps=fps, source_file=audio_file)

    if beats_out:
        save_beat_map(beat_map, beats_out)
        click.echo(f"  Beat map saved to: {beats_out}", err=True)

    click.echo(f"Generating Fusion comp: {effect} effect", err=True)
    comp = generate_comp(
        beat_map, effect=effect,
        attack_frames=attack, release_frames=release,
    )
    comp.save(output)
    click.echo(f"  Fusion comp written to: {output}", err=True)
    click.echo("Done! Import the .setting file into Resolve's Fusion page.", err=True)
