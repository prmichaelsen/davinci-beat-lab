"""System and user prompt construction for the AI director."""

from __future__ import annotations

from beatlab.presets import PRESETS


def build_system_prompt() -> str:
    """Build the system prompt with preset catalog and creative guidelines."""
    preset_lines = []
    for p in PRESETS.values():
        preset_lines.append(
            f"- **{p.name}**: {p.description}\n"
            f"  Node: {p.node_type}.{p.parameter} | "
            f"Base: {p.base_value} → Peak: {p.peak_value} | "
            f"Attack: {p.attack_frames}f, Release: {p.release_frames}f | "
            f"Curve: {p.curve}"
        )
    preset_catalog = "\n".join(preset_lines)

    return f"""You are an AI visual effects director for music videos and beat-synced content. Your job is to analyze audio section data and choose the best visual effects for each section of a song.

## Available Effect Presets

{preset_catalog}

## Custom Effects

You may also define custom effects beyond the preset catalog. A custom effect has:
- node_type: "Transform", "BrightnessContrast", or "Glow"
- parameter: the parameter to keyframe (e.g. "Size", "Gain", "Glow")
- base_value: resting value
- peak_value: effect peak value
- attack_frames: frames to reach peak
- release_frames: frames to return to base
- curve: "linear", "smooth", or "step"

## Creative Guidelines

1. **Match effects to energy**: Use subtle effects (zoom_pulse, glow_swell) for low-energy sections, intense effects (flash, hard_cut, zoom_bounce) for high-energy sections.
2. **Layer for impact**: High-energy sections like drops can combine multiple effects (e.g. flash + zoom_bounce).
3. **Maintain coherence**: Similar sections should use similar effects for visual consistency.
4. **Vary on repeats**: If a section type repeats (e.g. second chorus), introduce subtle variation — add an extra layered effect, adjust parameters slightly, or use a different curve.
5. **Use spectral data**: High spectral centroid suggests bright/aggressive music, low centroid suggests mellow/warm. Use this to inform effect choices.
6. **Intensity curves**: Use "exponential" for sections where you want beats to hit harder. Use "logarithmic" for sections where even quiet beats should be visible. Use "linear" as the default.

## Output Format

Respond with ONLY a JSON object (no markdown, no explanation). The JSON must follow this schema:

```json
{{
  "sections": [
    {{
      "section_index": 0,
      "presets": ["preset_name"],
      "custom_effects": [],
      "intensity_curve": "linear",
      "attack_frames": 2,
      "release_frames": 4
    }}
  ]
}}
```

Every section in the input must have a corresponding entry in your output."""


def build_user_prompt(
    beat_map: dict,
    user_prompt: str | None = None,
) -> str:
    """Build the user prompt from beat map data and optional creative direction."""
    sections = beat_map.get("sections", [])
    tempo = beat_map.get("tempo", 0)
    duration = beat_map.get("duration", 0)
    total_beats = len(beat_map.get("beats", []))

    lines = [
        f"## Track Info",
        f"- Tempo: {tempo:.1f} BPM",
        f"- Duration: {duration:.1f}s",
        f"- Total beats: {total_beats}",
        f"- Sections: {len(sections)}",
        "",
        "## Sections",
        "",
    ]

    for i, sec in enumerate(sections):
        sec_duration = sec.get("end_time", 0) - sec.get("start_time", 0)
        # Count beats in this section
        beat_count = sum(
            1 for b in beat_map.get("beats", [])
            if sec.get("start_time", 0) <= b.get("time", 0) < sec.get("end_time", 0)
        )
        avg_intensity = 0.0
        section_beats = [
            b for b in beat_map.get("beats", [])
            if sec.get("start_time", 0) <= b.get("time", 0) < sec.get("end_time", 0)
        ]
        if section_beats:
            avg_intensity = sum(b.get("intensity", 0) for b in section_beats) / len(section_beats)

        line = (
            f"### Section {i} ({sec.get('type', 'unknown')}, {sec.get('label', '')})\n"
            f"- Time: {sec.get('start_time', 0):.1f}s - {sec.get('end_time', 0):.1f}s "
            f"({sec_duration:.1f}s)\n"
            f"- Beats: {beat_count} | Avg intensity: {avg_intensity:.2f}"
        )

        spectral = sec.get("spectral", {})
        if spectral:
            line += (
                f"\n- Spectral: centroid={spectral.get('centroid', 0):.2f}, "
                f"rms={spectral.get('rms_energy', 0):.2f}, "
                f"rolloff={spectral.get('rolloff', 0):.2f}, "
                f"contrast={spectral.get('contrast', 0):.2f}"
            )

        lines.append(line)
        lines.append("")

    if user_prompt:
        lines.append(f"## Creative Direction")
        lines.append(f"{user_prompt}")
        lines.append("")

    lines.append("Generate the effect plan JSON for this track.")

    return "\n".join(lines)
