"""Beat map to Fusion composition generator."""

from __future__ import annotations

from beatlab.beat_map import load_beat_map
from beatlab.fusion.keyframes import KeyframeTrack
from beatlab.fusion.nodes import (
    FusionNode,
    make_brightness_contrast,
    make_glow,
    make_transform,
)
from beatlab.fusion.setting_writer import FusionComp


def generate_comp(
    beat_map: dict,
    effect: str = "zoom",
    attack_frames: int = 2,
    release_frames: int = 4,
) -> FusionComp:
    """Generate a Fusion comp from a beat map.

    Args:
        beat_map: Parsed beat map dict (from JSON).
        effect: Effect type — "zoom", "flash", "glow", or "all".
        attack_frames: Frames before beat peak.
        release_frames: Frames after beat peak to return to base.

    Returns:
        FusionComp ready to serialize.
    """
    comp = FusionComp()
    beats = beat_map["beats"]
    prev_node_name: str | None = None
    pos_x = 0.0

    if effect in ("zoom", "all"):
        node = make_transform(name="BeatZoom", source_op=prev_node_name, pos_x=pos_x)
        track = KeyframeTrack()
        for beat in beats:
            intensity = beat.get("intensity", 1.0)
            peak = 1.0 + 0.08 * intensity  # Scale zoom by intensity
            track.add_pulse(
                beat["frame"], base_value=1.0, peak_value=peak,
                attack_frames=attack_frames, release_frames=release_frames,
            )
        node.animated["Size"] = track
        comp.add_node(node)
        prev_node_name = node.name
        pos_x += 110

    if effect in ("flash", "all"):
        node = make_brightness_contrast(
            name="BeatFlash", source_op=prev_node_name, pos_x=pos_x,
        )
        track = KeyframeTrack()
        for beat in beats:
            intensity = beat.get("intensity", 1.0)
            peak = 1.0 + 0.4 * intensity  # Scale brightness by intensity
            track.add_pulse(
                beat["frame"], base_value=1.0, peak_value=peak,
                attack_frames=max(1, attack_frames // 2),
                release_frames=release_frames,
                interpolation="linear",
            )
        node.animated["Gain"] = track
        comp.add_node(node)
        prev_node_name = node.name
        pos_x += 110

    if effect in ("glow", "all"):
        node = make_glow(name="BeatGlow", source_op=prev_node_name, pos_x=pos_x)
        track = KeyframeTrack()
        for beat in beats:
            intensity = beat.get("intensity", 1.0)
            peak = 0.5 * intensity
            track.add_pulse(
                beat["frame"], base_value=0.0, peak_value=peak,
                attack_frames=attack_frames + 1,
                release_frames=release_frames + 2,
            )
        node.animated["Glow"] = track
        comp.add_node(node)
        prev_node_name = node.name
        pos_x += 110

    comp.active_tool = prev_node_name
    return comp


def generate_from_file(
    beat_map_path: str,
    output_path: str,
    effect: str = "zoom",
    attack_frames: int = 2,
    release_frames: int = 4,
) -> None:
    """Load a beat map JSON and generate a .setting file."""
    beat_map = load_beat_map(beat_map_path)
    comp = generate_comp(
        beat_map, effect=effect,
        attack_frames=attack_frames, release_frames=release_frames,
    )
    comp.save(output_path)
