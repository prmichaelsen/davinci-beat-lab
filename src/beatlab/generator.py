"""Beat map to Fusion composition generator with preset and section support."""

from __future__ import annotations

from beatlab.beat_map import load_beat_map
from beatlab.fusion.keyframes import KeyframeTrack
from beatlab.fusion.nodes import (
    make_brightness_contrast,
    make_glow,
    make_transform,
    FusionNode,
)
from beatlab.fusion.setting_writer import FusionComp
from beatlab.presets import PRESETS, EffectPreset, apply_intensity, presets_for_section


NODE_MAKERS = {
    "Transform": make_transform,
    "BrightnessContrast": make_brightness_contrast,
    "Glow": make_glow,
}


def _make_node_for_preset(preset: EffectPreset, name: str, source_op: str | None, pos_x: float) -> FusionNode:
    """Create the right Fusion node type for a preset."""
    maker = NODE_MAKERS.get(preset.node_type)
    if maker is None:
        # Fallback to BrightnessContrast for unknown types
        maker = make_brightness_contrast
    return maker(name=name, source_op=source_op, pos_x=pos_x)


def generate_comp(
    beat_map: dict,
    effect: str | None = None,
    preset_names: list[str] | None = None,
    attack_frames: int | None = None,
    release_frames: int | None = None,
    intensity_curve: str = "linear",
    section_mode: bool = False,
    overshoot: bool = False,
) -> FusionComp:
    """Generate a Fusion comp from a beat map.

    Args:
        beat_map: Parsed beat map dict (from JSON).
        effect: Legacy effect type — "zoom", "flash", "glow", or "all". Ignored if preset_names given.
        preset_names: List of preset names to apply (e.g. ["zoom_pulse", "flash"]).
        attack_frames: Override attack frames (None = use preset default).
        release_frames: Override release frames (None = use preset default).
        intensity_curve: Intensity mapping — "linear", "exponential", "logarithmic".
        section_mode: If True, vary presets based on detected sections in beat_map.
        overshoot: If True, add overshoot keyframe past peak before settling.

    Returns:
        FusionComp ready to serialize.
    """
    comp = FusionComp()
    beats = beat_map["beats"]
    sections = beat_map.get("sections", [])
    prev_node_name: str | None = None
    pos_x = 0.0

    # Resolve which presets to use
    if preset_names:
        presets = [PRESETS[n] for n in preset_names if n in PRESETS]
    elif effect == "all":
        presets = [PRESETS["zoom_pulse"], PRESETS["flash"], PRESETS["glow_swell"]]
    elif effect == "flash":
        presets = [PRESETS["flash"]]
    elif effect == "glow":
        presets = [PRESETS["glow_swell"]]
    else:
        presets = [PRESETS["zoom_pulse"]]

    if section_mode and sections:
        # Section-aware: build nodes for each unique preset across all sections
        _generate_section_aware(
            comp, beats, sections, presets, intensity_curve,
            attack_frames, release_frames, overshoot,
        )
    else:
        # Simple mode: apply chosen presets to all beats
        for preset in presets:
            node_name = f"Beat{preset.name.title().replace('_', '')}"
            node = _make_node_for_preset(preset, node_name, prev_node_name, pos_x)

            atk = attack_frames if attack_frames is not None else preset.attack_frames
            rel = release_frames if release_frames is not None else preset.release_frames

            track = KeyframeTrack()
            for beat in beats:
                intensity = beat.get("intensity", 1.0)
                peak = apply_intensity(preset, intensity, curve=intensity_curve)
                track.add_pulse(
                    beat["frame"], base_value=preset.base_value, peak_value=peak,
                    attack_frames=atk, release_frames=rel,
                    interpolation=preset.curve,
                )

            if overshoot and preset.node_type == "Transform":
                _add_overshoot(track, beats, preset, intensity_curve, atk, rel)

            node.animated[preset.parameter] = track
            comp.add_node(node)
            prev_node_name = node.name
            pos_x += 110

    comp.active_tool = prev_node_name
    return comp


def _generate_section_aware(
    comp: FusionComp,
    beats: list[dict],
    sections: list[dict],
    fallback_presets: list[EffectPreset],
    intensity_curve: str,
    attack_frames: int | None,
    release_frames: int | None,
    overshoot: bool,
) -> None:
    """Generate nodes with section-aware preset switching."""
    # Build a lookup: for each beat, determine its section type
    # Then for each unique (node_type, parameter) combo, create one node with all keyframes

    # Collect all presets we'll need across sections
    all_presets: dict[str, EffectPreset] = {}
    for sec in sections:
        for pname in presets_for_section(sec["type"]):
            if pname in PRESETS:
                all_presets[pname] = PRESETS[pname]
    # Also include fallbacks
    for p in fallback_presets:
        all_presets[p.name] = p

    # Group by (node_type, parameter) to avoid duplicate nodes
    grouped: dict[tuple[str, str], list[EffectPreset]] = {}
    for p in all_presets.values():
        key = (p.node_type, p.parameter)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(p)

    prev_node_name: str | None = None
    pos_x = 0.0

    for (node_type, parameter), preset_group in grouped.items():
        # Use the first preset as the base node config
        base_preset = preset_group[0]
        node_name = f"Beat{parameter}"
        node = _make_node_for_preset(base_preset, node_name, prev_node_name, pos_x)

        track = KeyframeTrack()
        for beat in beats:
            section_type = beat.get("section", "mid_energy")
            # Pick the best preset for this section from available ones
            section_presets = presets_for_section(section_type)
            preset = base_preset
            for sp_name in section_presets:
                if sp_name in all_presets and all_presets[sp_name].parameter == parameter:
                    preset = all_presets[sp_name]
                    break

            intensity = beat.get("intensity", 1.0)
            peak = apply_intensity(preset, intensity, curve=intensity_curve)
            atk = attack_frames if attack_frames is not None else preset.attack_frames
            rel = release_frames if release_frames is not None else preset.release_frames
            track.add_pulse(
                beat["frame"], base_value=preset.base_value, peak_value=peak,
                attack_frames=atk, release_frames=rel, interpolation=preset.curve,
            )

        node.animated[parameter] = track
        comp.add_node(node)
        prev_node_name = node.name
        pos_x += 110


def _add_overshoot(
    track: KeyframeTrack,
    beats: list[dict],
    preset: EffectPreset,
    intensity_curve: str,
    attack_frames: int,
    release_frames: int,
) -> None:
    """Insert overshoot keyframes: peak goes 20% past target, then settles.

    This modifies the track in-place by adjusting existing peak keyframes
    and inserting settle keyframes.
    """
    # We need to re-do the keyframes with overshoot. Clear and rebuild.
    original_kfs = list(track.keyframes)
    track.keyframes.clear()

    i = 0
    while i < len(original_kfs):
        kf = original_kfs[i]
        # Detect peak keyframes (value != base_value and next kf is base_value)
        is_peak = (
            kf.value != preset.base_value
            and i + 1 < len(original_kfs)
            and original_kfs[i + 1].value == preset.base_value
        )
        if is_peak:
            overshoot_val = kf.value + (kf.value - preset.base_value) * 0.2
            track.keyframes.append(kf)  # peak (unchanged)
            settle_frame = kf.frame + max(1, release_frames // 3)
            track.add(settle_frame, overshoot_val * 0.95, kf.interpolation)  # slight overshoot settle
        else:
            track.keyframes.append(kf)
        i += 1


def generate_from_file(
    beat_map_path: str,
    output_path: str,
    effect: str | None = "zoom",
    preset_names: list[str] | None = None,
    attack_frames: int | None = None,
    release_frames: int | None = None,
    intensity_curve: str = "linear",
    section_mode: bool = False,
    overshoot: bool = False,
) -> None:
    """Load a beat map JSON and generate a .setting file."""
    beat_map = load_beat_map(beat_map_path)
    comp = generate_comp(
        beat_map, effect=effect, preset_names=preset_names,
        attack_frames=attack_frames, release_frames=release_frames,
        intensity_curve=intensity_curve, section_mode=section_mode,
        overshoot=overshoot,
    )
    comp.save(output_path)
