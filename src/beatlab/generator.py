"""Beat map to Fusion composition generator with preset, section, and AI plan support."""

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
        maker = make_brightness_contrast
    return maker(name=name, source_op=source_op, pos_x=pos_x)


def _preset_from_custom(custom: dict) -> EffectPreset:
    """Create an ad-hoc EffectPreset from a custom effect dict in an AI plan."""
    return EffectPreset(
        name=f"custom_{custom.get('parameter', 'fx')}",
        description="AI-generated custom effect",
        node_type=custom.get("node_type", "BrightnessContrast"),
        parameter=custom.get("parameter", "Gain"),
        base_value=custom.get("base_value", 1.0),
        peak_value=custom.get("peak_value", 1.2),
        attack_frames=custom.get("attack_frames", 2),
        release_frames=custom.get("release_frames", 4),
        curve=custom.get("curve", "smooth"),
    )


def _get_section_for_beat(beat: dict, sections: list[dict]) -> int | None:
    """Find which section index a beat belongs to."""
    beat_time = beat.get("time", 0)
    for i, sec in enumerate(sections):
        if sec.get("start_time", 0) <= beat_time < sec.get("end_time", 0):
            return i
    return None


def generate_comp(
    beat_map: dict,
    effect: str | None = None,
    preset_names: list[str] | None = None,
    attack_frames: int | None = None,
    release_frames: int | None = None,
    intensity_curve: str = "linear",
    section_mode: bool = False,
    overshoot: bool = False,
    effect_plan: object | None = None,
) -> FusionComp:
    """Generate a Fusion comp from a beat map.

    Args:
        beat_map: Parsed beat map dict (from JSON).
        effect: Legacy effect type — "zoom", "flash", "glow", or "all".
        preset_names: List of preset names to apply.
        attack_frames: Override attack frames.
        release_frames: Override release frames.
        intensity_curve: Intensity mapping curve.
        section_mode: If True, vary presets based on detected sections.
        overshoot: If True, add overshoot keyframe past peak.
        effect_plan: EffectPlan from AI director (overrides all other preset selection).

    Returns:
        FusionComp ready to serialize.
    """
    comp = FusionComp()
    beats = beat_map["beats"]
    sections = beat_map.get("sections", [])

    if effect_plan is not None:
        _generate_from_plan(comp, beats, sections, effect_plan)
        return comp

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
        _generate_section_aware(
            comp, beats, sections, presets, intensity_curve,
            attack_frames, release_frames, overshoot,
        )
    else:
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


def _generate_from_plan(
    comp: FusionComp,
    beats: list[dict],
    sections: list[dict],
    effect_plan: object,
) -> None:
    """Generate nodes from an AI effect plan."""
    # Build a map of section_index → SectionPlan
    plan_map: dict[int, object] = {}
    for sp in effect_plan.sections:
        plan_map[sp.section_index] = sp

    # Collect all unique (node_type, parameter) combos across the plan
    all_presets_needed: dict[str, EffectPreset] = {}
    for sp in effect_plan.sections:
        for pname in sp.presets:
            if pname in PRESETS:
                all_presets_needed[pname] = PRESETS[pname]
        for custom in sp.custom_effects:
            p = _preset_from_custom(custom)
            all_presets_needed[p.name] = p

    # Group by (node_type, parameter)
    grouped: dict[tuple[str, str], list[str]] = {}
    for pname, p in all_presets_needed.items():
        key = (p.node_type, p.parameter)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(pname)

    prev_node_name: str | None = None
    pos_x = 0.0

    for (node_type, parameter), preset_names_group in grouped.items():
        base_preset = all_presets_needed[preset_names_group[0]]
        node_name = f"AI{parameter}"
        node = _make_node_for_preset(base_preset, node_name, prev_node_name, pos_x)

        track = KeyframeTrack()
        for beat in beats:
            sec_idx = _get_section_for_beat(beat, sections)
            sp = plan_map.get(sec_idx) if sec_idx is not None else None

            # Find the best preset for this beat's section
            preset = base_preset
            if sp is not None:
                # Check plan's presets for one matching this parameter
                for pname in sp.presets:
                    if pname in all_presets_needed and all_presets_needed[pname].parameter == parameter:
                        preset = all_presets_needed[pname]
                        break
                # Check custom effects
                for custom in sp.custom_effects:
                    cp = _preset_from_custom(custom)
                    if cp.parameter == parameter:
                        preset = cp
                        break

            curve = sp.intensity_curve if sp else "linear"
            atk = sp.attack_frames if sp and sp.attack_frames else preset.attack_frames
            rel = sp.release_frames if sp and sp.release_frames else preset.release_frames

            intensity = beat.get("intensity", 1.0)
            peak = apply_intensity(preset, intensity, curve=curve)
            track.add_pulse(
                beat["frame"], base_value=preset.base_value, peak_value=peak,
                attack_frames=atk, release_frames=rel,
                interpolation=preset.curve,
            )

        node.animated[parameter] = track
        comp.add_node(node)
        prev_node_name = node.name
        pos_x += 110

    comp.active_tool = prev_node_name


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
    all_presets: dict[str, EffectPreset] = {}
    for sec in sections:
        for pname in presets_for_section(sec["type"]):
            if pname in PRESETS:
                all_presets[pname] = PRESETS[pname]
    for p in fallback_presets:
        all_presets[p.name] = p

    grouped: dict[tuple[str, str], list[EffectPreset]] = {}
    for p in all_presets.values():
        key = (p.node_type, p.parameter)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(p)

    prev_node_name: str | None = None
    pos_x = 0.0

    for (node_type, parameter), preset_group in grouped.items():
        base_preset = preset_group[0]
        node_name = f"Beat{parameter}"
        node = _make_node_for_preset(base_preset, node_name, prev_node_name, pos_x)

        track = KeyframeTrack()
        for beat in beats:
            section_type = beat.get("section", "mid_energy")
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
    """Insert overshoot keyframes."""
    original_kfs = list(track.keyframes)
    track.keyframes.clear()

    i = 0
    while i < len(original_kfs):
        kf = original_kfs[i]
        is_peak = (
            kf.value != preset.base_value
            and i + 1 < len(original_kfs)
            and original_kfs[i + 1].value == preset.base_value
        )
        if is_peak:
            overshoot_val = kf.value + (kf.value - preset.base_value) * 0.2
            track.keyframes.append(kf)
            settle_frame = kf.frame + max(1, release_frames // 3)
            track.add(settle_frame, overshoot_val * 0.95, kf.interpolation)
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
    effect_plan: object | None = None,
) -> None:
    """Load a beat map JSON and generate a .setting file."""
    beat_map = load_beat_map(beat_map_path)
    comp = generate_comp(
        beat_map, effect=effect, preset_names=preset_names,
        attack_frames=attack_frames, release_frames=release_frames,
        intensity_curve=intensity_curve, section_mode=section_mode,
        overshoot=overshoot, effect_plan=effect_plan,
    )
    comp.save(output_path)
