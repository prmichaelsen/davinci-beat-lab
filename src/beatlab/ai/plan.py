"""Effect plan schema and validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from beatlab.presets import PRESETS


@dataclass
class SectionPlan:
    """Effect plan for a single section."""

    section_index: int
    presets: list[str] = field(default_factory=list)
    custom_effects: list[dict] = field(default_factory=list)
    sustained_effects: list[dict] = field(default_factory=list)
    intensity_curve: str = "linear"
    attack_frames: int | None = None
    release_frames: int | None = None
    style_prompt: str | None = None  # SD style for video render mode
    wan_denoise: float | None = None  # Wan2.1 denoising strength (0.0-1.0)
    transition_frames: int | None = None  # FILM transition frames at section boundary (2-30)


@dataclass
class EffectPlan:
    """Complete effect plan mapping sections to presets and parameters."""

    sections: list[SectionPlan] = field(default_factory=list)


def parse_effect_plan(text: str) -> EffectPlan:
    """Parse an effect plan from LLM response text.

    Handles raw JSON or JSON wrapped in markdown code fences.

    Raises:
        ValueError: If the response doesn't contain valid JSON.
    """
    # Try to extract JSON from markdown code fence
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    json_str = fence_match.group(1) if fence_match else text.strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse effect plan JSON from LLM response: {e}\n"
            f"Response was: {text[:500]}"
        )

    if "sections" not in data:
        raise ValueError("Effect plan JSON missing 'sections' key")

    sections = []
    for s in data["sections"]:
        # Support grouped sections: "section_indices": [0, 1, 2]
        indices = s.get("section_indices", None)
        if indices is None:
            indices = [s.get("section_index", 0)]

        for idx in indices:
            sections.append(SectionPlan(
                section_index=idx,
                presets=s.get("presets", []),
                custom_effects=s.get("custom_effects", []),
                sustained_effects=s.get("sustained_effects", []),
                intensity_curve=s.get("intensity_curve", "linear"),
                attack_frames=s.get("attack_frames"),
                release_frames=s.get("release_frames"),
                style_prompt=s.get("style_prompt"),
                wan_denoise=s.get("wan_denoise"),
                transition_frames=s.get("transition_frames"),
            ))

    return EffectPlan(sections=sections)


def validate_effect_plan(plan: EffectPlan) -> list[str]:
    """Validate an effect plan and return warnings.

    Returns a list of warning strings. Empty list means fully valid.
    """
    warnings = []
    available = set(PRESETS.keys())

    for sp in plan.sections:
        for preset_name in sp.presets:
            if preset_name not in available:
                warnings.append(
                    f"Section {sp.section_index}: unknown preset '{preset_name}' "
                    f"(available: {', '.join(sorted(available))})"
                )

        for i, custom in enumerate(sp.custom_effects):
            for required_key in ("node_type", "parameter", "base_value", "peak_value"):
                if required_key not in custom:
                    warnings.append(
                        f"Section {sp.section_index}: custom effect {i} "
                        f"missing required key '{required_key}'"
                    )

        if sp.wan_denoise is not None and not (0.0 <= sp.wan_denoise <= 1.0):
            warnings.append(
                f"Section {sp.section_index}: wan_denoise {sp.wan_denoise} "
                f"out of range (expected 0.0-1.0)"
            )

        if sp.transition_frames is not None and not (0 <= sp.transition_frames <= 60):
            warnings.append(
                f"Section {sp.section_index}: transition_frames {sp.transition_frames} "
                f"out of range (expected 1-60)"
            )

    return warnings
