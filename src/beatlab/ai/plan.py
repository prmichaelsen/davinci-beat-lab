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
    intensity_curve: str = "linear"
    attack_frames: int | None = None
    release_frames: int | None = None


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
        sections.append(SectionPlan(
            section_index=s.get("section_index", 0),
            presets=s.get("presets", []),
            custom_effects=s.get("custom_effects", []),
            intensity_curve=s.get("intensity_curve", "linear"),
            attack_frames=s.get("attack_frames"),
            release_frames=s.get("release_frames"),
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

    return warnings
