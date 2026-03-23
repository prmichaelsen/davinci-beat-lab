"""Effect preset library — configurable effect types for beat-synced visuals."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class EffectPreset:
    """Defines a single beat-synced effect type."""

    name: str
    description: str
    node_type: str        # Transform, BrightnessContrast, Glow
    parameter: str        # Which parameter to keyframe (Size, Gain, Glow, etc.)
    base_value: float     # Resting value
    peak_value: float     # Maximum effect value (at intensity=1.0)
    attack_frames: int    # Frames to reach peak
    release_frames: int   # Frames to return to base
    curve: str            # linear, smooth, step


# ── Preset Registry ──────────────────────────────────────────────────────────

PRESETS: dict[str, EffectPreset] = {}


def _register(preset: EffectPreset) -> EffectPreset:
    PRESETS[preset.name] = preset
    return preset


_register(EffectPreset(
    name="zoom_pulse",
    description="Gentle zoom in/out on each beat",
    node_type="Transform", parameter="Size",
    base_value=1.0, peak_value=1.15,
    attack_frames=2, release_frames=4, curve="smooth",
))

_register(EffectPreset(
    name="zoom_bounce",
    description="Snappy zoom with fast attack and slow release",
    node_type="Transform", parameter="Size",
    base_value=1.0, peak_value=1.25,
    attack_frames=1, release_frames=6, curve="smooth",
))

_register(EffectPreset(
    name="flash",
    description="Brightness flash on beat",
    node_type="BrightnessContrast", parameter="Gain",
    base_value=1.0, peak_value=1.8,
    attack_frames=1, release_frames=3, curve="linear",
))

_register(EffectPreset(
    name="glow_swell",
    description="Subtle bloom that swells on beat — use sparingly, can soften image at high values",
    node_type="Glow", parameter="Glow",
    base_value=0.0, peak_value=0.3,
    attack_frames=3, release_frames=6, curve="smooth",
))

_register(EffectPreset(
    name="hard_cut",
    description="Sharp brightness spike — instant on, instant off",
    node_type="BrightnessContrast", parameter="Gain",
    base_value=1.0, peak_value=2.5,
    attack_frames=0, release_frames=1, curve="step",
))

_register(EffectPreset(
    name="contrast_pop",
    description="Contrast boost on beat for punchy look",
    node_type="BrightnessContrast", parameter="Contrast",
    base_value=0.0, peak_value=0.5,
    attack_frames=1, release_frames=4, curve="smooth",
))


# ── Intensity Mapping ────────────────────────────────────────────────────────

def apply_intensity(
    preset: EffectPreset,
    intensity: float,
    curve: str = "linear",
) -> float:
    """Scale a preset's peak value by beat intensity.

    Args:
        preset: The effect preset.
        intensity: Beat intensity (0.0 to 1.0).
        curve: Mapping curve — "linear", "exponential", "logarithmic".

    Returns:
        Scaled peak value.
    """
    if curve == "exponential":
        mapped = intensity ** 2
    elif curve == "logarithmic":
        mapped = math.log1p(intensity * (math.e - 1)) if intensity > 0 else 0.0
    else:
        mapped = intensity

    delta = preset.peak_value - preset.base_value
    return preset.base_value + delta * mapped


# ── Section → Preset Mapping ─────────────────────────────────────────────────

SECTION_PRESET_MAP: dict[str, list[str]] = {
    "low_energy": ["zoom_pulse", "glow_swell"],
    "mid_energy": ["flash", "zoom_bounce"],
    "high_energy": ["hard_cut", "flash", "zoom_bounce"],
}


def presets_for_section(section_type: str) -> list[str]:
    """Return preset names appropriate for a section energy level."""
    return SECTION_PRESET_MAP.get(section_type, ["zoom_pulse"])


def list_presets() -> list[dict]:
    """Return all presets as dicts for display."""
    return [
        {
            "name": p.name,
            "description": p.description,
            "node": p.node_type,
            "parameter": p.parameter,
            "curve": p.curve,
        }
        for p in PRESETS.values()
    ]
