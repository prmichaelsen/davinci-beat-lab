"""Tests for effect presets, intensity mapping, and section detection."""

import math

from beatlab.presets import (
    PRESETS,
    EffectPreset,
    apply_intensity,
    list_presets,
    presets_for_section,
)


class TestPresetRegistry:
    def test_at_least_4_presets(self):
        assert len(PRESETS) >= 4

    def test_zoom_pulse_exists(self):
        p = PRESETS["zoom_pulse"]
        assert p.node_type == "Transform"
        assert p.parameter == "Size"
        assert p.base_value == 1.0
        assert p.peak_value == 1.08

    def test_flash_exists(self):
        p = PRESETS["flash"]
        assert p.node_type == "BrightnessContrast"
        assert p.curve == "linear"

    def test_hard_cut_exists(self):
        p = PRESETS["hard_cut"]
        assert p.curve == "step"

    def test_list_presets(self):
        result = list_presets()
        assert len(result) >= 4
        names = [p["name"] for p in result]
        assert "zoom_pulse" in names
        assert "flash" in names


class TestIntensityMapping:
    def test_linear_full_intensity(self):
        p = PRESETS["zoom_pulse"]
        val = apply_intensity(p, 1.0, "linear")
        assert val == p.peak_value

    def test_linear_zero_intensity(self):
        p = PRESETS["zoom_pulse"]
        val = apply_intensity(p, 0.0, "linear")
        assert val == p.base_value

    def test_linear_half_intensity(self):
        p = PRESETS["zoom_pulse"]
        val = apply_intensity(p, 0.5, "linear")
        expected = p.base_value + (p.peak_value - p.base_value) * 0.5
        assert abs(val - expected) < 1e-6

    def test_exponential_compresses_low(self):
        p = PRESETS["zoom_pulse"]
        linear_val = apply_intensity(p, 0.3, "linear")
        exp_val = apply_intensity(p, 0.3, "exponential")
        assert exp_val < linear_val  # exponential compresses low values

    def test_logarithmic_boosts_low(self):
        p = PRESETS["zoom_pulse"]
        linear_val = apply_intensity(p, 0.3, "linear")
        log_val = apply_intensity(p, 0.3, "logarithmic")
        assert log_val > linear_val  # logarithmic boosts low values


class TestSectionMapping:
    def test_low_energy_section(self):
        presets = presets_for_section("low_energy")
        assert "zoom_pulse" in presets
        assert "hard_cut" not in presets

    def test_high_energy_section(self):
        presets = presets_for_section("high_energy")
        assert "hard_cut" in presets or "flash" in presets

    def test_unknown_section_fallback(self):
        presets = presets_for_section("unknown_type")
        assert len(presets) > 0  # Should return default
