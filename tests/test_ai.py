"""Tests for AI effect director modules."""

import json

from beatlab.ai.plan import EffectPlan, SectionPlan, parse_effect_plan, validate_effect_plan
from beatlab.ai.prompt import build_system_prompt, build_user_prompt
from beatlab.ai.provider import LLMProvider


class MockProvider(LLMProvider):
    """Mock LLM provider that returns a fixed response."""

    def __init__(self, response: str):
        self._response = response

    def complete(self, system: str, user: str) -> str:
        return self._response


class TestEffectPlan:
    def test_parse_valid_json(self):
        text = json.dumps({
            "sections": [
                {
                    "section_index": 0,
                    "presets": ["zoom_pulse", "flash"],
                    "custom_effects": [],
                    "intensity_curve": "exponential",
                    "attack_frames": 1,
                    "release_frames": 3,
                }
            ]
        })
        plan = parse_effect_plan(text)
        assert len(plan.sections) == 1
        assert plan.sections[0].presets == ["zoom_pulse", "flash"]
        assert plan.sections[0].intensity_curve == "exponential"

    def test_parse_json_in_code_fence(self):
        text = '```json\n{"sections": [{"section_index": 0, "presets": ["flash"]}]}\n```'
        plan = parse_effect_plan(text)
        assert len(plan.sections) == 1
        assert plan.sections[0].presets == ["flash"]

    def test_parse_invalid_json_raises(self):
        try:
            parse_effect_plan("not json at all")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Failed to parse" in str(e)

    def test_parse_missing_sections_raises(self):
        try:
            parse_effect_plan('{"foo": "bar"}')
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "missing 'sections'" in str(e)

    def test_validate_unknown_preset_warns(self):
        plan = EffectPlan(sections=[
            SectionPlan(section_index=0, presets=["nonexistent_preset"]),
        ])
        warnings = validate_effect_plan(plan)
        assert len(warnings) == 1
        assert "unknown preset" in warnings[0]

    def test_validate_valid_preset_no_warnings(self):
        plan = EffectPlan(sections=[
            SectionPlan(section_index=0, presets=["zoom_pulse", "flash"]),
        ])
        warnings = validate_effect_plan(plan)
        assert len(warnings) == 0

    def test_validate_custom_effect_missing_key_warns(self):
        plan = EffectPlan(sections=[
            SectionPlan(
                section_index=0,
                presets=[],
                custom_effects=[{"node_type": "Transform"}],  # missing parameter, base_value, peak_value
            ),
        ])
        warnings = validate_effect_plan(plan)
        assert len(warnings) == 3  # missing parameter, base_value, peak_value

    def test_validate_complete_custom_effect_no_warnings(self):
        plan = EffectPlan(sections=[
            SectionPlan(
                section_index=0,
                presets=[],
                custom_effects=[{
                    "node_type": "Transform",
                    "parameter": "Size",
                    "base_value": 1.0,
                    "peak_value": 1.25,
                }],
            ),
        ])
        warnings = validate_effect_plan(plan)
        assert len(warnings) == 0


class TestPrompt:
    def test_system_prompt_includes_presets(self):
        prompt = build_system_prompt()
        assert "zoom_pulse" in prompt
        assert "flash" in prompt
        assert "glow_swell" in prompt
        assert "JSON" in prompt

    def test_user_prompt_includes_sections(self):
        beat_map = {
            "tempo": 120.0,
            "duration": 60.0,
            "beats": [{"time": 0.5, "frame": 15, "intensity": 0.8}],
            "sections": [
                {
                    "start_time": 0.0, "end_time": 30.0,
                    "type": "low_energy", "label": "verse",
                    "spectral": {"centroid": 0.3, "rms_energy": 0.2, "rolloff": 0.4, "contrast": 0.3},
                },
            ],
        }
        prompt = build_user_prompt(beat_map)
        assert "120.0 BPM" in prompt
        assert "low_energy" in prompt
        assert "centroid=0.30" in prompt

    def test_user_prompt_includes_creative_direction(self):
        beat_map = {"tempo": 120.0, "duration": 60.0, "beats": [], "sections": []}
        prompt = build_user_prompt(beat_map, user_prompt="make it dreamy")
        assert "make it dreamy" in prompt
        assert "Creative Direction" in prompt

    def test_user_prompt_no_direction(self):
        beat_map = {"tempo": 120.0, "duration": 60.0, "beats": [], "sections": []}
        prompt = build_user_prompt(beat_map)
        assert "Creative Direction" not in prompt


class TestDirector:
    def test_create_effect_plan_with_mock(self):
        from beatlab.ai.director import create_effect_plan

        response = json.dumps({
            "sections": [
                {"section_index": 0, "presets": ["zoom_pulse"], "custom_effects": []},
                {"section_index": 1, "presets": ["flash", "zoom_bounce"], "custom_effects": []},
            ]
        })
        provider = MockProvider(response)
        beat_map = {
            "tempo": 120.0, "duration": 60.0,
            "beats": [{"time": 0.5, "frame": 15, "intensity": 0.8}],
            "sections": [
                {"start_time": 0.0, "end_time": 30.0, "type": "low_energy", "label": "verse"},
                {"start_time": 30.0, "end_time": 60.0, "type": "high_energy", "label": "chorus"},
            ],
        }
        plan = create_effect_plan(beat_map, provider)
        assert len(plan.sections) == 2
        assert plan.sections[0].presets == ["zoom_pulse"]
        assert plan.sections[1].presets == ["flash", "zoom_bounce"]


class TestGeneratorWithPlan:
    def test_generate_from_plan(self):
        from beatlab.generator import generate_comp

        beat_map = {
            "version": "1.1", "source_file": "test.mp3",
            "duration": 4.0, "tempo": 120.0, "fps": 30,
            "beats": [
                {"time": 0.5, "frame": 15, "intensity": 0.8},
                {"time": 1.0, "frame": 30, "intensity": 1.0},
                {"time": 2.5, "frame": 75, "intensity": 0.6},
            ],
            "sections": [
                {"start_time": 0.0, "end_time": 2.0, "type": "low_energy", "label": "verse"},
                {"start_time": 2.0, "end_time": 4.0, "type": "high_energy", "label": "chorus"},
            ],
        }
        plan = EffectPlan(sections=[
            SectionPlan(section_index=0, presets=["zoom_pulse"], intensity_curve="linear"),
            SectionPlan(section_index=1, presets=["flash", "zoom_bounce"], intensity_curve="exponential"),
        ])

        comp = generate_comp(beat_map, effect_plan=plan)
        output = comp.serialize()
        assert "Transform" in output
        assert "BrightnessContrast" in output
        assert "BezierSpline" in output

    def test_generate_from_plan_with_custom_effect(self):
        from beatlab.generator import generate_comp

        beat_map = {
            "version": "1.1", "source_file": "test.mp3",
            "duration": 2.0, "tempo": 120.0, "fps": 30,
            "beats": [{"time": 0.5, "frame": 15, "intensity": 1.0}],
            "sections": [
                {"start_time": 0.0, "end_time": 2.0, "type": "high_energy", "label": "chorus"},
            ],
        }
        plan = EffectPlan(sections=[
            SectionPlan(
                section_index=0,
                presets=[],
                custom_effects=[{
                    "node_type": "Transform",
                    "parameter": "Size",
                    "base_value": 1.0,
                    "peak_value": 1.25,
                    "attack_frames": 1,
                    "release_frames": 8,
                    "curve": "smooth",
                }],
            ),
        ])
        comp = generate_comp(beat_map, effect_plan=plan)
        output = comp.serialize()
        assert "Transform" in output
        assert "1.25" in output
