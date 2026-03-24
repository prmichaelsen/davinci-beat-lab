"""Tests for plan patching."""

from beatlab.render.patcher import merge_plan, generate_patch_from_updates


class TestMergePlan:
    def _base_plan(self):
        return {
            "sections": [
                {"section_index": 0, "style_prompt": "dark moody", "presets": ["zoom_pulse"]},
                {"section_index": 1, "style_prompt": "bright neon", "presets": ["flash"]},
                {"section_index": 2, "style_prompt": "ethereal mist", "presets": ["glow_swell"]},
            ]
        }

    def test_patch_changes_style_prompt(self):
        base = self._base_plan()
        patch = {"sections": [{"section_index": 1, "style_prompt": "psychedelic fractal"}]}
        merged, changed = merge_plan(base, patch)
        assert 1 in changed
        sec1 = next(s for s in merged["sections"] if s["section_index"] == 1)
        assert sec1["style_prompt"] == "psychedelic fractal"
        assert sec1["presets"] == ["flash"]  # unchanged

    def test_unchanged_sections_not_in_changed(self):
        base = self._base_plan()
        patch = {"sections": [{"section_index": 1, "style_prompt": "psychedelic fractal"}]}
        _, changed = merge_plan(base, patch)
        assert 0 not in changed
        assert 2 not in changed

    def test_no_change_same_value(self):
        base = self._base_plan()
        patch = {"sections": [{"section_index": 1, "style_prompt": "bright neon"}]}
        _, changed = merge_plan(base, patch)
        assert len(changed) == 0

    def test_multiple_patches(self):
        base = self._base_plan()
        patch = {"sections": [
            {"section_index": 0, "style_prompt": "new 0"},
            {"section_index": 2, "style_prompt": "new 2"},
        ]}
        _, changed = merge_plan(base, patch)
        assert sorted(changed) == [0, 2]

    def test_preserves_all_sections(self):
        base = self._base_plan()
        patch = {"sections": [{"section_index": 1, "style_prompt": "new"}]}
        merged, _ = merge_plan(base, patch)
        assert len(merged["sections"]) == 3


class TestGeneratePatch:
    def test_generates_valid_patch(self):
        updates = [
            {"section_index": 88, "style_prompt": "new prompt"},
            {"section_index": 89, "style_prompt": "another prompt"},
        ]
        patch = generate_patch_from_updates(updates)
        assert len(patch["sections"]) == 2
        assert patch["sections"][0]["section_index"] == 88
