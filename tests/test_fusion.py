"""Tests for Fusion .setting generation."""

import json
import tempfile
from pathlib import Path

from beatlab.fusion.keyframes import Keyframe, KeyframeTrack
from beatlab.fusion.nodes import make_transform, make_brightness_contrast
from beatlab.fusion.setting_writer import FusionComp
from beatlab.generator import generate_comp


class TestKeyframe:
    def test_linear_keyframe(self):
        kf = Keyframe(frame=10, value=1.5, interpolation="linear")
        lua = kf.to_lua(prev_frame=0, next_frame=20)
        assert "[10]" in lua
        assert "1.5" in lua
        assert "Linear = true" in lua

    def test_smooth_keyframe_has_handles(self):
        kf = Keyframe(frame=10, value=1.0, interpolation="smooth")
        lua = kf.to_lua(prev_frame=0, next_frame=20)
        assert "LH" in lua
        assert "RH" in lua

    def test_first_keyframe_no_lh(self):
        kf = Keyframe(frame=0, value=1.0, interpolation="smooth")
        lua = kf.to_lua(prev_frame=None, next_frame=10)
        assert "LH" not in lua
        assert "RH" in lua


class TestKeyframeTrack:
    def test_add_pulse(self):
        track = KeyframeTrack()
        track.add_pulse(beat_frame=30, base_value=1.0, peak_value=1.1,
                        attack_frames=2, release_frames=4)
        assert len(track.keyframes) == 3  # pre-attack base, peak, release
        assert track.keyframes[0].frame == 28  # attack start
        assert track.keyframes[1].frame == 30  # peak
        assert track.keyframes[1].value == 1.1
        assert track.keyframes[2].frame == 34  # release

    def test_to_lua_entries(self):
        track = KeyframeTrack()
        track.add(0, 1.0)
        track.add(30, 1.5)
        entries = track.to_lua_entries()
        assert len(entries) == 2
        assert "[0]" in entries[0]
        assert "[30]" in entries[1]


class TestFusionComp:
    def test_serialize_produces_valid_structure(self):
        comp = FusionComp()
        node = make_transform(name="Transform1")
        node.inputs["Size"] = 1.0
        comp.add_node(node)

        output = comp.serialize()
        assert "Tools = ordered()" in output
        assert "Transform1 = Transform" in output
        assert "Size = Input { Value = 1.0, }" in output
        assert 'ActiveTool = "Transform1"' in output

    def test_animated_node_produces_spline(self):
        comp = FusionComp()
        node = make_transform(name="BeatZoom")
        track = KeyframeTrack()
        track.add(0, 1.0)
        track.add(15, 1.1)
        track.add(30, 1.0)
        node.animated["Size"] = track
        comp.add_node(node)

        output = comp.serialize()
        assert "BeatZoomSize = BezierSpline" in output
        assert 'SourceOp = "BeatZoomSize"' in output
        assert "KeyFrames" in output

    def test_save_and_read(self):
        comp = FusionComp()
        node = make_brightness_contrast(name="BC1")
        node.inputs["Gain"] = 1.2
        comp.add_node(node)

        with tempfile.NamedTemporaryFile(suffix=".setting", delete=False, mode="w") as f:
            path = f.name
        comp.save(path)
        content = Path(path).read_text()
        assert "BC1 = BrightnessContrast" in content


class TestGenerator:
    def _make_beat_map(self):
        return {
            "version": "1.0",
            "source_file": "test.mp3",
            "duration": 10.0,
            "tempo": 120.0,
            "fps": 30,
            "beats": [
                {"time": 0.5, "frame": 15, "intensity": 0.8},
                {"time": 1.0, "frame": 30, "intensity": 1.0},
                {"time": 1.5, "frame": 45, "intensity": 0.6},
            ],
            "onsets": [],
        }

    def test_generate_zoom(self):
        comp = generate_comp(self._make_beat_map(), effect="zoom")
        output = comp.serialize()
        assert "Transform" in output
        assert "BezierSpline" in output

    def test_generate_flash(self):
        comp = generate_comp(self._make_beat_map(), effect="flash")
        output = comp.serialize()
        assert "BrightnessContrast" in output
        assert "BezierSpline" in output

    def test_generate_all(self):
        comp = generate_comp(self._make_beat_map(), effect="all")
        output = comp.serialize()
        assert "Transform" in output
        assert "BrightnessContrast" in output
        assert "Glow" in output

    def test_generate_with_preset(self):
        comp = generate_comp(self._make_beat_map(), preset_names=["zoom_bounce", "flash"])
        output = comp.serialize()
        assert "Transform" in output
        assert "BrightnessContrast" in output

    def test_intensity_curve_exponential(self):
        comp = generate_comp(self._make_beat_map(), effect="zoom", intensity_curve="exponential")
        output = comp.serialize()
        assert "BezierSpline" in output

    def test_section_mode(self):
        beat_map = self._make_beat_map()
        beat_map["sections"] = [
            {"start_time": 0.0, "end_time": 1.0, "start_frame": 0, "end_frame": 30, "type": "low_energy", "label": "verse"},
            {"start_time": 1.0, "end_time": 2.0, "start_frame": 30, "end_frame": 60, "type": "high_energy", "label": "chorus"},
        ]
        beat_map["beats"][0]["section"] = "low_energy"
        beat_map["beats"][1]["section"] = "high_energy"
        beat_map["beats"][2]["section"] = "high_energy"
        comp = generate_comp(beat_map, section_mode=True)
        output = comp.serialize()
        assert "BezierSpline" in output

    def test_overshoot(self):
        comp = generate_comp(self._make_beat_map(), effect="zoom", overshoot=True)
        output = comp.serialize()
        assert "Transform" in output

    def test_generate_from_file(self):
        import tempfile
        from beatlab.generator import generate_from_file

        beat_map = self._make_beat_map()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(beat_map, f)
            json_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".setting", delete=False) as f:
            setting_path = f.name

        generate_from_file(json_path, setting_path, effect="zoom")
        content = Path(setting_path).read_text()
        assert "Transform" in content
