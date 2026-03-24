"""Tests for the render pipeline — frame params, ComfyUI workflow, cost estimation."""

from beatlab.render.frames import generate_frame_params
from beatlab.render.comfyui import build_img2img_workflow
from beatlab.render.cloud import estimate_cost


class TestFrameParams:
    def _make_beat_map(self):
        return {
            "tempo": 120.0, "duration": 4.0, "fps": 30,
            "beats": [
                {"time": 0.5, "frame": 15, "intensity": 0.8},
                {"time": 1.0, "frame": 30, "intensity": 1.0},
                {"time": 2.5, "frame": 75, "intensity": 0.5},
            ],
            "sections": [
                {"start_time": 0.0, "end_time": 2.0, "type": "low_energy"},
                {"start_time": 2.0, "end_time": 4.0, "type": "high_energy"},
            ],
        }

    def test_generates_params_for_all_frames(self):
        params = generate_frame_params(self._make_beat_map(), total_frames=120, fps=30)
        assert len(params) == 120

    def test_beat_frames_have_higher_denoise(self):
        params = generate_frame_params(
            self._make_beat_map(), total_frames=120, fps=30,
            base_denoise=0.3, beat_denoise=0.5,
        )
        # Frame 15 is a beat
        beat_param = params[14]  # 0-indexed, frame 15
        non_beat = params[0]     # frame 1, not a beat
        assert beat_param["denoise"] > non_beat["denoise"]

    def test_section_styles_applied(self):
        styles = {0: "dark moody", 1: "bright neon"}
        params = generate_frame_params(
            self._make_beat_map(), total_frames=120, fps=30,
            section_styles=styles,
        )
        # Frame 1 (time=0.03s) is in section 0
        assert params[0]["prompt"] == "dark moody"
        # Frame 90 (time=3.0s) is in section 1
        assert params[89]["prompt"] == "bright neon"

    def test_default_style_when_no_section(self):
        params = generate_frame_params(
            self._make_beat_map(), total_frames=120, fps=30,
            default_style="psychedelic",
        )
        assert params[0]["prompt"] == "psychedelic"

    def test_consistent_seed(self):
        params = generate_frame_params(
            self._make_beat_map(), total_frames=120, fps=30, seed=123,
        )
        assert all(p["seed"] == 123 for p in params)


class TestComfyUIWorkflow:
    def test_basic_workflow_structure(self):
        wf = build_img2img_workflow(
            image_name="test.png", prompt="psychedelic", negative_prompt="blurry",
            denoise=0.4, seed=42, model="sd_xl_base_1.0.safetensors",
            controlnet_model=None,
        )
        assert "1" in wf  # checkpoint loader
        assert "2" in wf  # load image
        assert "6" in wf  # ksampler
        assert "8" in wf  # save image
        assert wf["6"]["inputs"]["denoise"] == 0.4
        assert wf["6"]["inputs"]["seed"] == 42

    def test_workflow_with_controlnet(self):
        wf = build_img2img_workflow(
            image_name="test.png", prompt="psychedelic", negative_prompt="blurry",
            denoise=0.4, seed=42, model="sd_xl_base_1.0.safetensors",
            controlnet_model="diffusers_xl_canny_full.safetensors",
        )
        assert "10" in wf  # controlnet loader
        assert "11" in wf  # canny
        assert "12" in wf  # controlnet apply
        # KSampler should be wired to controlnet output
        assert wf["6"]["inputs"]["positive"] == ["12", 0]

    def test_workflow_without_controlnet(self):
        wf = build_img2img_workflow(
            image_name="test.png", prompt="test", negative_prompt="bad",
            denoise=0.3, seed=1, model="model.safetensors",
            controlnet_model=None,
        )
        assert "10" not in wf
        assert wf["6"]["inputs"]["positive"] == ["4", 0]


class TestCostEstimation:
    def test_basic_estimate(self):
        cost = estimate_cost(frame_count=54000, fps_render=7.5, price_per_hr=1.0)
        assert cost["frames"] == 54000
        assert cost["estimated_hours"] == 2.0
        assert cost["estimated_cost_usd"] == 2.0

    def test_small_video(self):
        cost = estimate_cost(frame_count=900, fps_render=7.5)
        assert cost["estimated_seconds"] == 120
        assert cost["estimated_cost_usd"] < 0.05

    def test_expensive_gpu(self):
        cost = estimate_cost(frame_count=54000, fps_render=10.0, price_per_hr=2.5)
        assert cost["estimated_cost_usd"] == 3.75


class TestPlanStylePrompt:
    def test_parse_style_prompt(self):
        import json
        from beatlab.ai.plan import parse_effect_plan
        text = json.dumps({
            "sections": [{
                "section_index": 0,
                "presets": ["zoom_pulse"],
                "style_prompt": "dark moody noir, film grain",
            }]
        })
        plan = parse_effect_plan(text)
        assert plan.sections[0].style_prompt == "dark moody noir, film grain"

    def test_parse_no_style_prompt(self):
        import json
        from beatlab.ai.plan import parse_effect_plan
        text = json.dumps({
            "sections": [{"section_index": 0, "presets": ["flash"]}]
        })
        plan = parse_effect_plan(text)
        assert plan.sections[0].style_prompt is None
