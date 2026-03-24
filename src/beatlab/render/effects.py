"""Beat-synced video effects via MoviePy — applies zoom, shake, brightness, glow, color grading."""

from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def apply_effects(
    video_path: str,
    output_path: str,
    beat_map: dict,
    effect_plan: object | None = None,
    fps: float | None = None,
) -> str:
    """Apply beat-synced effects to a video using MoviePy.

    Reads the beat map and effect plan, applies per-frame effects
    (zoom, shake, brightness, glow, color grading) synced to beats.

    Args:
        video_path: Input video path.
        output_path: Output video path.
        beat_map: Parsed beat map dict with beats and sections.
        effect_plan: EffectPlan from AI director (optional).
        fps: Override frame rate.

    Returns:
        output_path
    """
    from moviepy import VideoFileClip

    clip = VideoFileClip(video_path)
    video_fps = fps or clip.fps

    # Build beat lookup: time → intensity
    beat_times = []
    beat_intensities = []
    for b in beat_map.get("beats", []):
        if b.get("intensity", 0) > 0:
            beat_times.append(b["time"])
            beat_intensities.append(b["intensity"])

    beat_times = np.array(beat_times)
    beat_intensities = np.array(beat_intensities)

    # Build section lookup: time → section index
    sections = beat_map.get("sections", [])

    # Build plan map
    plan_map: dict[int, object] = {}
    if effect_plan is not None:
        for sp in effect_plan.sections:
            plan_map[sp.section_index] = sp

    # Precompute per-preset parameters from plan
    section_presets = {}
    for i, sec in enumerate(sections):
        sp = plan_map.get(i)
        if sp:
            section_presets[i] = {
                "presets": sp.presets,
                "intensity_curve": sp.intensity_curve,
                "sustained": getattr(sp, "sustained_effects", None) or [],
            }
        else:
            section_presets[i] = {
                "presets": ["zoom_pulse"],
                "intensity_curve": "linear",
                "sustained": [],
            }

    def get_section_index(t: float) -> int:
        """Find which section a timestamp belongs to."""
        for i, sec in enumerate(sections):
            if sec.get("start_time", 0) <= t < sec.get("end_time", 0):
                return i
        return 0

    def get_beat_intensity(t: float, attack: float = 0.08, release: float = 0.2) -> float:
        """Get the beat effect intensity at time t, with attack/release envelope."""
        if len(beat_times) == 0:
            return 0.0

        # Find nearest beat before or at t
        idx = np.searchsorted(beat_times, t, side="right") - 1
        if idx < 0:
            return 0.0

        beat_t = beat_times[idx]
        beat_i = beat_intensities[idx]
        dt = t - beat_t

        if dt < 0:
            return 0.0
        elif dt < attack:
            # Attack phase: ramp up
            return beat_i * (dt / attack)
        elif dt < attack + release:
            # Release phase: decay
            progress = (dt - attack) / release
            return beat_i * (1.0 - progress)
        else:
            return 0.0

    def get_sustained_value(t: float, param: str, default: float) -> float:
        """Get sustained effect value at time t with smooth transitions."""
        sec_idx = get_section_index(t)
        sp = section_presets.get(sec_idx, {})
        sustained = sp.get("sustained", [])

        for seff in sustained:
            params = seff.get("parameters", {})
            if param in params:
                return params[param]

        return default

    def process_frame(get_frame: Callable, t: float) -> np.ndarray:
        """Apply all effects to a single frame."""
        frame = get_frame(t).astype(np.float32)
        h, w = frame.shape[:2]

        intensity = get_beat_intensity(t)
        sec_idx = get_section_index(t)
        sp = section_presets.get(sec_idx, {})
        presets = sp.get("presets", [])

        # ── Zoom (zoom_pulse, zoom_bounce) ──
        zoom_amount = 0.0
        if "zoom_pulse" in presets:
            zoom_amount = max(zoom_amount, 0.15 * intensity)
        if "zoom_bounce" in presets:
            zoom_amount = max(zoom_amount, 0.25 * intensity)

        if zoom_amount > 0.001:
            scale = 1.0 + zoom_amount
            new_h, new_w = int(h * scale), int(w * scale)
            # Crop center after scaling
            from PIL import Image
            img = Image.fromarray(frame.astype(np.uint8))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            img = img.crop((left, top, left + w, top + h))
            frame = np.array(img).astype(np.float32)

        # ── Camera shake (shake_x, shake_y) ──
        shake_x = 0.0
        shake_y = 0.0
        if "shake_x" in presets:
            shake_x = 0.015 * w * intensity * math.sin(t * 47.0)  # Pseudo-random via sin
        if "shake_y" in presets:
            shake_y = 0.01 * h * intensity * math.cos(t * 53.0)

        if abs(shake_x) > 0.5 or abs(shake_y) > 0.5:
            dx = int(round(shake_x))
            dy = int(round(shake_y))
            shifted = np.zeros_like(frame)
            # Compute source and dest slices for the shift
            src_x0 = max(0, -dx)
            src_y0 = max(0, -dy)
            src_x1 = min(w, w - dx)
            src_y1 = min(h, h - dy)
            dst_x0 = max(0, dx)
            dst_y0 = max(0, dy)
            dst_x1 = min(w, w + dx)
            dst_y1 = min(h, h + dy)
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
            frame = shifted

        # ── Brightness / Flash (flash, hard_cut) ──
        brightness = 1.0
        if "flash" in presets:
            brightness += 0.8 * intensity
        if "hard_cut" in presets:
            brightness += 1.5 * intensity

        if brightness != 1.0:
            frame = frame * brightness

        # ── Contrast pop ──
        if "contrast_pop" in presets and intensity > 0.01:
            contrast = 0.5 * intensity
            mean = frame.mean()
            frame = (frame - mean) * (1.0 + contrast) + mean

        # ── Glow ──
        if "glow_swell" in presets and intensity > 0.01:
            glow_strength = 0.3 * intensity
            from PIL import ImageFilter, Image
            img = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))
            blurred = img.filter(ImageFilter.GaussianBlur(radius=8))
            blurred_arr = np.array(blurred).astype(np.float32)
            frame = frame * (1.0 - glow_strength) + blurred_arr * glow_strength

        # ── Sustained color grading ──
        master_gain = get_sustained_value(t, "MasterGain", 1.0)
        master_saturation = get_sustained_value(t, "MasterSaturation", 1.0)
        master_contrast = get_sustained_value(t, "MasterContrast", 0.0)
        master_lift = get_sustained_value(t, "MasterLift", 0.0)

        if master_gain != 1.0:
            frame = frame * master_gain

        if master_lift != 0.0:
            frame = frame + master_lift * 255.0

        if master_contrast != 0.0:
            mean = frame.mean()
            frame = (frame - mean) * (1.0 + master_contrast) + mean

        if master_saturation != 1.0:
            # Convert to grayscale and blend
            gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
            gray = np.stack([gray] * 3, axis=-1)
            frame = gray + master_saturation * (frame - gray)

        # Per-channel gain (color tinting)
        gain_r = get_sustained_value(t, "GainR", 1.0)
        gain_g = get_sustained_value(t, "GainG", 1.0)
        gain_b = get_sustained_value(t, "GainB", 1.0)
        if gain_r != 1.0 or gain_g != 1.0 or gain_b != 1.0:
            frame[..., 0] *= gain_r
            frame[..., 1] *= gain_g
            frame[..., 2] *= gain_b

        # Clamp
        frame = np.clip(frame, 0, 255)
        return frame.astype(np.uint8)

    _log("Applying beat-synced effects...")
    styled = clip.transform(process_frame)

    _log(f"Writing output: {output_path}")
    styled.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=video_fps,
        logger=None,
    )

    clip.close()
    _log("Effects applied.")
    return output_path
