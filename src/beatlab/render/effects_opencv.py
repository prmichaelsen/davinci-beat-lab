"""Beat-synced video effects via OpenCV — fast single-pass processing."""

from __future__ import annotations

import cv2
import math
import sys
import numpy as np
from datetime import datetime
from pathlib import Path


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def apply_effects(
    video_path: str,
    output_path: str,
    beat_map: dict,
    effect_plan: object | None = None,
    fps: float | None = None,
    glow: bool = False,
) -> str:
    """Apply beat-synced effects via OpenCV in a single pass.

    Args:
        video_path: Input video path.
        output_path: Output video path.
        beat_map: Parsed beat map dict with beats and sections.
        effect_plan: EffectPlan from AI director (optional).
        fps: Override frame rate.
        glow: Enable glow/bloom effect (slower).

    Returns:
        output_path
    """
    cap = cv2.VideoCapture(video_path)
    video_fps = fps or cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    _log(f"Applying beat-synced effects (OpenCV): {total_frames} frames, {w}x{h} @ {video_fps}fps")
    if glow:
        _log("  Glow enabled (slower)")

    # Build per-stem onset lookups
    stems = beat_map.get("stems", {})
    has_stems = bool(stems)

    def _build_onset_arrays(onsets):
        times = np.array([o["time"] for o in onsets if o.get("strength", 0) > 0])
        strengths = np.array([o["strength"] for o in onsets if o.get("strength", 0) > 0])
        return times, strengths

    def _adaptive_normalize(strengths, label=""):
        """Remap strengths to 0-1 based on the signal's own distribution.

        Uses percentile-based scaling:
        - p10 becomes the noise floor (mapped to 0)
        - p95 becomes full intensity (mapped to 1)
        - Everything between is linearly scaled
        This ensures effects trigger proportionally to what's loud *for this stem*.
        """
        if len(strengths) == 0:
            return strengths
        p10 = np.percentile(strengths, 10)
        p95 = np.percentile(strengths, 95)
        rng = p95 - p10
        if rng <= 0:
            return np.ones_like(strengths) * 0.5
        normalized = np.clip((strengths - p10) / rng, 0.0, 1.0)
        if label:
            _log(f"    {label}: raw range [{strengths.min():.4f} - {strengths.max():.4f}], "
                 f"p10={p10:.4f}, p95={p95:.4f} → normalized mean {normalized.mean():.2f}")
        return normalized

    if has_stems:
        drum_onsets = stems.get("drums", {}).get("onsets", [])
        bass_onsets = stems.get("bass", {}).get("onsets", [])
        bass_drops = stems.get("bass", {}).get("drops", [])
        other_onsets = stems.get("other", {}).get("onsets", [])
        vocal_presence = stems.get("vocals", {}).get("presence", [])

        drum_times, drum_strengths_raw = _build_onset_arrays(drum_onsets)
        bass_times, bass_strengths_raw = _build_onset_arrays(bass_onsets)
        bass_drop_times = np.array([d["time"] for d in bass_drops])
        bass_drop_intensities_raw = np.array([d["intensity"] for d in bass_drops]) if bass_drops else np.array([])
        other_times, other_strengths_raw = _build_onset_arrays(other_onsets)

        # Adaptive normalization per stem
        _log("  Adaptive normalization:")
        drum_strengths = _adaptive_normalize(drum_strengths_raw, "drums")
        bass_strengths = _adaptive_normalize(bass_strengths_raw, "bass")
        bass_drop_intensities = _adaptive_normalize(bass_drop_intensities_raw, "bass drops")
        other_strengths = _adaptive_normalize(other_strengths_raw, "other")

        _log(f"  Stem-routed effects: {len(drum_times)} drum hits, {len(bass_times)} bass hits, "
             f"{len(bass_drop_times)} bass drops, {len(other_times)} synth hits, {len(vocal_presence)} vocal regions")
    else:
        # Fallback: use full-mix beats for everything
        beats = beat_map.get("beats", [])
        drum_times = np.array([b["time"] for b in beats if b.get("intensity", 0) > 0])
        drum_strengths = np.array([b["intensity"] for b in beats if b.get("intensity", 0) > 0])
        bass_times = bass_drop_times = bass_drop_intensities = np.array([])
        bass_strengths = np.array([])
        other_times = other_strengths = np.array([])
        vocal_presence = []
        _log(f"  No stems — using full-mix beats ({len(drum_times)} beats)")

    sections = beat_map.get("sections", [])

    # Build plan map
    plan_map = {}
    if effect_plan is not None:
        for sp in effect_plan.sections:
            plan_map[sp.section_index] = sp

    section_presets = {}
    for i, sec in enumerate(sections):
        sp = plan_map.get(i)
        if sp:
            section_presets[i] = {"presets": sp.presets, "intensity_curve": sp.intensity_curve}
        else:
            section_presets[i] = {"presets": ["zoom_pulse"], "intensity_curve": "linear"}

    def get_section(t):
        for i, sec in enumerate(sections):
            if sec.get("start_time", 0) <= t < sec.get("end_time", float("inf")):
                return i
        return 0

    def _get_onset_intensity(t, times, strengths, attack=0.04, release=0.2):
        """Get intensity envelope value at time t from nearest onset."""
        if len(times) == 0:
            return 0.0
        idx = np.searchsorted(times, t, side="right") - 1
        if idx < 0:
            return 0.0
        dt = t - times[idx]
        if dt < 0:
            return 0.0
        si = strengths[idx]
        if dt < attack:
            return si * (dt / attack)
        elif dt < attack + release:
            return si * (1.0 - (dt - attack) / release)
        return 0.0

    def get_drum_intensity(t):
        return _get_onset_intensity(t, drum_times, drum_strengths, attack=0.03, release=0.15)

    def get_bass_intensity(t):
        # Combine bass onsets (fast) with bass drops (slow, heavy)
        onset_i = _get_onset_intensity(t, bass_times, bass_strengths, attack=0.05, release=0.3)
        drop_i = _get_onset_intensity(t, bass_drop_times, bass_drop_intensities, attack=0.02, release=0.5)
        return min(1.0, onset_i + drop_i)

    def get_other_intensity(t):
        return _get_onset_intensity(t, other_times, other_strengths, attack=0.06, release=0.35)

    def is_vocal(t):
        """Check if time t falls within a vocal presence region."""
        for region in vocal_presence:
            if region.get("start_time", 0) <= t <= region.get("end_time", 0):
                return True
        return False

    # Output — write raw frames, pipe through ffmpeg for encoding
    tmp_path = output_path + ".tmp.mp4"
    out = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (w, h))

    import time
    start_time = time.time()
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_num / video_fps
        sec_idx = get_section(t)
        sp = section_presets.get(sec_idx, {})
        presets = sp.get("presets", [])

        # Per-stem intensities
        di = get_drum_intensity(t)
        bi = get_bass_intensity(t)
        oi = get_other_intensity(t)
        vocal = is_vocal(t)

        # Suppress aggressive effects during vocal sections
        vocal_damp = 0.4 if vocal else 1.0

        # === ZOOM (zoom_pulse, zoom_bounce) — driven by BASS ===
        zoom_amount = 0.0
        if "zoom_pulse" in presets:
            zoom_amount = max(zoom_amount, 0.12 * bi)
        if "zoom_bounce" in presets:
            zoom_amount = max(zoom_amount, 0.20 * bi)

        if zoom_amount > 0.001:
            zoom = 1.0 + zoom_amount
            new_h, new_w = int(h * zoom), int(w * zoom)
            zoomed = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            top = (new_h - h) // 2
            left = (new_w - w) // 2
            frame = zoomed[top:top+h, left:left+w]

        # === CAMERA SHAKE (shake_x, shake_y) — driven by DRUMS ===
        shake_x, shake_y = 0, 0
        if "shake_x" in presets and di > 0.1:
            shake_x = int(8 * di * vocal_damp * math.sin(t * 47))
        if "shake_y" in presets and di > 0.1:
            shake_y = int(5 * di * vocal_damp * math.cos(t * 53))

        if abs(shake_x) > 0 or abs(shake_y) > 0:
            M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # === BRIGHTNESS FLASH (flash, hard_cut) — driven by DRUMS ===
        bright_alpha = 1.0
        bright_beta = 0
        if "flash" in presets and di > 0.05:
            bright_alpha += 0.3 * di * vocal_damp
            bright_beta = int(30 * di * vocal_damp)
        if "hard_cut" in presets and di > 0.05:
            bright_alpha += 0.8 * di * vocal_damp
            bright_beta = int(50 * di * vocal_damp)

        if bright_alpha != 1.0 or bright_beta != 0:
            frame = cv2.convertScaleAbs(frame, alpha=bright_alpha, beta=bright_beta)

        # === CONTRAST POP (contrast_pop) — driven by OTHER (synths/pads) ===
        if "contrast_pop" in presets and oi > 0.1:
            contrast = 1.0 + 0.4 * oi
            mean = np.mean(frame)
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=int(mean * (1 - contrast)))

        # === GLOW (glow_swell) — driven by OTHER (synths/pads), optional ===
        if glow and "glow_swell" in presets and oi > 0.05:
            alpha = 0.3 * oi
            blurred = cv2.GaussianBlur(frame, (0, 0), 8)
            frame = cv2.addWeighted(frame, 1.0 - alpha, blurred, alpha, 0)

        out.write(frame)
        frame_num += 1

        if frame_num % 1000 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_num / elapsed
            eta = (total_frames - frame_num) / fps_actual / 60
            _log(f"  [{frame_num}/{total_frames}] {fps_actual:.0f} fps, ETA {eta:.1f}m")

    cap.release()
    out.release()

    elapsed = time.time() - start_time
    _log(f"  Effects applied in {elapsed:.0f}s ({frame_num / elapsed:.0f} fps)")

    # Re-encode with ffmpeg
    import subprocess

    def _try_nvenc():
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=10,
            )
            return "h264_nvenc" in result.stdout
        except Exception:
            return False

    is_preview = locals().get("preview", False)
    if is_preview:
        encoder = "libx264"
        encode_opts = ["-preset", "ultrafast", "-crf", "28"]
    else:
        has_nvenc = _try_nvenc()
        encoder = "h264_nvenc" if has_nvenc else "libx264"
        encode_opts = ["-preset", "p4", "-rc", "vbr", "-cq", "18"] if has_nvenc else ["-preset", "fast", "-crf", "18"]

    _log(f"  Re-encoding with {encoder}{' (preview)' if is_preview else ''}...")
    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_path,
        "-i", video_path,
        "-map", "0:v", "-map", "1:a?",
        "-c:v", encoder, "-pix_fmt", "yuv420p", *encode_opts,
        "-c:a", "copy",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    Path(tmp_path).unlink(missing_ok=True)

    _log(f"  Done: {output_path}")
    return output_path


def apply_effects_ai(
    video_path: str,
    output_path: str,
    effect_events: list[dict],
    fps: float | None = None,
    time_offset: float = 0.0,
    hard_cuts: bool = False,
    preview: bool = False,
    effect_offsets: dict[str, float] | None = None,
) -> str:
    """Apply effects from Layer 3 AI-generated effect events.

    Each event is: {time, duration, effect, intensity, sustain?, stem_source, rationale}

    Args:
        video_path: Input video path.
        output_path: Output video path.
        effect_events: List of effect events from audio_intelligence Layer 3.
        fps: Override frame rate.
        time_offset: Offset added to event times (for clips trimmed from a longer video).
        preview: Half resolution + ultrafast encode for quick previews.

    Returns:
        output_path
    """
    cap = cv2.VideoCapture(video_path)
    video_fps = fps or cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Preview mode: half resolution
    if preview:
        w = w // 2
        h = h // 2
        _log(f"Applying AI-directed effects (PREVIEW): {total_frames} frames, {w}x{h} @ {video_fps}fps")
    else:
        _log(f"Applying AI-directed effects: {total_frames} frames, {w}x{h} @ {video_fps}fps")
    _log(f"  {len(effect_events)} effect events")

    # Apply per-effect time offsets (shift events earlier/later to align visual peak with audio peak)
    if effect_offsets:
        events = []
        for event in effect_events:
            offset_ms = effect_offsets.get(event.get("effect", ""), 0)
            if offset_ms:
                event = dict(event)
                event["time"] = max(0, event["time"] + offset_ms / 1000.0)
            events.append(event)
        _log(f"  Applied effect offsets: {effect_offsets}")
    else:
        events = list(effect_events)

    # Pre-sort events by time, filter hard_cuts if disabled
    events = sorted(events, key=lambda e: e["time"])
    if not hard_cuts:
        before = len(events)
        events = [e for e in events if e.get("effect") != "hard_cut"]
        diff = before - len(events)
        if diff:
            _log(f"  hard_cut disabled — filtered {diff} events")

    # Build event lookup — for each frame, compute active effect intensities
    def get_event_intensity(t: float, event: dict) -> float:
        """Get the intensity of an effect event at time t, including attack/sustain/release envelope."""
        event_time = event["time"] - time_offset
        duration = event.get("duration", 0.2)
        sustain = event.get("sustain") or 0.0
        intensity = event.get("intensity", 0.5)

        dt = t - event_time
        if dt < 0:
            return 0.0

        attack = min(0.04, duration * 0.2)
        release = duration - attack

        if sustain > 0:
            # Attack → sustain → release
            if dt < attack:
                return intensity * (dt / attack)
            elif dt < attack + sustain:
                return intensity
            elif dt < attack + sustain + release:
                return intensity * (1.0 - (dt - attack - sustain) / release)
            return 0.0
        else:
            # Attack → release
            if dt < attack:
                return intensity * (dt / attack)
            elif dt < attack + release:
                return intensity * (1.0 - (dt - attack) / release)
            return 0.0

    # Output
    tmp_path = output_path + ".tmp.mp4"
    out = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (w, h))

    import time
    start_time = time.time()
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if preview:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        t = frame_num / video_fps

        # Find active events — only check events within a reasonable window
        zoom_amount = 0.0
        zoom_bounce_active = False
        shake_x_val = 0
        shake_y_val = 0
        bright_alpha = 1.0
        bright_beta = 0
        contrast_amount = 0.0
        glow_amount = 0.0

        # First pass: check if any zoom_bounce is active (suppresses zoom_pulse)
        for event in events:
            event_time = event["time"] - time_offset
            max_dur = event.get("duration", 0.2) + (event.get("sustain") or 0.0) + 0.5
            if event_time > t + 0.1:
                break
            if event_time + max_dur < t:
                continue
            if event["effect"] == "zoom_bounce" and get_event_intensity(t, event) > 0.05:
                zoom_bounce_active = True
                break

        for event in events:
            event_time = event["time"] - time_offset
            max_dur = event.get("duration", 0.2) + (event.get("sustain") or 0.0) + 0.5
            if event_time > t + 0.1:
                break  # events are sorted, no more can be active
            if event_time + max_dur < t:
                continue  # event already finished

            ei = get_event_intensity(t, event)
            if ei < 0.01:
                continue

            effect = event["effect"]

            if effect == "zoom_pulse":
                # Suppress zoom_pulse when zoom_bounce is active
                if not zoom_bounce_active:
                    zoom_amount = max(zoom_amount, 0.12 * ei)
            elif effect == "zoom_bounce":
                zoom_amount = max(zoom_amount, 0.20 * ei)
            elif effect == "shake_x":
                shake_x_val += int(8 * ei * math.sin(t * 47))
            elif effect == "shake_y":
                shake_y_val += int(5 * ei * math.cos(t * 53))
            elif effect == "flash":
                # Flash disabled — too blinding. Treated as contrast_pop instead.
                contrast_amount = max(contrast_amount, 0.4 * ei)
            elif effect == "hard_cut":
                bright_alpha = max(bright_alpha, 1.0 + 0.8 * ei)
                bright_beta = max(bright_beta, int(50 * ei))
            elif effect == "contrast_pop":
                contrast_amount = max(contrast_amount, 0.4 * ei)
            elif effect == "glow_swell":
                glow_amount = max(glow_amount, 0.3 * ei)

        # Apply effects
        if zoom_amount > 0.001:
            zoom = 1.0 + zoom_amount
            new_h, new_w = int(h * zoom), int(w * zoom)
            zoomed = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            top = (new_h - h) // 2
            left = (new_w - w) // 2
            frame = zoomed[top:top+h, left:left+w]

        if abs(shake_x_val) > 0 or abs(shake_y_val) > 0:
            M = np.float32([[1, 0, shake_x_val], [0, 1, shake_y_val]])
            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        if bright_alpha != 1.0 or bright_beta != 0:
            frame = cv2.convertScaleAbs(frame, alpha=bright_alpha, beta=bright_beta)

        if contrast_amount > 0.01:
            contrast = 1.0 + contrast_amount
            mean = np.mean(frame)
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=int(mean * (1 - contrast)))

        if glow_amount > 0.01:
            blurred = cv2.GaussianBlur(frame, (0, 0), 8)
            frame = cv2.addWeighted(frame, 1.0 - glow_amount, blurred, glow_amount, 0)

        out.write(frame)
        frame_num += 1

        if frame_num % 1000 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_num / elapsed
            eta = (total_frames - frame_num) / fps_actual / 60
            _log(f"  [{frame_num}/{total_frames}] {fps_actual:.0f} fps, ETA {eta:.1f}m")

    cap.release()
    out.release()

    elapsed = time.time() - start_time
    _log(f"  Effects applied in {elapsed:.0f}s ({frame_num / elapsed:.0f} fps)")

    # Re-encode with ffmpeg
    import subprocess

    def _try_nvenc():
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=10,
            )
            return "h264_nvenc" in result.stdout
        except Exception:
            return False

    is_preview = locals().get("preview", False)
    if is_preview:
        encoder = "libx264"
        encode_opts = ["-preset", "ultrafast", "-crf", "28"]
    else:
        has_nvenc = _try_nvenc()
        encoder = "h264_nvenc" if has_nvenc else "libx264"
        encode_opts = ["-preset", "p4", "-rc", "vbr", "-cq", "18"] if has_nvenc else ["-preset", "fast", "-crf", "18"]

    _log(f"  Re-encoding with {encoder}{' (preview)' if is_preview else ''}...")
    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_path,
        "-i", video_path,
        "-map", "0:v", "-map", "1:a?",
        "-c:v", encoder, "-pix_fmt", "yuv420p", *encode_opts,
        "-c:a", "copy",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    Path(tmp_path).unlink(missing_ok=True)

    _log(f"  Done: {output_path}")
    return output_path
