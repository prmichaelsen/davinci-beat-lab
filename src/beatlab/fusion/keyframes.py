"""Keyframe and spline utilities for Fusion .setting generation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Keyframe:
    """A single keyframe in a BezierSpline."""

    frame: int
    value: float
    interpolation: str = "smooth"  # linear, smooth, step

    def to_lua(self, prev_frame: int | None, next_frame: int | None) -> str:
        """Serialize to Fusion .setting keyframe entry."""
        parts = [f"{self.value}"]

        if self.interpolation == "linear":
            flags = "Flags = { Linear = true, }"
        elif self.interpolation == "step":
            flags = "Flags = { StepIn = true, }"
        else:
            flags = None

        # Generate bezier handles for smooth interpolation
        if self.interpolation == "smooth":
            if prev_frame is not None:
                lh_x = self.frame - (self.frame - prev_frame) / 3.0
                parts.append(f"LH = {{ {lh_x:.1f}, {self.value} }}")
            if next_frame is not None:
                rh_x = self.frame + (next_frame - self.frame) / 3.0
                parts.append(f"RH = {{ {rh_x:.1f}, {self.value} }}")
        elif self.interpolation == "linear":
            if prev_frame is not None:
                lh_x = self.frame - (self.frame - prev_frame) / 3.0
                parts.append(f"LH = {{ {lh_x:.1f}, {self.value} }}")
            if next_frame is not None:
                rh_x = self.frame + (next_frame - self.frame) / 3.0
                parts.append(f"RH = {{ {rh_x:.1f}, {self.value} }}")

        if flags:
            parts.append(flags)

        return f"[{self.frame}] = {{ {', '.join(parts)}, }},"


@dataclass
class KeyframeTrack:
    """A sequence of keyframes for one parameter."""

    keyframes: list[Keyframe] = field(default_factory=list)

    def add(self, frame: int, value: float, interpolation: str = "smooth") -> None:
        self.keyframes.append(Keyframe(frame, value, interpolation))

    def add_pulse(
        self,
        beat_frame: int,
        base_value: float,
        peak_value: float,
        attack_frames: int = 2,
        release_frames: int = 4,
        interpolation: str = "smooth",
    ) -> None:
        """Add an attack-peak-release pulse at the given beat frame."""
        attack_start = max(0, beat_frame - attack_frames)

        # Only add pre-attack base if it won't overlap with a previous pulse
        if not self.keyframes or self.keyframes[-1].frame < attack_start - 1:
            self.add(attack_start, base_value, interpolation)

        self.add(beat_frame, peak_value, interpolation)
        self.add(beat_frame + release_frames, base_value, interpolation)

    def to_lua_entries(self) -> list[str]:
        """Serialize all keyframes to Lua entries."""
        sorted_kfs = sorted(self.keyframes, key=lambda k: k.frame)
        entries = []
        for i, kf in enumerate(sorted_kfs):
            prev_f = sorted_kfs[i - 1].frame if i > 0 else None
            next_f = sorted_kfs[i + 1].frame if i < len(sorted_kfs) - 1 else None
            entries.append(kf.to_lua(prev_f, next_f))
        return entries
