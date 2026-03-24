"""Persistent work directory for render pipeline resume support."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


class WorkDir:
    """Manages a persistent work directory for a render job.

    Each step writes its output here. On re-run, steps check if their
    output already exists and skip if so.
    """

    def __init__(self, video_path: str, base_dir: str = ".beatlab_work"):
        video_name = Path(video_path).stem
        self.root = Path(base_dir) / video_name
        self.root.mkdir(parents=True, exist_ok=True)

        self.audio_path = self.root / "audio.wav"
        self.beats_path = self.root / "beats.json"
        self.plan_path = self.root / "plan.json"
        self.params_path = self.root / "frame_params.json"
        self.frames_dir = self.root / "frames"
        self.styled_dir = self.root / "styled"
        self.status_path = self.root / "status.json"

    def has_audio(self) -> bool:
        return self.audio_path.exists() and self.audio_path.stat().st_size > 0

    def has_beats(self) -> bool:
        return self.beats_path.exists()

    def has_plan(self) -> bool:
        return self.plan_path.exists()

    def has_params(self) -> bool:
        return self.params_path.exists()

    def has_frames(self, expected_count: int | None = None) -> bool:
        if not self.frames_dir.exists():
            return False
        count = len(list(self.frames_dir.glob("frame_*.png")))
        if expected_count is not None:
            return count >= expected_count
        return count > 0

    def frame_count(self) -> int:
        if not self.frames_dir.exists():
            return 0
        return len(list(self.frames_dir.glob("frame_*.png")))

    def styled_count(self) -> int:
        if not self.styled_dir.exists():
            return 0
        return len(list(self.styled_dir.glob("frame_*.png")))

    def has_styled(self, expected_count: int | None = None) -> bool:
        if not self.styled_dir.exists():
            return False
        count = self.styled_count()
        if expected_count is not None:
            return count >= expected_count
        return count > 0

    def save_beats(self, beat_map: dict) -> None:
        with open(self.beats_path, "w") as f:
            json.dump(beat_map, f, indent=2)

    def load_beats(self) -> dict:
        with open(self.beats_path) as f:
            return json.load(f)

    def save_plan(self, plan_data: dict) -> None:
        with open(self.plan_path, "w") as f:
            json.dump(plan_data, f, indent=2)

    def load_plan(self) -> dict:
        with open(self.plan_path) as f:
            return json.load(f)

    def save_params(self, params: list[dict]) -> None:
        with open(self.params_path, "w") as f:
            json.dump(params, f, indent=2)

    def load_params(self) -> list[dict]:
        with open(self.params_path) as f:
            return json.load(f)

    def save_status(self, step: str, data: dict | None = None) -> None:
        status = {"last_completed_step": step}
        if data:
            status.update(data)
        with open(self.status_path, "w") as f:
            json.dump(status, f, indent=2)

    def load_status(self) -> dict:
        if self.status_path.exists():
            with open(self.status_path) as f:
                return json.load(f)
        return {}

    def ensure_frames_dir(self) -> str:
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        return str(self.frames_dir)

    def ensure_styled_dir(self) -> str:
        self.styled_dir.mkdir(parents=True, exist_ok=True)
        return str(self.styled_dir)

    def clean(self) -> None:
        """Wipe the entire work directory."""
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def summary(self) -> str:
        """Return a human-readable summary of cached state."""
        lines = [f"Work dir: {self.root}"]
        if self.has_audio():
            lines.append(f"  audio: cached")
        if self.has_beats():
            lines.append(f"  beats: cached")
        if self.has_plan():
            lines.append(f"  plan: cached")
        if self.has_frames():
            lines.append(f"  frames: {self.frame_count()} extracted")
        if self.has_styled():
            lines.append(f"  styled: {self.styled_count()} rendered")
        return "\n".join(lines)
