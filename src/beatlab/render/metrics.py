"""Processing metrics tracker — records step timings for real-world estimates."""

from __future__ import annotations

import time
from pathlib import Path
from datetime import datetime

from beatlab.db import get_setting, set_setting


def load_metrics(project_dir: str | Path) -> dict:
    """Load metrics from the project DB settings table."""
    project_dir = Path(project_dir)
    runs = get_setting(project_dir, "metrics_runs", [])
    step_averages = get_setting(project_dir, "metrics_step_averages", {})
    metrics = {}
    if runs:
        metrics["runs"] = runs
    if step_averages:
        metrics["step_averages"] = step_averages
    return metrics


def save_metrics(project_dir: str | Path, metrics: dict) -> None:
    """Save metrics to the project DB settings table."""
    project_dir = Path(project_dir)
    set_setting(project_dir, "metrics_runs", metrics.get("runs", []))
    set_setting(project_dir, "metrics_step_averages", metrics.get("step_averages", {}))


def record_step(
    project_dir: str | Path,
    video_name: str,
    step: str,
    duration_seconds: float,
    metadata: dict | None = None,
) -> None:
    """Record a completed step's timing and metadata.

    Args:
        project_dir: Path to the project directory containing project.db
        video_name: Name of the video being processed
        step: Step name (e.g. "audio_extract", "beat_analysis", "veo_render", "crossfade")
        duration_seconds: How long the step took
        metadata: Optional dict with step-specific info (frame_count, segment_count, etc.)
    """
    metrics = load_metrics(project_dir)

    if "runs" not in metrics:
        metrics["runs"] = []
    if "step_averages" not in metrics:
        metrics["step_averages"] = {}

    entry = {
        "video": video_name,
        "step": step,
        "duration_s": round(duration_seconds, 1),
        "timestamp": datetime.now().isoformat(),
    }
    if metadata:
        entry["metadata"] = metadata

    metrics["runs"].append(entry)

    # Update running averages per step
    _update_averages(metrics, step, duration_seconds, metadata)

    save_metrics(project_dir, metrics)


def _update_averages(metrics: dict, step: str, duration: float, metadata: dict | None) -> None:
    """Update per-step averages for estimation."""
    avgs = metrics.setdefault("step_averages", {})
    step_avg = avgs.setdefault(step, {"count": 0, "total_seconds": 0.0})

    step_avg["count"] += 1
    step_avg["total_seconds"] = round(step_avg["total_seconds"] + duration, 1)
    step_avg["avg_seconds"] = round(step_avg["total_seconds"] / step_avg["count"], 1)
    step_avg["last_seconds"] = round(duration, 1)

    # Track per-unit rates where applicable
    if metadata:
        if "segment_count" in metadata and metadata["segment_count"] > 0:
            per_segment = duration / metadata["segment_count"]
            step_avg["avg_per_segment"] = round(
                (step_avg.get("avg_per_segment", per_segment) + per_segment) / 2, 2
            )
        if "frame_count" in metadata and metadata["frame_count"] > 0:
            per_frame = duration / metadata["frame_count"]
            step_avg["avg_per_frame"] = round(
                (step_avg.get("avg_per_frame", per_frame) + per_frame) / 2, 2
            )


def estimate_step(
    project_dir: str | Path,
    step: str,
    segment_count: int | None = None,
    frame_count: int | None = None,
) -> dict | None:
    """Estimate how long a step will take based on historical data.

    Returns dict with estimated_seconds, basis (what the estimate is based on), or None if no data.
    """
    metrics = load_metrics(project_dir)
    avgs = metrics.get("step_averages", {}).get(step)

    if not avgs:
        return None

    # Prefer per-unit estimates if we have the count
    if segment_count and "avg_per_segment" in avgs:
        est = avgs["avg_per_segment"] * segment_count
        return {
            "estimated_seconds": round(est),
            "estimated_minutes": round(est / 60, 1),
            "basis": f"{avgs['avg_per_segment']:.1f}s/segment × {segment_count}",
            "confidence": "high" if avgs["count"] >= 3 else "low",
        }

    if frame_count and "avg_per_frame" in avgs:
        est = avgs["avg_per_frame"] * frame_count
        return {
            "estimated_seconds": round(est),
            "estimated_minutes": round(est / 60, 1),
            "basis": f"{avgs['avg_per_frame']:.2f}s/frame × {frame_count}",
            "confidence": "high" if avgs["count"] >= 3 else "low",
        }

    # Fall back to simple average
    est = avgs["avg_seconds"]
    return {
        "estimated_seconds": round(est),
        "estimated_minutes": round(est / 60, 1),
        "basis": f"average of {avgs['count']} runs",
        "confidence": "medium" if avgs["count"] >= 3 else "low",
    }


def format_estimate(step: str, estimate: dict | None) -> str:
    """Format an estimate for display."""
    if not estimate:
        return f"  {step}: no estimate (first run)"

    mins = estimate["estimated_minutes"]
    basis = estimate["basis"]
    conf = estimate["confidence"]

    if mins < 1:
        time_str = f"{estimate['estimated_seconds']}s"
    elif mins < 60:
        time_str = f"{mins:.0f}m"
    else:
        hours = mins / 60
        time_str = f"{hours:.1f}h"

    return f"  {step}: ~{time_str} ({basis}, {conf} confidence)"


class StepTimer:
    """Context manager that records step timing automatically."""

    def __init__(self, project_dir: str | Path, video_name: str, step: str, metadata: dict | None = None):
        self.project_dir = project_dir
        self.video_name = video_name
        self.step = step
        self.metadata = metadata
        self._start = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        duration = time.time() - self._start
        record_step(self.project_dir, self.video_name, self.step, duration, self.metadata)

    def set_metadata(self, **kwargs):
        """Update metadata during the step (e.g. after discovering segment count)."""
        if self.metadata is None:
            self.metadata = {}
        self.metadata.update(kwargs)
