"""Beat map generation — convert analysis results to frame-rate-aware JSON."""

from __future__ import annotations

import json
from pathlib import Path


def time_to_frame(time_sec: float, fps: float) -> int:
    """Convert a timestamp in seconds to the nearest frame number.

    Args:
        time_sec: Time in seconds.
        fps: Frames per second (e.g. 24, 29.97, 30, 60).

    Returns:
        Nearest frame number (0-indexed).
    """
    return round(time_sec * fps)


def _assign_section(beat_time: float, sections: list[dict]) -> str | None:
    """Find which section a beat belongs to by timestamp."""
    for sec in sections:
        if sec["start_time"] <= beat_time < sec["end_time"]:
            return sec["type"]
    return None


def create_beat_map(
    analysis: dict,
    fps: float,
    source_file: str,
    stem_analyses: dict | None = None,
) -> dict:
    """Convert analysis results to a frame-rate-aware beat map.

    Args:
        analysis: Dict from analyzer.analyze_audio().
        fps: Timeline frame rate.
        source_file: Original audio file path (for metadata).
        stem_analyses: Optional per-stem analysis from stems.analyze_all_stems().

    Returns:
        Beat map dict ready for JSON serialization.
    """
    sections = analysis.get("sections", [])
    has_sections = len(sections) > 0

    beats = []
    for b in analysis["beats"]:
        entry = {
            "time": b["time"],
            "frame": time_to_frame(b["time"], fps),
            "intensity": b["intensity"],
        }
        if has_sections:
            entry["section"] = _assign_section(b["time"], sections)
        beats.append(entry)

    onsets = [
        {
            "time": o["time"],
            "frame": time_to_frame(o["time"], fps),
            "strength": o["strength"],
        }
        for o in analysis["onsets"]
    ]

    has_stems = stem_analyses is not None and len(stem_analyses) > 0
    version = "2.0" if has_stems else (
        "1.2" if (has_sections and any("spectral" in s for s in sections)) else ("1.1" if has_sections else "1.0")
    )

    result = {
        "version": version,
        "source_file": str(Path(source_file).name),
        "duration": analysis["duration"],
        "tempo": analysis["tempo"],
        "fps": fps,
        "beats": beats,
        "onsets": onsets,
    }

    if has_sections:
        result["sections"] = [
            {
                "start_time": s["start_time"],
                "end_time": s["end_time"],
                "start_frame": time_to_frame(s["start_time"], fps),
                "end_frame": time_to_frame(s["end_time"], fps),
                "type": s["type"],
                "label": s["label"],
                **({"spectral": s["spectral"]} if "spectral" in s else {}),
            }
            for s in sections
        ]

    if has_stems:
        result["stems"] = _build_stems_data(stem_analyses, fps)

    return result


def _build_stems_data(stem_analyses: dict, fps: float) -> dict:
    """Convert per-stem analysis dicts into frame-aware beat map format."""
    stems = {}
    for stem_name, analysis in stem_analyses.items():
        stem_data = {}

        # Beats (drums primarily)
        if "beats" in analysis:
            stem_data["beats"] = [
                {
                    "time": b["time"],
                    "frame": time_to_frame(b["time"], fps),
                    "intensity": b["intensity"],
                    **({"downbeat": b["downbeat"]} if "downbeat" in b else {}),
                }
                for b in analysis["beats"]
            ]

        # Onsets
        if "onsets" in analysis:
            stem_data["onsets"] = [
                {
                    "time": o["time"],
                    "frame": time_to_frame(o["time"], fps),
                    "strength": o["strength"],
                }
                for o in analysis["onsets"]
            ]

        # Drops (bass)
        if "drops" in analysis:
            stem_data["drops"] = [
                {
                    "time": d["time"],
                    "frame": time_to_frame(d["time"], fps),
                    "intensity": d["intensity"],
                }
                for d in analysis["drops"]
            ]

        # Presence regions (vocals)
        if "presence" in analysis:
            stem_data["presence"] = [
                {
                    "start_time": p["start_time"],
                    "end_time": p["end_time"],
                    "start_frame": time_to_frame(p["start_time"], fps),
                    "end_frame": time_to_frame(p["end_time"], fps),
                }
                for p in analysis["presence"]
            ]

        # Sections (drums)
        if "sections" in analysis:
            stem_data["sections"] = [
                {
                    "start_time": s["start_time"],
                    "end_time": s["end_time"],
                    "start_frame": time_to_frame(s["start_time"], fps),
                    "end_frame": time_to_frame(s["end_time"], fps),
                    "type": s["type"],
                    "label": s["label"],
                }
                for s in analysis["sections"]
            ]

        # Tempo (drums)
        if "tempo" in analysis:
            stem_data["tempo"] = analysis["tempo"]

        stems[stem_name] = stem_data

    return stems


def save_beat_map(beat_map: dict, output_path: str) -> None:
    """Write beat map to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(beat_map, f, indent=2)


def load_beat_map(path: str) -> dict:
    """Load a beat map from a JSON file."""
    with open(path) as f:
        return json.load(f)
