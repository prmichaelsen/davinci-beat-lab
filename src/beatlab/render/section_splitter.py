"""Split long sections into sub-sections for natural playback speed."""

from __future__ import annotations

import json
import math
from pathlib import Path


def find_long_sections(
    plan: dict,
    sections: list[dict],
    max_duration: float = 8.0,
) -> list[dict]:
    """Find sections that exceed max_duration.

    Returns list of {section_index, duration, num_splits} dicts.
    """
    long = []
    for i, sec in enumerate(sections):
        dur = sec.get("end_time", 0) - sec.get("start_time", 0)
        if dur > max_duration:
            num_splits = math.ceil(dur / max_duration)
            long.append({
                "section_index": i,
                "start_time": sec.get("start_time", 0),
                "end_time": sec.get("end_time", 0),
                "duration": dur,
                "num_splits": num_splits,
                "type": sec.get("type", ""),
                "label": sec.get("label", ""),
            })
    return long


def generate_splits(
    plan: dict,
    sections: list[dict],
    max_duration: float = 8.0,
    existing_splits: dict | None = None,
) -> dict:
    """Generate a splits.json that defines how to break long sections into sub-sections.

    If existing_splits is provided, merges with it — further splitting any
    existing sub-sections that still exceed max_duration.

    Returns a splits dict:
    {
        "max_duration": 8.0,
        "splits": {
            "16": {
                "original": {start_time, end_time, duration},
                "sub_sections": [
                    {"sub_index": 0, "start_time": 123.8, "end_time": 131.8, "duration": 8.0},
                    ...
                ]
            }
        }
    }
    """
    # If we have existing splits, check if any sub-sections need further splitting
    if existing_splits:
        existing_map = existing_splits.get("splits", {})
        # Build effective section list: expand already-split sections
        effective_sections = []
        for i, sec in enumerate(sections):
            idx_str = str(i)
            if idx_str in existing_map:
                for sub in existing_map[idx_str]["sub_sections"]:
                    effective_sections.append({
                        "start_time": sub["start_time"],
                        "end_time": sub["end_time"],
                        "type": sec.get("type", ""),
                        "label": sec.get("label", ""),
                        "_parent_idx": i,
                        "_sub_index": sub["sub_index"],
                    })
            else:
                effective_sections.append(sec)

        # Find long sections in the effective list
        long_effective = find_long_sections(plan, effective_sections, max_duration)

        if not long_effective:
            return existing_splits  # Nothing more to split

        # Re-split: regenerate the full splits dict from scratch using effective sections
        sections = effective_sections

    long_sections = find_long_sections(plan, sections, max_duration)

    if not long_sections:
        return {"max_duration": max_duration, "splits": {}}

    # Find plan entries by section_index for style inheritance
    plan_by_idx = {}
    for s in plan.get("sections", []):
        plan_by_idx[s["section_index"]] = s

    splits = {}
    for ls in long_sections:
        idx = ls["section_index"]
        start = ls["start_time"]
        end = ls["end_time"]
        dur = ls["duration"]
        n = ls["num_splits"]
        chunk = dur / n

        sub_sections = []
        for j in range(n):
            sub_start = start + j * chunk
            sub_end = min(start + (j + 1) * chunk, end)
            sub_dur = sub_end - sub_start

            # Inherit style from parent section's plan entry
            parent_plan = plan_by_idx.get(idx, {})
            style = parent_plan.get("style_prompt", "")

            # Add slight variation for sub-sections after the first
            if j > 0 and style:
                style = f"{style}, continuation with subtle evolution"

            sub_sections.append({
                "sub_index": j,
                "start_time": round(sub_start, 2),
                "end_time": round(sub_end, 2),
                "duration": round(sub_dur, 2),
                "style_prompt": style,
                "transition_action": parent_plan.get("transition_action", ""),
                "wan_denoise": parent_plan.get("wan_denoise"),
            })

        splits[str(idx)] = {
            "original": {
                "start_time": start,
                "end_time": end,
                "duration": dur,
                "type": ls["type"],
                "label": ls["label"],
            },
            "sub_sections": sub_sections,
        }

    return {"max_duration": max_duration, "splits": splits}


def save_splits(splits: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)


def load_splits(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_stale_files(
    work_dir: str,
    splits: dict,
) -> list[str]:
    """Find files that need to be deleted/regenerated due to splits.

    For each split section N, the following are stale:
    - google_segments/segment_N-1_N.mp4 (transition INTO this section)
    - google_segments/segment_N_N+1.mp4 (transition OUT of this section)
    - google_remapped/remapped_N-1.mp4 and remapped_N.mp4
    - google_labeled/* for these indices
    - google_concat.mp4, google_muxed.mp4, google_output.mp4 (final outputs)

    Does NOT delete: styled images (we keep the original styled_N.png)
    """
    work = Path(work_dir)
    stale = []

    for idx_str in splits.get("splits", {}):
        idx = int(idx_str)

        # Segments involving this section
        for pattern in [
            f"google_segments/segment_{idx-1:03d}_{idx:03d}.mp4",
            f"google_segments/segment_{idx:03d}_{idx+1:03d}.mp4",
            f"google_remapped/remapped_{idx-1:03d}.mp4",
            f"google_remapped/remapped_{idx:03d}.mp4",
        ]:
            p = work / pattern
            if p.exists():
                stale.append(str(p))

        # Labeled versions
        for pattern in [
            f"google_labeled/labeled_{idx-1:03d}.mp4",
            f"google_labeled/labeled_{idx:03d}.mp4",
        ]:
            p = work / pattern
            if p.exists():
                stale.append(str(p))

    # Final assembly outputs are always stale when splits change
    for name in ["google_concat.mp4", "google_muxed.mp4", "google_output.mp4"]:
        p = work / name
        if p.exists():
            stale.append(str(p))

    return stale


def get_keyframe_timestamps(splits: dict, fps: float) -> list[dict]:
    """Get timestamps where new keyframe images need to be extracted from source video.

    Returns list of {section_index, sub_index, time, frame} for each sub-section boundary.
    """
    timestamps = []
    for idx_str, split_info in splits.get("splits", {}).items():
        idx = int(idx_str)
        for sub in split_info["sub_sections"]:
            timestamps.append({
                "section_index": idx,
                "sub_index": sub["sub_index"],
                "time": sub["start_time"],
                "frame": round(sub["start_time"] * fps),
                "style_prompt": sub.get("style_prompt", ""),
            })
    return timestamps
