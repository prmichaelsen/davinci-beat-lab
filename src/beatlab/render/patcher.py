"""Plan patching — merge partial plan updates into cached plans, detect changed sections."""

from __future__ import annotations

import json
from pathlib import Path


def load_patch(patch_path: str) -> dict:
    """Load a patch file. Returns dict with 'sections' list."""
    with open(patch_path) as f:
        return json.load(f)


def merge_plan(base_plan: dict, patch: dict) -> tuple[dict, list[int]]:
    """Merge a patch into a base plan.

    Patch sections override base sections by section_index.
    Only fields present in the patch are overwritten — missing fields keep base values.

    Args:
        base_plan: The full cached plan dict.
        patch: Partial plan dict with only sections to change.

    Returns:
        (merged_plan, changed_indices) — the full merged plan and list of section indices that changed.
    """
    # Index base sections
    base_by_idx: dict[int, dict] = {}
    for s in base_plan.get("sections", []):
        base_by_idx[s["section_index"]] = s

    changed: list[int] = []

    for patch_section in patch.get("sections", []):
        idx = patch_section["section_index"]

        if idx in base_by_idx:
            # Merge: patch fields override base, keep rest
            base = base_by_idx[idx]
            for key, value in patch_section.items():
                if key != "section_index" and base.get(key) != value:
                    changed.append(idx)
                    break
            for key, value in patch_section.items():
                base[key] = value
        else:
            # New section
            base_by_idx[idx] = patch_section
            changed.append(idx)

    # Dedupe changed list
    changed = sorted(set(changed))

    # Rebuild plan
    merged = dict(base_plan)
    merged["sections"] = sorted(base_by_idx.values(), key=lambda s: s["section_index"])

    return merged, changed


def save_plan(plan: dict, path: str) -> None:
    """Save a plan to JSON."""
    with open(path, "w") as f:
        json.dump(plan, f, indent=2)


def detect_stale_outputs(
    work_dir: str,
    changed_indices: list[int],
    output_pattern: str = "clip_{idx:03d}*",
) -> list[str]:
    """Find output files that need to be regenerated due to plan changes.

    Args:
        work_dir: The work directory path.
        changed_indices: Section indices that changed.
        output_pattern: Glob pattern for output files (with {idx} placeholder).

    Returns:
        List of file paths to delete before re-rendering.
    """
    stale = []
    work = Path(work_dir)

    for idx in changed_indices:
        # Look for clip files, styled frames, or other outputs matching this section
        for pattern in [
            f"clips/clip_{idx:03d}*",
            f"styled/section_{idx:03d}*",
            f"wan_clips/clip_{idx:03d}*",
            f"wan_clips/segment_{idx}*",
        ]:
            for f in work.glob(pattern):
                stale.append(str(f))

    return stale


def generate_patch_from_updates(updates: list[dict]) -> dict:
    """Generate a patch dict from a list of section updates.

    Each update is a dict with at least 'section_index' and any fields to change.
    Convenience function for Claude to call programmatically.

    Example:
        updates = [
            {"section_index": 88, "style_prompt": "new prompt here"},
            {"section_index": 89, "style_prompt": "another prompt", "candidates": 4},
        ]
        patch = generate_patch_from_updates(updates)
    """
    return {"sections": updates}
