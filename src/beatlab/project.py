"""Project load/save — unified access to split or legacy YAML formats."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_project(work_dir: Path) -> dict:
    """Load project data from YAML files.

    NOTE: This is a YAML-only reader for CLI/export use.
    The API server reads directly from SQLite via db.py — it does NOT call this function.

    Returns a unified dict with: meta, sections, keyframes, transitions, bin,
    transition_bin, watched_folders, _format, _active_timeline, _work_dir.
    """
    work_dir = Path(work_dir)

    if (work_dir / "timeline.yaml").exists():
        # Split format
        narrative = _load_yaml(work_dir / "narrative.yaml") or {}
        timeline_data = _load_yaml(work_dir / "timeline.yaml") or {}
        project = _load_yaml(work_dir / "project.yaml") or {}
        active = timeline_data.get("active_timeline", "default")
        tl = timeline_data.get("timelines", {}).get(active, {})

        # Enrich keyframes with section context from narrative
        sections = narrative.get("sections", [])
        section_by_label = {s["label"]: s for s in sections if "label" in s}
        keyframes = tl.get("keyframes", [])
        for kf in keyframes:
            section_label = kf.get("section", "")
            if section_label and section_label in section_by_label and not kf.get("context"):
                sec = section_by_label[section_label]
                kf["context"] = {
                    "mood": sec.get("mood", ""),
                    "energy": sec.get("energy", ""),
                    "instruments": sec.get("instruments", []),
                    "motifs": sec.get("motifs", []),
                    "events": sec.get("events", []),
                    "visual_direction": sec.get("visual_direction", ""),
                    "details": sec.get("notes", ""),
                }

        return {
            "meta": project.get("meta", {}),
            "sections": sections,
            "keyframes": keyframes,
            "transitions": tl.get("transitions", []),
            "bin": tl.get("bin", []),
            "transition_bin": tl.get("transition_bin", []),
            "watched_folders": project.get("watched_folders", []),
            "_format": "split",
            "_active_timeline": active,
            "_work_dir": str(work_dir),
        }
    elif (work_dir / "narrative_keyframes.yaml").exists():
        # Legacy format
        parsed = _load_yaml(work_dir / "narrative_keyframes.yaml") or {}
        parsed["_format"] = "legacy"
        parsed["_work_dir"] = str(work_dir)
        return parsed
    else:
        return {
            "meta": {},
            "sections": [],
            "keyframes": [],
            "transitions": [],
            "_format": "empty",
            "_work_dir": str(work_dir),
        }


def save_project(data: dict, work_dir: Path) -> None:
    """Save project data. Auto-splits legacy format on first write."""
    work_dir = Path(work_dir)
    fmt = data.get("_format", "legacy")

    if fmt == "legacy" and not (work_dir / "timeline.yaml").exists():
        # First write on legacy — split it
        split_narrative_yaml(work_dir, data)
    elif fmt == "split" or (work_dir / "timeline.yaml").exists():
        _save_split(data, work_dir)
    else:
        # Fallback — save as legacy
        _save_legacy(data, work_dir)


def split_narrative_yaml(work_dir: Path, data: dict | None = None) -> None:
    """Split narrative_keyframes.yaml into narrative.yaml + timeline.yaml + project.yaml."""
    if data is None:
        legacy = work_dir / "narrative_keyframes.yaml"
        if not legacy.exists():
            return
        data = _load_yaml(legacy) or {}

    # Already split?
    if (work_dir / "timeline.yaml").exists():
        return

    # Extract sections from keyframe contexts
    sections = []
    seen_sections = set()
    for kf in data.get("keyframes", []):
        ctx = kf.get("context")
        section_label = kf.get("section", "")
        if ctx and section_label and section_label not in seen_sections:
            seen_sections.add(section_label)
            sections.append({
                "id": f"section_{section_label}",
                "label": section_label,
                "start": kf.get("timestamp", "0:00"),
                "mood": ctx.get("mood", ""),
                "energy": ctx.get("energy", ""),
                "instruments": ctx.get("instruments", []),
                "motifs": ctx.get("motifs", []),
                "events": ctx.get("events", []),
                "visual_direction": ctx.get("visual_direction", ""),
                "notes": ctx.get("details", ""),
            })

    # narrative.yaml
    _save_yaml({"sections": sections}, work_dir / "narrative.yaml")

    # timeline.yaml — strip context from keyframes
    keyframes = []
    for kf in data.get("keyframes", []):
        kf_copy = dict(kf)
        kf_copy.pop("context", None)
        keyframes.append(kf_copy)

    timeline = {
        "active_timeline": "default",
        "timelines": {
            "default": {
                "keyframes": keyframes,
                "transitions": data.get("transitions", []),
                "bin": data.get("bin", []),
                "transition_bin": data.get("transition_bin", []),
            }
        }
    }
    _save_yaml(timeline, work_dir / "timeline.yaml")

    # project.yaml
    project = {
        "meta": data.get("meta", {}),
        "watched_folders": data.get("watched_folders", []),
    }
    _save_yaml(project, work_dir / "project.yaml")


def _save_split(data: dict, work_dir: Path) -> None:
    """Save in split format."""
    active = data.get("_active_timeline", "default")

    # Strip enriched context from keyframes before saving
    keyframes = []
    for kf in data.get("keyframes", []):
        kf_copy = dict(kf)
        kf_copy.pop("context", None)
        keyframes.append(kf_copy)

    # Read existing timeline.yaml to preserve other timelines
    existing_timeline = _load_yaml(work_dir / "timeline.yaml") or {}
    timelines = existing_timeline.get("timelines", {})
    timelines[active] = {
        "keyframes": keyframes,
        "transitions": data.get("transitions", []),
        "bin": data.get("bin", []),
        "transition_bin": data.get("transition_bin", []),
    }
    _save_yaml({
        "active_timeline": active,
        "timelines": timelines,
    }, work_dir / "timeline.yaml")

    # narrative.yaml — sections
    if "sections" in data:
        _save_yaml({"sections": data["sections"]}, work_dir / "narrative.yaml")

    # project.yaml — meta + watched_folders
    project = {}
    if "meta" in data:
        project["meta"] = data["meta"]
    if "watched_folders" in data:
        project["watched_folders"] = data["watched_folders"]
    if project:
        _save_yaml(project, work_dir / "project.yaml")


def _save_legacy(data: dict, work_dir: Path) -> None:
    """Save as legacy narrative_keyframes.yaml."""
    save_data = {k: v for k, v in data.items() if not k.startswith("_")}
    _save_yaml(save_data, work_dir / "narrative_keyframes.yaml")


def get_timelines(work_dir: Path) -> dict:
    """List available timelines."""
    work_dir = Path(work_dir)
    timeline_data = _load_yaml(work_dir / "timeline.yaml") or {}
    active = timeline_data.get("active_timeline", "default")
    names = list(timeline_data.get("timelines", {}).keys())
    return {"active": active, "timelines": names}


def switch_timeline(work_dir: Path, name: str) -> None:
    """Switch the active timeline."""
    work_dir = Path(work_dir)
    timeline_data = _load_yaml(work_dir / "timeline.yaml") or {}
    if name not in timeline_data.get("timelines", {}):
        raise ValueError(f"Timeline not found: {name}")
    timeline_data["active_timeline"] = name
    _save_yaml(timeline_data, work_dir / "timeline.yaml")


def create_timeline(work_dir: Path, name: str, copy_from: str | None = None) -> None:
    """Create a new timeline, optionally copying from an existing one."""
    work_dir = Path(work_dir)
    timeline_data = _load_yaml(work_dir / "timeline.yaml") or {"active_timeline": "default", "timelines": {}}
    timelines = timeline_data.get("timelines", {})
    if name in timelines:
        raise ValueError(f"Timeline already exists: {name}")
    if copy_from:
        if copy_from not in timelines:
            raise ValueError(f"Source timeline not found: {copy_from}")
        import copy
        timelines[name] = copy.deepcopy(timelines[copy_from])
    else:
        timelines[name] = {"keyframes": [], "transitions": [], "bin": [], "transition_bin": []}
    timeline_data["timelines"] = timelines
    _save_yaml(timeline_data, work_dir / "timeline.yaml")


def import_timeline(work_dir: Path, source_path: str, timeline_name: str | None = None) -> dict:
    """Import a timeline from another project's timeline.yaml."""
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    source_data = _load_yaml(source) or {}
    # If it's a timeline.yaml, get the active timeline
    if "timelines" in source_data:
        active = source_data.get("active_timeline", "default")
        tl = source_data["timelines"].get(active, {})
    else:
        # Maybe it's a legacy narrative_keyframes.yaml
        tl = {
            "keyframes": source_data.get("keyframes", []),
            "transitions": source_data.get("transitions", []),
            "bin": source_data.get("bin", []),
            "transition_bin": source_data.get("transition_bin", []),
        }

    work_dir = Path(work_dir)
    timeline_data = _load_yaml(work_dir / "timeline.yaml") or {"active_timeline": "default", "timelines": {}}
    target_name = timeline_name or timeline_data.get("active_timeline", "default")
    timeline_data["timelines"][target_name] = tl
    _save_yaml(timeline_data, work_dir / "timeline.yaml")

    return {
        "keyframes": len(tl.get("keyframes", [])),
        "transitions": len(tl.get("transitions", [])),
    }


# ── YAML I/O helpers ──

def _load_yaml(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def _save_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
