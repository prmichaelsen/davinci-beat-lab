"""Dump project.db (SQLite) to YAML files for pipeline consumption.

The db is the source of truth (written by the synthesizer frontend).
YAML is auxiliary — used for pipeline operations, export, and auditing.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import yaml


def dump_db_to_yaml(project_dir: str | Path) -> None:
    """Dump project.db to timeline.yaml + project.yaml + narrative.yaml.

    Skips if project.db doesn't exist. Overwrites existing YAML files.

    Args:
        project_dir: Path to the project work directory.
    """
    project_dir = Path(project_dir)
    db_path = project_dir / "project.db"

    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row

    try:
        # ── Meta → project.yaml ──
        meta = {}
        for row in conn.execute("SELECT key, value FROM meta"):
            key, value = row["key"], row["value"]
            # Try to parse JSON values (resolution is stored as "[1920, 1080]")
            try:
                meta[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                meta[key] = value

        project_yaml = {
            "meta": meta,
        }

        # ── Keyframes + Transitions → timeline.yaml ──
        keyframes = []
        for row in conn.execute("SELECT * FROM keyframes WHERE deleted_at IS NULL ORDER BY timestamp"):
            kf = {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "section": row["section"],
                "prompt": row["prompt"],
                "selected": row["selected"],
            }
            if row["source"]:
                kf["source"] = row["source"]
            if row["candidates"] and row["candidates"] != "[]":
                try:
                    kf["candidates"] = json.loads(row["candidates"])
                except json.JSONDecodeError:
                    pass
            if row["context"]:
                try:
                    kf["context"] = json.loads(row["context"])
                except json.JSONDecodeError:
                    kf["context"] = None
            keyframes.append(kf)

        # Bin (soft-deleted keyframes)
        kf_bin = []
        for row in conn.execute("SELECT * FROM keyframes WHERE deleted_at IS NOT NULL ORDER BY timestamp"):
            kf = {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "section": row["section"],
                "prompt": row["prompt"],
                "selected": row["selected"],
                "deleted_at": row["deleted_at"],
            }
            if row["source"]:
                kf["source"] = row["source"]
            kf_bin.append(kf)

        transitions = []
        for row in conn.execute("SELECT * FROM transitions WHERE deleted_at IS NULL ORDER BY from_kf"):
            tr = {
                "id": row["id"],
                "from": row["from_kf"],
                "to": row["to_kf"],
                "duration_seconds": row["duration_seconds"],
                "slots": row["slots"],
                "action": row["action"],
            }
            if row["use_global_prompt"]:
                tr["use_global_prompt"] = True
            if row["selected"] and row["selected"] != "[]":
                try:
                    tr["selected"] = json.loads(row["selected"])
                except json.JSONDecodeError:
                    pass
            if row["remap"]:
                try:
                    tr["remap"] = json.loads(row["remap"])
                except json.JSONDecodeError:
                    pass
            transitions.append(tr)

        # Transition bin
        tr_bin = []
        for row in conn.execute("SELECT * FROM transitions WHERE deleted_at IS NOT NULL ORDER BY from_kf"):
            tr = {
                "id": row["id"],
                "from": row["from_kf"],
                "to": row["to_kf"],
                "duration_seconds": row["duration_seconds"],
                "action": row["action"],
                "deleted_at": row["deleted_at"],
            }
            tr_bin.append(tr)

        timeline_yaml = {
            "active_timeline": "default",
            "timelines": {
                "default": {
                    "keyframes": keyframes,
                    "transitions": transitions,
                    "bin": kf_bin,
                    "transition_bin": tr_bin,
                }
            }
        }

        # ── Effects + Suppressions → beats.yaml ──
        effects = []
        for row in conn.execute("SELECT * FROM effects ORDER BY time"):
            effects.append({
                "id": row["id"],
                "type": row["type"],
                "time": row["time"],
                "intensity": row["intensity"],
                "duration": row["duration"],
            })

        suppressions = []
        for row in conn.execute("SELECT * FROM suppressions ORDER BY from_time"):
            sup = {
                "id": row["id"],
                "from_time": row["from_time"],
                "to_time": row["to_time"],
            }
            if row["effect_types"]:
                try:
                    sup["effect_types"] = json.loads(row["effect_types"])
                except json.JSONDecodeError:
                    sup["effect_types"] = row["effect_types"]
            suppressions.append(sup)

        beats_yaml = {
            "effects": effects,
            "suppressions": suppressions,
        }

        # ── Sections → narrative.yaml ──
        sections = []
        # Check if sections table exists (migration may not have run yet)
        has_sections_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sections'"
        ).fetchone()
        if has_sections_table:
            for row in conn.execute("SELECT * FROM sections ORDER BY sort_order"):
                sec = {
                    "id": row["id"],
                    "label": row["label"],
                    "start": row["start"],
                    "mood": row["mood"],
                    "energy": row["energy"],
                    "visual_direction": row["visual_direction"],
                    "notes": row["notes"],
                }
                if row["end"]:
                    sec["end"] = row["end"]
                for json_col in ("instruments", "motifs", "events"):
                    try:
                        sec[json_col] = json.loads(row[json_col])
                    except (json.JSONDecodeError, TypeError):
                        sec[json_col] = []
                sections.append(sec)

        if sections:
            _write_yaml({"sections": sections}, project_dir / "narrative.yaml")

        # ── Write files ──
        _write_yaml(project_yaml, project_dir / "project.yaml")
        _write_yaml(timeline_yaml, project_dir / "timeline.yaml")
        if effects or suppressions:
            _write_yaml(beats_yaml, project_dir / "beats.yaml")

        # Also write legacy narrative_keyframes.yaml for backward compat
        legacy = {
            "meta": meta,
            "keyframes": keyframes,
            "transitions": transitions,
            "bin": kf_bin,
            "transition_bin": tr_bin,
        }
        _write_yaml(legacy, project_dir / "narrative_keyframes.yaml")

    finally:
        conn.close()


def _write_yaml(data: dict, path: Path) -> None:
    """Write YAML with nice formatting."""
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=1000)
