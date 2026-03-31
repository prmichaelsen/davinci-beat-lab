"""SQLite storage layer for beatlab projects.

Replaces YAML read/write with instant SQL operations.
Each project gets its own `project.db` in its .beatlab_work directory.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from contextlib import contextmanager


# Per-project connection pool (one connection per thread)
_connections: dict[str, sqlite3.Connection] = {}
_conn_lock = threading.Lock()


_migrated_dbs: set[str] = set()  # tracks which DBs have been migrated this process

def get_db(project_dir: Path) -> sqlite3.Connection:
    """Get or create a SQLite connection for a project directory."""
    db_path = str(project_dir / "project.db")
    thread_key = f"{db_path}:{threading.current_thread().ident}"

    with _conn_lock:
        if thread_key not in _connections:
            conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            if db_path not in _migrated_dbs:
                _ensure_schema(conn)
                _migrated_dbs.add(db_path)
            _connections[thread_key] = conn
        return _connections[thread_key]


def close_db(project_dir: Path):
    """Close all connections for a project."""
    db_path = str(project_dir / "project.db")
    with _conn_lock:
        to_remove = [k for k in _connections if k.startswith(db_path)]
        for k in to_remove:
            _connections[k].close()
            del _connections[k]


@contextmanager
def transaction(project_dir: Path):
    """Context manager for a database transaction."""
    conn = get_db(project_dir)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _ensure_schema(conn: sqlite3.Connection):
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS keyframes (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            section TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            prompt TEXT NOT NULL DEFAULT '',
            selected INTEGER,
            candidates TEXT NOT NULL DEFAULT '[]',
            context TEXT,
            deleted_at TEXT
        );

        CREATE TABLE IF NOT EXISTS transitions (
            id TEXT PRIMARY KEY,
            from_kf TEXT NOT NULL,
            to_kf TEXT NOT NULL,
            duration_seconds REAL NOT NULL DEFAULT 0,
            slots INTEGER NOT NULL DEFAULT 1,
            action TEXT NOT NULL DEFAULT '',
            use_global_prompt INTEGER NOT NULL DEFAULT 0,
            selected TEXT NOT NULL DEFAULT '[]',
            remap TEXT NOT NULL DEFAULT '{"method":"linear","target_duration":0}',
            deleted_at TEXT
        );

        CREATE TABLE IF NOT EXISTS effects (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL DEFAULT 'pulse',
            time REAL NOT NULL DEFAULT 0,
            intensity REAL NOT NULL DEFAULT 0.8,
            duration REAL NOT NULL DEFAULT 0.2
        );

        CREATE TABLE IF NOT EXISTS suppressions (
            id TEXT PRIMARY KEY,
            from_time REAL NOT NULL,
            to_time REAL NOT NULL,
            effect_types TEXT
        );

        CREATE TABLE IF NOT EXISTS bench (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            source_path TEXT NOT NULL,
            label TEXT NOT NULL DEFAULT '',
            added_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS markers (
            id TEXT PRIMARY KEY,
            time REAL NOT NULL,
            label TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS tracks (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL DEFAULT 'Track 1',
            z_order INTEGER NOT NULL DEFAULT 0,
            blend_mode TEXT NOT NULL DEFAULT 'normal',
            base_opacity REAL NOT NULL DEFAULT 1.0,
            enabled INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS opacity_keyframes (
            id TEXT PRIMARY KEY,
            track_id TEXT NOT NULL,
            time REAL NOT NULL,
            opacity REAL NOT NULL DEFAULT 1.0
        );

        CREATE INDEX IF NOT EXISTS idx_keyframes_timestamp ON keyframes(timestamp);
        CREATE INDEX IF NOT EXISTS idx_keyframes_deleted ON keyframes(deleted_at);
        CREATE INDEX IF NOT EXISTS idx_transitions_from ON transitions(from_kf);
        CREATE INDEX IF NOT EXISTS idx_transitions_to ON transitions(to_kf);
        CREATE INDEX IF NOT EXISTS idx_transitions_deleted ON transitions(deleted_at);
        CREATE INDEX IF NOT EXISTS idx_opacity_kf_track ON opacity_keyframes(track_id, time);
    """)

    # ── Migration: add track_id to keyframes/transitions if missing ──
    cols = {row[1] for row in conn.execute("PRAGMA table_info(keyframes)").fetchall()}
    if "track_id" not in cols:
        conn.execute("ALTER TABLE keyframes ADD COLUMN track_id TEXT NOT NULL DEFAULT 'track_1'")
    cols = {row[1] for row in conn.execute("PRAGMA table_info(transitions)").fetchall()}
    if "track_id" not in cols:
        conn.execute("ALTER TABLE transitions ADD COLUMN track_id TEXT NOT NULL DEFAULT 'track_1'")

    # Ensure default track exists
    try:
        if not conn.execute("SELECT 1 FROM tracks WHERE id = 'track_1'").fetchone():
            conn.execute("INSERT OR IGNORE INTO tracks (id, name, z_order, blend_mode, base_opacity, enabled) VALUES ('track_1', 'Track 1', 0, 'normal', 1.0, 1)")
    except Exception:
        pass  # another thread may have inserted it


# ── Meta operations ─────────────────────────────────────────────────

def get_meta(project_dir: Path) -> dict:
    conn = get_db(project_dir)
    rows = conn.execute("SELECT key, value FROM meta").fetchall()
    meta = {}
    for row in rows:
        try:
            meta[row["key"]] = json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            meta[row["key"]] = row["value"]
    return meta


def set_meta(project_dir: Path, key: str, value):
    conn = get_db(project_dir)
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        (key, json.dumps(value) if not isinstance(value, str) else value),
    )
    conn.commit()


def set_meta_bulk(project_dir: Path, meta: dict):
    conn = get_db(project_dir)
    for key, value in meta.items():
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, json.dumps(value) if not isinstance(value, str) else value),
        )
    conn.commit()


# ── Keyframe operations ─────────────────────────────────────────────

def _row_to_keyframe(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "section": row["section"],
        "source": row["source"],
        "prompt": row["prompt"],
        "selected": row["selected"],
        "candidates": json.loads(row["candidates"]),
        "context": json.loads(row["context"]) if row["context"] else None,
        "track_id": row["track_id"] if "track_id" in row.keys() else "track_1",
        "deleted_at": row["deleted_at"],
    }


def get_keyframes(project_dir: Path, include_deleted: bool = False) -> list[dict]:
    conn = get_db(project_dir)
    if include_deleted:
        rows = conn.execute("SELECT * FROM keyframes ORDER BY timestamp").fetchall()
    else:
        rows = conn.execute("SELECT * FROM keyframes WHERE deleted_at IS NULL ORDER BY timestamp").fetchall()
    return [_row_to_keyframe(r) for r in rows]


def get_keyframe(project_dir: Path, kf_id: str) -> dict | None:
    conn = get_db(project_dir)
    row = conn.execute("SELECT * FROM keyframes WHERE id = ?", (kf_id,)).fetchone()
    return _row_to_keyframe(row) if row else None


def get_binned_keyframes(project_dir: Path) -> list[dict]:
    conn = get_db(project_dir)
    rows = conn.execute("SELECT * FROM keyframes WHERE deleted_at IS NOT NULL ORDER BY timestamp").fetchall()
    return [_row_to_keyframe(r) for r in rows]


def add_keyframe(project_dir: Path, kf: dict):
    conn = get_db(project_dir)
    conn.execute(
        """INSERT OR REPLACE INTO keyframes (id, timestamp, section, source, prompt, selected, candidates, context, deleted_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (kf["id"], kf["timestamp"], kf.get("section", ""), kf.get("source", ""),
         kf.get("prompt", ""), kf.get("selected"), json.dumps(kf.get("candidates", [])),
         json.dumps(kf.get("context")) if kf.get("context") else None, kf.get("deleted_at")),
    )
    conn.commit()


def update_keyframe(project_dir: Path, kf_id: str, **fields):
    conn = get_db(project_dir)
    sets = []
    values = []
    for key, val in fields.items():
        col = key
        if key == "candidates" or key == "context":
            val = json.dumps(val) if val is not None else None
        sets.append(f"{col} = ?")
        values.append(val)
    values.append(kf_id)
    conn.execute(f"UPDATE keyframes SET {', '.join(sets)} WHERE id = ?", values)
    conn.commit()


def delete_keyframe(project_dir: Path, kf_id: str, deleted_at: str):
    """Soft-delete a keyframe."""
    conn = get_db(project_dir)
    conn.execute("UPDATE keyframes SET deleted_at = ? WHERE id = ?", (deleted_at, kf_id))
    conn.commit()


def restore_keyframe(project_dir: Path, kf_id: str):
    conn = get_db(project_dir)
    conn.execute("UPDATE keyframes SET deleted_at = NULL WHERE id = ?", (kf_id,))
    conn.commit()


def next_keyframe_id(project_dir: Path) -> str:
    conn = get_db(project_dir)
    row = conn.execute(
        "SELECT id FROM keyframes ORDER BY CAST(SUBSTR(id, 4) AS INTEGER) DESC LIMIT 1"
    ).fetchone()
    if row:
        num = int(row["id"].replace("kf_", "")) + 1
    else:
        num = 1
    return f"kf_{num:03d}"


# ── Transition operations ───────────────────────────────────────────

def _row_to_transition(row: sqlite3.Row) -> dict:
    remap = json.loads(row["remap"]) if row["remap"] else {"method": "linear", "target_duration": 0}
    selected_raw = json.loads(row["selected"]) if row["selected"] else [None]
    # Flatten legacy [N] to N for frontend compat
    selected = selected_raw[0] if isinstance(selected_raw, list) and len(selected_raw) == 1 else selected_raw
    return {
        "id": row["id"],
        "from": row["from_kf"],
        "to": row["to_kf"],
        "duration_seconds": row["duration_seconds"],
        "slots": row["slots"],
        "action": row["action"],
        "use_global_prompt": bool(row["use_global_prompt"]),
        "selected": selected,
        "remap": remap,
        "track_id": row["track_id"] if "track_id" in row.keys() else "track_1",
        "deleted_at": row["deleted_at"],
    }


def get_transitions(project_dir: Path, include_deleted: bool = False) -> list[dict]:
    conn = get_db(project_dir)
    if include_deleted:
        rows = conn.execute("SELECT * FROM transitions").fetchall()
    else:
        rows = conn.execute("SELECT * FROM transitions WHERE deleted_at IS NULL").fetchall()
    return [_row_to_transition(r) for r in rows]


def get_transition(project_dir: Path, tr_id: str) -> dict | None:
    conn = get_db(project_dir)
    row = conn.execute("SELECT * FROM transitions WHERE id = ?", (tr_id,)).fetchone()
    return _row_to_transition(row) if row else None


def get_binned_transitions(project_dir: Path) -> list[dict]:
    conn = get_db(project_dir)
    rows = conn.execute("SELECT * FROM transitions WHERE deleted_at IS NOT NULL").fetchall()
    return [_row_to_transition(r) for r in rows]


def add_transition(project_dir: Path, tr: dict):
    conn = get_db(project_dir)
    selected = tr.get("selected")
    if isinstance(selected, (int, str)) and selected is not None:
        selected = [selected]
    elif selected is None:
        selected = [None]
    conn.execute(
        """INSERT OR REPLACE INTO transitions (id, from_kf, to_kf, duration_seconds, slots, action, use_global_prompt, selected, remap, deleted_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (tr["id"], tr.get("from", ""), tr.get("to", ""), tr.get("duration_seconds", 0),
         tr.get("slots", 1), tr.get("action", ""), int(tr.get("use_global_prompt", False)),
         json.dumps(selected), json.dumps(tr.get("remap", {"method": "linear", "target_duration": 0})),
         tr.get("deleted_at")),
    )
    conn.commit()


def update_transition(project_dir: Path, tr_id: str, **fields):
    conn = get_db(project_dir)
    sets = []
    values = []
    for key, val in fields.items():
        col = key
        if key == "from":
            col = "from_kf"
        elif key == "to":
            col = "to_kf"
        elif key == "selected":
            if isinstance(val, (int, str)) and val is not None:
                val = json.dumps([val])
            elif val is None:
                val = json.dumps([None])
            else:
                val = json.dumps(val)
        elif key == "remap":
            val = json.dumps(val)
        elif key == "use_global_prompt":
            val = int(val)
        sets.append(f"{col} = ?")
        values.append(val)
    values.append(tr_id)
    conn.execute(f"UPDATE transitions SET {', '.join(sets)} WHERE id = ?", values)
    conn.commit()


def delete_transition(project_dir: Path, tr_id: str, deleted_at: str):
    conn = get_db(project_dir)
    conn.execute("UPDATE transitions SET deleted_at = ? WHERE id = ?", (deleted_at, tr_id))
    conn.commit()


def restore_transition(project_dir: Path, tr_id: str):
    conn = get_db(project_dir)
    conn.execute("UPDATE transitions SET deleted_at = NULL WHERE id = ?", (tr_id,))
    conn.commit()


def next_transition_id(project_dir: Path) -> str:
    conn = get_db(project_dir)
    row = conn.execute(
        "SELECT id FROM transitions ORDER BY CAST(SUBSTR(id, 4) AS INTEGER) DESC LIMIT 1"
    ).fetchone()
    if row:
        num = int(row["id"].replace("tr_", "")) + 1
    else:
        num = 1
    return f"tr_{num:03d}"


def get_transitions_involving(project_dir: Path, kf_id: str) -> list[dict]:
    """Get all active transitions that reference a keyframe as from or to."""
    conn = get_db(project_dir)
    rows = conn.execute(
        "SELECT * FROM transitions WHERE deleted_at IS NULL AND (from_kf = ? OR to_kf = ?)",
        (kf_id, kf_id),
    ).fetchall()
    return [_row_to_transition(r) for r in rows]


# ── Effects / Suppressions ──────────────────────────────────────────

def get_effects(project_dir: Path) -> list[dict]:
    conn = get_db(project_dir)
    rows = conn.execute("SELECT * FROM effects ORDER BY time").fetchall()
    return [{"id": r["id"], "type": r["type"], "time": r["time"],
             "intensity": r["intensity"], "duration": r["duration"]} for r in rows]


def save_effects(project_dir: Path, effects: list[dict], suppressions: list[dict]):
    conn = get_db(project_dir)
    conn.execute("DELETE FROM effects")
    conn.execute("DELETE FROM suppressions")
    for fx in effects:
        conn.execute(
            "INSERT INTO effects (id, type, time, intensity, duration) VALUES (?, ?, ?, ?, ?)",
            (fx["id"], fx["type"], fx["time"], fx["intensity"], fx["duration"]),
        )
    for sup in suppressions:
        conn.execute(
            "INSERT INTO suppressions (id, from_time, to_time, effect_types) VALUES (?, ?, ?, ?)",
            (sup["id"], sup["from"], sup["to"],
             json.dumps(sup.get("effectTypes")) if sup.get("effectTypes") else None),
        )
    conn.commit()


def get_suppressions(project_dir: Path) -> list[dict]:
    conn = get_db(project_dir)
    rows = conn.execute("SELECT * FROM suppressions ORDER BY from_time").fetchall()
    return [{"id": r["id"], "from": r["from_time"], "to": r["to_time"],
             "effectTypes": json.loads(r["effect_types"]) if r["effect_types"] else None}
            for r in rows]


# ── Bench operations ─────────────────────────────────────────────────

def get_bench(project_dir: Path) -> list[dict]:
    """Get all benched items with usage tracking."""
    conn = get_db(project_dir)
    rows = conn.execute("SELECT * FROM bench ORDER BY added_at DESC").fetchall()

    # Build usage map: source_path -> list of (entity_id, timestamp)
    # Check active transitions for matching selected video paths
    active_trs = conn.execute(
        "SELECT id, from_kf, to_kf FROM transitions WHERE deleted_at IS NULL"
    ).fetchall()
    active_kfs = conn.execute(
        "SELECT id, timestamp, source FROM keyframes WHERE deleted_at IS NULL"
    ).fetchall()

    # For transitions, the source path is selected_transitions/{id}_slot_0.mp4
    # For keyframes, it's the source field or selected_keyframes/{id}.png
    tr_sources = {}
    for tr in active_trs:
        tr_path = f"selected_transitions/{tr['id']}_slot_0.mp4"
        tr_sources[tr_path] = tr
    kf_sources = {}
    for kf in active_kfs:
        kf_path = f"selected_keyframes/{kf['id']}.png"
        kf_sources[kf_path] = kf
        if kf["source"]:
            kf_sources[kf["source"]] = kf

    items = []
    for row in rows:
        src = row["source_path"]
        usages = []

        if row["type"] == "transition":
            # Find all transitions that use this video (by checking if the file content matches)
            # Simple heuristic: check if any transition's selected video path matches
            for tr_path, tr in tr_sources.items():
                # A bench item's source could be a pool path or a selected_transitions path
                # We track usage by the bench item's source_path appearing as a candidate
                pass
            # For now, scan transition candidates dirs for this source
            # This is expensive — we'll optimize later with a source_hash column
        elif row["type"] == "keyframe":
            for kf_path, kf in kf_sources.items():
                if kf["source"] == src:
                    usages.append({"entityId": kf["id"], "timestamp": kf["timestamp"]})

        items.append({
            "id": row["id"],
            "type": row["type"],
            "sourcePath": row["source_path"],
            "label": row["label"],
            "addedAt": row["added_at"],
            "usageCount": len(usages),
            "usages": usages,
        })

    return items


def add_to_bench(project_dir: Path, bench_type: str, source_path: str, label: str = "") -> str:
    """Add an item to the bench. Returns the bench ID."""
    conn = get_db(project_dir)
    from datetime import datetime, timezone
    import uuid

    bench_id = f"bench_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO bench (id, type, source_path, label, added_at) VALUES (?, ?, ?, ?, ?)",
        (bench_id, bench_type, source_path, label, now),
    )
    conn.commit()
    return bench_id


def remove_from_bench(project_dir: Path, bench_id: str):
    conn = get_db(project_dir)
    conn.execute("DELETE FROM bench WHERE id = ?", (bench_id,))
    conn.commit()


def get_bench_item(project_dir: Path, bench_id: str) -> dict | None:
    conn = get_db(project_dir)
    row = conn.execute("SELECT * FROM bench WHERE id = ?", (bench_id,)).fetchone()
    if not row:
        return None
    return {
        "id": row["id"], "type": row["type"],
        "sourcePath": row["source_path"], "label": row["label"],
        "addedAt": row["added_at"],
    }


# ── Track operations ───────────────────────────────────────────────

def get_tracks(project_dir: Path) -> list[dict]:
    conn = get_db(project_dir)
    rows = conn.execute("SELECT * FROM tracks ORDER BY z_order").fetchall()
    return [{
        "id": r["id"], "name": r["name"], "z_order": r["z_order"],
        "blend_mode": r["blend_mode"], "base_opacity": r["base_opacity"],
        "enabled": bool(r["enabled"]),
    } for r in rows]


def add_track(project_dir: Path, track: dict):
    conn = get_db(project_dir)
    conn.execute(
        "INSERT INTO tracks (id, name, z_order, blend_mode, base_opacity, enabled) VALUES (?, ?, ?, ?, ?, ?)",
        (track["id"], track.get("name", "New Track"), track.get("z_order", 0),
         track.get("blend_mode", "normal"), track.get("base_opacity", 1.0),
         1 if track.get("enabled", True) else 0),
    )
    conn.commit()


def update_track(project_dir: Path, track_id: str, **fields):
    conn = get_db(project_dir)
    sets = []
    values = []
    for key, val in fields.items():
        if key == "enabled":
            val = 1 if val else 0
        sets.append(f"{key} = ?")
        values.append(val)
    values.append(track_id)
    conn.execute(f"UPDATE tracks SET {', '.join(sets)} WHERE id = ?", values)
    conn.commit()


def delete_track(project_dir: Path, track_id: str):
    conn = get_db(project_dir)
    conn.execute("DELETE FROM tracks WHERE id = ?", (track_id,))
    conn.execute("DELETE FROM opacity_keyframes WHERE track_id = ?", (track_id,))
    # Soft-delete keyframes and transitions on this track
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("UPDATE keyframes SET deleted_at = ? WHERE track_id = ? AND deleted_at IS NULL", (now, track_id))
    conn.execute("UPDATE transitions SET deleted_at = ? WHERE track_id = ? AND deleted_at IS NULL", (now, track_id))
    conn.commit()


def reorder_tracks(project_dir: Path, track_ids: list[str]):
    conn = get_db(project_dir)
    for i, tid in enumerate(track_ids):
        conn.execute("UPDATE tracks SET z_order = ? WHERE id = ?", (i, tid))
    conn.commit()


# ── Opacity keyframe operations ────────────────────────────────────

def get_opacity_keyframes(project_dir: Path, track_id: str) -> list[dict]:
    conn = get_db(project_dir)
    rows = conn.execute("SELECT * FROM opacity_keyframes WHERE track_id = ? ORDER BY time", (track_id,)).fetchall()
    return [{"id": r["id"], "track_id": r["track_id"], "time": r["time"], "opacity": r["opacity"]} for r in rows]


def add_opacity_keyframe(project_dir: Path, okf_id: str, track_id: str, time: float, opacity: float):
    conn = get_db(project_dir)
    conn.execute("INSERT OR REPLACE INTO opacity_keyframes (id, track_id, time, opacity) VALUES (?, ?, ?, ?)",
                 (okf_id, track_id, time, opacity))
    conn.commit()


def update_opacity_keyframe(project_dir: Path, okf_id: str, **fields):
    conn = get_db(project_dir)
    sets = []
    values = []
    for key, val in fields.items():
        sets.append(f"{key} = ?")
        values.append(val)
    values.append(okf_id)
    conn.execute(f"UPDATE opacity_keyframes SET {', '.join(sets)} WHERE id = ?", values)
    conn.commit()


def delete_opacity_keyframe(project_dir: Path, okf_id: str):
    conn = get_db(project_dir)
    conn.execute("DELETE FROM opacity_keyframes WHERE id = ?", (okf_id,))
    conn.commit()


# ── Marker operations ──────────────────────────────────────────────

def get_markers(project_dir: Path) -> list[dict]:
    conn = get_db(project_dir)
    rows = conn.execute("SELECT id, time, label FROM markers ORDER BY time").fetchall()
    return [{"id": r["id"], "time": r["time"], "label": r["label"]} for r in rows]


def add_marker(project_dir: Path, marker_id: str, time: float, label: str = ""):
    conn = get_db(project_dir)
    conn.execute("INSERT OR REPLACE INTO markers (id, time, label) VALUES (?, ?, ?)", (marker_id, time, label))
    conn.commit()


def update_marker(project_dir: Path, marker_id: str, **fields):
    conn = get_db(project_dir)
    sets = []
    values = []
    for key, val in fields.items():
        sets.append(f"{key} = ?")
        values.append(val)
    values.append(marker_id)
    conn.execute(f"UPDATE markers SET {', '.join(sets)} WHERE id = ?", values)
    conn.commit()


def delete_marker(project_dir: Path, marker_id: str):
    conn = get_db(project_dir)
    conn.execute("DELETE FROM markers WHERE id = ?", (marker_id,))
    conn.commit()


# ── Timeline Validation ─────────────────────────────────────────────

def validate_timeline(project_dir: Path) -> list[str]:
    """Check timeline integrity. Returns list of warning strings (empty = healthy)."""
    warnings = []

    def parse_ts(ts):
        parts = str(ts).split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(ts) if isinstance(ts, (int, float)) else 0

    kfs = get_keyframes(project_dir)
    trs = get_transitions(project_dir)

    kf_times = {k["id"]: parse_ts(k["timestamp"]) for k in kfs}
    kf_sorted = sorted(kfs, key=lambda k: parse_ts(k["timestamp"]))

    # Build adjacency
    outgoing = {}  # kf_id -> list of tr
    incoming = {}  # kf_id -> list of tr
    for t in trs:
        outgoing.setdefault(t["from"], []).append(t)
        incoming.setdefault(t["to"], []).append(t)

    # Check each kf
    for i, kf in enumerate(kf_sorted):
        kf_id = kf["id"]
        outs = outgoing.get(kf_id, [])
        ins = incoming.get(kf_id, [])

        if len(outs) > 1:
            warnings.append(f"{kf_id}: {len(outs)} outgoing transitions")
        if len(ins) > 1:
            warnings.append(f"{kf_id}: {len(ins)} incoming transitions")
        if i > 0 and not ins:
            warnings.append(f"{kf_id}: no incoming transition")
        if i < len(kf_sorted) - 1 and not outs:
            warnings.append(f"{kf_id}: no outgoing transition")

    # Check transitions link to existing kfs and point forward in time
    active_kf_ids = set(kf_times.keys())
    for t in trs:
        if t["from"] not in active_kf_ids:
            warnings.append(f"{t['id']}: from_kf {t['from']} not found")
        elif t["to"] not in active_kf_ids:
            warnings.append(f"{t['id']}: to_kf {t['to']} not found")
        else:
            ft = kf_times[t["from"]]
            tt = kf_times[t["to"]]
            if ft > tt:
                warnings.append(f"{t['id']}: backwards {t['from']}({ft:.1f}s) -> {t['to']}({tt:.1f}s)")

    return warnings


# ── YAML Import / Export ────────────────────────────────────────────

def import_from_yaml(project_dir: Path):
    """Import project data from YAML files into SQLite."""
    from beatlab.project import load_project

    data = load_project(project_dir)
    if data.get("_format") == "empty":
        return

    conn = get_db(project_dir)

    # Clear existing data
    conn.execute("DELETE FROM keyframes")
    conn.execute("DELETE FROM transitions")
    conn.execute("DELETE FROM meta")
    conn.execute("DELETE FROM effects")
    conn.execute("DELETE FROM suppressions")

    # Meta
    meta = data.get("meta", {})
    for key, value in meta.items():
        if key.startswith("_"):
            continue
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            (key, json.dumps(value) if not isinstance(value, str) else value),
        )

    # Keyframes (active)
    for kf in data.get("keyframes", []):
        conn.execute(
            """INSERT INTO keyframes (id, timestamp, section, source, prompt, selected, candidates, context)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (kf["id"], kf.get("timestamp", "0:00"), kf.get("section", ""), kf.get("source", ""),
             kf.get("prompt", ""), kf.get("selected"), json.dumps(kf.get("candidates", [])),
             json.dumps(kf.get("context")) if kf.get("context") else None),
        )

    # Binned keyframes
    for kf in data.get("bin", []):
        conn.execute(
            """INSERT INTO keyframes (id, timestamp, section, source, prompt, selected, candidates, context, deleted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (kf["id"], kf.get("timestamp", "0:00"), kf.get("section", ""), kf.get("source", ""),
             kf.get("prompt", ""), kf.get("selected"), json.dumps(kf.get("candidates", [])),
             json.dumps(kf.get("context")) if kf.get("context") else None, kf.get("deleted_at", "")),
        )

    # Transitions (active)
    for tr in data.get("transitions", []):
        selected = tr.get("selected")
        if isinstance(selected, (int, str)) and selected is not None:
            selected = [selected]
        elif selected is None:
            selected = [None]
        conn.execute(
            """INSERT INTO transitions (id, from_kf, to_kf, duration_seconds, slots, action, use_global_prompt, selected, remap)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (tr["id"], tr.get("from", ""), tr.get("to", ""), tr.get("duration_seconds", 0),
             tr.get("slots", 1), tr.get("action", ""), int(tr.get("use_global_prompt", False)),
             json.dumps(selected), json.dumps(tr.get("remap", {"method": "linear", "target_duration": 0}))),
        )

    # Binned transitions
    for tr in data.get("transition_bin", []):
        selected = tr.get("selected")
        if isinstance(selected, (int, str)) and selected is not None:
            selected = [selected]
        elif selected is None:
            selected = [None]
        conn.execute(
            """INSERT INTO transitions (id, from_kf, to_kf, duration_seconds, slots, action, use_global_prompt, selected, remap, deleted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (tr["id"], tr.get("from", ""), tr.get("to", ""), tr.get("duration_seconds", 0),
             tr.get("slots", 1), tr.get("action", ""), int(tr.get("use_global_prompt", False)),
             json.dumps(selected), json.dumps(tr.get("remap", {"method": "linear", "target_duration": 0})),
             tr.get("deleted_at", "")),
        )

    # Watched folders as meta
    wf = data.get("watched_folders", [])
    if wf:
        conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)", ("watched_folders", json.dumps(wf)))

    # Import effects from beats.yaml
    import yaml
    effects_path = project_dir / "beats.yaml"
    if effects_path.exists():
        with open(effects_path) as f:
            effects_data = yaml.safe_load(f) or {}
        for fx in effects_data.get("effects", []):
            conn.execute(
                "INSERT INTO effects (id, type, time, intensity, duration) VALUES (?, ?, ?, ?, ?)",
                (fx["id"], fx.get("type", "pulse"), fx.get("time", 0), fx.get("intensity", 0.8), fx.get("duration", 0.2)),
            )
        for sup in effects_data.get("suppressions", []):
            conn.execute(
                "INSERT INTO suppressions (id, from_time, to_time, effect_types) VALUES (?, ?, ?, ?)",
                (sup["id"], sup.get("from", 0), sup.get("to", 0),
                 json.dumps(sup.get("effectTypes")) if sup.get("effectTypes") else None),
            )

    conn.commit()


def export_to_yaml(project_dir: Path):
    """Export SQLite data back to YAML files (split format)."""
    import yaml

    meta = get_meta(project_dir)
    keyframes = get_keyframes(project_dir)
    binned_kfs = get_binned_keyframes(project_dir)
    transitions = get_transitions(project_dir)
    binned_trs = get_binned_transitions(project_dir)

    # Strip internal fields from meta
    clean_meta = {k: v for k, v in meta.items() if not k.startswith("_") and k != "watched_folders"}
    watched = meta.get("watched_folders", [])
    if isinstance(watched, str):
        watched = json.loads(watched)

    # project.yaml
    project_data = {"meta": clean_meta, "watched_folders": watched}
    with open(project_dir / "project.yaml", "w") as f:
        yaml.dump(project_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Strip deleted_at from active items, convert back to YAML-friendly format
    def kf_to_yaml(kf):
        d = dict(kf)
        d.pop("deleted_at", None)
        return d

    def tr_to_yaml(tr):
        d = dict(tr)
        d.pop("deleted_at", None)
        # Restore YAML key names
        d["from"] = d.pop("from_kf", d.pop("from", ""))
        d["to"] = d.pop("to_kf", d.pop("to", ""))
        return d

    # timeline.yaml
    timeline = {
        "active_timeline": "default",
        "timelines": {
            "default": {
                "keyframes": [kf_to_yaml(kf) for kf in keyframes],
                "transitions": [tr_to_yaml(tr) for tr in transitions],
                "bin": [kf_to_yaml(kf) for kf in binned_kfs],
                "transition_bin": [tr_to_yaml(tr) for tr in binned_trs],
            }
        }
    }
    with open(project_dir / "timeline.yaml", "w") as f:
        yaml.dump(timeline, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=1000)
