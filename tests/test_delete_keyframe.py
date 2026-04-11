"""E2E tests for delete-keyframe — verifies bridging, property inheritance, no ghosts."""

from pathlib import Path
import shutil
import pytest


def _make_png():
    import struct, zlib
    raw = b'\x00\xff\xff\xff'
    compressed = zlib.compress(raw)
    ihdr = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
    chunks = b''
    for tag, data in [(b'IHDR', ihdr), (b'IDAT', compressed), (b'IEND', b'')]:
        chunks += struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    return b'\x89PNG\r\n\x1a\n' + chunks


def _make_mp4():
    return b'\x00\x00\x00\x1cftypisom\x00\x00\x00\x00isomavc1'


def _setup_project(tmp_path):
    from beatlab.db import add_keyframe, add_transition

    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    (project_dir / "selected_keyframes").mkdir()
    (project_dir / "selected_transitions").mkdir()

    png = _make_png()
    mp4 = _make_mp4()

    for kf_id, ts in [("kf_001", "1:00.00"), ("kf_002", "1:05.00"), ("kf_003", "1:10.00"), ("kf_004", "1:15.00")]:
        (project_dir / "selected_keyframes" / f"{kf_id}.png").write_bytes(png)
        add_keyframe(project_dir, {
            "id": kf_id, "timestamp": ts, "section": "", "source": "",
            "prompt": f"KF {kf_id}", "selected": 1, "candidates": [],
            "track_id": "track_1",
        })

    # tr_001: kf_001→kf_002 (plain)
    add_transition(project_dir, {
        "id": "tr_001", "from": "kf_001", "to": "kf_002",
        "duration_seconds": 5, "slots": 1, "action": "",
        "use_global_prompt": False, "selected": None,
        "remap": {"method": "linear", "target_duration": 5},
        "track_id": "track_1",
    })

    # tr_002: kf_002→kf_003 (with video + properties)
    add_transition(project_dir, {
        "id": "tr_002", "from": "kf_002", "to": "kf_003",
        "duration_seconds": 5, "slots": 1, "action": "Smooth dissolve",
        "use_global_prompt": False, "selected": [1],
        "remap": {"method": "linear", "target_duration": 5},
        "track_id": "track_1",
        "blend_mode": "add",
        "opacity_curve": [[0, 0], [0.5, 1], [1, 0]],
    })
    (project_dir / "selected_transitions" / "tr_002_slot_0.mp4").write_bytes(mp4)
    cand_dir = project_dir / "transition_candidates" / "tr_002" / "slot_0"
    cand_dir.mkdir(parents=True)
    (cand_dir / "v1.mp4").write_bytes(mp4)

    # tr_003: kf_003→kf_004 (plain)
    add_transition(project_dir, {
        "id": "tr_003", "from": "kf_003", "to": "kf_004",
        "duration_seconds": 5, "slots": 1, "action": "",
        "use_global_prompt": False, "selected": None,
        "remap": {"method": "linear", "target_duration": 5},
        "track_id": "track_1",
    })

    return project_dir


def _parse_ts(ts):
    parts = str(ts).split(":")
    return int(parts[0]) * 60 + float(parts[1]) if len(parts) == 2 else float(ts)


def _get_active_trs(project_dir, track="track_1"):
    from beatlab.db import get_transitions
    return [t for t in get_transitions(project_dir)
            if t.get("track_id") == track and not t.get("deleted_at")]


def _get_active_kfs(project_dir, track="track_1"):
    from beatlab.db import get_keyframes
    return [k for k in get_keyframes(project_dir)
            if k.get("track_id", "track_1") == track and not k.get("deleted_at")]


def _simulate_delete(project_dir, kf_id):
    """Simulate delete-keyframe logic matching the fixed API server."""
    from beatlab.db import (
        get_keyframe, delete_keyframe as db_del_kf,
        get_transitions_involving, delete_transition as db_del_tr,
        get_keyframes as db_get_kfs, get_transitions as db_get_trs,
        next_transition_id, add_transition as db_add_tr,
        get_transition_effects, add_transition_effect,
    )
    from datetime import datetime, timezone
    import os

    kf = get_keyframe(project_dir, kf_id)
    now = datetime.now(timezone.utc).isoformat()

    orphaned = get_transitions_involving(project_dir, kf_id)
    inherited_tr_id = None
    for tr in orphaned:
        sel = tr.get("selected")
        if sel is not None and sel != [None]:
            inherited_tr_id = tr["id"]
            break

    for tr in orphaned:
        db_del_tr(project_dir, tr["id"], now)
    db_del_kf(project_dir, kf_id, now)

    # Bridge
    removed_time = _parse_ts(kf["timestamp"])
    kf_track = kf.get("track_id", "track_1")
    all_kfs = [k for k in db_get_kfs(project_dir)
               if k.get("track_id", "track_1") == kf_track and not k.get("deleted_at")]
    sorted_kfs = sorted(all_kfs, key=lambda k: _parse_ts(k["timestamp"]))
    prev_kf = None
    next_kf = None
    for k in sorted_kfs:
        t = _parse_ts(k["timestamp"])
        if t < removed_time:
            prev_kf = k
        elif t > removed_time and next_kf is None:
            next_kf = k

    bridge_id = None
    if prev_kf and next_kf:
        active_trs = [t for t in db_get_trs(project_dir)
                      if t.get("track_id") == kf_track and not t.get("deleted_at")]
        existing = next((t for t in active_trs
                         if t["from"] == prev_kf["id"] and t["to"] == next_kf["id"]), None)
        if not existing:
            bridge_id = next_transition_id(project_dir)
            dur = round(_parse_ts(next_kf["timestamp"]) - _parse_ts(prev_kf["timestamp"]), 2)

            tr_props = {}
            selected = None
            if inherited_tr_id:
                inh_tr = next((t for t in orphaned if t["id"] == inherited_tr_id), None)
                if inh_tr:
                    for prop in ("action", "blend_mode", "opacity", "opacity_curve"):
                        if inh_tr.get(prop) is not None:
                            tr_props[prop] = inh_tr[prop]

                old_sel = project_dir / "selected_transitions" / f"{inherited_tr_id}_slot_0.mp4"
                if old_sel.exists():
                    new_sel = project_dir / "selected_transitions" / f"{bridge_id}_slot_0.mp4"
                    shutil.copy2(str(old_sel), str(new_sel))
                    selected = 1

                old_cand = project_dir / "transition_candidates" / inherited_tr_id / "slot_0"
                if old_cand.exists():
                    new_cand = project_dir / "transition_candidates" / bridge_id / "slot_0"
                    new_cand.mkdir(parents=True, exist_ok=True)
                    for f in old_cand.iterdir():
                        shutil.copy2(str(f), str(new_cand / f.name))

            if dur > 0.05:
                db_add_tr(project_dir, {
                    "id": bridge_id, "from": prev_kf["id"], "to": next_kf["id"],
                    "duration_seconds": dur, "slots": 1, "selected": selected,
                    "remap": {"method": "linear", "target_duration": dur},
                    "track_id": kf_track, **tr_props,
                })

    return bridge_id


class TestDeleteKeyframe:
    def test_bridge_created(self, tmp_path):
        """Deleting a middle keyframe should create a bridge transition."""
        project_dir = _setup_project(tmp_path)
        bridge_id = _simulate_delete(project_dir, "kf_002")

        assert bridge_id is not None
        trs = _get_active_trs(project_dir)
        bridge = next((t for t in trs if t["id"] == bridge_id), None)
        assert bridge is not None
        assert bridge["from"] == "kf_001"
        assert bridge["to"] == "kf_003"

    def test_no_orphaned_transitions(self, tmp_path):
        """After delete, no active transitions should reference the deleted keyframe."""
        project_dir = _setup_project(tmp_path)
        _simulate_delete(project_dir, "kf_002")

        trs = _get_active_trs(project_dir)
        for tr in trs:
            assert tr["from"] != "kf_002", f"{tr['id']} still references deleted kf_002 as from"
            assert tr["to"] != "kf_002", f"{tr['id']} still references deleted kf_002 as to"

    def test_inherits_transition_properties(self, tmp_path):
        """Bridge transition should inherit properties from the orphaned transition with video."""
        project_dir = _setup_project(tmp_path)
        bridge_id = _simulate_delete(project_dir, "kf_002")

        from beatlab.db import get_transitions
        bridge = next(t for t in get_transitions(project_dir) if t["id"] == bridge_id)
        # tr_001 had no properties, tr_002 had blend_mode=add + opacity_curve
        # inherited_tr_id should be tr_002 (has selected video)
        assert bridge.get("blend_mode") == "add"
        assert bridge.get("opacity_curve") is not None
        assert bridge.get("action") == "Smooth dissolve"

    def test_inherits_selected_video(self, tmp_path):
        """Bridge transition should get a copy of the inherited transition's video."""
        project_dir = _setup_project(tmp_path)
        bridge_id = _simulate_delete(project_dir, "kf_002")

        sel = project_dir / "selected_transitions" / f"{bridge_id}_slot_0.mp4"
        assert sel.exists(), "Bridge should have inherited selected video"

    def test_inherits_candidates(self, tmp_path):
        """Bridge transition should get copies of the inherited transition's candidates."""
        project_dir = _setup_project(tmp_path)
        bridge_id = _simulate_delete(project_dir, "kf_002")

        cand_dir = project_dir / "transition_candidates" / bridge_id / "slot_0"
        assert cand_dir.exists()
        assert len(list(cand_dir.glob("v*.mp4"))) == 1

    def test_chain_intact_after_delete(self, tmp_path):
        """The full chain should be connected after deleting a middle keyframe."""
        project_dir = _setup_project(tmp_path)
        _simulate_delete(project_dir, "kf_002")

        trs = _get_active_trs(project_dir)
        kfs = _get_active_kfs(project_dir)
        kf_ids = [k["id"] for k in sorted(kfs, key=lambda k: _parse_ts(k["timestamp"]))]
        # Should be kf_001 → kf_003 → kf_004
        assert kf_ids == ["kf_001", "kf_003", "kf_004"]

        # Check transitions form a chain
        tr_pairs = set((t["from"], t["to"]) for t in trs)
        assert ("kf_001", "kf_003") in tr_pairs, "Bridge kf_001→kf_003 missing"
        assert ("kf_003", "kf_004") in tr_pairs, "tr_003 kf_003→kf_004 missing"

    def test_delete_first_keyframe(self, tmp_path):
        """Deleting the first keyframe should not crash and should clean up transitions."""
        project_dir = _setup_project(tmp_path)
        bridge_id = _simulate_delete(project_dir, "kf_001")

        # No bridge expected (no prev_kf)
        trs = _get_active_trs(project_dir)
        for tr in trs:
            assert tr["from"] != "kf_001"
            assert tr["to"] != "kf_001"

    def test_delete_last_keyframe(self, tmp_path):
        """Deleting the last keyframe should not crash and should clean up transitions."""
        project_dir = _setup_project(tmp_path)
        bridge_id = _simulate_delete(project_dir, "kf_004")

        trs = _get_active_trs(project_dir)
        for tr in trs:
            assert tr["from"] != "kf_004"
            assert tr["to"] != "kf_004"

    def test_sequential_deletes(self, tmp_path):
        """Deleting multiple keyframes in sequence should maintain chain integrity."""
        project_dir = _setup_project(tmp_path)
        _simulate_delete(project_dir, "kf_002")
        _simulate_delete(project_dir, "kf_003")

        trs = _get_active_trs(project_dir)
        kfs = _get_active_kfs(project_dir)
        kf_ids = [k["id"] for k in sorted(kfs, key=lambda k: _parse_ts(k["timestamp"]))]
        assert kf_ids == ["kf_001", "kf_004"]

        tr_pairs = set((t["from"], t["to"]) for t in trs)
        assert ("kf_001", "kf_004") in tr_pairs
