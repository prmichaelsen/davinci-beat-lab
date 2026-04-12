"""E2E tests for batch-delete-keyframes — verifies bridging, property inheritance, no cross-track."""

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

    # Track 1: 5 keyframes
    for kf_id, ts in [("kf_001", "1:00.00"), ("kf_002", "1:05.00"), ("kf_003", "1:10.00"),
                       ("kf_004", "1:15.00"), ("kf_005", "1:20.00")]:
        (project_dir / "selected_keyframes" / f"{kf_id}.png").write_bytes(png)
        add_keyframe(project_dir, {
            "id": kf_id, "timestamp": ts, "section": "", "source": "",
            "prompt": f"KF {kf_id}", "selected": 1, "candidates": [], "track_id": "track_1",
        })

    # Track 2: 2 keyframes at overlapping times
    for kf_id, ts in [("kf_t2_001", "1:04.00"), ("kf_t2_002", "1:11.00")]:
        (project_dir / "selected_keyframes" / f"{kf_id}.png").write_bytes(png)
        add_keyframe(project_dir, {
            "id": kf_id, "timestamp": ts, "section": "", "source": "",
            "prompt": f"KF {kf_id}", "selected": 1, "candidates": [], "track_id": "track_2",
        })

    # Transitions with video + properties on track 1
    add_transition(project_dir, {"id": "tr_001", "from": "kf_001", "to": "kf_002",
        "duration_seconds": 5, "slots": 1, "action": "", "use_global_prompt": False,
        "selected": None, "remap": {"method": "linear", "target_duration": 5}, "track_id": "track_1"})

    add_transition(project_dir, {"id": "tr_002", "from": "kf_002", "to": "kf_003",
        "duration_seconds": 5, "slots": 1, "action": "Smooth", "use_global_prompt": False,
        "selected": [1], "remap": {"method": "linear", "target_duration": 5}, "track_id": "track_1",
        "blend_mode": "add", "opacity_curve": [[0, 0], [0.5, 1], [1, 0]]})
    (project_dir / "selected_transitions" / "tr_002_slot_0.mp4").write_bytes(mp4)
    cand_dir = project_dir / "transition_candidates" / "tr_002" / "slot_0"
    cand_dir.mkdir(parents=True)
    (cand_dir / "v1.mp4").write_bytes(mp4)

    add_transition(project_dir, {"id": "tr_003", "from": "kf_003", "to": "kf_004",
        "duration_seconds": 5, "slots": 1, "action": "", "use_global_prompt": False,
        "selected": None, "remap": {"method": "linear", "target_duration": 5}, "track_id": "track_1"})

    add_transition(project_dir, {"id": "tr_004", "from": "kf_004", "to": "kf_005",
        "duration_seconds": 5, "slots": 1, "action": "", "use_global_prompt": False,
        "selected": None, "remap": {"method": "linear", "target_duration": 5}, "track_id": "track_1"})

    # Track 2 transition
    add_transition(project_dir, {"id": "tr_t2_001", "from": "kf_t2_001", "to": "kf_t2_002",
        "duration_seconds": 7, "slots": 1, "action": "", "use_global_prompt": False,
        "selected": None, "remap": {"method": "linear", "target_duration": 7}, "track_id": "track_2"})

    return project_dir


def _parse_ts(ts):
    parts = str(ts).split(":")
    return int(parts[0]) * 60 + float(parts[1]) if len(parts) == 2 else float(ts)


def _get_active(project_dir, table, track=None):
    if table == "transitions":
        from beatlab.db import get_transitions
        items = [t for t in get_transitions(project_dir) if not t.get("deleted_at")]
    else:
        from beatlab.db import get_keyframes
        items = [k for k in get_keyframes(project_dir) if not k.get("deleted_at")]
    if track:
        items = [i for i in items if i.get("track_id", "track_1") == track]
    return items


def _simulate_batch_delete(project_dir, kf_ids):
    """Simulate batch-delete-keyframes matching the fixed API server."""
    from beatlab.db import (
        get_keyframe, delete_keyframe as db_del_kf, get_keyframes as db_get_kfs,
        get_transitions_involving, delete_transition as db_del_tr,
        next_transition_id, add_transition as db_add_tr, get_transitions as db_get_trs,
        get_transition_effects, add_transition_effect,
    )
    from datetime import datetime, timezone
    import os

    now = datetime.now(timezone.utc).isoformat()

    inherited_videos = {}
    for kf_id in kf_ids:
        kf = get_keyframe(project_dir, kf_id)
        if not kf:
            continue
        track = kf.get("track_id", "track_1")
        for tr in get_transitions_involving(project_dir, kf_id):
            sel = tr.get("selected")
            if sel is not None and sel != [None] and track not in inherited_videos:
                inherited_videos[track] = tr
            db_del_tr(project_dir, tr["id"], now)
        db_del_kf(project_dir, kf_id, now)

    tracks_affected = set()
    for kf_id in kf_ids:
        kf = get_keyframe(project_dir, kf_id)
        if kf:
            tracks_affected.add(kf.get("track_id", "track_1"))

    for track in tracks_affected:
        track_kfs = [k for k in db_get_kfs(project_dir)
                     if k.get("track_id", "track_1") == track and not k.get("deleted_at")]
        sorted_kfs = sorted(track_kfs, key=lambda k: _parse_ts(k["timestamp"]))

        active_trs = [t for t in db_get_trs(project_dir)
                      if t.get("track_id") == track and not t.get("deleted_at")]
        existing_pairs = set((t["from"], t["to"]) for t in active_trs)

        inh_tr = inherited_videos.get(track)

        for i in range(len(sorted_kfs) - 1):
            a = sorted_kfs[i]
            b = sorted_kfs[i + 1]
            if (a["id"], b["id"]) not in existing_pairs:
                dur = round(_parse_ts(b["timestamp"]) - _parse_ts(a["timestamp"]), 2)
                if dur <= 0.05:
                    continue

                tr_id = next_transition_id(project_dir)
                tr_props = {}
                selected = None

                if inh_tr:
                    for prop in ("action", "blend_mode", "opacity", "opacity_curve"):
                        if inh_tr.get(prop) is not None:
                            tr_props[prop] = inh_tr[prop]

                    old_sel = project_dir / "selected_transitions" / f"{inh_tr['id']}_slot_0.mp4"
                    if old_sel.exists():
                        new_sel = project_dir / "selected_transitions" / f"{tr_id}_slot_0.mp4"
                        shutil.copy2(str(old_sel), str(new_sel))
                        selected = 1

                db_add_tr(project_dir, {
                    "id": tr_id, "from": a["id"], "to": b["id"],
                    "duration_seconds": dur, "slots": 1, "selected": selected,
                    "remap": {"method": "linear", "target_duration": dur},
                    "track_id": track, **tr_props,
                })


class TestBatchDeleteKeyframe:
    def test_single_delete_bridges(self, tmp_path):
        """Deleting one middle keyframe should create a bridge."""
        project_dir = _setup_project(tmp_path)
        _simulate_batch_delete(project_dir, ["kf_003"])

        trs = _get_active(project_dir, "transitions", "track_1")
        pairs = set((t["from"], t["to"]) for t in trs)
        assert ("kf_002", "kf_004") in pairs, "Bridge kf_002→kf_004 missing"

    def test_consecutive_delete_bridges(self, tmp_path):
        """Deleting two consecutive middle keyframes should bridge across both."""
        project_dir = _setup_project(tmp_path)
        _simulate_batch_delete(project_dir, ["kf_002", "kf_003"])

        trs = _get_active(project_dir, "transitions", "track_1")
        pairs = set((t["from"], t["to"]) for t in trs)
        assert ("kf_001", "kf_004") in pairs, "Bridge kf_001→kf_004 missing"

    def test_no_orphaned_transitions(self, tmp_path):
        """No active transitions should reference deleted keyframes."""
        project_dir = _setup_project(tmp_path)
        _simulate_batch_delete(project_dir, ["kf_002", "kf_003"])

        trs = _get_active(project_dir, "transitions", "track_1")
        for tr in trs:
            assert tr["from"] != "kf_002" and tr["from"] != "kf_003"
            assert tr["to"] != "kf_002" and tr["to"] != "kf_003"

    def test_inherits_video(self, tmp_path):
        """Bridge should inherit video from deleted transition that had one."""
        project_dir = _setup_project(tmp_path)
        _simulate_batch_delete(project_dir, ["kf_003"])

        trs = _get_active(project_dir, "transitions", "track_1")
        bridge = next((t for t in trs if t["from"] == "kf_002" and t["to"] == "kf_004"), None)
        assert bridge is not None
        sel = project_dir / "selected_transitions" / f"{bridge['id']}_slot_0.mp4"
        assert sel.exists(), "Bridge should have inherited video"

    def test_inherits_properties(self, tmp_path):
        """Bridge should inherit blend_mode and opacity_curve from deleted transition."""
        project_dir = _setup_project(tmp_path)
        _simulate_batch_delete(project_dir, ["kf_003"])

        trs = _get_active(project_dir, "transitions", "track_1")
        bridge = next((t for t in trs if t["from"] == "kf_002" and t["to"] == "kf_004"), None)
        assert bridge is not None
        assert bridge.get("blend_mode") == "add"
        assert bridge.get("opacity_curve") is not None

    def test_no_cross_track_bridges(self, tmp_path):
        """Deleting track_1 keyframes should not create bridges on track_2."""
        project_dir = _setup_project(tmp_path)
        t2_trs_before = _get_active(project_dir, "transitions", "track_2")

        _simulate_batch_delete(project_dir, ["kf_002", "kf_003"])

        t2_trs_after = _get_active(project_dir, "transitions", "track_2")
        # Track 2 should be unchanged
        assert len(t2_trs_after) == len(t2_trs_before), \
            f"Track 2 transitions changed: {len(t2_trs_before)} → {len(t2_trs_after)}"

    def test_no_zero_length_bridges(self, tmp_path):
        """No bridge transitions should have zero or negative duration."""
        project_dir = _setup_project(tmp_path)
        _simulate_batch_delete(project_dir, ["kf_002", "kf_003", "kf_004"])

        trs = _get_active(project_dir, "transitions", "track_1")
        kfs = _get_active(project_dir, "keyframes", "track_1")
        kf_map = {k["id"]: k for k in kfs}
        for tr in trs:
            fk = kf_map.get(tr["from"])
            tk = kf_map.get(tr["to"])
            if fk and tk:
                dur = _parse_ts(tk["timestamp"]) - _parse_ts(fk["timestamp"])
                assert dur > 0.05, f"{tr['id']} has zero/negative duration: {dur}"

    def test_chain_intact(self, tmp_path):
        """After batch delete, the remaining chain should be fully connected."""
        project_dir = _setup_project(tmp_path)
        _simulate_batch_delete(project_dir, ["kf_002", "kf_004"])

        kfs = _get_active(project_dir, "keyframes", "track_1")
        trs = _get_active(project_dir, "transitions", "track_1")
        kf_ids = [k["id"] for k in sorted(kfs, key=lambda k: _parse_ts(k["timestamp"]))]
        assert kf_ids == ["kf_001", "kf_003", "kf_005"]

        pairs = set((t["from"], t["to"]) for t in trs)
        assert ("kf_001", "kf_003") in pairs
        assert ("kf_003", "kf_005") in pairs

    def test_delete_all_middle(self, tmp_path):
        """Deleting all middle keyframes should bridge first to last."""
        project_dir = _setup_project(tmp_path)
        _simulate_batch_delete(project_dir, ["kf_002", "kf_003", "kf_004"])

        kfs = _get_active(project_dir, "keyframes", "track_1")
        trs = _get_active(project_dir, "transitions", "track_1")
        kf_ids = [k["id"] for k in sorted(kfs, key=lambda k: _parse_ts(k["timestamp"]))]
        assert kf_ids == ["kf_001", "kf_005"]

        pairs = set((t["from"], t["to"]) for t in trs)
        assert ("kf_001", "kf_005") in pairs
