"""E2E tests for duplicate-keyframe — verifies properties copy, no ghosts, no overlaps."""

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
    """Minimal valid-ish mp4 bytes (just enough for file existence checks)."""
    return b'\x00\x00\x00\x1cftypisom\x00\x00\x00\x00isomavc1'


def _setup_project(tmp_path):
    from beatlab.db import add_keyframe, add_transition

    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    (project_dir / "selected_keyframes").mkdir()
    (project_dir / "selected_transitions").mkdir()

    png = _make_png()
    mp4 = _make_mp4()

    for kf_id, ts in [("kf_001", "1:00.00"), ("kf_002", "1:10.00"), ("kf_003", "1:20.00")]:
        (project_dir / "selected_keyframes" / f"{kf_id}.png").write_bytes(png)
        add_keyframe(project_dir, {
            "id": kf_id, "timestamp": ts, "section": "", "source": "",
            "prompt": f"KF {kf_id}", "selected": 1, "candidates": [],
            "track_id": "track_1",
        })

    # Transition with properties: opacity curve, blend mode, selected video
    add_transition(project_dir, {
        "id": "tr_001", "from": "kf_001", "to": "kf_002",
        "duration_seconds": 10, "slots": 1, "action": "Smooth transition",
        "use_global_prompt": False, "selected": [1],
        "remap": {"method": "linear", "target_duration": 10},
        "track_id": "track_1",
        "blend_mode": "add",
        "opacity_curve": [[0, 0], [0.5, 1], [1, 0]],
        "saturation_curve": [[0, 1], [1, 0.5]],
    })

    # Create selected video for tr_001
    (project_dir / "selected_transitions" / "tr_001_slot_0.mp4").write_bytes(mp4)

    # Create candidates for tr_001
    cand_dir = project_dir / "transition_candidates" / "tr_001" / "slot_0"
    cand_dir.mkdir(parents=True)
    (cand_dir / "v1.mp4").write_bytes(mp4)
    (cand_dir / "v2.mp4").write_bytes(mp4)

    # Plain transition
    add_transition(project_dir, {
        "id": "tr_002", "from": "kf_002", "to": "kf_003",
        "duration_seconds": 10, "slots": 1, "action": "",
        "use_global_prompt": False, "selected": None,
        "remap": {"method": "linear", "target_duration": 10},
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


def _find_overlaps(transitions, keyframes):
    kf_map = {k["id"]: k for k in keyframes}
    items = []
    for tr in transitions:
        fk = kf_map.get(tr.get("from", ""))
        tk = kf_map.get(tr.get("to", ""))
        if not fk or not tk:
            continue
        items.append((_parse_ts(fk["timestamp"]), _parse_ts(tk["timestamp"]), tr["id"]))
    items.sort()
    overlaps = []
    for i in range(1, len(items)):
        if items[i][0] < items[i - 1][1]:
            overlaps.append((items[i - 1][2], items[i][2]))
    return overlaps


def _simulate_duplicate(project_dir, source_id, timestamp):
    """Simulate duplicate-keyframe logic matching the API server."""
    from beatlab.db import (
        add_keyframe as db_add_kf, get_keyframes as db_get_kfs,
        get_keyframe as db_get_kf, get_transitions as db_get_trs,
        add_transition as db_add_tr, delete_transition as db_del_tr,
        next_keyframe_id, next_transition_id,
        get_transition_effects, add_transition_effect,
        update_transition,
    )
    from datetime import datetime, timezone

    source_kf = db_get_kf(project_dir, source_id)
    track_id = source_kf.get("track_id", "track_1")
    new_id = next_keyframe_id(project_dir)
    new_time = _parse_ts(timestamp)

    # Copy keyframe image
    src_sel = project_dir / "selected_keyframes" / f"{source_id}.png"
    if src_sel.exists():
        shutil.copy2(str(src_sel), str(project_dir / "selected_keyframes" / f"{new_id}.png"))

    db_add_kf(project_dir, {
        "id": new_id, "timestamp": timestamp, "section": "",
        "source": "", "prompt": source_kf.get("prompt", ""),
        "selected": source_kf.get("selected"), "candidates": [],
        "track_id": track_id,
    })

    all_kfs = [k for k in db_get_kfs(project_dir)
               if k.get("track_id", "track_1") == track_id and not k.get("deleted_at")]
    sorted_kfs = sorted(all_kfs, key=lambda k: _parse_ts(k["timestamp"]))
    new_idx = next(i for i, k in enumerate(sorted_kfs) if k["id"] == new_id)
    prev_kf = sorted_kfs[new_idx - 1] if new_idx > 0 else None
    next_kf = sorted_kfs[new_idx + 1] if new_idx < len(sorted_kfs) - 1 else None

    all_trs = [t for t in db_get_trs(project_dir)
               if t.get("track_id", "track_1") == track_id and not t.get("deleted_at")]

    # Find and remove ALL transitions spanning across the new keyframe's position
    kf_time_map = {k["id"]: _parse_ts(k["timestamp"]) for k in sorted_kfs}
    spanning_trs = []
    for t in all_trs:
        from_time = kf_time_map.get(t.get("from", ""))
        to_time = kf_time_map.get(t.get("to", ""))
        if from_time is not None and to_time is not None:
            if from_time < new_time < to_time:
                spanning_trs.append(t)
    old_tr = spanning_trs[0] if spanning_trs else None
    for t in spanning_trs:
        db_del_tr(project_dir, t["id"], datetime.now(timezone.utc).isoformat())

    existing_from_prev = any(t["from"] == prev_kf["id"] and t["to"] == new_id for t in all_trs) if prev_kf else False
    existing_to_next = any(t["from"] == new_id and t["to"] == next_kf["id"] for t in all_trs) if next_kf else False

    tr_props = {}
    if old_tr:
        for prop in ("action", "blend_mode", "opacity", "opacity_curve",
                     "saturation_curve", "label", "label_color"):
            if old_tr.get(prop) is not None:
                tr_props[prop] = old_tr[prop]

    created_tr_ids = []

    if prev_kf and not existing_from_prev:
        dur = round(new_time - _parse_ts(prev_kf["timestamp"]), 2)
        if dur > 0.05:
            tr_id = next_transition_id(project_dir)
            db_add_tr(project_dir, {
                "id": tr_id, "from": prev_kf["id"], "to": new_id,
                "duration_seconds": dur, "slots": 1, "selected": None,
                "remap": {"method": "linear", "target_duration": dur},
                "track_id": track_id, **tr_props,
            })
            created_tr_ids.append(tr_id)

            if old_tr:
                old_sel = project_dir / "selected_transitions" / f"{old_tr['id']}_slot_0.mp4"
                if old_sel.exists():
                    new_sel = project_dir / "selected_transitions" / f"{tr_id}_slot_0.mp4"
                    shutil.copy2(str(old_sel), str(new_sel))
                    update_transition(project_dir, tr_id, selected=[1])

                old_cand = project_dir / "transition_candidates" / old_tr["id"]
                if old_cand.is_dir():
                    new_cand = project_dir / "transition_candidates" / tr_id
                    new_cand.mkdir(parents=True, exist_ok=True)
                    for slot_dir in old_cand.iterdir():
                        if slot_dir.is_dir():
                            dst_slot = new_cand / slot_dir.name
                            dst_slot.mkdir(parents=True, exist_ok=True)
                            for f in slot_dir.iterdir():
                                shutil.copy2(str(f), str(dst_slot / f.name))

    if next_kf and not existing_to_next:
        dur = round(_parse_ts(next_kf["timestamp"]) - new_time, 2)
        if dur > 0.05:
            tr_id = next_transition_id(project_dir)
            db_add_tr(project_dir, {
                "id": tr_id, "from": new_id, "to": next_kf["id"],
                "duration_seconds": dur, "slots": 1, "selected": None,
                "remap": {"method": "linear", "target_duration": dur},
                "track_id": track_id, **tr_props,
            })
            created_tr_ids.append(tr_id)

            if old_tr:
                old_sel = project_dir / "selected_transitions" / f"{old_tr['id']}_slot_0.mp4"
                if old_sel.exists():
                    new_sel = project_dir / "selected_transitions" / f"{tr_id}_slot_0.mp4"
                    shutil.copy2(str(old_sel), str(new_sel))
                    update_transition(project_dir, tr_id, selected=[1])

                old_cand = project_dir / "transition_candidates" / old_tr["id"]
                if old_cand.is_dir():
                    new_cand = project_dir / "transition_candidates" / tr_id
                    new_cand.mkdir(parents=True, exist_ok=True)
                    for slot_dir in old_cand.iterdir():
                        if slot_dir.is_dir():
                            dst_slot = new_cand / slot_dir.name
                            dst_slot.mkdir(parents=True, exist_ok=True)
                            for f in slot_dir.iterdir():
                                shutil.copy2(str(f), str(dst_slot / f.name))

    return new_id, created_tr_ids


class TestDuplicateKeyframe:
    def test_splits_transition_no_overlaps(self, tmp_path):
        """Duplicating into a spanning transition should split it without overlaps."""
        project_dir = _setup_project(tmp_path)

        new_id, tr_ids = _simulate_duplicate(project_dir, "kf_001", "1:05.00")
        assert len(tr_ids) == 2

        trs = _get_active_trs(project_dir)
        kfs = _get_active_kfs(project_dir)
        overlaps = _find_overlaps(trs, kfs)
        assert len(overlaps) == 0, f"Overlaps: {overlaps}"

    def test_copies_transition_properties(self, tmp_path):
        """New transitions should inherit properties from the spanning transition."""
        project_dir = _setup_project(tmp_path)

        new_id, tr_ids = _simulate_duplicate(project_dir, "kf_001", "1:05.00")
        from beatlab.db import get_transitions
        for tr_id in tr_ids:
            tr = next(t for t in get_transitions(project_dir) if t["id"] == tr_id)
            assert tr.get("blend_mode") == "add", f"{tr_id} missing blend_mode"
            assert tr.get("opacity_curve") is not None, f"{tr_id} missing opacity_curve"
            assert tr.get("action") == "Smooth transition", f"{tr_id} missing action"

    def test_copies_selected_video(self, tmp_path):
        """New transitions should get a copy of the spanning transition's selected video."""
        project_dir = _setup_project(tmp_path)

        new_id, tr_ids = _simulate_duplicate(project_dir, "kf_001", "1:05.00")
        for tr_id in tr_ids:
            sel = project_dir / "selected_transitions" / f"{tr_id}_slot_0.mp4"
            assert sel.exists(), f"{tr_id} missing selected video"

    def test_copies_transition_candidates(self, tmp_path):
        """New transitions should get copies of the spanning transition's candidate videos."""
        project_dir = _setup_project(tmp_path)

        new_id, tr_ids = _simulate_duplicate(project_dir, "kf_001", "1:05.00")
        for tr_id in tr_ids:
            cand_dir = project_dir / "transition_candidates" / tr_id / "slot_0"
            assert cand_dir.exists(), f"{tr_id} missing candidates dir"
            assert len(list(cand_dir.glob("v*.mp4"))) == 2, f"{tr_id} should have 2 candidates"

    def test_no_zero_length_transitions(self, tmp_path):
        """Duplicating at an existing keyframe's timestamp should not create zero-length transitions."""
        project_dir = _setup_project(tmp_path)

        new_id, tr_ids = _simulate_duplicate(project_dir, "kf_001", "1:10.00")
        # At 1:10 is kf_002 — one transition should be zero-length and skipped
        trs = _get_active_trs(project_dir)
        for tr in trs:
            kfs = _get_active_kfs(project_dir)
            kf_map = {k["id"]: k for k in kfs}
            fk = kf_map.get(tr.get("from", ""))
            tk = kf_map.get(tr.get("to", ""))
            if fk and tk:
                dur = _parse_ts(tk["timestamp"]) - _parse_ts(fk["timestamp"])
                assert dur > 0.05, f"{tr['id']} has zero/negative duration: {dur}"

    def test_double_duplicate_no_overlaps(self, tmp_path):
        """Duplicating twice near the same spot should not create overlapping transitions."""
        project_dir = _setup_project(tmp_path)

        _simulate_duplicate(project_dir, "kf_001", "1:03.00")
        _simulate_duplicate(project_dir, "kf_001", "1:06.00")

        trs = _get_active_trs(project_dir)
        kfs = _get_active_kfs(project_dir)
        overlaps = _find_overlaps(trs, kfs)
        assert len(overlaps) == 0, f"Overlaps after double duplicate: {overlaps}"

    def test_spanning_transition_not_immediate_neighbors(self, tmp_path):
        """Regression: duplicating a keyframe into the middle of a transition that spans
        across multiple keyframes (e.g., tr from kf_001 to kf_003, insert between kf_002 and kf_003)
        should still find and remove the spanning transition, not leave it as a ghost."""
        from beatlab.db import add_keyframe, add_transition, get_transitions, get_keyframes

        project_dir = tmp_path / "test_span"
        project_dir.mkdir()
        (project_dir / "selected_keyframes").mkdir()
        (project_dir / "selected_transitions").mkdir()

        png = _make_png()

        # 4 keyframes: kf_001 at 1:00, kf_002 at 1:10, kf_003 at 1:20, kf_004 at 1:30
        for kf_id, ts in [("kf_001", "1:00"), ("kf_002", "1:10"), ("kf_003", "1:20"), ("kf_004", "1:30")]:
            (project_dir / "selected_keyframes" / f"{kf_id}.png").write_bytes(png)
            add_keyframe(project_dir, {
                "id": kf_id, "timestamp": ts, "section": "", "source": "",
                "prompt": "", "selected": 1, "candidates": [], "track_id": "track_1",
            })

        # Normal transitions: kf_001→kf_002, kf_002→kf_003, kf_003→kf_004
        add_transition(project_dir, {
            "id": "tr_001", "from": "kf_001", "to": "kf_002",
            "duration_seconds": 10, "slots": 1, "selected": None,
            "remap": {"method": "linear", "target_duration": 10}, "track_id": "track_1",
        })
        add_transition(project_dir, {
            "id": "tr_002", "from": "kf_002", "to": "kf_003",
            "duration_seconds": 10, "slots": 1, "selected": None,
            "remap": {"method": "linear", "target_duration": 10}, "track_id": "track_1",
        })
        add_transition(project_dir, {
            "id": "tr_003", "from": "kf_003", "to": "kf_004",
            "duration_seconds": 10, "slots": 1, "selected": None,
            "remap": {"method": "linear", "target_duration": 10}, "track_id": "track_1",
        })

        # Now simulate a BAD state: a spanning transition from kf_001 to kf_003
        # (as if tr_002 got corrupted or a paste operation created it)
        add_transition(project_dir, {
            "id": "tr_bad", "from": "kf_001", "to": "kf_003",
            "duration_seconds": 20, "slots": 1, "selected": None,
            "remap": {"method": "linear", "target_duration": 20}, "track_id": "track_1",
        })

        # Duplicate kf_002 to 1:15 (between kf_002 at 1:10 and kf_003 at 1:20)
        # The spanning tr_bad (kf_001→kf_003) covers this time range
        new_id, tr_ids = _simulate_duplicate(project_dir, "kf_002", "1:15")

        # The spanning transition tr_bad should have been found and deleted
        trs = _get_active_trs(project_dir)
        tr_ids_active = [t["id"] for t in trs]
        assert "tr_bad" not in tr_ids_active, \
            f"Spanning transition tr_bad should have been deleted but is still active: {tr_ids_active}"

        # No overlaps should exist
        kfs = _get_active_kfs(project_dir)
        overlaps = _find_overlaps(trs, kfs)
        assert len(overlaps) == 0, f"Overlaps after spanning transition fix: {overlaps}"

    def test_no_cross_track_transitions(self, tmp_path):
        """All new transitions should be on the same track as the source keyframe."""
        project_dir = _setup_project(tmp_path)

        new_id, tr_ids = _simulate_duplicate(project_dir, "kf_001", "1:05.00")
        from beatlab.db import get_transitions
        for tr_id in tr_ids:
            tr = next(t for t in get_transitions(project_dir) if t["id"] == tr_id)
            assert tr.get("track_id") == "track_1", f"{tr_id} on wrong track: {tr.get('track_id')}"
