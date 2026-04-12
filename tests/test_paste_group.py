"""E2E tests for paste-group — verifies no duplicate/overlapping transitions on paste."""

from pathlib import Path
import shutil
import pytest


def _setup_project(tmp_path: Path):
    """Create a minimal project with keyframes and transitions for paste testing."""
    from beatlab.db import add_keyframe, add_transition

    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    (project_dir / "selected_keyframes").mkdir()
    (project_dir / "selected_transitions").mkdir()

    # 1x1 white PNG
    import struct, zlib
    def _make_png():
        raw = b'\x00\xff\xff\xff'
        compressed = zlib.compress(raw)
        ihdr = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
        chunks = b''
        for tag, data in [(b'IHDR', ihdr), (b'IDAT', compressed), (b'IEND', b'')]:
            chunks += struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
        return b'\x89PNG\r\n\x1a\n' + chunks

    png = _make_png()

    # Create 3 keyframes on track_1: 1:00, 1:05, 1:10
    for kf_id, ts in [("kf_001", "1:00.00"), ("kf_002", "1:05.00"), ("kf_003", "1:10.00")]:
        (project_dir / "selected_keyframes" / f"{kf_id}.png").write_bytes(png)
        add_keyframe(project_dir, {
            "id": kf_id, "timestamp": ts, "section": "", "source": "",
            "prompt": f"KF {kf_id}", "selected": 1, "candidates": [],
            "track_id": "track_1",
        })

    # Transitions: kf_001→kf_002, kf_002→kf_003
    add_transition(project_dir, {
        "id": "tr_001", "from": "kf_001", "to": "kf_002",
        "duration_seconds": 5, "slots": 1, "action": "", "use_global_prompt": False,
        "selected": None, "remap": {"method": "linear", "target_duration": 5},
        "track_id": "track_1", "opacity_curve": [[0, 0], [0.5, 1], [1, 0]],
    })
    add_transition(project_dir, {
        "id": "tr_002", "from": "kf_002", "to": "kf_003",
        "duration_seconds": 5, "slots": 1, "action": "", "use_global_prompt": False,
        "selected": None, "remap": {"method": "linear", "target_duration": 5},
        "track_id": "track_1",
    })

    return project_dir


def _parse_ts(ts):
    parts = str(ts).split(":")
    return int(parts[0]) * 60 + float(parts[1]) if len(parts) == 2 else float(ts)


def _get_active(project_dir, table="transitions", track=None):
    if table == "transitions":
        from beatlab.db import get_transitions
        items = [t for t in get_transitions(project_dir) if not t.get("deleted_at")]
    else:
        from beatlab.db import get_keyframes
        items = [k for k in get_keyframes(project_dir) if not k.get("deleted_at")]
    if track:
        items = [i for i in items if i.get("track_id", "track_1") == track]
    return items


def _find_overlaps(transitions, keyframes):
    kf_map = {k["id"]: k for k in keyframes}
    items = []
    for tr in transitions:
        from_kf = kf_map.get(tr.get("from", ""))
        to_kf = kf_map.get(tr.get("to", ""))
        if not from_kf or not to_kf:
            continue
        ft = _parse_ts(from_kf["timestamp"])
        tt = _parse_ts(to_kf["timestamp"])
        items.append((ft, tt, tr["id"]))
    items.sort()
    overlaps = []
    for i in range(1, len(items)):
        if items[i][0] < items[i - 1][1]:
            overlaps.append((items[i - 1][2], items[i][2]))
    return overlaps


def _simulate_paste(project_dir, kf_ids, target_time, target_track):
    """Simulate the paste-group logic."""
    from beatlab.db import (
        get_keyframe as db_get_kf, add_keyframe as db_add_kf,
        get_transitions as db_get_trs, add_transition as db_add_tr,
        next_keyframe_id, next_transition_id,
        get_transition_effects, add_transition_effect,
    )

    def secs_to_ts(s):
        m = int(s) // 60
        return f"{m}:{s - m * 60:05.2f}"

    src_kfs = []
    for kid in kf_ids:
        kf = db_get_kf(project_dir, kid)
        if kf and not kf.get("deleted_at"):
            src_kfs.append(kf)
    src_kfs.sort(key=lambda k: _parse_ts(k["timestamp"]))
    min_time = _parse_ts(src_kfs[0]["timestamp"])

    id_map = {}
    created_kfs = []
    for src in src_kfs:
        offset = _parse_ts(src["timestamp"]) - min_time
        new_time = target_time + offset
        new_id = next_keyframe_id(project_dir)
        id_map[src["id"]] = new_id

        src_sel = project_dir / "selected_keyframes" / f"{src['id']}.png"
        if src_sel.exists():
            dst_sel = project_dir / "selected_keyframes" / f"{new_id}.png"
            shutil.copy2(str(src_sel), str(dst_sel))

        db_add_kf(project_dir, {
            "id": new_id, "timestamp": secs_to_ts(new_time), "section": "",
            "source": "", "prompt": src.get("prompt", ""), "selected": src.get("selected"),
            "candidates": [], "track_id": target_track,
        })
        created_kfs.append({"id": new_id, "timestamp": secs_to_ts(new_time)})

    src_kf_set = set(kf_ids)
    all_trs = db_get_trs(project_dir)
    internal_trs = [t for t in all_trs
                    if t["from"] in src_kf_set and t["to"] in src_kf_set
                    and not t.get("deleted_at")]

    from beatlab.db import get_keyframes as db_get_kfs_paste
    all_kfs_paste = {k["id"]: k for k in db_get_kfs_paste(project_dir) if not k.get("deleted_at")}
    target_trs = [t for t in all_trs
                  if t.get("track_id") == target_track and not t.get("deleted_at")]
    existing_ranges = []
    for t in target_trs:
        fk = all_kfs_paste.get(t["from"])
        tk = all_kfs_paste.get(t["to"])
        if fk and tk:
            existing_ranges.append((_parse_ts(fk["timestamp"]), _parse_ts(tk["timestamp"])))

    created_trs = []
    for src_tr in internal_trs:
        new_from = id_map.get(src_tr["from"])
        new_to = id_map.get(src_tr["to"])
        if not new_from or not new_to:
            continue

        from_ts = _parse_ts(next((k["timestamp"] for k in created_kfs if k["id"] == new_from), "0"))
        to_ts = _parse_ts(next((k["timestamp"] for k in created_kfs if k["id"] == new_to), "0"))
        if to_ts - from_ts <= 0.05:
            continue

        if any(ef < to_ts and et > from_ts for ef, et in existing_ranges):
            continue

        new_tr_id = next_transition_id(project_dir)

        # Copy transition candidates
        src_cand = project_dir / "transition_candidates" / src_tr["id"]
        if src_cand.is_dir():
            dst_cand = project_dir / "transition_candidates" / new_tr_id
            dst_cand.mkdir(parents=True, exist_ok=True)
            for slot_dir in src_cand.iterdir():
                if slot_dir.is_dir():
                    dst_slot = dst_cand / slot_dir.name
                    dst_slot.mkdir(parents=True, exist_ok=True)
                    for f in slot_dir.iterdir():
                        shutil.copy2(str(f), str(dst_slot / f.name))

        # Copy selected video
        src_sel = project_dir / "selected_transitions" / f"{src_tr['id']}_slot_0.mp4"
        if src_sel.exists():
            dst_sel = project_dir / "selected_transitions" / f"{new_tr_id}_slot_0.mp4"
            shutil.copy2(str(src_sel), str(dst_sel))

        db_add_tr(project_dir, {
            "id": new_tr_id, "from": new_from, "to": new_to,
            "duration_seconds": src_tr.get("duration_seconds", 0),
            "slots": 1, "action": src_tr.get("action", ""),
            "use_global_prompt": False, "selected": src_tr.get("selected"),
            "remap": src_tr.get("remap", {}),
            "track_id": target_track,
            "opacity_curve": src_tr.get("opacity_curve"),
            "blend_mode": src_tr.get("blend_mode", ""),
        })
        created_trs.append({"id": new_tr_id, "from": new_from, "to": new_to})

    return created_kfs, created_trs


class TestPasteGroup:
    def test_paste_to_different_track(self, tmp_path):
        """Pasting to a different track should create clean non-overlapping transitions."""
        project_dir = _setup_project(tmp_path)

        created_kfs, created_trs = _simulate_paste(
            project_dir, ["kf_001", "kf_002", "kf_003"],
            target_time=120, target_track="track_2"
        )

        assert len(created_kfs) == 3
        assert len(created_trs) == 2

        trs = _get_active(project_dir, "transitions", "track_2")
        kfs = _get_active(project_dir, "keyframes", "track_2")
        overlaps = _find_overlaps(trs, kfs)
        assert len(overlaps) == 0, f"Overlaps on track_2: {overlaps}"

    def test_paste_same_track_no_overlap(self, tmp_path):
        """Pasting to the same track at a different time should not overlap existing transitions."""
        project_dir = _setup_project(tmp_path)

        created_kfs, created_trs = _simulate_paste(
            project_dir, ["kf_001", "kf_002", "kf_003"],
            target_time=180, target_track="track_1"
        )

        assert len(created_trs) == 2

        trs = _get_active(project_dir, "transitions", "track_1")
        kfs = _get_active(project_dir, "keyframes", "track_1")
        overlaps = _find_overlaps(trs, kfs)
        assert len(overlaps) == 0, f"Overlaps on track_1: {overlaps}"

    def test_double_paste_no_duplicates(self, tmp_path):
        """Pasting the same group twice to the same location should not create duplicate transitions."""
        project_dir = _setup_project(tmp_path)

        _simulate_paste(project_dir, ["kf_001", "kf_002"], target_time=120, target_track="track_2")
        trs_after_first = _get_active(project_dir, "transitions", "track_2")

        _simulate_paste(project_dir, ["kf_001", "kf_002"], target_time=120, target_track="track_2")
        trs_after_second = _get_active(project_dir, "transitions", "track_2")

        # Second paste creates new KFs (different IDs) so new transitions are expected
        # but they should not overlap
        kfs = _get_active(project_dir, "keyframes", "track_2")
        overlaps = _find_overlaps(trs_after_second, kfs)
        assert len(overlaps) == 0, f"Overlaps after double paste: {overlaps}"

    def test_paste_preserves_effects(self, tmp_path):
        """Pasted transitions should preserve opacity curves and blend modes."""
        project_dir = _setup_project(tmp_path)

        created_kfs, created_trs = _simulate_paste(
            project_dir, ["kf_001", "kf_002"],
            target_time=120, target_track="track_2"
        )

        assert len(created_trs) == 1
        from beatlab.db import get_transitions
        new_tr = next(t for t in get_transitions(project_dir) if t["id"] == created_trs[0]["id"])
        # tr_001 had an opacity_curve
        assert new_tr.get("opacity_curve") is not None, "Opacity curve should be preserved"

    def test_paste_filters_deleted_transitions(self, tmp_path):
        """Deleted source transitions should not be pasted."""
        project_dir = _setup_project(tmp_path)

        from beatlab.db import delete_transition
        from datetime import datetime, UTC
        delete_transition(project_dir, "tr_001", datetime.now(UTC).isoformat())

        created_kfs, created_trs = _simulate_paste(
            project_dir, ["kf_001", "kf_002", "kf_003"],
            target_time=120, target_track="track_2"
        )

        # tr_001 was deleted, so only tr_002 should be pasted
        assert len(created_trs) == 1

    def test_paste_preserves_selected_video_index(self, tmp_path):
        """Pasted transition should have correct selected index matching the copied candidate."""
        from beatlab.db import add_transition, add_keyframe, get_transitions, update_transition
        import hashlib

        project_dir = _setup_project(tmp_path)

        # Add selected video as v3 (not v1) to tr_001
        mp4 = b'\x00\x00\x00\x1cftypisom\x00\x00\x00\x00isomavc1'
        mp4_v3 = mp4 + b'v3_unique_data'

        cand_dir = project_dir / "transition_candidates" / "tr_001" / "slot_0"
        cand_dir.mkdir(parents=True)
        (cand_dir / "v1.mp4").write_bytes(mp4)
        (cand_dir / "v2.mp4").write_bytes(mp4 + b'v2')
        (cand_dir / "v3.mp4").write_bytes(mp4_v3)

        # Select v3
        sel_path = project_dir / "selected_transitions" / "tr_001_slot_0.mp4"
        sel_path.write_bytes(mp4_v3)
        update_transition(project_dir, "tr_001", selected=[3])

        # Paste
        created_kfs, created_trs = _simulate_paste(
            project_dir, ["kf_001", "kf_002"],
            target_time=120, target_track="track_2"
        )

        assert len(created_trs) == 1
        new_tr_id = created_trs[0]["id"]

        # Verify selected video file matches v3
        new_sel = project_dir / "selected_transitions" / f"{new_tr_id}_slot_0.mp4"
        assert new_sel.exists(), "Pasted transition should have selected video"

        new_sel_hash = hashlib.md5(new_sel.read_bytes()).hexdigest()
        v3_hash = hashlib.md5(mp4_v3).hexdigest()
        assert new_sel_hash == v3_hash, "Selected video content should match v3"

        # Verify selected index points to the right candidate
        new_tr = next(t for t in get_transitions(project_dir) if t["id"] == new_tr_id)
        sel = new_tr.get("selected")

        # Check that the candidate at the selected index matches
        new_cand_dir = project_dir / "transition_candidates" / new_tr_id / "slot_0"
        if new_cand_dir.exists():
            import json
            sel_idx = json.loads(sel) if isinstance(sel, str) else sel
            if isinstance(sel_idx, list):
                sel_idx = sel_idx[0]
            if sel_idx is not None:
                cand_file = new_cand_dir / f"v{sel_idx}.mp4"
                if cand_file.exists():
                    cand_hash = hashlib.md5(cand_file.read_bytes()).hexdigest()
                    assert cand_hash == v3_hash, \
                        f"Candidate v{sel_idx} should match selected video (got {cand_hash[:8]} vs {v3_hash[:8]})"
