"""Integration tests for the REST API server.

Spins up the API server on a random port against a temp project directory,
then exercises the key REST endpoints end-to-end through the DB layer.
"""

import json
import shutil
import tempfile
import threading
from http.server import HTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from beatlab.api_server import make_handler
from beatlab.db import (
    close_db, get_keyframes, get_transitions, set_meta,
    _migrated_dbs,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def project_env():
    """Create a temp work dir with a single project and start the API server."""
    work_dir = Path(tempfile.mkdtemp())
    project_name = "test_project"
    project_dir = work_dir / project_name
    project_dir.mkdir()

    # Seed metadata so the project is recognized
    set_meta(project_dir, "title", "Test Project")
    set_meta(project_dir, "fps", 24)

    Handler = make_handler(work_dir)
    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield {
        "work_dir": work_dir,
        "project_dir": project_dir,
        "project_name": project_name,
        "base_url": f"http://127.0.0.1:{port}",
    }

    server.shutdown()
    close_db(project_dir)
    # Clear migration cache so next test gets a fresh DB
    db_path = str(project_dir / "project.db")
    _migrated_dbs.discard(db_path)
    shutil.rmtree(work_dir)


def api(env, method, path, body=None):
    """Helper: send a JSON request to the test server."""
    from urllib.error import HTTPError
    url = f"{env['base_url']}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        resp = urlopen(req)
        return json.loads(resp.read())
    except HTTPError as e:
        error_body = e.read().decode()
        raise AssertionError(f"HTTP {e.code} on {method} {path}: {error_body}") from e


def get_editor_data(env):
    """Fetch keyframes + transitions via the API (avoids cross-thread DB issues)."""
    return api(env, "GET", f"/api/projects/{env['project_name']}/keyframes")


def active_transitions(env):
    """Get non-deleted transitions via API."""
    data = get_editor_data(env)
    return data.get("transitions", [])


def parse_ts(ts):
    """Parse timestamp string to seconds."""
    parts = str(ts).split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(ts)


def assert_timeline_integrity(env, track_id="track_1"):
    """Validate that the timeline chain is consistent for a given track.

    Rules:
    - Keyframes sorted by timestamp form a linear chain
    - Each adjacent pair has exactly one transition connecting them
    - No orphaned transitions (from/to pointing at missing kf)
    - No duplicate transitions between the same pair
    - Transition durations match keyframe time gaps
    - All transitions' from/to keyframes are on the same track
    """
    data = get_editor_data(env)
    kfs = [k for k in data["keyframes"] if k.get("trackId", "track_1") == track_id]
    trs = [t for t in data["transitions"] if t.get("trackId", "track_1") == track_id]
    kf_ids = {k["id"] for k in kfs}

    # Sort keyframes by time
    sorted_kfs = sorted(kfs, key=lambda k: parse_ts(k["timestamp"]))

    # Build transition lookup
    tr_by_pair = {}
    for tr in trs:
        pair = (tr["from"], tr["to"])
        assert pair not in tr_by_pair, f"Duplicate transition for {pair}: {tr['id']} and {tr_by_pair[pair]['id']}"
        tr_by_pair[pair] = tr

    # Check no orphaned transitions
    for tr in trs:
        assert tr["from"] in kf_ids, f"Transition {tr['id']} has orphaned 'from': {tr['from']}"
        assert tr["to"] in kf_ids, f"Transition {tr['id']} has orphaned 'to': {tr['to']}"

    # Check chain: each adjacent pair should have a transition
    for i in range(len(sorted_kfs) - 1):
        kf_a = sorted_kfs[i]
        kf_b = sorted_kfs[i + 1]
        pair = (kf_a["id"], kf_b["id"])
        assert pair in tr_by_pair, \
            f"Missing transition {kf_a['id']} ({kf_a['timestamp']}) -> {kf_b['id']} ({kf_b['timestamp']})"

        # Verify duration approximately matches time gap
        tr = tr_by_pair[pair]
        expected_dur = round(parse_ts(kf_b["timestamp"]) - parse_ts(kf_a["timestamp"]), 2)
        assert abs(tr["durationSeconds"] - expected_dur) < 0.05, \
            f"Transition {tr['id']} duration {tr['durationSeconds']} != expected {expected_dur}"

    # No extra transitions beyond the chain
    expected_pairs = {(sorted_kfs[i]["id"], sorted_kfs[i + 1]["id"]) for i in range(len(sorted_kfs) - 1)}
    actual_pairs = set(tr_by_pair.keys())
    extra = actual_pairs - expected_pairs
    assert not extra, f"Extra transitions not in chain: {extra}"

    return sorted_kfs, trs


# ── Add Keyframe ────────────────────────────────────────────────────


class TestAddKeyframe:
    def test_add_single_keyframe(self, project_env):
        env = project_env
        result = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
            "timestamp": "0:05.00",
        })
        assert result["success"] is True
        kf_id = result["keyframe"]["id"]

        data = get_editor_data(env)
        assert any(k["id"] == kf_id for k in data["keyframes"])

    def test_add_keyframe_with_track(self, project_env):
        env = project_env
        result = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
            "timestamp": "0:10.00",
            "trackId": "track_2",
        })
        kf_id = result["keyframe"]["id"]

        data = get_editor_data(env)
        kf = next(k for k in data["keyframes"] if k["id"] == kf_id)
        assert kf["trackId"] == "track_2"

    def test_add_keyframe_creates_transitions_to_neighbors(self, project_env):
        """Adding 3 keyframes should produce 2 transitions linking them."""
        env = project_env

        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        trs = active_transitions(env)
        assert len(trs) >= 2

    def test_add_keyframe_relinks_spanning_transition(self, project_env):
        """Inserting a kf into an existing transition should relink, not delete."""
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})
        kf1_id = r1["keyframe"]["id"]
        kf2_id = r2["keyframe"]["id"]

        trs_before = active_transitions(env)
        spanning = next((t for t in trs_before if t["from"] == kf1_id and t["to"] == kf2_id), None)
        assert spanning is not None, "Should have a transition between kf1 and kf2"
        original_tr_id = spanning["id"]

        # Insert kf in the middle
        r3 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        kf3_id = r3["keyframe"]["id"]

        trs_after = active_transitions(env)

        # Original transition should still exist, relinked to kf3
        relinked = next((t for t in trs_after if t["id"] == original_tr_id), None)
        assert relinked is not None, "Original transition should be preserved"
        assert relinked["from"] == kf1_id
        assert relinked["to"] == kf3_id

        # New transition from kf3 -> kf2
        bridge = next((t for t in trs_after if t["from"] == kf3_id and t["to"] == kf2_id), None)
        assert bridge is not None, "Should have a new transition from inserted kf to next"

    def test_add_keyframe_different_tracks_independent(self, project_env):
        """Keyframes on different tracks shouldn't create cross-track transitions."""
        env = project_env

        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00", "trackId": "track_1"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00", "trackId": "track_1"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:02.00", "trackId": "track_2"})

        data = get_editor_data(env)
        kf_map = {k["id"]: k for k in data["keyframes"]}
        for tr in data["transitions"]:
            from_kf = kf_map[tr["from"]]
            to_kf = kf_map[tr["to"]]
            assert from_kf["trackId"] == to_kf["trackId"], \
                f"Transition {tr['id']} crosses tracks: {from_kf['trackId']} -> {to_kf['trackId']}"


# ── Delete Keyframe ─────────────────────────────────────────────────


class TestDeleteKeyframe:
    def test_delete_keyframe_removes_from_active(self, project_env):
        env = project_env
        r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        kf_id = r["keyframe"]["id"]

        api(env, "POST", f"/api/projects/{env['project_name']}/delete-keyframe", {"keyframeId": kf_id})

        data = get_editor_data(env)
        assert not any(k["id"] == kf_id for k in data["keyframes"])

    def test_delete_middle_keyframe_bridges_neighbors(self, project_env):
        """Deleting the middle kf should create a bridge transition.
        Note: bridging runs in a background thread, so we poll briefly."""
        import time
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        r3 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})
        kf1 = r1["keyframe"]["id"]
        kf2 = r2["keyframe"]["id"]
        kf3 = r3["keyframe"]["id"]

        api(env, "POST", f"/api/projects/{env['project_name']}/delete-keyframe", {"keyframeId": kf2})

        # Bridge runs in a background thread — poll up to 2s
        bridge = None
        for _ in range(20):
            trs = active_transitions(env)
            bridge = next((t for t in trs if t["from"] == kf1 and t["to"] == kf3), None)
            if bridge:
                break
            time.sleep(0.1)
        assert bridge is not None, f"Should bridge {kf1} -> {kf3} after deleting {kf2}"


# ── Update Timestamp ────────────────────────────────────────────────


class TestUpdateTimestamp:
    def test_update_timestamp(self, project_env):
        env = project_env
        r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        kf_id = r["keyframe"]["id"]

        api(env, "POST", f"/api/projects/{env['project_name']}/update-timestamp", {
            "keyframeId": kf_id,
            "newTimestamp": "0:08.00",
        })

        data = get_editor_data(env)
        kf = next(k for k in data["keyframes"] if k["id"] == kf_id)
        assert kf["timestamp"] == "0:08.00"


# ── Track Operations ────────────────────────────────────────────────


class TestTracks:
    def test_add_track(self, project_env):
        env = project_env
        result = api(env, "POST", f"/api/projects/{env['project_name']}/tracks/add", {})
        assert result.get("success") is True
        assert result.get("id") is not None

    def test_update_track(self, project_env):
        env = project_env
        r = api(env, "POST", f"/api/projects/{env['project_name']}/tracks/add", {})
        track_id = r["id"]

        api(env, "POST", f"/api/projects/{env['project_name']}/tracks/update", {
            "id": track_id,
            "name": "My Track",
            "blend_mode": "multiply",
            "base_opacity": 0.5,
        })

        from beatlab.db import get_tracks
        tracks = get_tracks(env["project_dir"])
        track = next(t for t in tracks if t["id"] == track_id)
        assert track["name"] == "My Track"
        assert track["blend_mode"] == "multiply"
        assert track["base_opacity"] == 0.5


# ── Style Operations (blend mode / opacity) ─────────────────────────


class TestStyle:
    def test_update_keyframe_style(self, project_env):
        env = project_env
        r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        kf_id = r["keyframe"]["id"]

        api(env, "POST", f"/api/projects/{env['project_name']}/update-keyframe-style", {
            "keyframeId": kf_id,
            "blendMode": "screen",
            "opacity": 0.7,
        })

        from beatlab.db import get_keyframe
        kf = get_keyframe(env["project_dir"], kf_id)
        assert kf["blend_mode"] == "screen"
        assert kf["opacity"] == pytest.approx(0.7)

    def test_update_transition_style_with_opacity_curve(self, project_env):
        env = project_env
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})

        trs = active_transitions(env)
        assert len(trs) > 0
        tr_id = trs[0]["id"]

        curve = [[0, 0], [0.5, 1], [1, 0.5]]
        api(env, "POST", f"/api/projects/{env['project_name']}/update-transition-style", {
            "transitionId": tr_id,
            "blendMode": "overlay",
            "opacityCurve": curve,
        })

        from beatlab.db import get_transition
        tr = get_transition(env["project_dir"], tr_id)
        assert tr["blend_mode"] == "overlay"
        assert tr["opacity_curve"] == curve


# ── Label Operations ────────────────────────────────────────────────


class TestLabels:
    def test_update_keyframe_label(self, project_env):
        env = project_env
        r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        kf_id = r["keyframe"]["id"]

        api(env, "POST", f"/api/projects/{env['project_name']}/update-keyframe-label", {
            "keyframeId": kf_id,
            "label": "Intro Shot",
            "labelColor": "#ff0000",
        })

        from beatlab.db import get_keyframe
        kf = get_keyframe(env["project_dir"], kf_id)
        assert kf["label"] == "Intro Shot"
        assert kf["label_color"] == "#ff0000"

    def test_update_transition_label(self, project_env):
        env = project_env
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})

        trs = active_transitions(env)
        tr_id = trs[0]["id"]

        api(env, "POST", f"/api/projects/{env['project_name']}/update-transition-label", {
            "transitionId": tr_id,
            "label": "Cross Dissolve",
            "labelColor": "#00ff00",
            "tags": ["dissolve", "smooth"],
        })

        from beatlab.db import get_transition
        tr = get_transition(env["project_dir"], tr_id)
        assert tr["label"] == "Cross Dissolve"
        assert tr["label_color"] == "#00ff00"
        assert tr["tags"] == ["dissolve", "smooth"]


# ── Get Keyframes (editor data) ─────────────────────────────────────


class TestGetKeyframes:
    def test_returns_keyframes_and_transitions(self, project_env):
        env = project_env

        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        result = get_editor_data(env)

        assert "keyframes" in result
        assert "transitions" in result
        assert len(result["keyframes"]) == 2
        assert len(result["transitions"]) >= 1

    def test_includes_track_id(self, project_env):
        env = project_env

        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
            "timestamp": "0:05.00", "trackId": "track_2",
        })

        result = get_editor_data(env)
        kf = result["keyframes"][0]
        assert kf["trackId"] == "track_2"

    def test_includes_blend_mode_and_opacity(self, project_env):
        env = project_env

        r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        kf_id = r["keyframe"]["id"]

        api(env, "POST", f"/api/projects/{env['project_name']}/update-keyframe-style", {
            "keyframeId": kf_id, "blendMode": "difference", "opacity": 0.3,
        })

        result = get_editor_data(env)
        kf = next(k for k in result["keyframes"] if k["id"] == kf_id)
        assert kf["blendMode"] == "difference"
        assert kf["opacity"] == pytest.approx(0.3)

    def test_transition_includes_opacity_curve(self, project_env):
        env = project_env

        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})

        trs = active_transitions(env)
        tr_id = trs[0]["id"]
        curve = [[0, 1], [0.5, 0], [1, 1]]

        api(env, "POST", f"/api/projects/{env['project_name']}/update-transition-style", {
            "transitionId": tr_id, "opacityCurve": curve,
        })

        result = get_editor_data(env)
        tr = next(t for t in result["transitions"] if t["id"] == tr_id)
        assert tr["opacityCurve"] == curve


# ── Drag-Drop: assign-pool-video ────────────────────────────────────


def _make_fake_video(project_dir, rel_path):
    """Create a tiny placeholder file to simulate a video on disk."""
    p = project_dir / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00" * 64)
    return rel_path


def _make_fake_image(project_dir, rel_path):
    """Create a tiny placeholder file to simulate an image on disk."""
    p = project_dir / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 56)
    return rel_path


class TestAssignPoolVideo:
    def test_assigns_video_as_candidate(self, project_env):
        """assign-pool-video should copy the video as a candidate and set it selected."""
        env = project_env
        pd = env["project_dir"]

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        trs = active_transitions(env)
        tr_id = trs[0]["id"]

        pool_path = _make_fake_video(pd, "pool/segments/test_clip.mp4")

        result = api(env, "POST", f"/api/projects/{env['project_name']}/assign-pool-video", {
            "transitionId": tr_id,
            "poolPath": pool_path,
        })

        assert result["success"] is True
        assert result["variant"] == 1

        # Candidate file should exist
        cand = pd / "transition_candidates" / tr_id / "slot_0" / "v1.mp4"
        assert cand.exists()

        # Selected file should exist
        sel = pd / "selected_transitions" / f"{tr_id}_slot_0.mp4"
        assert sel.exists()

        # DB should reflect selected variant
        from beatlab.db import get_transition
        tr = get_transition(pd, tr_id)
        assert tr["selected"] == 1

    def test_assigns_multiple_videos_increments_variant(self, project_env):
        """Assigning multiple videos should create v1, v2, etc."""
        env = project_env
        pd = env["project_dir"]

        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        trs = active_transitions(env)
        tr_id = trs[0]["id"]

        _make_fake_video(pd, "pool/segments/clip_a.mp4")
        _make_fake_video(pd, "pool/segments/clip_b.mp4")

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/assign-pool-video", {
            "transitionId": tr_id, "poolPath": "pool/segments/clip_a.mp4",
        })
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/assign-pool-video", {
            "transitionId": tr_id, "poolPath": "pool/segments/clip_b.mp4",
        })

        assert r1["variant"] == 1
        assert r2["variant"] == 2

        assert (pd / "transition_candidates" / tr_id / "slot_0" / "v1.mp4").exists()
        assert (pd / "transition_candidates" / tr_id / "slot_0" / "v2.mp4").exists()


# ── Drag-Drop: duplicate-transition-video ───────────────────────────


class TestDuplicateTransitionVideo:
    def test_copies_candidates_and_selected(self, project_env):
        """duplicate-transition-video should copy all candidates and selected video."""
        env = project_env
        pd = env["project_dir"]

        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})

        trs = active_transitions(env)
        assert len(trs) >= 2
        source_tr = trs[0]
        target_tr = trs[1]

        # Seed source with candidates and selected
        src_cand_dir = pd / "transition_candidates" / source_tr["id"] / "slot_0"
        src_cand_dir.mkdir(parents=True, exist_ok=True)
        (src_cand_dir / "v1.mp4").write_bytes(b"video_a")
        (src_cand_dir / "v2.mp4").write_bytes(b"video_b")

        src_sel = pd / "selected_transitions" / f"{source_tr['id']}_slot_0.mp4"
        src_sel.parent.mkdir(parents=True, exist_ok=True)
        src_sel.write_bytes(b"selected_video")

        from beatlab.db import update_transition
        update_transition(pd, source_tr["id"], selected=1)

        # Duplicate
        result = api(env, "POST", f"/api/projects/{env['project_name']}/duplicate-transition-video", {
            "sourceId": source_tr["id"],
            "targetId": target_tr["id"],
        })
        assert result["success"] is True

        # Target should have candidates copied
        dst_cand_dir = pd / "transition_candidates" / target_tr["id"] / "slot_0"
        assert (dst_cand_dir / "v1.mp4").exists()
        assert (dst_cand_dir / "v2.mp4").exists()
        assert (dst_cand_dir / "v1.mp4").read_bytes() == b"video_a"

        # Target should have selected video copied
        dst_sel = pd / "selected_transitions" / f"{target_tr['id']}_slot_0.mp4"
        assert dst_sel.exists()
        assert dst_sel.read_bytes() == b"selected_video"

        # Target DB should have selected variant set
        from beatlab.db import get_transition
        tr = get_transition(pd, target_tr["id"])
        assert tr["selected"] == 1

    def test_copies_action_prompt(self, project_env):
        """duplicate-transition-video should also copy the action prompt."""
        env = project_env
        pd = env["project_dir"]

        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})

        trs = active_transitions(env)
        source_tr = trs[0]
        target_tr = trs[1]

        # Set action on source
        from beatlab.db import update_transition
        update_transition(pd, source_tr["id"], action="Slow zoom into face")

        # Need a selected video for the copy to trigger
        src_sel = pd / "selected_transitions" / f"{source_tr['id']}_slot_0.mp4"
        src_sel.parent.mkdir(parents=True, exist_ok=True)
        src_sel.write_bytes(b"video")
        update_transition(pd, source_tr["id"], selected=1)

        api(env, "POST", f"/api/projects/{env['project_name']}/duplicate-transition-video", {
            "sourceId": source_tr["id"],
            "targetId": target_tr["id"],
        })

        from beatlab.db import get_transition
        tr = get_transition(pd, target_tr["id"])
        assert tr["action"] == "Slow zoom into face"

    def test_does_not_overwrite_existing_candidates(self, project_env):
        """If target already has a v1.mp4, it should not be overwritten."""
        env = project_env
        pd = env["project_dir"]

        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})

        trs = active_transitions(env)
        source_tr = trs[0]
        target_tr = trs[1]

        # Seed source
        src_dir = pd / "transition_candidates" / source_tr["id"] / "slot_0"
        src_dir.mkdir(parents=True, exist_ok=True)
        (src_dir / "v1.mp4").write_bytes(b"source_v1")

        # Seed target with existing v1
        dst_dir = pd / "transition_candidates" / target_tr["id"] / "slot_0"
        dst_dir.mkdir(parents=True, exist_ok=True)
        (dst_dir / "v1.mp4").write_bytes(b"original_target_v1")

        api(env, "POST", f"/api/projects/{env['project_name']}/duplicate-transition-video", {
            "sourceId": source_tr["id"],
            "targetId": target_tr["id"],
        })

        # Target v1 should NOT be overwritten
        assert (dst_dir / "v1.mp4").read_bytes() == b"original_target_v1"


# ── Drag-Drop: assign-keyframe-image ────────────────────────────────


class TestAssignKeyframeImage:
    def test_assigns_image_and_creates_candidate(self, project_env):
        """assign-keyframe-image should copy image as selected and as a candidate."""
        env = project_env
        pd = env["project_dir"]

        r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        kf_id = r["keyframe"]["id"]

        img_path = _make_fake_image(pd, "pool/stills/test_image.png")

        result = api(env, "POST", f"/api/projects/{env['project_name']}/assign-keyframe-image", {
            "keyframeId": kf_id,
            "sourcePath": img_path,
        })

        assert result["success"] is True
        assert result["selected"] == 1

        # Selected keyframe image should exist
        sel = pd / "selected_keyframes" / f"{kf_id}.png"
        assert sel.exists()

        # Candidate should exist
        cand = pd / "keyframe_candidates" / "candidates" / f"section_{kf_id}" / "v1.png"
        assert cand.exists()

        # DB should reflect selected variant
        from beatlab.db import get_keyframe
        kf = get_keyframe(pd, kf_id)
        assert kf["selected"] == 1

    def test_assigns_multiple_images_increments_variant(self, project_env):
        """Assigning multiple images should create v1, v2, etc."""
        env = project_env
        pd = env["project_dir"]

        r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        kf_id = r["keyframe"]["id"]

        _make_fake_image(pd, "pool/stills/img_a.png")
        _make_fake_image(pd, "pool/stills/img_b.png")

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/assign-keyframe-image", {
            "keyframeId": kf_id, "sourcePath": "pool/stills/img_a.png",
        })
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/assign-keyframe-image", {
            "keyframeId": kf_id, "sourcePath": "pool/stills/img_b.png",
        })

        assert r1["selected"] == 1
        assert r2["selected"] == 2

        cand_dir = pd / "keyframe_candidates" / "candidates" / f"section_{kf_id}"
        assert (cand_dir / "v1.png").exists()
        assert (cand_dir / "v2.png").exists()


# ── Duplicate Keyframe ──────────────────────────────────────────────


class TestDuplicateKeyframe:
    def test_duplicate_creates_new_kf_with_candidates(self, project_env):
        env = project_env
        pd = env["project_dir"]

        r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        kf_id = r["keyframe"]["id"]

        # Seed candidates
        cand_dir = pd / "keyframe_candidates" / "candidates" / f"section_{kf_id}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        (cand_dir / "v1.png").write_bytes(b"img1")
        (cand_dir / "v2.png").write_bytes(b"img2")

        result = api(env, "POST", f"/api/projects/{env['project_name']}/duplicate-keyframe", {
            "keyframeId": kf_id,
            "timestamp": "0:10.00",
        })
        assert result["success"] is True
        new_id = result["keyframe"]["id"]
        assert new_id != kf_id

        # New kf should appear in timeline
        data = get_editor_data(env)
        assert any(k["id"] == new_id for k in data["keyframes"])

        # Candidates should be copied
        new_cand_dir = pd / "keyframe_candidates" / "candidates" / f"section_{new_id}"
        assert (new_cand_dir / "v1.png").exists()
        assert (new_cand_dir / "v2.png").exists()

    def test_duplicate_does_not_cross_tracks(self, project_env):
        """Duplicating a kf should only wire transitions to same-track neighbors."""
        env = project_env

        # Track 1: kf at 0s and 10s
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
            "timestamp": "0:00.00", "trackId": "track_1",
        })
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
            "timestamp": "0:10.00", "trackId": "track_1",
        })

        # Track 2: kf at 5s
        r_t2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
            "timestamp": "0:05.00", "trackId": "track_2",
        })

        # Duplicate the track_2 kf to 7s — should NOT create transitions to track_1 kfs
        result = api(env, "POST", f"/api/projects/{env['project_name']}/duplicate-keyframe", {
            "keyframeId": r_t2["keyframe"]["id"],
            "timestamp": "0:07.00",
        })
        new_id = result["keyframe"]["id"]

        data = get_editor_data(env)
        kf_map = {k["id"]: k for k in data["keyframes"]}
        for tr in data["transitions"]:
            from_track = kf_map[tr["from"]]["trackId"]
            to_track = kf_map[tr["to"]]["trackId"]
            assert from_track == to_track, \
                f"Cross-track transition {tr['id']}: {tr['from']} ({from_track}) -> {tr['to']} ({to_track})"

    def test_duplicate_wires_to_same_track_neighbors(self, project_env):
        """Duplicating a kf between same-track neighbors should produce correct transitions."""
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
            "timestamp": "0:00.00", "trackId": "track_2",
        })
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
            "timestamp": "0:10.00", "trackId": "track_2",
        })

        # Also add a kf on track_1 at 5s to potentially confuse the neighbor search
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
            "timestamp": "0:05.00", "trackId": "track_1",
        })

        # Duplicate r1 to 5s on track_2
        result = api(env, "POST", f"/api/projects/{env['project_name']}/duplicate-keyframe", {
            "keyframeId": r1["keyframe"]["id"],
            "timestamp": "0:05.00",
        })
        new_id = result["keyframe"]["id"]

        # Should have transitions: r1 -> new_id -> r2, all on track_2
        data = get_editor_data(env)
        t2_trs = [t for t in data["transitions"] if t.get("trackId") == "track_2"]
        assert len(t2_trs) == 2

        has_prev = any(t["from"] == r1["keyframe"]["id"] and t["to"] == new_id for t in t2_trs)
        has_next = any(t["from"] == new_id and t["to"] == r2["keyframe"]["id"] for t in t2_trs)
        assert has_prev, "Should have transition from source to duplicate"
        assert has_next, "Should have transition from duplicate to next"


# ── Batch Delete Keyframes ──────────────────────────────────────────


class TestBatchDeleteKeyframes:
    def test_batch_delete_multiple(self, project_env):
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        r3 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})

        api(env, "POST", f"/api/projects/{env['project_name']}/batch-delete-keyframes", {
            "keyframeIds": [r1["keyframe"]["id"], r3["keyframe"]["id"]],
        })

        data = get_editor_data(env)
        ids = [k["id"] for k in data["keyframes"]]
        assert r1["keyframe"]["id"] not in ids
        assert r2["keyframe"]["id"] in ids
        assert r3["keyframe"]["id"] not in ids


# ── Restore Keyframe ────────────────────────────────────────────────


class TestRestoreKeyframe:
    def test_restore_returns_to_timeline(self, project_env):
        env = project_env

        r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        kf_id = r["keyframe"]["id"]

        api(env, "POST", f"/api/projects/{env['project_name']}/delete-keyframe", {"keyframeId": kf_id})

        data = get_editor_data(env)
        assert not any(k["id"] == kf_id for k in data["keyframes"])

        api(env, "POST", f"/api/projects/{env['project_name']}/restore-keyframe", {"keyframeId": kf_id})

        data = get_editor_data(env)
        assert any(k["id"] == kf_id for k in data["keyframes"])


# ── Delete / Restore Transition ─────────────────────────────────────


class TestDeleteRestoreTransition:
    def test_delete_transition(self, project_env):
        env = project_env

        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        trs = active_transitions(env)
        assert len(trs) == 1
        tr_id = trs[0]["id"]

        api(env, "POST", f"/api/projects/{env['project_name']}/delete-transition", {"transitionId": tr_id})

        trs = active_transitions(env)
        assert not any(t["id"] == tr_id for t in trs)

    def test_restore_transition(self, project_env):
        env = project_env

        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        trs = active_transitions(env)
        tr_id = trs[0]["id"]

        api(env, "POST", f"/api/projects/{env['project_name']}/delete-transition", {"transitionId": tr_id})
        api(env, "POST", f"/api/projects/{env['project_name']}/restore-transition", {"transitionId": tr_id})

        trs = active_transitions(env)
        assert any(t["id"] == tr_id for t in trs)


# ── Update Transition Action / Remap ────────────────────────────────


class TestTransitionUpdates:
    def test_update_transition_action(self, project_env):
        env = project_env

        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        trs = active_transitions(env)
        tr_id = trs[0]["id"]

        api(env, "POST", f"/api/projects/{env['project_name']}/update-transition-action", {
            "transitionId": tr_id,
            "action": "Dolly zoom into subject",
        })

        from beatlab.db import get_transition
        tr = get_transition(env["project_dir"], tr_id)
        assert tr["action"] == "Dolly zoom into subject"

    def test_update_transition_remap(self, project_env):
        env = project_env

        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        trs = active_transitions(env)
        tr_id = trs[0]["id"]

        curve = [[0, 0], [0.3, 0.7], [1, 1]]
        api(env, "POST", f"/api/projects/{env['project_name']}/update-transition-remap", {
            "transitionId": tr_id,
            "targetDuration": 8.0,
            "method": "curve",
            "curvePoints": curve,
        })

        from beatlab.db import get_transition
        tr = get_transition(env["project_dir"], tr_id)
        assert tr["remap"]["method"] == "curve"
        assert tr["remap"]["target_duration"] == 8.0
        assert tr["remap"]["curve_points"] == curve


# ── Bin (soft-deleted items) ────────────────────────────────────────


class TestBin:
    def test_bin_lists_deleted_items(self, project_env):
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        kf1 = r1["keyframe"]["id"]

        api(env, "POST", f"/api/projects/{env['project_name']}/delete-keyframe", {"keyframeId": kf1})

        result = api(env, "GET", f"/api/projects/{env['project_name']}/bin")
        binned_ids = [k["id"] for k in result.get("bin", [])]
        assert kf1 in binned_ids


# ── Select Keyframes / Transitions ──────────────────────────────────


class TestSelections:
    def test_select_keyframe_variant(self, project_env):
        env = project_env
        pd = env["project_dir"]

        r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        kf_id = r["keyframe"]["id"]

        # Seed candidates
        cand_dir = pd / "keyframe_candidates" / "candidates" / f"section_{kf_id}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        (cand_dir / "v1.png").write_bytes(b"img1")
        (cand_dir / "v2.png").write_bytes(b"img2")
        # Create selected keyframe dir
        sel_dir = pd / "selected_keyframes"
        sel_dir.mkdir(parents=True, exist_ok=True)

        api(env, "POST", f"/api/projects/{env['project_name']}/select-keyframes", {
            "selections": {kf_id: 2},
        })

        from beatlab.db import get_keyframe
        kf = get_keyframe(pd, kf_id)
        assert kf["selected"] == 2

    def test_select_transition_variant(self, project_env):
        env = project_env
        pd = env["project_dir"]

        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        trs = active_transitions(env)
        tr_id = trs[0]["id"]

        # Seed candidates
        cand_dir = pd / "transition_candidates" / tr_id / "slot_0"
        cand_dir.mkdir(parents=True, exist_ok=True)
        (cand_dir / "v1.mp4").write_bytes(b"vid1")
        (cand_dir / "v2.mp4").write_bytes(b"vid2")
        sel_dir = pd / "selected_transitions"
        sel_dir.mkdir(parents=True, exist_ok=True)

        api(env, "POST", f"/api/projects/{env['project_name']}/select-transitions", {
            "selections": {tr_id: [2]},
        })

        from beatlab.db import get_transition
        tr = get_transition(pd, tr_id)
        assert tr["selected"] == 2


# ── Markers ─────────────────────────────────────────────────────────


class TestMarkers:
    def test_add_and_list_markers(self, project_env):
        env = project_env

        api(env, "POST", f"/api/projects/{env['project_name']}/markers/add", {
            "time": 5.0,
            "label": "Chorus",
        })
        api(env, "POST", f"/api/projects/{env['project_name']}/markers/add", {
            "time": 10.0,
            "label": "Bridge",
        })

        result = api(env, "GET", f"/api/projects/{env['project_name']}/markers")
        markers = result.get("markers", [])
        assert len(markers) == 2
        labels = {m["label"] for m in markers}
        assert "Chorus" in labels
        assert "Bridge" in labels

    def test_update_marker(self, project_env):
        env = project_env

        api(env, "POST", f"/api/projects/{env['project_name']}/markers/add", {
            "time": 5.0,
            "label": "Intro",
        })

        result = api(env, "GET", f"/api/projects/{env['project_name']}/markers")
        marker_id = result["markers"][0]["id"]

        api(env, "POST", f"/api/projects/{env['project_name']}/markers/update", {
            "id": marker_id,
            "time": 7.5,
            "label": "Verse 1",
        })

        result = api(env, "GET", f"/api/projects/{env['project_name']}/markers")
        m = result["markers"][0]
        assert m["label"] == "Verse 1"
        assert m["time"] == 7.5

    def test_remove_marker(self, project_env):
        env = project_env

        api(env, "POST", f"/api/projects/{env['project_name']}/markers/add", {
            "time": 5.0,
            "label": "Drop",
        })

        result = api(env, "GET", f"/api/projects/{env['project_name']}/markers")
        marker_id = result["markers"][0]["id"]

        api(env, "POST", f"/api/projects/{env['project_name']}/markers/remove", {
            "id": marker_id,
        })

        result = api(env, "GET", f"/api/projects/{env['project_name']}/markers")
        assert len(result.get("markers", [])) == 0


# ── Effects ─────────────────────────────────────────────────────────


class TestEffects:
    def test_save_and_load_effects(self, project_env):
        env = project_env

        effects = [
            {"id": "fx_1", "type": "zoom", "time": 5.0, "intensity": 1.5, "duration": 0.3},
            {"id": "fx_2", "type": "shake", "time": 10.0, "intensity": 2.0, "duration": 0.5},
        ]
        api(env, "POST", f"/api/projects/{env['project_name']}/effects", {
            "effects": effects,
            "suppressions": [],
        })

        result = api(env, "GET", f"/api/projects/{env['project_name']}/effects")
        assert len(result.get("effects", [])) == 2


# ── Update Prompt ───────────────────────────────────────────────────


class TestUpdatePrompt:
    def test_update_keyframe_prompt(self, project_env):
        env = project_env

        r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        kf_id = r["keyframe"]["id"]

        api(env, "POST", f"/api/projects/{env['project_name']}/update-prompt", {
            "keyframeId": kf_id,
            "prompt": "Ethereal forest with glowing mushrooms",
        })

        from beatlab.db import get_keyframe
        kf = get_keyframe(env["project_dir"], kf_id)
        assert kf["prompt"] == "Ethereal forest with glowing mushrooms"


# ── Update Meta ─────────────────────────────────────────────────────


class TestUpdateMeta:
    def test_update_motion_prompt(self, project_env):
        env = project_env

        api(env, "POST", f"/api/projects/{env['project_name']}/update-meta", {
            "motion_prompt": "Slow cinematic pans with shallow depth of field",
        })

        from beatlab.db import get_meta
        meta = get_meta(env["project_dir"])
        assert meta["motion_prompt"] == "Slow cinematic pans with shallow depth of field"


# ── Track Reorder ───────────────────────────────────────────────────


class TestTrackReorder:
    def test_reorder_tracks(self, project_env):
        env = project_env

        api(env, "POST", f"/api/projects/{env['project_name']}/tracks/add", {})
        api(env, "POST", f"/api/projects/{env['project_name']}/tracks/add", {})

        from beatlab.db import get_tracks
        tracks = get_tracks(env["project_dir"])
        ids = [t["id"] for t in sorted(tracks, key=lambda t: t["z_order"])]

        # Reverse the order
        api(env, "POST", f"/api/projects/{env['project_name']}/tracks/reorder", {
            "trackIds": list(reversed(ids)),
        })

        tracks_after = get_tracks(env["project_dir"])
        ids_after = [t["id"] for t in sorted(tracks_after, key=lambda t: t["z_order"])]
        assert ids_after == list(reversed(ids))


# ── Track Delete ────────────────────────────────────────────────────


class TestTrackDelete:
    def test_delete_track(self, project_env):
        env = project_env

        r = api(env, "POST", f"/api/projects/{env['project_name']}/tracks/add", {})
        track_id = r["id"]

        from beatlab.db import get_tracks
        assert any(t["id"] == track_id for t in get_tracks(env["project_dir"]))

        api(env, "POST", f"/api/projects/{env['project_name']}/tracks/delete", {
            "id": track_id,
        })

        assert not any(t["id"] == track_id for t in get_tracks(env["project_dir"]))


# ══════════════════════════════════════════════════════════════════════
# Extended Keyframe Tests
# ══════════════════════════════════════════════════════════════════════


class TestAddKeyframeExtended:
    def test_first_keyframe_no_transitions(self, project_env):
        """A single keyframe on a track should have zero transitions."""
        env = project_env
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        data = get_editor_data(env)
        assert len(data["keyframes"]) == 1
        assert len(data["transitions"]) == 0

    def test_two_keyframes_one_transition(self, project_env):
        """Two keyframes should produce exactly one transition."""
        env = project_env
        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})

        assert_timeline_integrity(env)

        trs = active_transitions(env)
        assert len(trs) == 1
        assert trs[0]["from"] == r1["keyframe"]["id"]
        assert trs[0]["to"] == r2["keyframe"]["id"]
        assert trs[0]["durationSeconds"] == pytest.approx(10.0, abs=0.05)

    def test_add_five_keyframes_produces_chain(self, project_env):
        """5 keyframes added in random order should form a clean 4-transition chain."""
        env = project_env
        for ts in ["0:20.00", "0:00.00", "0:15.00", "0:05.00", "0:10.00"]:
            api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": ts})

        sorted_kfs, trs = assert_timeline_integrity(env)
        assert len(sorted_kfs) == 5
        assert len(trs) == 4

    def test_insert_at_beginning(self, project_env):
        """Inserting before all existing kfs creates correct chain."""
        env = project_env
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:20.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:01.00"})

        assert_timeline_integrity(env)

    def test_insert_at_end(self, project_env):
        """Inserting after all existing kfs creates correct chain."""
        env = project_env
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:30.00"})

        assert_timeline_integrity(env)

    def test_relink_preserves_transition_metadata(self, project_env):
        """Inserting into a transition should preserve the original tr's action and label."""
        env = project_env
        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})

        trs = active_transitions(env)
        tr_id = trs[0]["id"]

        # Set metadata on the transition
        api(env, "POST", f"/api/projects/{env['project_name']}/update-transition-action", {
            "transitionId": tr_id, "action": "Dolly zoom",
        })
        api(env, "POST", f"/api/projects/{env['project_name']}/update-transition-label", {
            "transitionId": tr_id, "label": "Hero Shot", "labelColor": "#ff0000",
        })

        # Insert kf in the middle
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        # The original transition should still have its metadata
        trs_after = active_transitions(env)
        relinked = next(t for t in trs_after if t["id"] == tr_id)
        assert relinked["action"] == "Dolly zoom"
        assert relinked["label"] == "Hero Shot"
        assert relinked["labelColor"] == "#ff0000"

    def test_transition_durations_correct_after_insert(self, project_env):
        """After inserting at 5s between 0s and 10s, durations should be 5s and 5s."""
        env = project_env
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        sorted_kfs, trs = assert_timeline_integrity(env)
        durations = sorted([t["durationSeconds"] for t in trs])
        assert durations == [pytest.approx(5.0, abs=0.05), pytest.approx(5.0, abs=0.05)]

    def test_add_with_prompt_and_section(self, project_env):
        env = project_env
        r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
            "timestamp": "0:05.00",
            "prompt": "Neon city at night",
            "section": "Verse 1",
        })

        data = get_editor_data(env)
        kf = next(k for k in data["keyframes"] if k["id"] == r["keyframe"]["id"])
        assert kf["prompt"] == "Neon city at night"
        assert kf["section"] == "Verse 1"


# ══════════════════════════════════════════════════════════════════════
# Extended Delete Keyframe Tests
# ══════════════════════════════════════════════════════════════════════


class TestDeleteKeyframeExtended:
    def test_delete_first_keyframe(self, project_env):
        """Deleting the first kf should leave remaining chain intact."""
        import time
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        r3 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})

        api(env, "POST", f"/api/projects/{env['project_name']}/delete-keyframe", {
            "keyframeId": r1["keyframe"]["id"],
        })
        time.sleep(0.5)  # bridge thread

        data = get_editor_data(env)
        kf_ids = [k["id"] for k in data["keyframes"]]
        assert r1["keyframe"]["id"] not in kf_ids
        assert r2["keyframe"]["id"] in kf_ids
        assert r3["keyframe"]["id"] in kf_ids

        # Remaining 2 kfs should have one transition
        trs = data["transitions"]
        assert len(trs) == 1
        assert trs[0]["from"] == r2["keyframe"]["id"]
        assert trs[0]["to"] == r3["keyframe"]["id"]

    def test_delete_last_keyframe(self, project_env):
        """Deleting the last kf should leave remaining chain intact."""
        import time
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        r3 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})

        api(env, "POST", f"/api/projects/{env['project_name']}/delete-keyframe", {
            "keyframeId": r3["keyframe"]["id"],
        })
        time.sleep(0.5)

        data = get_editor_data(env)
        trs = data["transitions"]
        assert len(trs) == 1
        assert trs[0]["from"] == r1["keyframe"]["id"]
        assert trs[0]["to"] == r2["keyframe"]["id"]

    def test_delete_only_keyframe(self, project_env):
        """Deleting the sole kf should leave empty timeline."""
        env = project_env

        r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/delete-keyframe", {
            "keyframeId": r["keyframe"]["id"],
        })

        data = get_editor_data(env)
        assert len(data["keyframes"]) == 0
        assert len(data["transitions"]) == 0

    def test_delete_and_restore_preserves_chain(self, project_env):
        """Delete middle kf, then restore it — transitions should reconnect."""
        import time
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        r3 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})

        assert_timeline_integrity(env)

        api(env, "POST", f"/api/projects/{env['project_name']}/delete-keyframe", {
            "keyframeId": r2["keyframe"]["id"],
        })
        time.sleep(0.5)  # wait for bridge

        # After delete: kf1 -> kf3 bridge
        data = get_editor_data(env)
        assert len(data["keyframes"]) == 2

        api(env, "POST", f"/api/projects/{env['project_name']}/restore-keyframe", {
            "keyframeId": r2["keyframe"]["id"],
        })

        # After restore: kf1 -> kf2 -> kf3 should have 2 transitions
        data = get_editor_data(env)
        assert len(data["keyframes"]) == 3


# ══════════════════════════════════════════════════════════════════════
# Extended Transition Tests
# ══════════════════════════════════════════════════════════════════════


class TestTransitionExtended:
    def test_transition_track_matches_keyframes(self, project_env):
        """Transition trackId should match the track of its from/to keyframes."""
        env = project_env
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
            "timestamp": "0:00.00", "trackId": "track_2",
        })
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
            "timestamp": "0:05.00", "trackId": "track_2",
        })

        data = get_editor_data(env)
        trs = [t for t in data["transitions"] if t.get("trackId") == "track_2"]
        assert len(trs) == 1

    def test_remap_curve_round_trip(self, project_env):
        """Setting a remap curve should persist and round-trip through the API."""
        env = project_env
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})

        trs = active_transitions(env)
        tr_id = trs[0]["id"]

        curve = [[0, 0], [0.2, 0.5], [0.8, 0.6], [1, 1]]
        api(env, "POST", f"/api/projects/{env['project_name']}/update-transition-remap", {
            "transitionId": tr_id, "targetDuration": 10.0, "method": "curve", "curvePoints": curve,
        })

        data = get_editor_data(env)
        tr = next(t for t in data["transitions"] if t["id"] == tr_id)
        assert tr["remap"]["method"] == "curve"
        # API returns snake_case for remap internals
        assert tr["remap"].get("curvePoints") == curve or tr["remap"].get("curve_points") == curve

    def test_opacity_curve_round_trip(self, project_env):
        """Setting an opacity curve should persist and round-trip through the API."""
        env = project_env
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})

        trs = active_transitions(env)
        tr_id = trs[0]["id"]

        curve = [[0, 0], [0.3, 1], [0.7, 1], [1, 0]]
        api(env, "POST", f"/api/projects/{env['project_name']}/update-transition-style", {
            "transitionId": tr_id, "opacityCurve": curve,
        })

        data = get_editor_data(env)
        tr = next(t for t in data["transitions"] if t["id"] == tr_id)
        assert tr["opacityCurve"] == curve

    def test_clear_opacity_curve(self, project_env):
        """Setting opacity curve to null should clear it."""
        env = project_env
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})

        trs = active_transitions(env)
        tr_id = trs[0]["id"]

        api(env, "POST", f"/api/projects/{env['project_name']}/update-transition-style", {
            "transitionId": tr_id, "opacityCurve": [[0, 0], [1, 1]],
        })
        api(env, "POST", f"/api/projects/{env['project_name']}/update-transition-style", {
            "transitionId": tr_id, "opacityCurve": None,
        })

        data = get_editor_data(env)
        tr = next(t for t in data["transitions"] if t["id"] == tr_id)
        assert tr["opacityCurve"] is None

    def test_blend_mode_round_trip_kf_and_tr(self, project_env):
        """Blend modes set on kf and tr should round-trip through the API."""
        env = project_env
        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})
        kf_id = r1["keyframe"]["id"]
        trs = active_transitions(env)
        tr_id = trs[0]["id"]

        api(env, "POST", f"/api/projects/{env['project_name']}/update-keyframe-style", {
            "keyframeId": kf_id, "blendMode": "multiply", "opacity": 0.5,
        })
        api(env, "POST", f"/api/projects/{env['project_name']}/update-transition-style", {
            "transitionId": tr_id, "blendMode": "screen",
        })

        data = get_editor_data(env)
        kf = next(k for k in data["keyframes"] if k["id"] == kf_id)
        tr = next(t for t in data["transitions"] if t["id"] == tr_id)
        assert kf["blendMode"] == "multiply"
        assert kf["opacity"] == pytest.approx(0.5)
        assert tr["blendMode"] == "screen"


# ══════════════════════════════════════════════════════════════════════
# Timeline Integrity Tests (complex multi-operation scenarios)
# ══════════════════════════════════════════════════════════════════════


class TestTimelineIntegrity:
    def test_build_10_keyframe_chain(self, project_env):
        """Build a 10-kf chain and validate full integrity."""
        env = project_env
        for i in range(10):
            api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
                "timestamp": f"0:{i * 5:05.2f}",
            })

        sorted_kfs, trs = assert_timeline_integrity(env)
        assert len(sorted_kfs) == 10
        assert len(trs) == 9

    def test_insert_between_every_pair(self, project_env):
        """Create 3 kfs, then insert between each pair. Chain stays valid."""
        env = project_env
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:20.00"})
        assert_timeline_integrity(env)

        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        assert_timeline_integrity(env)

        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:15.00"})
        sorted_kfs, trs = assert_timeline_integrity(env)
        assert len(sorted_kfs) == 5
        assert len(trs) == 4

    def test_delete_every_other_keyframe(self, project_env):
        """Create 5 kfs, delete #2 and #4, verify chain integrity."""
        import time
        env = project_env

        ids = []
        for i in range(5):
            r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
                "timestamp": f"0:{i * 5:05.2f}",
            })
            ids.append(r["keyframe"]["id"])

        assert_timeline_integrity(env)

        # Delete kf at index 1 and 3
        api(env, "POST", f"/api/projects/{env['project_name']}/delete-keyframe", {"keyframeId": ids[1]})
        time.sleep(0.5)
        api(env, "POST", f"/api/projects/{env['project_name']}/delete-keyframe", {"keyframeId": ids[3]})
        time.sleep(0.5)

        sorted_kfs, trs = assert_timeline_integrity(env)
        assert len(sorted_kfs) == 3
        assert len(trs) == 2
        remaining_ids = [k["id"] for k in sorted_kfs]
        assert remaining_ids == [ids[0], ids[2], ids[4]]

    def test_multi_track_isolation(self, project_env):
        """Two tracks built independently should not interfere."""
        env = project_env

        # Track 1: 3 keyframes
        for ts in ["0:00.00", "0:05.00", "0:10.00"]:
            api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
                "timestamp": ts, "trackId": "track_1",
            })
        # Track 2: 2 keyframes at overlapping times
        for ts in ["0:02.00", "0:08.00"]:
            api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
                "timestamp": ts, "trackId": "track_2",
            })

        assert_timeline_integrity(env, track_id="track_1")
        assert_timeline_integrity(env, track_id="track_2")

        data = get_editor_data(env)
        t1_trs = [t for t in data["transitions"] if t.get("trackId") == "track_1"]
        t2_trs = [t for t in data["transitions"] if t.get("trackId") == "track_2"]
        assert len(t1_trs) == 2
        assert len(t2_trs) == 1

    def test_insert_delete_insert_cycle(self, project_env):
        """Insert kf, delete it, insert again at same spot — chain stays valid."""
        import time
        env = project_env

        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})
        assert_timeline_integrity(env)

        r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        assert_timeline_integrity(env)

        api(env, "POST", f"/api/projects/{env['project_name']}/delete-keyframe", {"keyframeId": r["keyframe"]["id"]})
        time.sleep(0.5)

        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        assert_timeline_integrity(env)

    def test_timestamp_update_maintains_chain(self, project_env):
        """Moving a kf via update-timestamp should update adjacent transition durations."""
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        r3 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})
        assert_timeline_integrity(env)

        # Move middle kf from 5s to 7s
        api(env, "POST", f"/api/projects/{env['project_name']}/update-timestamp", {
            "keyframeId": r2["keyframe"]["id"],
            "newTimestamp": "0:07.00",
        })

        # Durations should now be 7s and 3s
        data = get_editor_data(env)
        trs = sorted(data["transitions"], key=lambda t: t["durationSeconds"])
        assert trs[0]["durationSeconds"] == pytest.approx(3.0, abs=0.05)
        assert trs[1]["durationSeconds"] == pytest.approx(7.0, abs=0.05)

    def test_batch_delete_preserves_integrity(self, project_env):
        """Batch deleting non-adjacent kfs should leave a valid chain."""
        import time
        env = project_env

        ids = []
        for i in range(6):
            r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
                "timestamp": f"0:{i * 3:05.2f}",
            })
            ids.append(r["keyframe"]["id"])

        assert_timeline_integrity(env)

        # Batch delete indices 1, 3, 4
        api(env, "POST", f"/api/projects/{env['project_name']}/batch-delete-keyframes", {
            "keyframeIds": [ids[1], ids[3], ids[4]],
        })
        time.sleep(1.0)  # bridges run in background

        sorted_kfs, trs = assert_timeline_integrity(env)
        assert len(sorted_kfs) == 3
        remaining_ids = [k["id"] for k in sorted_kfs]
        assert remaining_ids == [ids[0], ids[2], ids[5]]

    def test_rapid_inserts_no_duplicate_transitions(self, project_env):
        """Rapidly adding keyframes should not create duplicate transitions."""
        env = project_env

        for i in range(8):
            api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
                "timestamp": f"0:{i * 2:05.2f}",
            })

        data = get_editor_data(env)
        trs = data["transitions"]

        # Check no duplicates
        pairs = [(t["from"], t["to"]) for t in trs]
        assert len(pairs) == len(set(pairs)), f"Duplicate transitions found: {pairs}"

        assert_timeline_integrity(env)


# ══════════════════════════════════════════════════════════════════════
# Unlink Keyframe Tests
# ══════════════════════════════════════════════════════════════════════


class TestUnlinkKeyframe:
    def test_unlink_both_sides(self, project_env):
        """Unlinking both sides removes all transitions touching the kf."""
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        r3 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})

        trs_before = active_transitions(env)
        assert len(trs_before) == 2

        result = api(env, "POST", f"/api/projects/{env['project_name']}/unlink-keyframe", {
            "keyframeId": r2["keyframe"]["id"],
        })
        assert result["success"] is True
        assert len(result["deleted"]) == 2

        # kf2 should still exist, but with no transitions
        data = get_editor_data(env)
        assert len(data["keyframes"]) == 3
        assert len(data["transitions"]) == 0

    def test_unlink_left_only(self, project_env):
        """Unlinking left removes only the incoming transition."""
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        r3 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})

        result = api(env, "POST", f"/api/projects/{env['project_name']}/unlink-keyframe", {
            "keyframeId": r2["keyframe"]["id"],
            "side": "left",
        })
        assert len(result["deleted"]) == 1

        trs = active_transitions(env)
        assert len(trs) == 1
        # Only outgoing transition should remain: kf2 -> kf3
        assert trs[0]["from"] == r2["keyframe"]["id"]
        assert trs[0]["to"] == r3["keyframe"]["id"]

    def test_unlink_right_only(self, project_env):
        """Unlinking right removes only the outgoing transition."""
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        r3 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})

        result = api(env, "POST", f"/api/projects/{env['project_name']}/unlink-keyframe", {
            "keyframeId": r2["keyframe"]["id"],
            "side": "right",
        })
        assert len(result["deleted"]) == 1

        trs = active_transitions(env)
        assert len(trs) == 1
        # Only incoming transition should remain: kf1 -> kf2
        assert trs[0]["from"] == r1["keyframe"]["id"]
        assert trs[0]["to"] == r2["keyframe"]["id"]

    def test_unlink_first_keyframe(self, project_env):
        """Unlinking the first kf should remove its one outgoing transition."""
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        result = api(env, "POST", f"/api/projects/{env['project_name']}/unlink-keyframe", {
            "keyframeId": r1["keyframe"]["id"],
        })
        assert len(result["deleted"]) == 1
        assert len(active_transitions(env)) == 0

    def test_unlink_already_unlinked_is_noop(self, project_env):
        """Unlinking a kf with no transitions should succeed with 0 deleted."""
        env = project_env

        r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        result = api(env, "POST", f"/api/projects/{env['project_name']}/unlink-keyframe", {
            "keyframeId": r["keyframe"]["id"],
        })
        assert result["success"] is True
        assert len(result["deleted"]) == 0

    def test_unlink_preserves_keyframe(self, project_env):
        """Unlinking should only remove transitions, not the keyframe itself."""
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        api(env, "POST", f"/api/projects/{env['project_name']}/unlink-keyframe", {
            "keyframeId": r1["keyframe"]["id"],
        })

        data = get_editor_data(env)
        assert len(data["keyframes"]) == 2

    def test_insert_between_unlinked_does_not_span(self, project_env):
        """Adding a kf between two unlinked kfs creates transitions to both neighbors."""
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:10.00"})

        # Unlink — removes the kf1->kf2 transition
        api(env, "POST", f"/api/projects/{env['project_name']}/unlink-keyframe", {
            "keyframeId": r1["keyframe"]["id"],
        })
        assert len(active_transitions(env)) == 0

        # Insert between — no spanning tr exists, so new trs to both neighbors
        r3 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        kf3 = r3["keyframe"]["id"]

        trs = active_transitions(env)
        has_from_kf1 = any(t["from"] == r1["keyframe"]["id"] and t["to"] == kf3 for t in trs)
        has_to_kf2 = any(t["from"] == kf3 and t["to"] == r2["keyframe"]["id"] for t in trs)
        assert has_from_kf1, "Should create kf1 -> kf_new"
        assert has_to_kf2, "Should create kf_new -> kf2"


# ══════════════════════════════════════════════════════════════════════
# Paste Group Tests
# ══════════════════════════════════════════════════════════════════════


class TestPasteGroup:
    def test_paste_three_kfs_with_transitions(self, project_env):
        """Paste 3 kfs with 2 connecting transitions at a new time."""
        env = project_env
        pd = env["project_dir"]

        # Create source group: 3 kfs at 0s, 5s, 10s
        ids = []
        for ts in ["0:00.00", "0:05.00", "0:10.00"]:
            r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": ts})
            ids.append(r["keyframe"]["id"])

        # Paste at 30s
        result = api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
            "keyframeIds": ids,
            "targetTime": "0:30.00",
            "targetTrackId": "track_1",
        })

        assert result["success"] is True
        assert len(result["keyframes"]) == 3
        assert len(result["transitions"]) == 2

        # Verify new kfs are at 30s, 35s, 40s
        data = get_editor_data(env)
        new_ids = {k["id"] for k in result["keyframes"]}
        new_kfs = sorted(
            [k for k in data["keyframes"] if k["id"] in new_ids],
            key=lambda k: parse_ts(k["timestamp"]),
        )
        assert len(new_kfs) == 3
        assert parse_ts(new_kfs[0]["timestamp"]) == pytest.approx(30.0, abs=0.05)
        assert parse_ts(new_kfs[1]["timestamp"]) == pytest.approx(35.0, abs=0.05)
        assert parse_ts(new_kfs[2]["timestamp"]) == pytest.approx(40.0, abs=0.05)

    def test_paste_to_different_track(self, project_env):
        """Pasted kfs and trs should be on the target track."""
        env = project_env

        ids = []
        for ts in ["0:00.00", "0:05.00"]:
            r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
                "timestamp": ts, "trackId": "track_1",
            })
            ids.append(r["keyframe"]["id"])

        result = api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
            "keyframeIds": ids,
            "targetTime": "0:20.00",
            "targetTrackId": "track_2",
        })

        data = get_editor_data(env)
        new_ids = {k["id"] for k in result["keyframes"]}
        new_tr_ids = {t["id"] for t in result["transitions"]}

        for kf in data["keyframes"]:
            if kf["id"] in new_ids:
                assert kf["trackId"] == "track_2"
        for tr in data["transitions"]:
            if tr["id"] in new_tr_ids:
                assert tr["trackId"] == "track_2"

    def test_paste_copies_kf_candidates(self, project_env):
        """Pasted kfs should have copies of source candidate files."""
        env = project_env
        pd = env["project_dir"]

        r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        kf_id = r["keyframe"]["id"]

        # Seed candidates on source
        cand_dir = pd / "keyframe_candidates" / "candidates" / f"section_{kf_id}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        (cand_dir / "v1.png").write_bytes(b"image_data")

        result = api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
            "keyframeIds": [kf_id],
            "targetTime": "0:20.00",
            "targetTrackId": "track_1",
        })

        new_id = result["keyframes"][0]["id"]
        new_cand = pd / "keyframe_candidates" / "candidates" / f"section_{new_id}" / "v1.png"
        assert new_cand.exists()
        assert new_cand.read_bytes() == b"image_data"

    def test_paste_copies_tr_videos(self, project_env):
        """Pasted transitions should have copies of source video candidates."""
        env = project_env
        pd = env["project_dir"]

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        trs = active_transitions(env)
        tr_id = trs[0]["id"]

        # Seed video on source transition
        cand_dir = pd / "transition_candidates" / tr_id / "slot_0"
        cand_dir.mkdir(parents=True, exist_ok=True)
        (cand_dir / "v1.mp4").write_bytes(b"video_data")

        sel_dir = pd / "selected_transitions"
        sel_dir.mkdir(parents=True, exist_ok=True)
        (sel_dir / f"{tr_id}_slot_0.mp4").write_bytes(b"selected_video")

        from beatlab.db import update_transition
        update_transition(pd, tr_id, selected=1, action="Slow zoom")

        result = api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
            "keyframeIds": [r1["keyframe"]["id"], r2["keyframe"]["id"]],
            "targetTime": "0:20.00",
            "targetTrackId": "track_1",
        })

        assert len(result["transitions"]) == 1
        new_tr_id = result["transitions"][0]["id"]

        # Video candidates copied
        new_cand = pd / "transition_candidates" / new_tr_id / "slot_0" / "v1.mp4"
        assert new_cand.exists()
        assert new_cand.read_bytes() == b"video_data"

        # Selected video copied
        new_sel = pd / "selected_transitions" / f"{new_tr_id}_slot_0.mp4"
        assert new_sel.exists()

        # Action prompt copied
        from beatlab.db import get_transition
        tr = get_transition(pd, new_tr_id)
        assert tr["action"] == "Slow zoom"

    def test_paste_group_is_self_contained(self, project_env):
        """Pasted group should not create transitions to existing kfs outside the group."""
        env = project_env

        # Existing kf at 25s
        api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:25.00"})

        # Create source: 2 kfs
        ids = []
        for ts in ["0:00.00", "0:05.00"]:
            r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": ts})
            ids.append(r["keyframe"]["id"])

        result = api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
            "keyframeIds": ids,
            "targetTime": "0:20.00",
            "targetTrackId": "track_1",
        })

        new_ids = {k["id"] for k in result["keyframes"]}
        new_tr_ids = {t["id"] for t in result["transitions"]}

        # Only transitions within the pasted group should exist
        data = get_editor_data(env)
        for tr in data["transitions"]:
            if tr["id"] in new_tr_ids:
                assert tr["from"] in new_ids and tr["to"] in new_ids, \
                    f"Pasted transition {tr['id']} links outside group: {tr['from']} -> {tr['to']}"

    def test_paste_empty_is_error(self, project_env):
        """Pasting with no keyframe IDs should return 400."""
        env = project_env
        try:
            api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
                "keyframeIds": [],
                "targetTime": "0:10.00",
                "targetTrackId": "track_1",
            })
            assert False, "Should have raised"
        except AssertionError as e:
            assert "400" in str(e)

    def test_paste_from_track1_to_track2(self, project_env):
        """Copy kfs from track_1, paste to track_2 — all new items should be on track_2."""
        env = project_env

        # Build chain on track_1
        ids = []
        for ts in ["0:00.00", "0:05.00", "0:10.00"]:
            r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
                "timestamp": ts, "trackId": "track_1",
            })
            ids.append(r["keyframe"]["id"])

        # Verify source is on track_1
        data = get_editor_data(env)
        for kf in data["keyframes"]:
            if kf["id"] in ids:
                assert kf["trackId"] == "track_1"

        # Paste to track_2 at 30s
        result = api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
            "keyframeIds": ids,
            "targetTime": "0:30.00",
            "targetTrackId": "track_2",
        })

        assert len(result["keyframes"]) == 3
        assert len(result["transitions"]) == 2

        # All new kfs on track_2
        data = get_editor_data(env)
        new_kf_ids = {k["id"] for k in result["keyframes"]}
        new_tr_ids = {t["id"] for t in result["transitions"]}
        for kf in data["keyframes"]:
            if kf["id"] in new_kf_ids:
                assert kf["trackId"] == "track_2", f"{kf['id']} on {kf['trackId']}, expected track_2"
        for tr in data["transitions"]:
            if tr["id"] in new_tr_ids:
                assert tr["trackId"] == "track_2", f"{tr['id']} on {tr['trackId']}, expected track_2"

        # Source kfs still on track_1 (not mutated)
        for kf in data["keyframes"]:
            if kf["id"] in ids:
                assert kf["trackId"] == "track_1"

    def test_paste_preserves_relative_timing(self, project_env):
        """Pasted kfs should maintain their relative spacing."""
        env = project_env

        # Source: 2s, 7s, 12s (gaps of 5s each)
        ids = []
        for ts in ["0:02.00", "0:07.00", "0:12.00"]:
            r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": ts})
            ids.append(r["keyframe"]["id"])

        # Paste at 1:00 (60s)
        result = api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
            "keyframeIds": ids,
            "targetTime": "1:00.00",
            "targetTrackId": "track_1",
        })

        data = get_editor_data(env)
        new_ids = {k["id"] for k in result["keyframes"]}
        new_kfs = sorted(
            [k for k in data["keyframes"] if k["id"] in new_ids],
            key=lambda k: parse_ts(k["timestamp"]),
        )

        # Should be at 60s, 65s, 70s
        times = [parse_ts(k["timestamp"]) for k in new_kfs]
        assert times[0] == pytest.approx(60.0, abs=0.05)
        assert times[1] == pytest.approx(65.0, abs=0.05)
        assert times[2] == pytest.approx(70.0, abs=0.05)

        # Transition durations should be 5s each
        new_tr_ids = {t["id"] for t in result["transitions"]}
        new_trs = [t for t in data["transitions"] if t["id"] in new_tr_ids]
        for tr in new_trs:
            assert tr["durationSeconds"] == pytest.approx(5.0, abs=0.05)

    def test_paste_single_kf_no_transitions(self, project_env):
        """Pasting a single kf should create 1 kf and 0 transitions."""
        env = project_env

        r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        result = api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
            "keyframeIds": [r["keyframe"]["id"]],
            "targetTime": "0:30.00",
            "targetTrackId": "track_1",
        })

        assert len(result["keyframes"]) == 1
        assert len(result["transitions"]) == 0

    def test_paste_does_not_mutate_source(self, project_env):
        """Source kfs/trs should be completely unchanged after paste."""
        env = project_env
        pd = env["project_dir"]

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
            "timestamp": "0:00.00", "prompt": "Original prompt",
        })
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        kf1 = r1["keyframe"]["id"]

        # Snapshot source state
        data_before = get_editor_data(env)
        src_kf = next(k for k in data_before["keyframes"] if k["id"] == kf1)
        src_trs = [t for t in data_before["transitions"]]

        # Paste
        api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
            "keyframeIds": [r1["keyframe"]["id"], r2["keyframe"]["id"]],
            "targetTime": "0:30.00",
            "targetTrackId": "track_2",
        })

        # Source should be identical
        data_after = get_editor_data(env)
        src_kf_after = next(k for k in data_after["keyframes"] if k["id"] == kf1)
        assert src_kf_after["prompt"] == "Original prompt"
        assert src_kf_after["trackId"] == "track_1"
        assert src_kf_after["timestamp"] == src_kf["timestamp"]

    def test_paste_subset_of_chain(self, project_env):
        """Pasting 2 of 4 kfs should only copy the transition between those 2."""
        env = project_env

        ids = []
        for ts in ["0:00.00", "0:05.00", "0:10.00", "0:15.00"]:
            r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": ts})
            ids.append(r["keyframe"]["id"])

        # Paste only kf[1] and kf[2] (5s and 10s)
        result = api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
            "keyframeIds": [ids[1], ids[2]],
            "targetTime": "0:30.00",
            "targetTrackId": "track_1",
        })

        assert len(result["keyframes"]) == 2
        assert len(result["transitions"]) == 1  # only the tr between kf[1]->kf[2]

    def test_paste_non_adjacent_kfs(self, project_env):
        """Pasting kfs that aren't adjacent should not copy the transitions between them."""
        env = project_env

        ids = []
        for ts in ["0:00.00", "0:05.00", "0:10.00", "0:15.00"]:
            r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": ts})
            ids.append(r["keyframe"]["id"])

        # Paste kf[0] and kf[3] (0s and 15s) — they're not adjacent, no direct tr
        result = api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
            "keyframeIds": [ids[0], ids[3]],
            "targetTime": "0:30.00",
            "targetTrackId": "track_1",
        })

        assert len(result["keyframes"]) == 2
        assert len(result["transitions"]) == 0  # no tr between kf[0] and kf[3]

    def test_paste_copies_blend_mode_and_opacity(self, project_env):
        """Pasted kfs should preserve blend mode and opacity from source."""
        env = project_env

        r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})
        kf_id = r["keyframe"]["id"]

        api(env, "POST", f"/api/projects/{env['project_name']}/update-keyframe-style", {
            "keyframeId": kf_id, "blendMode": "screen", "opacity": 0.7,
        })

        result = api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
            "keyframeIds": [kf_id],
            "targetTime": "0:30.00",
            "targetTrackId": "track_1",
        })

        data = get_editor_data(env)
        new_id = result["keyframes"][0]["id"]
        new_kf = next(k for k in data["keyframes"] if k["id"] == new_id)
        assert new_kf["blendMode"] == "screen"
        assert new_kf["opacity"] == pytest.approx(0.7)

    def test_paste_copies_tr_remap_and_opacity_curve(self, project_env):
        """Pasted transitions should preserve remap curves and opacity curves."""
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        trs = active_transitions(env)
        tr_id = trs[0]["id"]

        remap_curve = [[0, 0], [0.3, 0.7], [1, 1]]
        opacity_curve = [[0, 0], [0.5, 1], [1, 0]]
        api(env, "POST", f"/api/projects/{env['project_name']}/update-transition-remap", {
            "transitionId": tr_id, "targetDuration": 5.0, "method": "curve", "curvePoints": remap_curve,
        })
        api(env, "POST", f"/api/projects/{env['project_name']}/update-transition-style", {
            "transitionId": tr_id, "blendMode": "multiply", "opacityCurve": opacity_curve,
        })

        result = api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
            "keyframeIds": [r1["keyframe"]["id"], r2["keyframe"]["id"]],
            "targetTime": "0:30.00",
            "targetTrackId": "track_1",
        })

        data = get_editor_data(env)
        new_tr_id = result["transitions"][0]["id"]
        new_tr = next(t for t in data["transitions"] if t["id"] == new_tr_id)
        assert new_tr["blendMode"] == "multiply"
        assert new_tr["opacityCurve"] == opacity_curve
        assert new_tr["remap"]["method"] == "curve"

    def test_paste_with_deleted_source_kf(self, project_env):
        """Pasting with a deleted kf ID should skip it gracefully."""
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        api(env, "POST", f"/api/projects/{env['project_name']}/delete-keyframe", {
            "keyframeId": r1["keyframe"]["id"],
        })

        result = api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
            "keyframeIds": [r1["keyframe"]["id"], r2["keyframe"]["id"]],
            "targetTime": "0:30.00",
            "targetTrackId": "track_1",
        })

        # Should only paste the non-deleted kf
        assert len(result["keyframes"]) == 1

    def test_paste_twice_creates_independent_copies(self, project_env):
        """Pasting the same group twice should create two independent sets."""
        env = project_env

        ids = []
        for ts in ["0:00.00", "0:05.00"]:
            r = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": ts})
            ids.append(r["keyframe"]["id"])

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
            "keyframeIds": ids, "targetTime": "0:20.00", "targetTrackId": "track_1",
        })
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
            "keyframeIds": ids, "targetTime": "0:40.00", "targetTrackId": "track_1",
        })

        # All IDs should be unique
        all_ids = [k["id"] for k in r1["keyframes"]] + [k["id"] for k in r2["keyframes"]]
        assert len(all_ids) == len(set(all_ids)), "Pasted IDs should be unique"

        # Both pastes should have their own transitions
        assert len(r1["transitions"]) == 1
        assert len(r2["transitions"]) == 1
        assert r1["transitions"][0]["id"] != r2["transitions"][0]["id"]

    def test_paste_cross_track_kfs_all_go_to_target(self, project_env):
        """Pasting kfs from mixed tracks should place all on the target track."""
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
            "timestamp": "0:00.00", "trackId": "track_1",
        })
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {
            "timestamp": "0:05.00", "trackId": "track_2",
        })

        result = api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
            "keyframeIds": [r1["keyframe"]["id"], r2["keyframe"]["id"]],
            "targetTime": "0:30.00",
            "targetTrackId": "track_3",
        })

        data = get_editor_data(env)
        new_ids = {k["id"] for k in result["keyframes"]}
        for kf in data["keyframes"]:
            if kf["id"] in new_ids:
                assert kf["trackId"] == "track_3"

    def test_paste_copies_all_curves(self, project_env):
        """Pasted transitions should copy all curve types."""
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        trs = active_transitions(env)
        tr_id = trs[0]["id"]

        curves = {
            "opacityCurve": [[0, 0], [0.5, 1], [1, 0.5]],
            "redCurve": [[0, 1], [1, 0.5]],
            "greenCurve": [[0, 0.8], [1, 1]],
            "blueCurve": [[0, 1], [0.5, 0], [1, 1]],
            "blackCurve": [[0, 0], [1, 0.3]],
            "hueShiftCurve": [[0, 0], [1, 0.5]],
            "saturationCurve": [[0, 1], [1, 0.2]],
        }
        api(env, "POST", f"/api/projects/{env['project_name']}/update-transition-style", {
            "transitionId": tr_id, **curves,
        })

        result = api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
            "keyframeIds": [r1["keyframe"]["id"], r2["keyframe"]["id"]],
            "targetTime": "0:30.00",
            "targetTrackId": "track_1",
        })

        data = get_editor_data(env)
        new_tr = next(t for t in data["transitions"] if t["id"] == result["transitions"][0]["id"])
        for key, expected in curves.items():
            assert new_tr[key] == expected, f"{key}: {new_tr.get(key)} != {expected}"

    def test_paste_copies_adjustment_flag(self, project_env):
        """Pasted adjustment transitions should preserve the isAdjustment flag."""
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        trs = active_transitions(env)
        tr_id = trs[0]["id"]

        api(env, "POST", f"/api/projects/{env['project_name']}/update-transition-style", {
            "transitionId": tr_id, "isAdjustment": True,
        })

        result = api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
            "keyframeIds": [r1["keyframe"]["id"], r2["keyframe"]["id"]],
            "targetTime": "0:30.00",
            "targetTrackId": "track_1",
        })

        data = get_editor_data(env)
        new_tr = next(t for t in data["transitions"] if t["id"] == result["transitions"][0]["id"])
        assert new_tr["isAdjustment"] is True

    def test_paste_copies_transition_effects(self, project_env):
        """Pasted transitions should copy their strobe/effects."""
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        trs = active_transitions(env)
        tr_id = trs[0]["id"]

        # Add a strobe effect
        fx_result = api(env, "POST", f"/api/projects/{env['project_name']}/transition-effects/add", {
            "transitionId": tr_id,
            "type": "strobe",
            "params": {"period": 0.125, "duty": 0.5},
        })
        assert fx_result["success"] is True

        result = api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
            "keyframeIds": [r1["keyframe"]["id"], r2["keyframe"]["id"]],
            "targetTime": "0:30.00",
            "targetTrackId": "track_1",
        })

        data = get_editor_data(env)
        new_tr = next(t for t in data["transitions"] if t["id"] == result["transitions"][0]["id"])
        assert len(new_tr["effects"]) == 1
        assert new_tr["effects"][0]["type"] == "strobe"
        assert new_tr["effects"][0]["params"]["period"] == pytest.approx(0.125)
        assert new_tr["effects"][0]["params"]["duty"] == pytest.approx(0.5)

    def test_paste_copies_labels_and_tags(self, project_env):
        """Pasted transitions should copy labels, colors, and tags."""
        env = project_env

        r1 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:00.00"})
        r2 = api(env, "POST", f"/api/projects/{env['project_name']}/add-keyframe", {"timestamp": "0:05.00"})

        trs = active_transitions(env)
        tr_id = trs[0]["id"]

        api(env, "POST", f"/api/projects/{env['project_name']}/update-transition-label", {
            "transitionId": tr_id,
            "label": "Hero Dissolve",
            "labelColor": "#ff00ff",
            "tags": ["hero", "dissolve"],
        })

        result = api(env, "POST", f"/api/projects/{env['project_name']}/paste-group", {
            "keyframeIds": [r1["keyframe"]["id"], r2["keyframe"]["id"]],
            "targetTime": "0:30.00",
            "targetTrackId": "track_1",
        })

        data = get_editor_data(env)
        new_tr = next(t for t in data["transitions"] if t["id"] == result["transitions"][0]["id"])
        assert new_tr["label"] == "Hero Dissolve"
        assert new_tr["labelColor"] == "#ff00ff"
        assert new_tr["tags"] == ["hero", "dissolve"]
