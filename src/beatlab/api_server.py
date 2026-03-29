"""SceneCraft REST API server — exposes pipeline operations for the synthesizer frontend."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote


def _log(msg: str):
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def make_handler(work_dir: Path):
    """Create a request handler class with the work_dir baked in."""

    class SceneCraftHandler(BaseHTTPRequestHandler):
        """REST API handler for SceneCraft pipeline operations."""

        # ── Routing ──────────────────────────────────────────────

        def do_GET(self):
            parsed = urlparse(self.path)
            path = unquote(parsed.path)

            # GET /api/projects
            if path == "/api/projects":
                return self._handle_list_projects()

            # GET /api/browse?path=subdir (browse .beatlab_work root)
            if path == "/api/browse":
                query = parsed.query
                subpath = ""
                if query:
                    for param in query.split("&"):
                        if param.startswith("path="):
                            subpath = unquote(param[5:])
                return self._handle_browse(subpath)

            # GET /api/projects/:name/keyframes
            m = re.match(r"^/api/projects/([^/]+)/keyframes$", path)
            if m:
                return self._handle_get_keyframes(m.group(1))

            # GET /api/projects/:name/beats
            m = re.match(r"^/api/projects/([^/]+)/beats$", path)
            if m:
                return self._handle_get_beats(m.group(1))

            # GET /api/projects/:name/ls (directory listing, optional ?path=subdir)
            m = re.match(r"^/api/projects/([^/]+)/ls$", path)
            if m:
                query = parsed.query
                subpath = ""
                if query:
                    for param in query.split("&"):
                        if param.startswith("path="):
                            subpath = unquote(param[5:])
                return self._handle_ls(m.group(1), subpath)

            # GET /api/projects/:name/bin
            m = re.match(r"^/api/projects/([^/]+)/bin$", path)
            if m:
                return self._handle_get_bin(m.group(1))

            # GET /api/projects/:name/watched-folders
            m = re.match(r"^/api/projects/([^/]+)/watched-folders$", path)
            if m:
                return self._handle_get_watched_folders(m.group(1))

            # GET /api/projects/:name/narrative
            m = re.match(r"^/api/projects/([^/]+)/narrative$", path)
            if m:
                return self._handle_get_narrative(m.group(1))

            # GET /api/projects/:name/timelines
            m = re.match(r"^/api/projects/([^/]+)/timelines$", path)
            if m:
                return self._handle_get_timelines(m.group(1))

            # GET /api/projects/:name/settings
            m = re.match(r"^/api/projects/([^/]+)/settings$", path)
            if m:
                return self._handle_get_settings(m.group(1))

            # GET /api/projects/:name/audio-intelligence
            m = re.match(r"^/api/projects/([^/]+)/audio-intelligence$", path)
            if m:
                return self._handle_get_audio_intelligence(m.group(1))

            # GET /api/projects/:name/version/history
            m = re.match(r"^/api/projects/([^/]+)/version/history$", path)
            if m:
                return self._handle_version_history(m.group(1))

            # GET /api/projects/:name/version/diff
            m = re.match(r"^/api/projects/([^/]+)/version/diff$", path)
            if m:
                return self._handle_version_diff(m.group(1))

            # GET /api/projects/:name/effects
            m = re.match(r"^/api/projects/([^/]+)/effects$", path)
            if m:
                return self._handle_get_effects(m.group(1))

            # GET /api/projects/:name/files/(.*)
            m = re.match(r"^/api/projects/([^/]+)/files/(.+)$", path)
            if m:
                return self._handle_serve_file(m.group(1), m.group(2))

            self._error(404, "NOT_FOUND", f"No route: GET {path}")

        def do_POST(self):
            parsed = urlparse(self.path)
            path = unquote(parsed.path)

            # POST /api/projects/:name/select-keyframes
            m = re.match(r"^/api/projects/([^/]+)/select-keyframes$", path)
            if m:
                return self._handle_select_keyframes(m.group(1))

            # POST /api/projects/:name/select-slot-keyframes
            m = re.match(r"^/api/projects/([^/]+)/select-slot-keyframes$", path)
            if m:
                return self._handle_select_slot_keyframes(m.group(1))

            # POST /api/projects/:name/select-transitions
            m = re.match(r"^/api/projects/([^/]+)/select-transitions$", path)
            if m:
                return self._handle_select_transitions(m.group(1))

            # POST /api/projects/:name/update-timestamp
            m = re.match(r"^/api/projects/([^/]+)/update-timestamp$", path)
            if m:
                return self._handle_update_timestamp(m.group(1))

            # POST /api/projects/:name/add-keyframe
            m = re.match(r"^/api/projects/([^/]+)/add-keyframe$", path)
            if m:
                return self._handle_add_keyframe(m.group(1))

            # POST /api/projects/:name/delete-keyframe
            m = re.match(r"^/api/projects/([^/]+)/delete-keyframe$", path)
            if m:
                return self._handle_delete_keyframe(m.group(1))

            # POST /api/projects/:name/restore-keyframe
            m = re.match(r"^/api/projects/([^/]+)/restore-keyframe$", path)
            if m:
                return self._handle_restore_keyframe(m.group(1))

            # POST /api/projects/:name/delete-transition
            m = re.match(r"^/api/projects/([^/]+)/delete-transition$", path)
            if m:
                return self._handle_delete_transition(m.group(1))

            # POST /api/projects/:name/restore-transition
            m = re.match(r"^/api/projects/([^/]+)/restore-transition$", path)
            if m:
                return self._handle_restore_transition(m.group(1))

            # POST /api/projects/:name/update-transition-action
            m = re.match(r"^/api/projects/([^/]+)/update-transition-action$", path)
            if m:
                return self._handle_update_transition_action(m.group(1))

            # POST /api/projects/:name/update-transition-remap
            m = re.match(r"^/api/projects/([^/]+)/update-transition-remap$", path)
            if m:
                return self._handle_update_transition_remap(m.group(1))

            # POST /api/projects/:name/generate-transition-action
            m = re.match(r"^/api/projects/([^/]+)/generate-transition-action$", path)
            if m:
                return self._handle_generate_transition_action(m.group(1))

            # POST /api/projects/:name/generate-slot-keyframe-candidates
            m = re.match(r"^/api/projects/([^/]+)/generate-slot-keyframe-candidates$", path)
            if m:
                return self._handle_generate_slot_keyframe_candidates(m.group(1))

            # POST /api/projects/:name/generate-keyframe-candidates
            m = re.match(r"^/api/projects/([^/]+)/generate-keyframe-candidates$", path)
            if m:
                return self._handle_generate_keyframe_candidates(m.group(1))

            # POST /api/projects/:name/generate-transition-candidates
            m = re.match(r"^/api/projects/([^/]+)/generate-transition-candidates$", path)
            if m:
                return self._handle_generate_transition_candidates(m.group(1))

            # POST /api/projects/:name/update-meta
            m = re.match(r"^/api/projects/([^/]+)/update-meta$", path)
            if m:
                return self._handle_update_meta(m.group(1))

            # POST /api/projects/:name/effects (add/update/delete effects)
            m = re.match(r"^/api/projects/([^/]+)/effects$", path)
            if m:
                return self._handle_update_effects(m.group(1))

            # POST /api/projects/:name/import
            m = re.match(r"^/api/projects/([^/]+)/import$", path)
            if m:
                return self._handle_import(m.group(1))

            # POST /api/projects/:name/settings
            m = re.match(r"^/api/projects/([^/]+)/settings$", path)
            if m:
                return self._handle_update_settings(m.group(1))

            # POST /api/projects/:name/watch-folder
            m = re.match(r"^/api/projects/([^/]+)/watch-folder$", path)
            if m:
                return self._handle_watch_folder(m.group(1))

            # POST /api/projects/:name/unwatch-folder
            m = re.match(r"^/api/projects/([^/]+)/unwatch-folder$", path)
            if m:
                return self._handle_unwatch_folder(m.group(1))

            # POST /api/projects/:name/narrative
            m = re.match(r"^/api/projects/([^/]+)/narrative$", path)
            if m:
                return self._handle_update_narrative(m.group(1))

            # POST /api/projects/:name/timeline/switch
            m = re.match(r"^/api/projects/([^/]+)/timeline/switch$", path)
            if m:
                return self._handle_timeline_switch(m.group(1))

            # POST /api/projects/:name/timeline/import
            m = re.match(r"^/api/projects/([^/]+)/timeline/import$", path)
            if m:
                return self._handle_timeline_import(m.group(1))

            # POST /api/projects/:name/timeline/create
            m = re.match(r"^/api/projects/([^/]+)/timeline/create$", path)
            if m:
                return self._handle_timeline_create(m.group(1))

            # POST /api/projects/:name/version/commit
            m = re.match(r"^/api/projects/([^/]+)/version/commit$", path)
            if m:
                return self._handle_version_commit(m.group(1))

            # POST /api/projects/:name/version/checkout
            m = re.match(r"^/api/projects/([^/]+)/version/checkout$", path)
            if m:
                return self._handle_version_checkout(m.group(1))

            # POST /api/projects/:name/version/branch
            m = re.match(r"^/api/projects/([^/]+)/version/branch$", path)
            if m:
                return self._handle_version_branch(m.group(1))

            # POST /api/projects/:name/version/delete-branch
            m = re.match(r"^/api/projects/([^/]+)/version/delete-branch$", path)
            if m:
                return self._handle_version_delete_branch(m.group(1))

            self._error(404, "NOT_FOUND", f"No route: POST {path}")

        def do_HEAD(self):
            """Handle HEAD requests — browsers send these for video preload/metadata."""
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            m = re.match(r"^/api/projects/([^/]+)/files/(.+)$", path)
            if m:
                project_name, file_path = m.group(1), m.group(2)
                full_path = (work_dir / project_name / file_path).resolve()
                if not str(full_path).startswith(str(work_dir.resolve())) or not full_path.exists():
                    self.send_response(404)
                    self.end_headers()
                    return
                file_size = full_path.stat().st_size
                content_type = mimetypes.guess_type(str(full_path))[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(file_size))
                self.send_header("Accept-Ranges", "bytes")
                self._cors_headers()
                self.end_headers()
                return
            self.send_response(405)
            self.end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors_headers()
            self.end_headers()

        # ── Handlers ─────────────────────────────────────────────

        def _handle_browse(self, subpath: str):
            """GET /api/browse?path=subdir — browse .beatlab_work directory tree."""
            target = (work_dir / subpath).resolve() if subpath else work_dir.resolve()

            if not str(target).startswith(str(work_dir.resolve())):
                return self._error(403, "FORBIDDEN", "Path traversal denied")

            if not target.is_dir():
                return self._error(404, "NOT_FOUND", f"Directory not found: {subpath or '/'}")

            # Use os.scandir for fast directory listing (single syscall, cached stat)
            entries = []
            with os.scandir(target) as scanner:
                items = sorted(scanner, key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
                for entry in items:
                    is_dir = entry.is_dir(follow_symlinks=False)
                    rel = str(Path(entry.path).relative_to(work_dir.resolve()))
                    info: dict = {"name": entry.name, "path": rel, "isDirectory": is_dir}
                    if not is_dir:
                        ext = Path(entry.name).suffix.lower()
                        if ext in ('.png', '.jpg', '.jpeg', '.webp'):
                            info["type"] = "image"
                        elif ext in ('.mp4', '.webm', '.mov'):
                            info["type"] = "video"
                        else:
                            info["type"] = "other"
                    entries.append(info)

            self._json_response({"path": subpath or "", "entries": entries})

        def _handle_list_projects(self):
            """GET /api/projects — list all projects in work dir."""
            projects = []
            for entry in sorted(work_dir.iterdir()):
                if not entry.is_dir():
                    continue
                files = list(entry.iterdir())
                filenames = [f.name for f in files]
                has_audio = any(f.endswith((".wav", ".mp3")) for f in filenames)
                has_video = any(f.endswith(".mp4") for f in filenames)
                has_yaml = "narrative_keyframes.yaml" in filenames or "timeline.yaml" in filenames
                has_beats = "beats.json" in filenames

                projects.append({
                    "name": entry.name,
                    "hasAudio": has_audio,
                    "hasVideo": has_video,
                    "hasYaml": has_yaml,
                    "hasBeats": has_beats,
                    "fileCount": len(files),
                    "modified": entry.stat().st_mtime * 1000,
                })

            self._json_response(projects)

        def _handle_get_keyframes(self, project_name: str):
            """GET /api/projects/:name/keyframes — load keyframe data for editor."""
            from beatlab.project import load_project
            project_dir = work_dir / project_name
            if not project_dir.is_dir():
                return self._error(404, "NOT_FOUND", f"Project not found: {project_name}")

            if not self._has_project_yaml(project_name):
                return self._json_response({
                    "meta": {"title": project_name, "fps": 24, "resolution": [1920, 1080]},
                    "keyframes": [],
                    "audioFile": None,
                    "projectName": project_name,
                })

            parsed = load_project(project_dir)

            meta = parsed.get("meta", {})
            result_meta = {
                "title": meta.get("title", project_name),
                "fps": meta.get("fps", 24),
                "resolution": meta.get("resolution", [1920, 1080]),
                "motionPrompt": meta.get("motion_prompt", ""),
                "defaultTransitionPrompt": meta.get("default_transition_prompt", "Smooth cinematic transition"),
            }

            keyframes = []
            for kf in parsed.get("keyframes", []):
                kf_id = kf.get("id", "")
                img_path = project_dir / "selected_keyframes" / f"{kf_id}.png"
                has_selected = img_path.exists()

                # Find candidates
                candidates_dir = project_dir / "keyframe_candidates" / "candidates" / f"section_{kf_id}"
                candidate_files = []
                if candidates_dir.exists():
                    candidate_files = sorted([
                        f"keyframe_candidates/candidates/section_{kf_id}/{f.name}"
                        for f in candidates_dir.glob("v*.png")
                    ])

                ctx = kf.get("context", {})
                keyframes.append({
                    "id": kf_id,
                    "timestamp": kf.get("timestamp", "0:00"),
                    "section": kf.get("section", ""),
                    "prompt": kf.get("prompt", ""),
                    "selected": kf.get("selected"),
                    "hasSelectedImage": has_selected,
                    "candidates": candidate_files,
                    "context": {
                        "mood": ctx.get("mood", ""),
                        "energy": ctx.get("energy", ""),
                        "instruments": ctx.get("instruments", []),
                        "motifs": ctx.get("motifs", []),
                        "events": ctx.get("events", []),
                        "visual_direction": ctx.get("visual_direction", ""),
                        "details": ctx.get("details", ""),
                    } if ctx else None,
                })

            # Find audio file
            audio_file = None
            for candidate in ("audio.wav", "audio.mp3"):
                if (project_dir / candidate).exists():
                    audio_file = candidate
                    break

            # Parse transitions
            transitions = []
            tr_candidates_root = project_dir / "transition_candidates"
            for tr in parsed.get("transitions", []):
                tr_id = tr.get("id", "")
                # Scan disk for video candidates per slot
                slot_candidates = {}
                tr_dir = tr_candidates_root / tr_id
                if tr_dir.exists():
                    for slot_dir in sorted(tr_dir.iterdir()):
                        if slot_dir.is_dir():
                            videos = sorted([
                                f"transition_candidates/{tr_id}/{slot_dir.name}/{f.name}"
                                for f in slot_dir.glob("v*.mp4")
                            ])
                            if videos:
                                slot_candidates[slot_dir.name] = videos

                # Check for selected transition videos
                selected_tr_dir = project_dir / "selected_transitions"
                has_selected_videos = []
                for slot_idx in range(tr.get("slots", 1)):
                    sel_path = selected_tr_dir / f"{tr_id}_slot_{slot_idx}.mp4"
                    has_selected_videos.append(sel_path.exists())

                # selected is a list: [variant_or_path_per_slot]
                selected_list = tr.get("selected", [])
                if not isinstance(selected_list, list):
                    selected_list = []

                # Scan for slot keyframe candidates (intermediate keyframe images for multi-slot transitions)
                slot_kf_candidates = {}
                selected_slot_kfs = {}
                slot_kf_root = project_dir / "slot_keyframe_candidates" / "candidates"
                selected_slot_kf_dir = project_dir / "selected_slot_keyframes"
                num_slots = tr.get("slots", 1)
                if num_slots > 1:
                    for slot_idx in range(num_slots - 1):
                        slot_key = f"{tr_id}_slot_{slot_idx}"
                        section_dir = slot_kf_root / f"section_{slot_key}"
                        if section_dir.exists():
                            images = sorted([
                                f"slot_keyframe_candidates/candidates/section_{slot_key}/{f.name}"
                                for f in section_dir.glob("v*.png")
                            ])
                            if images:
                                slot_kf_candidates[slot_key] = images
                        # Check for selected slot keyframe
                        sel_path = selected_slot_kf_dir / f"{slot_key}.png"
                        if sel_path.exists():
                            # Find which variant is selected from YAML
                            slot_kf_selected = tr.get("selected_slot_keyframes", {})
                            if isinstance(slot_kf_selected, dict):
                                selected_slot_kfs[slot_key] = slot_kf_selected.get(slot_key)

                transitions.append({
                    "id": tr_id,
                    "from": tr.get("from", ""),
                    "to": tr.get("to", ""),
                    "durationSeconds": tr.get("duration_seconds", 0),
                    "slots": tr.get("slots", 1),
                    "action": tr.get("action", ""),
                    "useGlobalPrompt": tr.get("use_global_prompt", True),
                    "candidates": slot_candidates,
                    "hasSelectedVideos": has_selected_videos,
                    "selected": selected_list,
                    "remap": tr.get("remap", {"method": "linear", "target_duration": 0}),
                    "slotKeyframeCandidates": slot_kf_candidates,
                    "selectedSlotKeyframes": selected_slot_kfs,
                    "slotActions": tr.get("slot_actions", []),
                })

            self._json_response({
                "meta": result_meta,
                "keyframes": keyframes,
                "transitions": transitions,
                "audioFile": audio_file,
                "projectName": project_name,
            })

        def _handle_get_beats(self, project_name: str):
            """GET /api/projects/:name/beats — load beats.json."""
            beats_path = work_dir / project_name / "beats.json"
            if not beats_path.exists():
                return self._error(404, "NOT_FOUND", "No beats.json found")

            with open(beats_path) as f:
                data = json.load(f)
            self._json_response(data)

        def _handle_select_keyframes(self, project_name: str):
            """POST /api/projects/:name/select-keyframes — apply keyframe selections."""
            body = self._read_json_body()
            if body is None:
                return

            selections = body.get("selections", {})
            if not selections:
                return self._error(400, "BAD_REQUEST", "Missing 'selections' in body")

            yaml_path = self._require_yaml_path(project_name)
            if yaml_path is None:
                return

            try:
                from beatlab.render.narrative import apply_keyframe_selection
                apply_keyframe_selection(str(yaml_path), selections)
                self._json_response({"success": True, "applied": len(selections)})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_select_slot_keyframes(self, project_name: str):
            """POST /api/projects/:name/select-slot-keyframes — apply slot selections."""
            body = self._read_json_body()
            if body is None:
                return

            selections = body.get("selections", {})
            if not selections:
                return self._error(400, "BAD_REQUEST", "Missing 'selections' in body")

            yaml_path = self._require_yaml_path(project_name)
            if yaml_path is None:
                return

            try:
                from beatlab.render.narrative import apply_slot_keyframe_selection
                apply_slot_keyframe_selection(str(yaml_path), selections)
                self._json_response({"success": True, "applied": len(selections)})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_select_transitions(self, project_name: str):
            """POST /api/projects/:name/select-transitions — apply transition video selections.

            Body: { "selections": { "tr_001_slot_0": 2, "tr_005": 1 } }
            Keys are "tr_NNN_slot_N" or "tr_NNN" (shorthand for slot_0).
            Values are 1-based variant indices.
            """
            body = self._read_json_body()
            if body is None:
                return

            selections = body.get("selections", {})
            if not selections:
                return self._error(400, "BAD_REQUEST", "Missing 'selections' in body")

            yaml_path = self._require_yaml_path(project_name)
            if yaml_path is None:
                return

            try:
                from beatlab.render.narrative import apply_transition_selection
                apply_transition_selection(str(yaml_path), selections)
                self._json_response({"success": True, "applied": len(selections)})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_update_timestamp(self, project_name: str):
            """POST /api/projects/:name/update-timestamp — update a keyframe timestamp."""
            body = self._read_json_body()
            if body is None:
                return

            kf_id = body.get("keyframeId")
            new_timestamp = body.get("newTimestamp")
            if not kf_id or new_timestamp is None:
                return self._error(400, "BAD_REQUEST", "Missing 'keyframeId' or 'newTimestamp'")

            yaml_path = self._require_yaml_path(project_name)
            if yaml_path is None:
                return

            try:
                # Read, find keyframe, update timestamp, write back
                content = yaml_path.read_text()
                # Match both "- id: kf_001" and "  id: kf_001" (pyyaml sorts keys, so id may not be first)
                id_match = re.search(rf"^[ -]+id: {re.escape(kf_id)}\s*$", content, re.MULTILINE)
                if not id_match:
                    return self._error(404, "NOT_FOUND", f"Keyframe {kf_id} not found")
                idx = id_match.start()

                ts_pattern = re.compile(r"\n(\s+)timestamp:\s*'?([^'\n]+)'?")
                after = content[idx:]
                match = ts_pattern.search(after)
                if not match:
                    return self._error(500, "INTERNAL_ERROR", "Timestamp field not found")

                full_match = match.group(0)
                indent = match.group(1)
                replacement = f"\n{indent}timestamp: '{new_timestamp}'"
                updated = content[:idx] + after.replace(full_match, replacement, 1)
                yaml_path.write_text(updated)

                self._json_response({"success": True, "keyframeId": kf_id, "newTimestamp": new_timestamp})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_get_bin(self, project_name: str):
            """GET /api/projects/:name/bin — list binned (soft-deleted) keyframes."""
            from beatlab.project import load_project
            project_dir = work_dir / project_name
            if not project_dir.is_dir():
                return self._json_response({"bin": []})
            parsed = load_project(project_dir)

            project_dir = work_dir / project_name
            bin_entries = []
            for kf in parsed.get("bin", []):
                kf_id = kf.get("id", "")
                img_path = project_dir / "selected_keyframes" / f"{kf_id}.png"
                bin_entries.append({
                    "id": kf_id,
                    "deleted_at": kf.get("deleted_at", ""),
                    "timestamp": kf.get("timestamp", "0:00"),
                    "section": kf.get("section", ""),
                    "prompt": kf.get("prompt", ""),
                    "hasSelectedImage": img_path.exists(),
                })

            transition_bin = []
            for tr in parsed.get("transition_bin", []):
                transition_bin.append({
                    "id": tr.get("id", ""),
                    "deleted_at": tr.get("deleted_at", ""),
                    "from": tr.get("from", ""),
                    "to": tr.get("to", ""),
                    "durationSeconds": tr.get("duration_seconds", 0),
                    "slots": tr.get("slots", 1),
                })

            self._json_response({"bin": bin_entries, "transitionBin": transition_bin})

        def _handle_add_keyframe(self, project_name: str):
            """POST /api/projects/:name/add-keyframe — create a new keyframe at a given timestamp."""
            body = self._read_json_body()
            if body is None:
                return

            timestamp = body.get("timestamp")
            if not timestamp:
                return self._error(400, "BAD_REQUEST", "Missing 'timestamp'")

            section = body.get("section", "")
            prompt = body.get("prompt", "")

            yaml_path = self._require_yaml_path(project_name)
            if yaml_path is None:
                return

            try:
                import yaml as pyyaml
                with open(yaml_path) as f:
                    parsed = pyyaml.safe_load(f)

                keyframes = parsed.get("keyframes", [])
                bin_list = parsed.get("bin", [])

                # Compute next sequential ID from keyframes + bin
                max_num = 0
                for kf in keyframes + bin_list:
                    kf_id = kf.get("id", "")
                    if kf_id.startswith("kf_"):
                        try:
                            num = int(kf_id[3:])
                            if num > max_num:
                                max_num = num
                        except ValueError:
                            pass
                new_id = f"kf_{max_num + 1:03d}"

                new_kf = {
                    "id": new_id,
                    "timestamp": timestamp,
                    "section": section,
                    "prompt": prompt,
                    "candidates": [],
                    "selected": None,
                }

                keyframes.append(new_kf)

                # Sort by timestamp
                def parse_ts(ts):
                    parts = str(ts).split(":")
                    if len(parts) == 2:
                        return int(parts[0]) * 60 + float(parts[1])
                    return 0
                keyframes.sort(key=lambda kf: parse_ts(kf.get("timestamp", "0:00")))

                parsed["keyframes"] = keyframes

                with open(yaml_path, "w") as f:
                    pyyaml.dump(parsed, f, default_flow_style=False, allow_unicode=True, width=1000)

                self._json_response({"success": True, "keyframe": {"id": new_id, "timestamp": timestamp, "section": section, "prompt": prompt}})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_delete_keyframe(self, project_name: str):
            """POST /api/projects/:name/delete-keyframe — soft-delete a keyframe to bin."""
            body = self._read_json_body()
            if body is None:
                return

            kf_id = body.get("keyframeId")
            if not kf_id:
                return self._error(400, "BAD_REQUEST", "Missing 'keyframeId'")

            yaml_path = self._require_yaml_path(project_name)
            if yaml_path is None:
                return

            try:
                import yaml as pyyaml
                from datetime import datetime, timezone
                with open(yaml_path) as f:
                    parsed = pyyaml.safe_load(f)

                keyframes = parsed.get("keyframes", [])
                idx = next((i for i, kf in enumerate(keyframes) if kf.get("id") == kf_id), -1)
                if idx == -1:
                    return self._error(404, "NOT_FOUND", f"Keyframe {kf_id} not found")

                removed = keyframes.pop(idx)
                removed["deleted_at"] = datetime.now(timezone.utc).isoformat()

                bin_list = parsed.get("bin", [])
                bin_list.append(removed)
                parsed["bin"] = bin_list
                parsed["keyframes"] = keyframes

                with open(yaml_path, "w") as f:
                    pyyaml.dump(parsed, f, default_flow_style=False, allow_unicode=True, width=1000)

                self._json_response({"success": True, "binned": {"id": kf_id, "deleted_at": removed["deleted_at"]}})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_restore_keyframe(self, project_name: str):
            """POST /api/projects/:name/restore-keyframe — restore a keyframe from bin."""
            body = self._read_json_body()
            if body is None:
                return

            kf_id = body.get("keyframeId")
            if not kf_id:
                return self._error(400, "BAD_REQUEST", "Missing 'keyframeId'")

            yaml_path = self._require_yaml_path(project_name)
            if yaml_path is None:
                return

            try:
                import yaml as pyyaml
                with open(yaml_path) as f:
                    parsed = pyyaml.safe_load(f)

                bin_list = parsed.get("bin", [])
                idx = next((i for i, kf in enumerate(bin_list) if kf.get("id") == kf_id), -1)
                if idx == -1:
                    return self._error(404, "NOT_FOUND", f"Keyframe {kf_id} not in bin")

                restored = bin_list.pop(idx)
                del restored["deleted_at"]

                keyframes = parsed.get("keyframes", [])
                keyframes.append(restored)
                # Sort by timestamp
                def parse_ts(ts):
                    parts = str(ts).split(":")
                    if len(parts) == 2:
                        return int(parts[0]) * 60 + float(parts[1])
                    return 0
                keyframes.sort(key=lambda kf: parse_ts(kf.get("timestamp", "0:00")))

                parsed["keyframes"] = keyframes
                parsed["bin"] = bin_list

                with open(yaml_path, "w") as f:
                    pyyaml.dump(parsed, f, default_flow_style=False, allow_unicode=True, width=1000)

                self._json_response({"success": True, "keyframe": {"id": kf_id}})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_delete_transition(self, project_name: str):
            """POST /api/projects/:name/delete-transition — soft-delete a transition to bin."""
            body = self._read_json_body()
            if body is None:
                return

            tr_id = body.get("transitionId")
            if not tr_id:
                return self._error(400, "BAD_REQUEST", "Missing 'transitionId'")

            yaml_path = self._require_yaml_path(project_name)
            if yaml_path is None:
                return

            try:
                import yaml as pyyaml
                from datetime import datetime, timezone
                with open(yaml_path) as f:
                    parsed = pyyaml.safe_load(f)

                transitions = parsed.get("transitions", [])
                idx = next((i for i, tr in enumerate(transitions) if tr.get("id") == tr_id), -1)
                if idx == -1:
                    return self._error(404, "NOT_FOUND", f"Transition {tr_id} not found")

                removed = transitions.pop(idx)
                removed["deleted_at"] = datetime.now(timezone.utc).isoformat()

                tr_bin = parsed.get("transition_bin", [])
                tr_bin.append(removed)
                parsed["transition_bin"] = tr_bin
                parsed["transitions"] = transitions

                with open(yaml_path, "w") as f:
                    pyyaml.dump(parsed, f, default_flow_style=False, allow_unicode=True, width=1000)

                self._json_response({"success": True, "binned": {"id": tr_id, "deleted_at": removed["deleted_at"]}})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_restore_transition(self, project_name: str):
            """POST /api/projects/:name/restore-transition — restore a transition from bin."""
            body = self._read_json_body()
            if body is None:
                return

            tr_id = body.get("transitionId")
            if not tr_id:
                return self._error(400, "BAD_REQUEST", "Missing 'transitionId'")

            yaml_path = self._require_yaml_path(project_name)
            if yaml_path is None:
                return

            try:
                import yaml as pyyaml
                with open(yaml_path) as f:
                    parsed = pyyaml.safe_load(f)

                tr_bin = parsed.get("transition_bin", [])
                idx = next((i for i, tr in enumerate(tr_bin) if tr.get("id") == tr_id), -1)
                if idx == -1:
                    return self._error(404, "NOT_FOUND", f"Transition {tr_id} not in bin")

                restored = tr_bin.pop(idx)
                del restored["deleted_at"]

                transitions = parsed.get("transitions", [])
                transitions.append(restored)
                parsed["transitions"] = transitions
                parsed["transition_bin"] = tr_bin

                with open(yaml_path, "w") as f:
                    pyyaml.dump(parsed, f, default_flow_style=False, allow_unicode=True, width=1000)

                self._json_response({"success": True, "transition": {"id": tr_id}})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_update_transition_action(self, project_name: str):
            """POST /api/projects/:name/update-transition-action — update a transition's action prompt."""
            body = self._read_json_body()
            if body is None:
                return

            tr_id = body.get("transitionId")
            action = body.get("action")
            slot_actions = body.get("slotActions")  # optional: per-slot prompts
            use_global = body.get("useGlobalPrompt")
            if not tr_id:
                return self._error(400, "BAD_REQUEST", "Missing 'transitionId'")

            yaml_path = self._require_yaml_path(project_name)
            if yaml_path is None:
                return

            try:
                import yaml as pyyaml
                with open(yaml_path) as f:
                    parsed = pyyaml.safe_load(f)

                transitions = parsed.get("transitions", [])
                tr = next((t for t in transitions if t.get("id") == tr_id), None)
                if not tr:
                    return self._error(404, "NOT_FOUND", f"Transition {tr_id} not found")

                if action is not None:
                    tr["action"] = action
                if slot_actions is not None:
                    tr["slot_actions"] = slot_actions
                if use_global is not None:
                    tr["use_global_prompt"] = use_global

                parsed["transitions"] = transitions
                with open(yaml_path, "w") as f:
                    pyyaml.dump(parsed, f, default_flow_style=False, allow_unicode=True, width=1000)

                self._json_response({"success": True})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_update_transition_remap(self, project_name: str):
            """POST /api/projects/:name/update-transition-remap — update a transition's remap/duration."""
            body = self._read_json_body()
            if body is None:
                return

            tr_id = body.get("transitionId")
            target_duration = body.get("targetDuration")
            method = body.get("method")
            if not tr_id:
                return self._error(400, "BAD_REQUEST", "Missing 'transitionId'")

            yaml_path = self._require_yaml_path(project_name)
            if yaml_path is None:
                return

            try:
                import yaml as pyyaml
                with open(yaml_path) as f:
                    parsed = pyyaml.safe_load(f)

                transitions = parsed.get("transitions", [])
                tr = next((t for t in transitions if t.get("id") == tr_id), None)
                if not tr:
                    return self._error(404, "NOT_FOUND", f"Transition {tr_id} not found")

                remap = tr.get("remap", {"method": "linear", "target_duration": 0})
                if target_duration is not None:
                    remap["target_duration"] = target_duration
                if method is not None:
                    remap["method"] = method
                tr["remap"] = remap

                parsed["transitions"] = transitions
                with open(yaml_path, "w") as f:
                    pyyaml.dump(parsed, f, default_flow_style=False, allow_unicode=True, width=1000)

                self._json_response({"success": True, "transitionId": tr_id, "remap": remap})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_generate_transition_action(self, project_name: str):
            """POST /api/projects/:name/generate-transition-action — LLM-generate action for a single transition."""
            body = self._read_json_body()
            if body is None:
                return

            tr_id = body.get("transitionId")
            if not tr_id:
                return self._error(400, "BAD_REQUEST", "Missing 'transitionId'")

            yaml_path = self._require_yaml_path(project_name)
            if yaml_path is None:
                return

            try:
                import yaml as pyyaml
                import base64
                import os

                with open(yaml_path) as f:
                    parsed = pyyaml.safe_load(f)

                transitions = parsed.get("transitions", [])
                tr = next((t for t in transitions if t.get("id") == tr_id), None)
                if not tr:
                    return self._error(404, "NOT_FOUND", f"Transition {tr_id} not found")

                kf_by_id = {kf["id"]: kf for kf in parsed.get("keyframes", [])}
                from_kf = kf_by_id.get(tr["from"])
                to_kf = kf_by_id.get(tr["to"])
                if not from_kf or not to_kf:
                    return self._error(400, "BAD_REQUEST", f"Keyframes {tr['from']} or {tr['to']} not found")

                project_dir = work_dir / project_name
                selected_dir = project_dir / "selected_keyframes"
                from_img = selected_dir / f"{tr['from']}.png"
                to_img = selected_dir / f"{tr['to']}.png"

                if not from_img.exists() or not to_img.exists():
                    return self._error(400, "BAD_REQUEST", "Selected keyframe images not found — run keyframe selection first")

                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    return self._error(500, "INTERNAL_ERROR", "ANTHROPIC_API_KEY not set")

                from anthropic import Anthropic
                client = Anthropic(api_key=api_key)

                from_b64 = base64.b64encode(from_img.read_bytes()).decode()
                to_b64 = base64.b64encode(to_img.read_bytes()).decode()
                from_ctx = from_kf.get("context", {})
                to_ctx = to_kf.get("context", {})
                master_prompt = parsed.get("meta", {}).get("prompt", "")
                master_context = f"Overall creative direction: {master_prompt}\n\n" if master_prompt else ""

                n_slots = tr.get("slots", 1)
                selected_slot_kf_dir = project_dir / "selected_slot_keyframes"

                if n_slots <= 1:
                    # Single-slot: generate one action from the two keyframes
                    user_content = [
                        {"type": "text", "text": f"You are a visual effects director for a music video. {master_context}Describe the ideal visual transition between these two keyframes.\n\n"},
                        {"type": "text", "text": f"FROM keyframe ({tr['from']}):\n"
                            f"  Timestamp: {from_kf['timestamp']}\n"
                            f"  Mood: {from_ctx.get('mood', 'unknown')}\n"
                            f"  Energy: {from_ctx.get('energy', 'unknown')}\n"
                            f"  Instruments: {', '.join(from_ctx.get('instruments', []))}\n"
                            f"  Visual direction: {from_ctx.get('visual_direction', '')}\n\n"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": from_b64}},
                        {"type": "text", "text": f"\nTO keyframe ({tr['to']}):\n"
                            f"  Timestamp: {to_kf['timestamp']}\n"
                            f"  Mood: {to_ctx.get('mood', 'unknown')}\n"
                            f"  Energy: {to_ctx.get('energy', 'unknown')}\n"
                            f"  Instruments: {', '.join(to_ctx.get('instruments', []))}\n"
                            f"  Visual direction: {to_ctx.get('visual_direction', '')}\n\n"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": to_b64}},
                        {"type": "text", "text": f"\nTransition duration: {tr['duration_seconds']}s.\n\n"
                            "Write a concise cinematic transition description (1-3 sentences) that describes the visual journey "
                            "from the first image to the second, considering the musical context. "
                            "Focus on motion, transformation, and mood shift. "
                            "This will be used as a prompt for Veo video generation.\n\n"
                            "Reply with ONLY the transition description, no preamble."},
                    ]

                    response = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=300,
                        messages=[{"role": "user", "content": user_content}],
                    )

                    action = response.content[0].text.strip()
                    tr["action"] = action
                else:
                    # Multi-slot: build the chain of images (from_kf -> intermediate_0 -> ... -> to_kf)
                    chain_images = [from_img]
                    for s in range(n_slots - 1):
                        slot_kf_path = selected_slot_kf_dir / f"{tr_id}_slot_{s}.png"
                        if slot_kf_path.exists():
                            chain_images.append(slot_kf_path)
                        else:
                            chain_images.append(None)
                    chain_images.append(to_img)

                    # Generate one prompt per slot
                    slot_actions = []
                    slot_duration = tr["duration_seconds"] / n_slots
                    for s in range(n_slots):
                        start_img_path = chain_images[s]
                        end_img_path = chain_images[s + 1]
                        if not start_img_path or not end_img_path or not start_img_path.exists() or not end_img_path.exists():
                            slot_actions.append(f"Smooth cinematic transition (slot {s})")
                            continue

                        s_b64 = base64.b64encode(start_img_path.read_bytes()).decode()
                        e_b64 = base64.b64encode(end_img_path.read_bytes()).decode()

                        user_content = [
                            {"type": "text", "text": f"You are a visual effects director for a music video. {master_context}"
                                f"This is slot {s + 1} of {n_slots} in a multi-slot transition from {tr['from']} to {tr['to']}.\n\n"},
                            {"type": "text", "text": "START frame for this slot:\n"},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": s_b64}},
                            {"type": "text", "text": "\nEND frame for this slot:\n"},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": e_b64}},
                            {"type": "text", "text": f"\nSlot duration: {slot_duration:.1f}s.\n\n"
                                "Write a concise cinematic description (1-3 sentences) of what happens visually during this slot. "
                                "The start and end frames may look similar — describe the motion, energy, and subtle transformations "
                                "that should occur between them. Focus on camera movement, lighting shifts, and particle/element behavior. "
                                "This will be used as a prompt for Veo video generation.\n\n"
                                "Reply with ONLY the description, no preamble."},
                        ]

                        response = client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=300,
                            messages=[{"role": "user", "content": user_content}],
                        )
                        slot_actions.append(response.content[0].text.strip())

                    tr["slot_actions"] = slot_actions
                    # Also set action to slot 0's action as a summary/fallback
                    if slot_actions:
                        tr["action"] = slot_actions[0]

                with open(yaml_path, "w") as f:
                    pyyaml.dump(parsed, f, default_flow_style=False, allow_unicode=True, width=1000)

                self._json_response({"success": True, "action": tr.get("action", ""), "slotActions": tr.get("slot_actions", [])})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_generate_slot_keyframe_candidates(self, project_name: str):
            """POST /api/projects/:name/generate-slot-keyframe-candidates — generate intermediate keyframe images for multi-slot transitions."""
            body = self._read_json_body()
            if body is None:
                return

            tr_id = body.get("transitionId")  # optional — generate for specific transition or all

            yaml_path = self._require_yaml_path(project_name)
            if yaml_path is None:
                return

            from beatlab.ws_server import job_manager
            job_id = job_manager.create_job("slot_keyframe_candidates", total=0, meta={"transitionId": tr_id or "all", "project": project_name})

            def _run():
                try:
                    from beatlab.render.narrative import generate_slot_keyframe_candidates
                    generate_slot_keyframe_candidates(str(yaml_path), vertex=False)

                    # Collect results
                    project_dir = work_dir / project_name
                    slot_kf_dir = project_dir / "slot_keyframe_candidates" / "candidates"
                    candidates = {}
                    if slot_kf_dir.exists():
                        for section_dir in sorted(slot_kf_dir.iterdir()):
                            if section_dir.is_dir() and section_dir.name.startswith("section_"):
                                slot_key = section_dir.name.replace("section_", "")
                                images = sorted([
                                    f"slot_keyframe_candidates/candidates/{section_dir.name}/{f.name}"
                                    for f in section_dir.glob("v*.png")
                                ])
                                if images:
                                    candidates[slot_key] = images

                    job_manager.complete_job(job_id, {"candidates": candidates})
                except Exception as e:
                    job_manager.fail_job(job_id, str(e))

            import threading
            threading.Thread(target=_run, daemon=True).start()
            self._json_response({"jobId": job_id})

        def _handle_generate_keyframe_candidates(self, project_name: str):
            """POST /api/projects/:name/generate-keyframe-candidates — async Imagen generation with WebSocket progress."""
            body = self._read_json_body()
            if body is None:
                return

            kf_id = body.get("keyframeId")
            count = body.get("count", 4)  # how many NEW candidates to generate
            if not kf_id:
                return self._error(400, "BAD_REQUEST", "Missing 'keyframeId'")

            yaml_path = self._require_yaml_path(project_name)
            if yaml_path is None:
                return

            # Count existing candidates so we generate beyond them (v5, v6, etc.)
            project_dir = work_dir / project_name
            candidates_dir = project_dir / "keyframe_candidates" / "candidates" / f"section_{kf_id}"
            existing_count = len(list(candidates_dir.glob("v*.png"))) if candidates_dir.exists() else 0
            total_count = existing_count + count

            from beatlab.ws_server import job_manager
            job_id = job_manager.create_job("keyframe_candidates", total=count, meta={"keyframeId": kf_id, "project": project_name})

            def _run():
                try:
                    from beatlab.render.narrative import generate_keyframe_candidates
                    generate_keyframe_candidates(
                        str(yaml_path),
                        vertex=True,
                        candidates_per_slot=total_count,
                        segment_filter={kf_id},
                    )

                    # Collect results
                    project_dir = work_dir / project_name
                    candidates_dir = project_dir / "keyframe_candidates" / "candidates" / f"section_{kf_id}"
                    candidates = []
                    if candidates_dir.exists():
                        candidates = sorted([
                            f"keyframe_candidates/candidates/section_{kf_id}/{f.name}"
                            for f in candidates_dir.glob("v*.png")
                        ])

                    job_manager.complete_job(job_id, {"keyframeId": kf_id, "candidates": candidates})
                except Exception as e:
                    job_manager.fail_job(job_id, str(e))

            import threading
            threading.Thread(target=_run, daemon=True).start()
            self._json_response({"jobId": job_id, "keyframeId": kf_id})

        def _handle_generate_transition_candidates(self, project_name: str):
            """POST /api/projects/:name/generate-transition-candidates — async Veo generation with WebSocket progress."""
            body = self._read_json_body()
            if body is None:
                return

            tr_id = body.get("transitionId")
            count = body.get("count", 4)  # how many NEW candidates to generate
            slot_index = body.get("slotIndex")  # optional: generate for a single slot only
            if not tr_id:
                return self._error(400, "BAD_REQUEST", "Missing 'transitionId'")

            yaml_path = self._require_yaml_path(project_name)
            if yaml_path is None:
                return

            # Count existing candidates so we generate beyond them (v5, v6, etc.)
            project_dir = work_dir / project_name
            tr_candidates_dir = project_dir / "transition_candidates" / tr_id
            existing_count = 0
            if tr_candidates_dir.exists():
                if slot_index is not None:
                    slot_dir = tr_candidates_dir / f"slot_{slot_index}"
                    if slot_dir.exists():
                        existing_count = len(list(slot_dir.glob("v*.mp4")))
                else:
                    for slot_dir in tr_candidates_dir.iterdir():
                        if slot_dir.is_dir():
                            existing_count = max(existing_count, len(list(slot_dir.glob("v*.mp4"))))
            total_count = existing_count + count

            from beatlab.ws_server import job_manager
            job_id = job_manager.create_job("transition_candidates", total=count, meta={"transitionId": tr_id, "project": project_name})

            def _run():
                try:
                    from beatlab.render.narrative import generate_transition_candidates
                    from beatlab.render.google_video import PromptRejectedError

                    job_manager.update_progress(job_id, 0, "Starting Veo generation...")

                    def _on_status(msg):
                        job_manager.update_progress(job_id, 0, msg)

                    try:
                        generate_transition_candidates(
                            str(yaml_path),
                            vertex=True,
                            candidates_per_slot=total_count,
                            segment_filter={tr_id},
                            slot_filter={slot_index} if slot_index is not None else None,
                            on_status=_on_status,
                        )
                    except PromptRejectedError as pre:
                        job_manager.update_progress(job_id, 0, f"Prompt issue: {str(pre)[:100]}")
                        # Some candidates may still have been generated — continue to collect

                    # Collect results
                    project_dir = work_dir / project_name
                    tr_candidates_dir = project_dir / "transition_candidates" / tr_id
                    candidates = {}
                    if tr_candidates_dir.exists():
                        for slot_dir in sorted(tr_candidates_dir.iterdir()):
                            if slot_dir.is_dir():
                                videos = sorted([
                                    f"transition_candidates/{tr_id}/{slot_dir.name}/{f.name}"
                                    for f in slot_dir.glob("v*.mp4")
                                ])
                                candidates[slot_dir.name] = videos

                    job_manager.complete_job(job_id, {"transitionId": tr_id, "candidates": candidates})
                except Exception as e:
                    err = str(e)
                    if "transient" in err.lower() or "None" in err:
                        job_manager.fail_job(job_id, f"Veo returned empty results after retries. This is usually transient — try again. ({err[:80]})")
                    else:
                        job_manager.fail_job(job_id, err)

            import threading
            threading.Thread(target=_run, daemon=True).start()
            self._json_response({"jobId": job_id, "transitionId": tr_id})

        def _handle_update_meta(self, project_name: str):
            """POST /api/projects/:name/update-meta — update project meta fields."""
            body = self._read_json_body()
            if body is None:
                return

            yaml_path = self._require_yaml_path(project_name)
            if yaml_path is None:
                return

            try:
                import yaml as pyyaml
                with open(yaml_path) as f:
                    parsed = pyyaml.safe_load(f)

                meta = parsed.get("meta", {})
                # Only allow updating specific safe fields
                for key in ("motion_prompt", "default_transition_prompt"):
                    if key in body:
                        meta[key] = body[key]

                parsed["meta"] = meta
                with open(yaml_path, "w") as f:
                    pyyaml.dump(parsed, f, default_flow_style=False, allow_unicode=True, width=1000)

                self._json_response({"success": True, "meta": meta})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_get_audio_intelligence(self, project_name: str):
            """GET /api/projects/:name/audio-intelligence — return processed beat events from audio intelligence file."""
            project_dir = work_dir / project_name

            # Determine which file to use
            import yaml as pyyaml
            settings_path = project_dir / "settings.yaml"
            ai_file = None
            if settings_path.exists():
                with open(settings_path) as f:
                    s = pyyaml.safe_load(f) or {}
                ai_file = s.get("audio_intelligence_file")

            # Auto-detect latest if not configured
            if not ai_file:
                candidates = sorted(project_dir.glob("audio_intelligence*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                if candidates:
                    ai_file = candidates[0].name

            if not ai_file or not (project_dir / ai_file).exists():
                # Fallback: return empty (frontend falls back to beats.json)
                return self._json_response({"activeFile": None, "events": [], "sections": [], "rules": []})

            try:
                import json as _json
                with open(project_dir / ai_file) as f:
                    data = _json.load(f)

                events = data.get("layer3_events", [])
                sections = data.get("layer2", [])
                rules = data.get("layer3_rules", [])

                # List available files
                available = sorted([f.name for f in project_dir.glob("audio_intelligence*.json")], reverse=True)

                self._json_response({
                    "activeFile": ai_file,
                    "availableFiles": available,
                    "events": events,
                    "sections": sections,
                    "ruleCount": len(rules),
                })
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_get_settings(self, project_name: str):
            """GET /api/projects/:name/settings — read project settings from settings.yaml."""
            import yaml as pyyaml
            settings_path = work_dir / project_name / "settings.yaml"
            defaults = {
                "preview_quality": 50,
                "audio_intelligence_file": None,
                "render_preview_fps": 24,
            }
            if settings_path.exists():
                with open(settings_path) as f:
                    saved = pyyaml.safe_load(f) or {}
                defaults.update(saved)

            # Also list available audio intelligence files
            project_dir = work_dir / project_name
            ai_files = sorted([
                f.name for f in project_dir.glob("audio_intelligence*.json")
            ], reverse=True)

            self._json_response({**defaults, "available_audio_intelligence_files": ai_files})

        def _handle_update_settings(self, project_name: str):
            """POST /api/projects/:name/settings — update project settings in settings.yaml."""
            body = self._read_json_body()
            if body is None:
                return

            import yaml as pyyaml
            settings_path = work_dir / project_name / "settings.yaml"

            existing = {}
            if settings_path.exists():
                with open(settings_path) as f:
                    existing = pyyaml.safe_load(f) or {}

            # Only allow known fields
            allowed = {"preview_quality", "audio_intelligence_file", "render_preview_fps"}
            for key in allowed:
                if key in body:
                    existing[key] = body[key]

            with open(settings_path, "w") as f:
                pyyaml.dump(existing, f, default_flow_style=False, allow_unicode=True)

            self._json_response({"success": True, **existing})

        def _handle_get_watched_folders(self, project_name: str):
            """GET /api/projects/:name/watched-folders — list persisted watched folders."""
            from beatlab.project import load_project
            project_dir = work_dir / project_name
            parsed = load_project(project_dir) if project_dir.is_dir() else {}
            self._json_response({"watchedFolders": parsed.get("watched_folders", [])})

        def _handle_get_effects(self, project_name: str):
            """GET /api/projects/:name/effects — load user-authored effects from beats.yaml."""
            effects_path = work_dir / project_name / "beats.yaml"
            if not effects_path.exists():
                return self._json_response({"effects": [], "suppressions": []})

            import yaml as pyyaml
            with open(effects_path) as f:
                parsed = pyyaml.safe_load(f) or {}

            self._json_response({
                "effects": parsed.get("effects", []),
                "suppressions": parsed.get("suppressions", []),
            })

        def _handle_update_effects(self, project_name: str):
            """POST /api/projects/:name/effects — update user-authored effects in beats.yaml.

            Body: { "effects": [...], "suppressions": [...] }
            Replaces the entire effects file with the provided data.
            """
            body = self._read_json_body()
            if body is None:
                return

            try:
                import yaml as pyyaml
                effects_path = work_dir / project_name / "beats.yaml"

                data = {}
                if effects_path.exists():
                    with open(effects_path) as f:
                        data = pyyaml.safe_load(f) or {}

                if "effects" in body:
                    data["effects"] = body["effects"]
                if "suppressions" in body:
                    data["suppressions"] = body["suppressions"]

                with open(effects_path, "w") as f:
                    pyyaml.dump(data, f, default_flow_style=False, allow_unicode=True, width=1000)

                self._json_response({"success": True})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_watch_folder(self, project_name: str):
            """POST /api/projects/:name/watch-folder — start watching a folder for auto-import."""
            body = self._read_json_body()
            if body is None:
                return

            folder_path = body.get("folderPath")
            if not folder_path:
                return self._error(400, "BAD_REQUEST", "Missing 'folderPath'")

            from beatlab.ws_server import folder_watcher
            if not folder_watcher:
                return self._error(500, "INTERNAL_ERROR", "Folder watcher not initialized")

            try:
                result = folder_watcher.add_watch(project_name, folder_path)

                # Persist to YAML
                from beatlab.project import load_project, save_project
                project_dir = work_dir / project_name
                data = load_project(project_dir)
                watched = data.get("watched_folders", [])
                if folder_path not in watched:
                    watched.append(folder_path)
                data["watched_folders"] = watched
                save_project(data, project_dir)

                self._json_response({"success": True, **result})
            except Exception as e:
                self._error(400, "BAD_REQUEST", str(e))

        def _handle_unwatch_folder(self, project_name: str):
            """POST /api/projects/:name/unwatch-folder — stop watching a folder."""
            body = self._read_json_body()
            if body is None:
                return

            folder_path = body.get("folderPath")
            if not folder_path:
                return self._error(400, "BAD_REQUEST", "Missing 'folderPath'")

            from beatlab.ws_server import folder_watcher
            if folder_watcher:
                folder_watcher.remove_watch(project_name, folder_path)

            # Remove from YAML
            from beatlab.project import load_project, save_project
            project_dir = work_dir / project_name
            if project_dir.is_dir():
                data = load_project(project_dir)
                watched = data.get("watched_folders", [])
                if folder_path in watched:
                    watched.remove(folder_path)
                data["watched_folders"] = watched
                save_project(data, project_dir)

            self._json_response({"success": True})

        def _handle_import(self, project_name: str):
            """POST /api/projects/:name/import — bulk import images as keyframes and videos as transitions.

            Body: { "sourcePath": "/absolute/path/to/dir/or/file", "timestamp": "0:00" }
            - If sourcePath is a directory, all images/videos inside are imported
            - If sourcePath is a file, just that file
            - Images (.png, .jpg, .jpeg, .webp) → keyframes, copied to selected_keyframes/
            - Videos (.mp4, .webm, .mov) → transitions, copied to selected_transitions/
            - All imported items go to the bin for user review
            - timestamp is the starting timestamp for imported keyframes (auto-increments by 1s per keyframe)
            """
            body = self._read_json_body()
            if body is None:
                return

            source_path = body.get("sourcePath")
            start_timestamp = body.get("timestamp", "0:00")
            if not source_path:
                return self._error(400, "BAD_REQUEST", "Missing 'sourcePath'")

            # Support both absolute paths and paths relative to work_dir
            source = Path(source_path)
            if not source.is_absolute():
                source = (work_dir / source_path).resolve()
            if not source.exists():
                return self._error(404, "NOT_FOUND", f"Source path not found: {source_path}")

            try:
                from beatlab.project import load_project, save_project
                import shutil
                from datetime import datetime, timezone

                project_dir = work_dir / project_name
                parsed = load_project(project_dir)
                if parsed.get("_format") == "empty":
                    parsed = {"meta": {"title": project_name, "fps": 24, "resolution": [1920, 1080]}, "keyframes": [], "transitions": [], "_format": "legacy", "_work_dir": str(project_dir)}

                keyframes = parsed.get("keyframes", [])
                transitions = parsed.get("transitions", [])
                kf_bin = parsed.get("bin", [])
                tr_bin = parsed.get("transition_bin", [])

                # Find next IDs
                all_kf_ids = [kf.get("id", "") for kf in keyframes + kf_bin]
                max_kf = max((int(m.group(1)) for kid in all_kf_ids if (m := __import__('re').match(r'kf_(\d+)', kid))), default=0)

                all_tr_ids = [tr.get("id", "") for tr in transitions + tr_bin]
                max_tr = max((int(m.group(1)) for tid in all_tr_ids if (m := __import__('re').match(r'tr_(\d+)', tid))), default=0)

                # Collect files
                IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}
                VIDEO_EXTS = {'.mp4', '.webm', '.mov'}

                files = []
                if source.is_dir():
                    files = sorted(source.iterdir())
                else:
                    files = [source]

                now = datetime.now(timezone.utc).isoformat()
                selected_kf_dir = project_dir / "selected_keyframes"
                selected_kf_dir.mkdir(parents=True, exist_ok=True)
                selected_tr_dir = project_dir / "selected_transitions"
                selected_tr_dir.mkdir(parents=True, exist_ok=True)

                # Parse starting timestamp
                def parse_ts(ts):
                    parts = str(ts).split(":")
                    if len(parts) == 2:
                        return int(parts[0]) * 60 + float(parts[1])
                    return 0

                def format_ts(seconds):
                    m = int(seconds // 60)
                    s = seconds % 60
                    whole = int(s)
                    frac = s - whole
                    if frac < 0.005:
                        return f"{m}:{whole:02d}"
                    return f"{m}:{whole:02d}.{round(frac * 100):02d}"

                current_ts = parse_ts(start_timestamp)
                imported_kf = []
                imported_tr = []

                for f in files:
                    if not f.is_file():
                        continue
                    ext = f.suffix.lower()

                    if ext in IMAGE_EXTS:
                        max_kf += 1
                        kf_id = f"kf_{max_kf:03d}"
                        # Copy to selected_keyframes
                        dest = selected_kf_dir / f"{kf_id}.png"
                        if ext == '.png':
                            shutil.copy2(str(f), str(dest))
                        else:
                            # Convert to PNG via copy (browser handles format)
                            shutil.copy2(str(f), str(dest))

                        kf_entry = {
                            "id": kf_id,
                            "timestamp": format_ts(current_ts),
                            "section": "",
                            "source": str(f),
                            "prompt": f"Imported from {f.name}",
                            "context": None,
                            "candidates": [],
                            "selected": 1,
                            "deleted_at": now,
                        }
                        kf_bin.append(kf_entry)
                        imported_kf.append(kf_id)
                        current_ts += 1.0

                    elif ext in VIDEO_EXTS:
                        max_tr += 1
                        tr_id = f"tr_{max_tr:03d}"
                        # Copy to selected_transitions
                        dest = selected_tr_dir / f"{tr_id}_slot_0{ext}"
                        shutil.copy2(str(f), str(dest))

                        tr_entry = {
                            "id": tr_id,
                            "from": "",
                            "to": "",
                            "duration_seconds": 0,
                            "slots": 1,
                            "action": f"Imported from {f.name}",
                            "candidates": [],
                            "selected": [],
                            "remap": {"method": "linear", "target_duration": 0},
                            "deleted_at": now,
                        }
                        tr_bin.append(tr_entry)
                        imported_tr.append(tr_id)

                parsed["bin"] = kf_bin
                parsed["transition_bin"] = tr_bin

                save_project(parsed, project_dir)

                self._json_response({
                    "success": True,
                    "imported": {
                        "keyframes": imported_kf,
                        "transitions": imported_tr,
                    },
                    "summary": f"{len(imported_kf)} keyframe(s), {len(imported_tr)} transition(s) imported to bin",
                })
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_ls(self, project_name: str, subpath: str):
            """GET /api/projects/:name/ls?path=subdir — list directory contents."""
            project_root = (work_dir / project_name).resolve()
            target = (project_root / subpath).resolve()

            # Path traversal prevention
            if not str(target).startswith(str(project_root)):
                return self._error(403, "FORBIDDEN", "Path traversal denied")

            if not target.is_dir():
                return self._error(404, "NOT_FOUND", f"Directory not found: {subpath or '/'}")

            # Use os.scandir for fast listing (no per-file stat on network mounts)
            entries = []
            with os.scandir(target) as scanner:
                items = sorted(scanner, key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
                for entry in items:
                    is_dir = entry.is_dir(follow_symlinks=False)
                    rel = str(Path(entry.path).relative_to(project_root))
                    info: dict = {"name": entry.name, "path": rel, "isDirectory": is_dir}
                    entries.append(info)

            self._json_response(entries)

        def _handle_serve_file(self, project_name: str, file_path: str):
            """GET /api/projects/:name/files/* — serve project files with Range support and caching."""
            full_path = (work_dir / project_name / file_path).resolve()

            # Path traversal prevention
            if not str(full_path).startswith(str(work_dir.resolve())):
                return self._error(403, "FORBIDDEN", "Path traversal denied")

            if not full_path.exists():
                return self._error(404, "NOT_FOUND", f"File not found: {file_path}")

            file_stat = full_path.stat()
            file_size = file_stat.st_size
            content_type = mimetypes.guess_type(str(full_path))[0] or "application/octet-stream"

            # ETag and Last-Modified for cache validation
            from email.utils import formatdate
            etag = f'"{file_size:x}-{int(file_stat.st_mtime):x}"'
            last_modified = formatdate(file_stat.st_mtime, usegmt=True)

            # Check If-None-Match (ETag) → 304 Not Modified
            if_none_match = self.headers.get("If-None-Match")
            if if_none_match and if_none_match.strip('" ') == etag.strip('"'):
                self.send_response(304)
                self.send_header("ETag", etag)
                self._cors_headers()
                self.end_headers()
                return

            # Check If-Modified-Since → 304 Not Modified
            if_modified = self.headers.get("If-Modified-Since")
            if if_modified:
                from email.utils import parsedate_to_datetime
                try:
                    cached_time = parsedate_to_datetime(if_modified).timestamp()
                    if file_stat.st_mtime <= cached_time:
                        self.send_response(304)
                        self.send_header("ETag", etag)
                        self._cors_headers()
                        self.end_headers()
                        return
                except Exception:
                    pass

            def _cache_headers():
                self.send_header("Cache-Control", "public, max-age=3600, immutable")
                self.send_header("ETag", etag)
                self.send_header("Last-Modified", last_modified)

            # Handle Range requests for audio/video streaming
            range_header = self.headers.get("Range")
            if range_header:
                m = re.match(r"bytes=(\d+)-(\d*)", range_header)
                if m:
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else file_size - 1
                    end = min(end, file_size - 1)
                    length = end - start + 1

                    self.send_response(206)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(length))
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.send_header("Accept-Ranges", "bytes")
                    _cache_headers()
                    self._cors_headers()
                    self.end_headers()

                    try:
                        with open(full_path, "rb") as f:
                            f.seek(start)
                            self.wfile.write(f.read(length))
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return

            # Full file response
            try:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(file_size))
                self.send_header("Accept-Ranges", "bytes")
                _cache_headers()
                self._cors_headers()
                self.end_headers()

                with open(full_path, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        # ── Narrative / Timeline Handlers ─────────────────────────

        def _handle_get_narrative(self, project_name: str):
            """GET /api/projects/:name/narrative — return sections from narrative.yaml."""
            from beatlab.project import load_project
            project_dir = work_dir / project_name
            if not project_dir.is_dir():
                return self._error(404, "NOT_FOUND", f"Project not found: {project_name}")
            data = load_project(project_dir)
            self._json_response({"sections": data.get("sections", [])})

        def _handle_update_narrative(self, project_name: str):
            """POST /api/projects/:name/narrative — update sections."""
            from beatlab.project import load_project, save_project
            body = self._read_json_body()
            if body is None:
                return
            project_dir = work_dir / project_name
            if not project_dir.is_dir():
                return self._error(404, "NOT_FOUND", f"Project not found: {project_name}")
            data = load_project(project_dir)
            data["sections"] = body.get("sections", data.get("sections", []))
            save_project(data, project_dir)
            self._json_response({"success": True, "sections": len(data["sections"])})

        def _handle_get_timelines(self, project_name: str):
            """GET /api/projects/:name/timelines — list available timelines."""
            from beatlab.project import get_timelines
            project_dir = work_dir / project_name
            if not project_dir.is_dir():
                return self._error(404, "NOT_FOUND", f"Project not found: {project_name}")
            try:
                result = get_timelines(project_dir)
                self._json_response(result)
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_timeline_switch(self, project_name: str):
            """POST /api/projects/:name/timeline/switch — switch active timeline."""
            from beatlab.project import switch_timeline
            body = self._read_json_body()
            if body is None:
                return
            name = body.get("name")
            if not name:
                return self._error(400, "BAD_REQUEST", "Missing 'name'")
            project_dir = work_dir / project_name
            if not project_dir.is_dir():
                return self._error(404, "NOT_FOUND", f"Project not found: {project_name}")
            try:
                switch_timeline(project_dir, name)
                self._json_response({"success": True, "active": name})
            except ValueError as e:
                self._error(404, "NOT_FOUND", str(e))

        def _handle_timeline_import(self, project_name: str):
            """POST /api/projects/:name/timeline/import — import timeline from source."""
            from beatlab.project import import_timeline
            body = self._read_json_body()
            if body is None:
                return
            source_path = body.get("sourcePath")
            timeline_name = body.get("timelineName")
            if not source_path:
                return self._error(400, "BAD_REQUEST", "Missing 'sourcePath'")
            project_dir = work_dir / project_name
            if not project_dir.is_dir():
                return self._error(404, "NOT_FOUND", f"Project not found: {project_name}")
            try:
                result = import_timeline(project_dir, source_path, timeline_name)
                self._json_response({"success": True, **result})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_timeline_create(self, project_name: str):
            """POST /api/projects/:name/timeline/create — create new timeline."""
            from beatlab.project import create_timeline
            body = self._read_json_body()
            if body is None:
                return
            name = body.get("name")
            copy_from = body.get("copyFrom")
            if not name:
                return self._error(400, "BAD_REQUEST", "Missing 'name'")
            project_dir = work_dir / project_name
            if not project_dir.is_dir():
                return self._error(404, "NOT_FOUND", f"Project not found: {project_name}")
            try:
                create_timeline(project_dir, name, copy_from)
                self._json_response({"success": True, "name": name})
            except ValueError as e:
                self._error(400, "BAD_REQUEST", str(e))

        # ── Git Version Handlers ─────────────────────────────────

        def _ensure_git_repo(self, project_dir: Path):
            """Lazily initialize git repo in project directory."""
            if not (project_dir / ".git").exists():
                import subprocess as sp
                sp.run(["git", "init"], cwd=project_dir, capture_output=True, check=True)
                sp.run(["git", "add", "-A"], cwd=project_dir, capture_output=True, check=True)
                sp.run(["git", "commit", "-m", "Initial project state"], cwd=project_dir, capture_output=True, check=True)

        def _handle_version_commit(self, project_name: str):
            """POST /api/projects/:name/version/commit"""
            import subprocess as sp
            body = self._read_json_body()
            if body is None:
                return
            message = body.get("message", "Save")
            project_dir = work_dir / project_name
            if not project_dir.is_dir():
                return self._error(404, "NOT_FOUND", f"Project not found: {project_name}")

            self._ensure_git_repo(project_dir)

            # Stage all changes
            sp.run(["git", "add", "-A"], cwd=project_dir, capture_output=True, check=True)

            # Check if there's anything to commit
            status = sp.run(["git", "status", "--porcelain"], cwd=project_dir, capture_output=True, text=True)
            if not status.stdout.strip():
                return self._json_response({"success": True, "noChanges": True})

            # Commit
            result = sp.run(["git", "commit", "-m", message], cwd=project_dir, capture_output=True, text=True)
            if result.returncode != 0:
                return self._error(500, "GIT_ERROR", result.stderr.strip())

            # Get SHA
            sha = sp.run(["git", "rev-parse", "--short", "HEAD"], cwd=project_dir, capture_output=True, text=True)
            self._json_response({"success": True, "sha": sha.stdout.strip(), "message": message})

        def _handle_version_history(self, project_name: str):
            """GET /api/projects/:name/version/history"""
            import subprocess as sp
            project_dir = work_dir / project_name
            if not project_dir.is_dir():
                return self._error(404, "NOT_FOUND", f"Project not found: {project_name}")

            self._ensure_git_repo(project_dir)

            # Parse limit from query
            parsed = urlparse(self.path)
            limit = 20
            if parsed.query:
                for param in parsed.query.split("&"):
                    if param.startswith("limit="):
                        try:
                            limit = int(param[6:])
                        except ValueError:
                            pass

            # Get log
            log = sp.run(
                ["git", "log", f"--max-count={limit}", "--format=%H|%h|%s|%aI"],
                cwd=project_dir, capture_output=True, text=True,
            )
            commits = []
            for line in log.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|", 3)
                if len(parts) >= 4:
                    commits.append({
                        "sha": parts[1],
                        "fullSha": parts[0],
                        "message": parts[2],
                        "date": parts[3],
                    })

            # Get current branch
            branch = sp.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=project_dir, capture_output=True, text=True,
            )

            # Get all branches
            branches_result = sp.run(
                ["git", "branch", "--list", "--format=%(refname:short)"],
                cwd=project_dir, capture_output=True, text=True,
            )
            branches = [b.strip() for b in branches_result.stdout.strip().split("\n") if b.strip()]

            self._json_response({
                "commits": commits,
                "branch": branch.stdout.strip(),
                "branches": branches,
            })

        def _handle_version_checkout(self, project_name: str):
            """POST /api/projects/:name/version/checkout — restore to commit as new commit."""
            import subprocess as sp
            body = self._read_json_body()
            if body is None:
                return
            sha = body.get("sha")
            if not sha:
                return self._error(400, "BAD_REQUEST", "Missing 'sha'")

            project_dir = work_dir / project_name
            if not project_dir.is_dir():
                return self._error(404, "NOT_FOUND", f"Project not found: {project_name}")

            self._ensure_git_repo(project_dir)

            # Get original commit message
            msg_result = sp.run(
                ["git", "log", "-1", "--format=%s", sha],
                cwd=project_dir, capture_output=True, text=True,
            )
            original_msg = msg_result.stdout.strip() if msg_result.returncode == 0 else "unknown"

            # Restore files from that commit
            result = sp.run(
                ["git", "checkout", sha, "--", "."],
                cwd=project_dir, capture_output=True, text=True,
            )
            if result.returncode != 0:
                return self._error(500, "GIT_ERROR", result.stderr.strip())

            # Commit the restoration as a new commit
            sp.run(["git", "add", "-A"], cwd=project_dir, capture_output=True, check=True)
            restore_msg = f"Restored to: {original_msg}"
            sp.run(["git", "commit", "-m", restore_msg], cwd=project_dir, capture_output=True, text=True)

            new_sha = sp.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=project_dir, capture_output=True, text=True,
            )
            self._json_response({"success": True, "sha": new_sha.stdout.strip(), "message": restore_msg})

        def _handle_version_branch(self, project_name: str):
            """POST /api/projects/:name/version/branch — create or switch branch."""
            import subprocess as sp
            body = self._read_json_body()
            if body is None:
                return
            name = body.get("name")
            create = body.get("create", False)
            if not name:
                return self._error(400, "BAD_REQUEST", "Missing 'name'")

            project_dir = work_dir / project_name
            if not project_dir.is_dir():
                return self._error(404, "NOT_FOUND", f"Project not found: {project_name}")

            self._ensure_git_repo(project_dir)

            if create:
                result = sp.run(
                    ["git", "checkout", "-b", name],
                    cwd=project_dir, capture_output=True, text=True,
                )
            else:
                result = sp.run(
                    ["git", "checkout", name],
                    cwd=project_dir, capture_output=True, text=True,
                )

            if result.returncode != 0:
                return self._error(500, "GIT_ERROR", result.stderr.strip())

            self._json_response({"success": True, "branch": name})

        def _handle_version_diff(self, project_name: str):
            """GET /api/projects/:name/version/diff — show uncommitted changes."""
            import subprocess as sp
            project_dir = work_dir / project_name
            if not project_dir.is_dir():
                return self._error(404, "NOT_FOUND", f"Project not found: {project_name}")

            self._ensure_git_repo(project_dir)

            # Parse optional from/to query params
            parsed = urlparse(self.path)
            from_ref = None
            to_ref = None
            if parsed.query:
                for param in parsed.query.split("&"):
                    if param.startswith("from="):
                        from_ref = unquote(param[5:])
                    elif param.startswith("to="):
                        to_ref = unquote(param[3:])

            if from_ref and to_ref:
                # Diff between two commits
                result = sp.run(
                    ["git", "diff", "--name-status", from_ref, to_ref],
                    cwd=project_dir, capture_output=True, text=True,
                )
            else:
                # Uncommitted changes
                result = sp.run(
                    ["git", "status", "--porcelain"],
                    cwd=project_dir, capture_output=True, text=True,
                )

            files = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                if from_ref and to_ref:
                    # git diff --name-status format: "M\tfile.txt"
                    parts = line.split("\t", 1)
                    status_map = {"M": "modified", "A": "added", "D": "deleted", "R": "renamed"}
                    status = status_map.get(parts[0].strip(), parts[0].strip())
                    filepath = parts[1] if len(parts) > 1 else ""
                else:
                    # git status --porcelain format: "XY file.txt"
                    status_code = line[:2].strip()
                    filepath = line[3:].strip()
                    status_map = {"M": "modified", "A": "added", "D": "deleted", "??": "untracked", "R": "renamed"}
                    status = status_map.get(status_code, status_code)

                is_binary = any(filepath.endswith(ext) for ext in
                                (".png", ".jpg", ".jpeg", ".mp4", ".wav", ".mp3", ".zip"))
                files.append({"path": filepath, "status": status, "binary": is_binary})

            self._json_response({"files": files, "hasChanges": len(files) > 0})

        def _handle_version_delete_branch(self, project_name: str):
            """POST /api/projects/:name/version/delete-branch"""
            import subprocess as sp
            body = self._read_json_body()
            if body is None:
                return
            name = body.get("name")
            if not name:
                return self._error(400, "BAD_REQUEST", "Missing 'name'")

            project_dir = work_dir / project_name
            if not project_dir.is_dir():
                return self._error(404, "NOT_FOUND", f"Project not found: {project_name}")

            self._ensure_git_repo(project_dir)

            # Check not current branch
            current = sp.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=project_dir, capture_output=True, text=True,
            )
            if current.stdout.strip() == name:
                return self._error(400, "BAD_REQUEST", f"Cannot delete current branch: {name}")

            result = sp.run(
                ["git", "branch", "-D", name],
                cwd=project_dir, capture_output=True, text=True,
            )
            if result.returncode != 0:
                return self._error(500, "GIT_ERROR", result.stderr.strip())

            self._json_response({"success": True})

        # ── Helpers ──────────────────────────────────────────────

        def _get_yaml_path(self, project_name: str) -> Path | None:
            """Get the narrative_keyframes.yaml path for a project.

            For handlers that call render/narrative.py functions (which expect
            the legacy YAML path). Returns the path if it exists, None otherwise.
            """
            project_dir = work_dir / project_name
            legacy = project_dir / "narrative_keyframes.yaml"
            if legacy.exists():
                return legacy
            return None

        def _require_yaml_path(self, project_name: str) -> Path | None:
            """Get YAML path or send 404 error. Returns None if error was sent."""
            path = self._get_yaml_path(project_name)
            if path is None:
                self._error(404, "NOT_FOUND", "No narrative_keyframes.yaml found")
                return None
            return path

        def _has_project_yaml(self, project_name: str) -> bool:
            """Check if a project has any YAML data (split or legacy)."""
            project_dir = work_dir / project_name
            return (
                (project_dir / "narrative_keyframes.yaml").exists()
                or (project_dir / "timeline.yaml").exists()
            )

        def _read_json_body(self) -> dict | None:
            """Read and parse JSON body from request."""
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                self._error(400, "BAD_REQUEST", "Empty body")
                return None
            try:
                body = self.rfile.read(length)
                return json.loads(body)
            except json.JSONDecodeError as e:
                self._error(400, "BAD_REQUEST", f"Invalid JSON: {e}")
                return None

        def _json_response(self, obj, status: int = 200):
            """Send a JSON response."""
            data = json.dumps(obj).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self._cors_headers()
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Client disconnected before response was sent

        def _error(self, status: int, code: str, message: str):
            """Send a JSON error response."""
            self._json_response({"error": message, "code": code}, status=status)

        def _cors_headers(self):
            """Add CORS headers for cross-origin requests from the synthesizer."""
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def log_message(self, format, *args):
            # Quiet default logging — we use _log() for important events
            pass

    return SceneCraftHandler


def run_server(host: str = "0.0.0.0", port: int = 8888, work_dir: str | None = None):
    """Start the SceneCraft REST API server."""
    wd = Path(work_dir) if work_dir else Path.cwd() / ".beatlab_work"
    if not wd.exists():
        print(f"Work directory not found: {wd}", file=sys.stderr)
        print("Run from the project root or specify --work-dir.", file=sys.stderr)
        raise SystemExit(1)

    handler = make_handler(wd)
    server = HTTPServer((host, port), handler)

    # Start WebSocket server for real-time job progress
    ws_port = port + 1
    from beatlab.ws_server import start_ws_server, FolderWatcher
    import beatlab.ws_server as _ws_mod
    _ws_mod.folder_watcher = FolderWatcher(wd)
    start_ws_server(host, ws_port)

    # Restore persisted folder watches from project YAMLs
    from beatlab.project import load_project
    restored_watches = 0
    for project_dir in wd.iterdir():
        if not project_dir.is_dir():
            continue
        try:
            parsed = load_project(project_dir)
            if parsed.get("_format") == "empty":
                continue
            for folder_path in parsed.get("watched_folders", []):
                try:
                    _ws_mod.folder_watcher.add_watch(project_dir.name, folder_path)
                    restored_watches += 1
                except Exception as e:
                    _log(f"  Warning: could not restore watch {folder_path} for {project_dir.name}: {e}")
        except Exception:
            pass

    _log(f"SceneCraft API server running at http://{host}:{port}")
    _log(f"SceneCraft WebSocket server at ws://{host}:{ws_port}")
    _log(f"  Work dir: {wd}")
    _log(f"  Projects: {len([d for d in wd.iterdir() if d.is_dir()])}")
    if restored_watches:
        _log(f"  Restored watches: {restored_watches}")
    _log("")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("Shutting down.")
        server.shutdown()
