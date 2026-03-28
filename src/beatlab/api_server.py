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

            # POST /api/projects/:name/generate-transition-action
            m = re.match(r"^/api/projects/([^/]+)/generate-transition-action$", path)
            if m:
                return self._handle_generate_transition_action(m.group(1))

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

            entries = []
            for entry in sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
                rel = str(entry.resolve().relative_to(work_dir.resolve()))
                info = {"name": entry.name, "path": rel, "isDirectory": entry.is_dir()}
                if not entry.is_dir():
                    ext = entry.suffix.lower()
                    info["size"] = entry.stat().st_size
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
                has_yaml = "narrative_keyframes.yaml" in filenames
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
            project_dir = work_dir / project_name
            if not project_dir.is_dir():
                return self._error(404, "NOT_FOUND", f"Project not found: {project_name}")

            yaml_path = project_dir / "narrative_keyframes.yaml"
            if not yaml_path.exists():
                return self._json_response({
                    "meta": {"title": project_name, "fps": 24, "resolution": [1920, 1080]},
                    "keyframes": [],
                    "audioFile": None,
                    "projectName": project_name,
                })

            import yaml as pyyaml
            with open(yaml_path) as f:
                parsed = pyyaml.safe_load(f)

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

            yaml_path = work_dir / project_name / "narrative_keyframes.yaml"
            if not yaml_path.exists():
                return self._error(404, "NOT_FOUND", "No narrative_keyframes.yaml found")

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

            yaml_path = work_dir / project_name / "narrative_keyframes.yaml"
            if not yaml_path.exists():
                return self._error(404, "NOT_FOUND", "No narrative_keyframes.yaml found")

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

            yaml_path = work_dir / project_name / "narrative_keyframes.yaml"
            if not yaml_path.exists():
                return self._error(404, "NOT_FOUND", "No narrative_keyframes.yaml found")

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

            yaml_path = work_dir / project_name / "narrative_keyframes.yaml"
            if not yaml_path.exists():
                return self._error(404, "NOT_FOUND", "No narrative_keyframes.yaml found")

            try:
                # Read, find keyframe, update timestamp, write back
                content = yaml_path.read_text()
                id_pattern = f"- id: {kf_id}"
                idx = content.find(id_pattern)
                if idx == -1:
                    return self._error(404, "NOT_FOUND", f"Keyframe {kf_id} not found")

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
            yaml_path = work_dir / project_name / "narrative_keyframes.yaml"
            if not yaml_path.exists():
                return self._json_response({"bin": []})

            import yaml as pyyaml
            with open(yaml_path) as f:
                parsed = pyyaml.safe_load(f)

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

        def _handle_delete_keyframe(self, project_name: str):
            """POST /api/projects/:name/delete-keyframe — soft-delete a keyframe to bin."""
            body = self._read_json_body()
            if body is None:
                return

            kf_id = body.get("keyframeId")
            if not kf_id:
                return self._error(400, "BAD_REQUEST", "Missing 'keyframeId'")

            yaml_path = work_dir / project_name / "narrative_keyframes.yaml"
            if not yaml_path.exists():
                return self._error(404, "NOT_FOUND", "No narrative_keyframes.yaml found")

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

            yaml_path = work_dir / project_name / "narrative_keyframes.yaml"
            if not yaml_path.exists():
                return self._error(404, "NOT_FOUND", "No narrative_keyframes.yaml found")

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

            yaml_path = work_dir / project_name / "narrative_keyframes.yaml"
            if not yaml_path.exists():
                return self._error(404, "NOT_FOUND", "No narrative_keyframes.yaml found")

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

            yaml_path = work_dir / project_name / "narrative_keyframes.yaml"
            if not yaml_path.exists():
                return self._error(404, "NOT_FOUND", "No narrative_keyframes.yaml found")

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
            use_global = body.get("useGlobalPrompt")
            if not tr_id:
                return self._error(400, "BAD_REQUEST", "Missing 'transitionId'")

            yaml_path = work_dir / project_name / "narrative_keyframes.yaml"
            if not yaml_path.exists():
                return self._error(404, "NOT_FOUND", "No narrative_keyframes.yaml found")

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
                if use_global is not None:
                    tr["use_global_prompt"] = use_global

                parsed["transitions"] = transitions
                with open(yaml_path, "w") as f:
                    pyyaml.dump(parsed, f, default_flow_style=False, allow_unicode=True, width=1000)

                self._json_response({"success": True})
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

            yaml_path = work_dir / project_name / "narrative_keyframes.yaml"
            if not yaml_path.exists():
                return self._error(404, "NOT_FOUND", "No narrative_keyframes.yaml found")

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
                    {"type": "text", "text": f"\nTransition duration: {tr['duration_seconds']}s, {tr['slots']} slot(s).\n\n"
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

                with open(yaml_path, "w") as f:
                    pyyaml.dump(parsed, f, default_flow_style=False, allow_unicode=True, width=1000)

                self._json_response({"success": True, "action": action})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_generate_keyframe_candidates(self, project_name: str):
            """POST /api/projects/:name/generate-keyframe-candidates — async Imagen generation with WebSocket progress."""
            body = self._read_json_body()
            if body is None:
                return

            kf_id = body.get("keyframeId")
            count = body.get("count")
            if not kf_id:
                return self._error(400, "BAD_REQUEST", "Missing 'keyframeId'")

            yaml_path = work_dir / project_name / "narrative_keyframes.yaml"
            if not yaml_path.exists():
                return self._error(404, "NOT_FOUND", "No narrative_keyframes.yaml found")

            from beatlab.ws_server import job_manager
            n_candidates = count or 4
            job_id = job_manager.create_job("keyframe_candidates", total=n_candidates, meta={"keyframeId": kf_id, "project": project_name})

            def _run():
                try:
                    from beatlab.render.narrative import generate_keyframe_candidates
                    generate_keyframe_candidates(
                        str(yaml_path),
                        vertex=False,
                        candidates_per_slot=count,
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
            count = body.get("count")
            if not tr_id:
                return self._error(400, "BAD_REQUEST", "Missing 'transitionId'")

            yaml_path = work_dir / project_name / "narrative_keyframes.yaml"
            if not yaml_path.exists():
                return self._error(404, "NOT_FOUND", "No narrative_keyframes.yaml found")

            from beatlab.ws_server import job_manager
            job_id = job_manager.create_job("transition_candidates", total=0, meta={"transitionId": tr_id, "project": project_name})

            def _run():
                try:
                    from beatlab.render.narrative import generate_transition_candidates
                    generate_transition_candidates(
                        str(yaml_path),
                        vertex=False,
                        candidates_per_slot=count,
                        segment_filter={tr_id},
                    )

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
                    job_manager.fail_job(job_id, str(e))

            import threading
            threading.Thread(target=_run, daemon=True).start()
            self._json_response({"jobId": job_id, "transitionId": tr_id})

        def _handle_update_meta(self, project_name: str):
            """POST /api/projects/:name/update-meta — update project meta fields."""
            body = self._read_json_body()
            if body is None:
                return

            yaml_path = work_dir / project_name / "narrative_keyframes.yaml"
            if not yaml_path.exists():
                return self._error(404, "NOT_FOUND", "No narrative_keyframes.yaml found")

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
                import yaml as pyyaml
                import shutil
                from datetime import datetime, timezone

                yaml_path = work_dir / project_name / "narrative_keyframes.yaml"
                project_dir = work_dir / project_name

                # Load or create YAML
                if yaml_path.exists():
                    with open(yaml_path) as f:
                        parsed = pyyaml.safe_load(f) or {}
                else:
                    parsed = {"meta": {"title": project_name, "fps": 24, "resolution": [1920, 1080]}, "keyframes": [], "transitions": []}

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

                with open(yaml_path, "w") as f:
                    pyyaml.dump(parsed, f, default_flow_style=False, allow_unicode=True, width=1000)

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

            entries = []
            for entry in sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
                rel = str(entry.resolve().relative_to(project_root))
                info = {"name": entry.name, "path": rel, "isDirectory": entry.is_dir()}
                if not entry.is_dir():
                    info["size"] = entry.stat().st_size
                entries.append(info)

            self._json_response(entries)

        def _handle_serve_file(self, project_name: str, file_path: str):
            """GET /api/projects/:name/files/* — serve project files with Range support."""
            full_path = (work_dir / project_name / file_path).resolve()

            # Path traversal prevention
            if not str(full_path).startswith(str(work_dir.resolve())):
                return self._error(403, "FORBIDDEN", "Path traversal denied")

            if not full_path.exists():
                return self._error(404, "NOT_FOUND", f"File not found: {file_path}")

            file_size = full_path.stat().st_size
            content_type = mimetypes.guess_type(str(full_path))[0] or "application/octet-stream"

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
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self._cors_headers()
            self.end_headers()

            try:
                with open(full_path, "rb") as f:
                    # Stream in 64KB chunks to avoid loading large files into memory
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        # ── Helpers ──────────────────────────────────────────────

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
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self._cors_headers()
            self.end_headers()
            self.wfile.write(data)

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
    from beatlab.ws_server import start_ws_server
    start_ws_server(host, ws_port)

    _log(f"SceneCraft API server running at http://{host}:{port}")
    _log(f"SceneCraft WebSocket server at ws://{host}:{ws_port}")
    _log(f"  Work dir: {wd}")
    _log(f"  Projects: {len([d for d in wd.iterdir() if d.is_dir()])}")
    _log("")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("Shutting down.")
        server.shutdown()
