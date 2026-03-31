"""SceneCraft REST API server — exposes pipeline operations for the synthesizer frontend."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, unquote


def _log(msg: str):
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def make_handler(work_dir: Path):
    """Create a request handler class with the work_dir baked in."""
    import threading
    _project_locks: dict[str, threading.Lock] = {}
    _locks_lock = threading.Lock()

    def _get_project_lock(project_name: str) -> threading.Lock:
        """Get a per-project lock for serializing YAML and git operations."""
        with _locks_lock:
            if project_name not in _project_locks:
                _project_locks[project_name] = threading.Lock()
            return _project_locks[project_name]

    def _yaml_lock(project_name: str):
        """Context manager for serializing YAML read-modify-write operations on a project."""
        return _get_project_lock(project_name)

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

            # GET /api/projects/:name/bench
            m = re.match(r"^/api/projects/([^/]+)/bench$", path)
            if m:
                return self._handle_get_bench(m.group(1))

            # GET /api/projects/:name/section-settings?section=...
            m = re.match(r"^/api/projects/([^/]+)/section-settings$", path)
            if m:
                return self._handle_get_section_settings(m.group(1))

            # GET /api/projects/:name/audio-intelligence
            m = re.match(r"^/api/projects/([^/]+)/audio-intelligence$", path)
            if m:
                return self._handle_get_audio_intelligence(m.group(1))

            # GET /api/projects/:name/descriptions
            m = re.match(r"^/api/projects/([^/]+)/descriptions$", path)
            if m:
                return self._handle_get_descriptions(m.group(1))

            # GET /api/projects/:name/staging/:stagingId
            m = re.match(r"^/api/projects/([^/]+)/staging/([^/]+)$", path)
            if m:
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None:
                    return
                staging_dir = project_dir / "staging" / m.group(2)
                if not staging_dir.is_dir():
                    return self._json_response({"candidates": []})
                candidates = sorted([
                    f"staging/{m.group(2)}/{f.name}"
                    for f in staging_dir.glob("v*.png")
                ], key=lambda p: int(p.rsplit("v", 1)[-1].split(".")[0]))
                return self._json_response({"candidates": candidates})

            # GET /api/projects/:name/download-preview?start=X&end=Y
            m = re.match(r"^/api/projects/([^/]+)/download-preview$", path)
            if m:
                return self._handle_download_preview(m.group(1))

            # GET /api/projects/:name/pool
            m = re.match(r"^/api/projects/([^/]+)/pool$", path)
            if m:
                return self._handle_get_pool(m.group(1))

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

            # Structural timeline mutations need a per-project lock to prevent
            # read-modify-write races (e.g., two concurrent deletes creating duplicate bridges).
            # Simple field updates (prompt, action, remap, select) and long-running ops don't need it.
            _proj_match = re.match(r"^/api/projects/([^/]+)/", path)
            _proj_name = _proj_match.group(1) if _proj_match else None
            _route_name = path.rsplit("/", 1)[-1] if "/" in path else ""
            _structural_routes = {
                "add-keyframe", "duplicate-keyframe", "delete-keyframe", "restore-keyframe",
                "delete-transition", "restore-transition",
                "split-transition", "insert-pool-item",
                "version/commit",
            }
            _use_lock = _proj_name and _route_name in _structural_routes

            if _use_lock:
                _get_project_lock(_proj_name).acquire()
            try:
                result = self._do_POST(path)
                # Validate timeline after structural mutations
                if _use_lock and _proj_name:
                    try:
                        from beatlab.db import validate_timeline
                        project_dir = work_dir / _proj_name
                        if (project_dir / "project.db").exists():
                            warnings = validate_timeline(project_dir)
                            if warnings:
                                _log(f"⚠ Timeline validation ({_route_name}): {len(warnings)} issues")
                                for w in warnings[:10]:
                                    _log(f"  - {w}")
                                # Send warnings via WebSocket
                                try:
                                    from beatlab.ws_server import job_manager as _jm
                                    _jm._broadcast({"type": "timeline_warning", "route": _route_name, "warnings": warnings})
                                except Exception:
                                    pass
                    except Exception as ve:
                        _log(f"  Validation error: {ve}")
                return result
            finally:
                if _use_lock:
                    _get_project_lock(_proj_name).release()

        def _do_POST(self, path):

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

            # POST /api/projects/:name/update-prompt
            m = re.match(r"^/api/projects/([^/]+)/update-prompt$", path)
            if m:
                return self._handle_update_prompt(m.group(1))

            # POST /api/projects/:name/add-keyframe
            m = re.match(r"^/api/projects/([^/]+)/add-keyframe$", path)
            if m:
                return self._handle_add_keyframe(m.group(1))

            # POST /api/projects/:name/duplicate-keyframe
            m = re.match(r"^/api/projects/([^/]+)/duplicate-keyframe$", path)
            if m:
                return self._handle_duplicate_keyframe(m.group(1))

            # POST /api/projects/:name/delete-keyframe
            m = re.match(r"^/api/projects/([^/]+)/delete-keyframe$", path)
            if m:
                return self._handle_delete_keyframe(m.group(1))

            # POST /api/projects/:name/restore-keyframe
            m = re.match(r"^/api/projects/([^/]+)/restore-keyframe$", path)
            if m:
                return self._handle_restore_keyframe(m.group(1))

            # POST /api/projects/:name/set-base-image
            m = re.match(r"^/api/projects/([^/]+)/set-base-image$", path)
            if m:
                return self._handle_set_base_image(m.group(1))

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

            # POST /api/projects/:name/enhance-transition-action
            m = re.match(r"^/api/projects/([^/]+)/enhance-transition-action$", path)
            if m:
                return self._handle_enhance_transition_action(m.group(1))

            # POST /api/projects/:name/assign-pool-video
            m = re.match(r"^/api/projects/([^/]+)/assign-pool-video$", path)
            if m:
                return self._handle_assign_pool_video(m.group(1))

            # POST /api/projects/:name/bench/add
            m = re.match(r"^/api/projects/([^/]+)/bench/add$", path)
            if m:
                return self._handle_bench_add(m.group(1))

            # POST /api/projects/:name/bench/remove
            m = re.match(r"^/api/projects/([^/]+)/bench/remove$", path)
            if m:
                return self._handle_bench_remove(m.group(1))

            # POST /api/projects/:name/split-transition
            m = re.match(r"^/api/projects/([^/]+)/split-transition$", path)
            if m:
                return self._handle_split_transition(m.group(1))

            # POST /api/projects/:name/insert-pool-item
            m = re.match(r"^/api/projects/([^/]+)/insert-pool-item$", path)
            if m:
                return self._handle_insert_pool_item(m.group(1))

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

            # POST /api/projects/:name/promote-staged-candidate
            m = re.match(r"^/api/projects/([^/]+)/promote-staged-candidate$", path)
            if m:
                return self._handle_promote_staged_candidate(m.group(1))

            # POST /api/projects/:name/generate-staged-candidate
            m = re.match(r"^/api/projects/([^/]+)/generate-staged-candidate$", path)
            if m:
                return self._handle_generate_staged_candidate(m.group(1))

            # POST /api/projects/:name/suggest-keyframe-prompts
            m = re.match(r"^/api/projects/([^/]+)/suggest-keyframe-prompts$", path)
            if m:
                return self._handle_suggest_keyframe_prompts(m.group(1))

            # POST /api/projects/:name/enhance-keyframe-prompt
            m = re.match(r"^/api/projects/([^/]+)/enhance-keyframe-prompt$", path)
            if m:
                return self._handle_enhance_keyframe_prompt(m.group(1))

            # POST /api/projects/:name/section-settings
            m = re.match(r"^/api/projects/([^/]+)/section-settings$", path)
            if m:
                return self._handle_section_settings(m.group(1))

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
            project_dir = self._get_project_dir(project_name)
            if project_dir is None:
                return self._error(404, "NOT_FOUND", f"Project not found: {project_name}")

            from beatlab.db import get_meta, get_keyframes as db_get_keyframes, get_transitions as db_get_transitions

            # Check if DB exists (auto-import happens in _get_project_dir)
            if not (project_dir / "project.db").exists():
                return self._json_response({
                    "meta": {"title": project_name, "fps": 24, "resolution": [1920, 1080]},
                    "keyframes": [], "transitions": [], "audioFile": None, "projectName": project_name,
                })

            meta = get_meta(project_dir)
            result_meta = {
                "title": meta.get("title", project_name),
                "fps": meta.get("fps", 24),
                "resolution": meta.get("resolution", [1920, 1080]),
                "motionPrompt": meta.get("motion_prompt", ""),
                "defaultTransitionPrompt": meta.get("default_transition_prompt", "Smooth cinematic transition"),
            }

            keyframes = []
            for kf in db_get_keyframes(project_dir):
                kf_id = kf["id"]
                img_path = project_dir / "selected_keyframes" / f"{kf_id}.png"
                has_selected = img_path.exists()

                candidates_dir = project_dir / "keyframe_candidates" / "candidates" / f"section_{kf_id}"
                candidate_files = []
                if candidates_dir.exists():
                    candidate_files = sorted([
                        f"keyframe_candidates/candidates/section_{kf_id}/{f.name}"
                        for f in candidates_dir.glob("v*.png")
                    ])

                ctx = kf.get("context")
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

            # Parse transitions from DB
            transitions = []
            tr_candidates_root = project_dir / "transition_candidates"
            for tr in db_get_transitions(project_dir):
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

                # selected: from DB it's already flattened; wrap back to list for API compat
                sel = tr.get("selected")
                selected_list = sel if isinstance(sel, list) else [sel]

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

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                import shutil
                from beatlab.db import update_keyframe
                selected_dir = project_dir / "selected_keyframes"
                selected_dir.mkdir(parents=True, exist_ok=True)

                for kf_id, variant in selections.items():
                    _log(f"select-keyframes: {kf_id} v{variant}")
                    cand_dir = project_dir / "keyframe_candidates" / "candidates" / f"section_{kf_id}"
                    source = cand_dir / f"v{variant}.png"
                    if source.exists():
                        shutil.copy2(str(source), str(selected_dir / f"{kf_id}.png"))
                        _log(f"  copied {source} -> {selected_dir / f'{kf_id}.png'}")
                    else:
                        _log(f"  WARNING: candidate not found: {source}")
                    update_keyframe(project_dir, kf_id, selected=variant)

                self._json_response({"success": True, "applied": len(selections)})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_select_slot_keyframes(self, project_name: str):
            """POST /api/projects/:name/select-slot-keyframes — apply slot keyframe selections."""
            body = self._read_json_body()
            if body is None:
                return

            selections = body.get("selections", {})
            if not selections:
                return self._error(400, "BAD_REQUEST", "Missing 'selections' in body")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                import shutil
                selected_dir = project_dir / "selected_slot_keyframes"
                selected_dir.mkdir(parents=True, exist_ok=True)
                slot_kf_root = project_dir / "slot_keyframe_candidates" / "candidates"

                for slot_key, variant in selections.items():
                    source = slot_kf_root / f"section_{slot_key}" / f"v{variant}.png"
                    if source.exists():
                        shutil.copy2(str(source), str(selected_dir / f"{slot_key}.png"))

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

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                import shutil
                from beatlab.db import update_transition
                selected_dir = project_dir / "selected_transitions"
                selected_dir.mkdir(parents=True, exist_ok=True)
                tr_candidates_root = project_dir / "transition_candidates"

                for key, variant in selections.items():
                    if "_slot_" in key:
                        tr_id, slot_part = key.rsplit("_slot_", 1)
                        slot_idx = int(slot_part)
                    else:
                        tr_id = key
                        slot_idx = 0

                    source = tr_candidates_root / tr_id / f"slot_{slot_idx}" / f"v{variant}.mp4"
                    dest = selected_dir / f"{tr_id}_slot_{slot_idx}.mp4"
                    if source.exists():
                        shutil.copy2(str(source), str(dest))

                    update_transition(project_dir, tr_id, selected=variant)

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

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                from beatlab.db import update_keyframe, get_transitions, update_transition

                def parse_ts(ts):
                    parts = str(ts).split(":")
                    if len(parts) == 2:
                        return int(parts[0]) * 60 + float(parts[1])
                    return float(ts) if isinstance(ts, (int, float)) else 0

                update_keyframe(project_dir, kf_id, timestamp=new_timestamp)

                # Update duration_seconds on adjacent transitions
                new_time = parse_ts(new_timestamp)
                all_trs = get_transitions(project_dir)
                for tr in all_trs:
                    if tr["from"] == kf_id or tr["to"] == kf_id:
                        other_id = tr["to"] if tr["from"] == kf_id else tr["from"]
                        from beatlab.db import get_keyframe
                        other_kf = get_keyframe(project_dir, other_id)
                        if other_kf:
                            other_time = parse_ts(other_kf["timestamp"])
                            dur = round(abs(new_time - other_time), 2)
                            update_transition(project_dir, tr["id"], duration_seconds=dur)

                self._json_response({"success": True, "keyframeId": kf_id, "newTimestamp": new_timestamp})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_update_prompt(self, project_name: str):
            """POST /api/projects/:name/update-prompt — update a keyframe's prompt."""
            body = self._read_json_body()
            if body is None:
                return

            kf_id = body.get("keyframeId")
            prompt = body.get("prompt")
            if not kf_id or prompt is None:
                return self._error(400, "BAD_REQUEST", "Missing 'keyframeId' or 'prompt'")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                from beatlab.db import update_keyframe, get_keyframe
                _log(f"update-prompt: {kf_id} prompt={repr(prompt[:60])}")
                kf = get_keyframe(project_dir, kf_id)
                if not kf:
                    _log(f"  NOT FOUND: {kf_id}")
                    return self._error(404, "NOT_FOUND", f"Keyframe {kf_id} not found")

                update_keyframe(project_dir, kf_id, prompt=prompt)
                _log(f"  saved prompt for {kf_id}")
                self._json_response({"success": True, "keyframeId": kf_id})
            except Exception as e:
                _log(f"  FAILED: {e}")
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_get_bin(self, project_name: str):
            """GET /api/projects/:name/bin — list binned (soft-deleted) keyframes and transitions."""
            project_dir = self._get_project_dir(project_name)
            if project_dir is None:
                return self._json_response({"bin": [], "transitionBin": []})

            from beatlab.db import get_binned_keyframes, get_binned_transitions

            bin_entries = []
            for kf in get_binned_keyframes(project_dir):
                img_path = project_dir / "selected_keyframes" / f"{kf['id']}.png"
                bin_entries.append({
                    "id": kf["id"],
                    "deleted_at": kf.get("deleted_at", ""),
                    "timestamp": kf.get("timestamp", "0:00"),
                    "section": kf.get("section", ""),
                    "prompt": kf.get("prompt", ""),
                    "hasSelectedImage": img_path.exists(),
                })

            transition_bin = []
            for tr in get_binned_transitions(project_dir):
                transition_bin.append({
                    "id": tr["id"],
                    "deleted_at": tr.get("deleted_at", ""),
                    "from": tr.get("from", ""),
                    "to": tr.get("to", ""),
                    "durationSeconds": tr.get("duration_seconds", 0),
                    "slots": tr.get("slots", 1),
                })

            self._json_response({"bin": bin_entries, "transitionBin": transition_bin})

        def _handle_get_descriptions(self, project_name: str):
            """GET /api/projects/:name/descriptions — parse descriptions.md into structured sections."""
            project_dir = work_dir / project_name
            desc_path = project_dir / "descriptions.md"
            if not desc_path.exists():
                return self._json_response({"sections": []})

            content = desc_path.read_text()
            sections = []
            # Split on ## Section N headers
            import re as _re
            parts = _re.split(r'^## (Section \d+.*?)$', content, flags=_re.MULTILINE)
            # parts = [preamble, header1, body1, header2, body2, ...]
            for i in range(1, len(parts), 2):
                header = parts[i].strip()
                body = parts[i + 1].strip() if i + 1 < len(parts) else ""

                # Extract section index from header: "Section 0 (verse, low_energy)"
                idx_match = _re.match(r'Section (\d+)', header)
                section_index = int(idx_match.group(1)) if idx_match else -1
                label = header

                # Extract time range from body: "**Time**: 0.0s - 16.0s"
                time_match = _re.search(r'\*\*Time\*\*:\s*([\d.]+)s\s*-\s*([\d.]+)s', body)
                start_time = float(time_match.group(1)) if time_match else 0
                end_time = float(time_match.group(2)) if time_match else 0

                sections.append({
                    "sectionIndex": section_index,
                    "label": label,
                    "startTime": start_time,
                    "endTime": end_time,
                    "content": body,
                })

            self._json_response({"sections": sections})

        def _handle_get_pool(self, project_name: str):
            """GET /api/projects/:name/pool — list pool assets (unassigned keyframe images and video segments)."""
            project_dir = work_dir / project_name
            pool_dir = project_dir / "pool"
            if not pool_dir.is_dir():
                return self._json_response({"keyframes": [], "segments": []})

            kf_dir = pool_dir / "keyframes"
            seg_dir = pool_dir / "segments"

            keyframes = []
            if kf_dir.is_dir():
                for f in sorted(kf_dir.iterdir()):
                    if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'):
                        keyframes.append({
                            "name": f.name,
                            "path": f"pool/keyframes/{f.name}",
                            "size": f.stat().st_size,
                        })

            segments = []
            if seg_dir.is_dir():
                for f in sorted(seg_dir.iterdir()):
                    if f.suffix.lower() in ('.mp4', '.webm', '.mov'):
                        segments.append({
                            "name": f.name,
                            "path": f"pool/segments/{f.name}",
                            "size": f.stat().st_size,
                        })

            self._json_response({"keyframes": keyframes, "segments": segments})

        def _handle_add_keyframe(self, project_name: str):
            """POST /api/projects/:name/add-keyframe — create a new keyframe at a given timestamp."""
            body = self._read_json_body()
            if body is None:
                return

            timestamp = body.get("timestamp")
            if not timestamp:
                return self._error(400, "BAD_REQUEST", "Missing 'timestamp'")

            # Validate timestamp format and range
            def _parse_ts_val(ts):
                parts = str(ts).split(":")
                if len(parts) == 2:
                    return int(parts[0]) * 60 + float(parts[1])
                return float(ts) if isinstance(ts, (int, float)) else -1

            ts_seconds = _parse_ts_val(timestamp)
            if ts_seconds < 0 or ts_seconds > 7200:  # max 2 hours
                return self._error(400, "BAD_REQUEST", f"Invalid timestamp: {timestamp} ({ts_seconds}s)")

            _log(f"add-keyframe: {project_name} at {timestamp} ({ts_seconds:.2f}s)")

            section = body.get("section", "")
            prompt = body.get("prompt", "")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                from beatlab.db import (
                    add_keyframe as db_add_kf, get_keyframes as db_get_kfs,
                    next_keyframe_id, next_transition_id,
                    add_transition as db_add_tr, delete_transition as db_del_tr,
                    get_transitions as db_get_trs, transaction,
                )

                def parse_ts(ts):
                    parts = str(ts).split(":")
                    if len(parts) == 2:
                        return int(parts[0]) * 60 + float(parts[1])
                    return float(ts) if isinstance(ts, (int, float)) else 0

                new_id = next_keyframe_id(project_dir)
                new_time = parse_ts(timestamp)

                new_kf = {
                    "id": new_id, "timestamp": timestamp, "section": section,
                    "source": f"selected_keyframes/{new_id}.png", "prompt": prompt,
                    "candidates": [], "selected": None,
                }
                db_add_kf(project_dir, new_kf)

                # Find timeline neighbors
                all_kfs = db_get_kfs(project_dir)
                sorted_kfs = sorted(all_kfs, key=lambda k: parse_ts(k["timestamp"]))
                new_idx = next((i for i, k in enumerate(sorted_kfs) if k["id"] == new_id), -1)
                prev_kf = sorted_kfs[new_idx - 1] if new_idx > 0 else None
                next_kf = sorted_kfs[new_idx + 1] if new_idx < len(sorted_kfs) - 1 else None

                # Wire transitions: split spanning transition or create new ones
                from datetime import datetime, timezone
                if prev_kf and next_kf:
                    all_trs = db_get_trs(project_dir)
                    old_tr = next((t for t in all_trs if t["from"] == prev_kf["id"] and t["to"] == next_kf["id"]), None)
                    if old_tr:
                        db_del_tr(project_dir, old_tr["id"], datetime.now(timezone.utc).isoformat())

                prev_time = parse_ts(prev_kf["timestamp"]) if prev_kf else None
                next_time = parse_ts(next_kf["timestamp"]) if next_kf else None

                if prev_kf:
                    dur_before = round(new_time - prev_time, 2)
                    tr1_id = next_transition_id(project_dir)
                    db_add_tr(project_dir, {
                        "id": tr1_id, "from": prev_kf["id"], "to": new_id,
                        "duration_seconds": dur_before, "slots": 1,
                        "action": "", "use_global_prompt": False, "selected": None,
                        "remap": {"method": "linear", "target_duration": dur_before},
                    })
                if next_kf:
                    dur_after = round(next_time - new_time, 2)
                    tr2_id = next_transition_id(project_dir)
                    db_add_tr(project_dir, {
                        "id": tr2_id, "from": new_id, "to": next_kf["id"],
                        "duration_seconds": dur_after, "slots": 1,
                        "action": "", "use_global_prompt": False, "selected": None,
                        "remap": {"method": "linear", "target_duration": dur_after},
                    })
                _log(f"  Wired: {prev_kf['id'] if prev_kf else '(start)'} -> {new_id} -> {next_kf['id'] if next_kf else '(end)'}")

                _log(f"  Created {new_id} at {timestamp}")
                self._json_response({"success": True, "keyframe": {"id": new_id, "timestamp": timestamp, "section": section, "prompt": prompt}})
            except Exception as e:
                _log(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_duplicate_keyframe(self, project_name: str):
            """POST /api/projects/:name/duplicate-keyframe — duplicate a keyframe with candidates at a new timestamp."""
            body = self._read_json_body()
            if body is None:
                return

            source_id = body.get("keyframeId")
            timestamp = body.get("timestamp")
            if not source_id or not timestamp:
                return self._error(400, "BAD_REQUEST", "Missing 'keyframeId' or 'timestamp'")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                from beatlab.db import (
                    add_keyframe as db_add_kf, get_keyframes as db_get_kfs,
                    get_keyframe as db_get_kf,
                    next_keyframe_id, next_transition_id,
                    add_transition as db_add_tr, delete_transition as db_del_tr,
                    get_transitions as db_get_trs,
                )
                import shutil

                source_kf = db_get_kf(project_dir, source_id)
                if not source_kf:
                    return self._error(404, "NOT_FOUND", f"Keyframe {source_id} not found")

                def parse_ts(ts):
                    parts = str(ts).split(":")
                    if len(parts) == 2:
                        return int(parts[0]) * 60 + float(parts[1])
                    return float(ts) if isinstance(ts, (int, float)) else 0

                new_id = next_keyframe_id(project_dir)
                new_time = parse_ts(timestamp)

                # Copy candidate files from disk
                src_candidates_dir = project_dir / "keyframe_candidates" / "candidates" / f"section_{source_id}"
                dst_candidates_dir = project_dir / "keyframe_candidates" / "candidates" / f"section_{new_id}"
                new_candidates = []
                if src_candidates_dir.exists():
                    dst_candidates_dir.mkdir(parents=True, exist_ok=True)
                    for f in sorted(src_candidates_dir.iterdir()):
                        if f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                            dest = dst_candidates_dir / f.name
                            shutil.copy2(str(f), str(dest))
                            new_candidates.append(f"keyframe_candidates/candidates/section_{new_id}/{f.name}")

                # If no files on disk, use DB candidates (rewrite paths to new id)
                if not new_candidates and source_kf.get("candidates"):
                    src_prefix = f"section_{source_id}/"
                    dst_prefix = f"section_{new_id}/"
                    for cand_path in source_kf["candidates"]:
                        src_file = project_dir / cand_path
                        if src_file.exists():
                            dst_path = cand_path.replace(src_prefix, dst_prefix)
                            dst_file = project_dir / dst_path
                            dst_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(str(src_file), str(dst_file))
                            new_candidates.append(dst_path)

                # Copy selected keyframe image
                src_selected = project_dir / "selected_keyframes" / f"{source_id}.png"
                dst_selected = project_dir / "selected_keyframes" / f"{new_id}.png"
                if src_selected.exists():
                    shutil.copy2(str(src_selected), str(dst_selected))

                new_kf = {
                    "id": new_id, "timestamp": timestamp,
                    "section": source_kf.get("section", ""),
                    "source": f"selected_keyframes/{new_id}.png",
                    "prompt": source_kf.get("prompt", ""),
                    "candidates": new_candidates,
                    "selected": source_kf.get("selected"),
                }
                db_add_kf(project_dir, new_kf)

                # Wire up transitions (same logic as add-keyframe)
                all_kfs = db_get_kfs(project_dir)
                sorted_kfs = sorted(all_kfs, key=lambda k: parse_ts(k["timestamp"]))
                new_idx = next((i for i, k in enumerate(sorted_kfs) if k["id"] == new_id), -1)
                prev_kf = sorted_kfs[new_idx - 1] if new_idx > 0 else None
                next_kf = sorted_kfs[new_idx + 1] if new_idx < len(sorted_kfs) - 1 else None

                from datetime import datetime, timezone
                if prev_kf and next_kf:
                    all_trs = db_get_trs(project_dir)
                    old_tr = next((t for t in all_trs if t["from"] == prev_kf["id"] and t["to"] == next_kf["id"]), None)
                    if old_tr:
                        db_del_tr(project_dir, old_tr["id"], datetime.now(timezone.utc).isoformat())

                prev_time = parse_ts(prev_kf["timestamp"]) if prev_kf else None
                next_time = parse_ts(next_kf["timestamp"]) if next_kf else None

                if prev_kf:
                    dur_before = round(new_time - prev_time, 2)
                    tr1_id = next_transition_id(project_dir)
                    db_add_tr(project_dir, {
                        "id": tr1_id, "from": prev_kf["id"], "to": new_id,
                        "duration_seconds": dur_before, "slots": 1,
                        "action": "", "use_global_prompt": False, "selected": None,
                        "remap": {"method": "linear", "target_duration": dur_before},
                    })
                if next_kf:
                    dur_after = round(next_time - new_time, 2)
                    tr2_id = next_transition_id(project_dir)
                    db_add_tr(project_dir, {
                        "id": tr2_id, "from": new_id, "to": next_kf["id"],
                        "duration_seconds": dur_after, "slots": 1,
                        "action": "", "use_global_prompt": False, "selected": None,
                        "remap": {"method": "linear", "target_duration": dur_after},
                    })

                _log(f"  Duplicated {source_id} -> {new_id} at {timestamp} ({len(new_candidates)} candidates copied)")
                self._json_response({"success": True, "keyframe": {"id": new_id, "timestamp": timestamp}})
            except Exception as e:
                _log(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_delete_keyframe(self, project_name: str):
            """POST /api/projects/:name/delete-keyframe — soft-delete a keyframe to bin."""
            body = self._read_json_body()
            if body is None:
                return

            kf_id = body.get("keyframeId")
            if not kf_id:
                return self._error(400, "BAD_REQUEST", "Missing 'keyframeId'")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                from beatlab.db import (
                    get_keyframe, delete_keyframe as db_del_kf, get_keyframes as db_get_kfs,
                    get_transitions_involving, delete_transition as db_del_tr,
                    next_transition_id, add_transition as db_add_tr,
                )
                from datetime import datetime, timezone
                import shutil as _shutil

                def parse_ts(ts):
                    parts = str(ts).split(":")
                    if len(parts) == 2:
                        return int(parts[0]) * 60 + float(parts[1])
                    return float(ts) if isinstance(ts, (int, float)) else 0

                kf = get_keyframe(project_dir, kf_id)
                if not kf:
                    return self._error(404, "NOT_FOUND", f"Keyframe {kf_id} not found")

                now = datetime.now(timezone.utc).isoformat()
                _log(f"[delete-kf] {kf_id}")

                # Soft-delete orphaned transitions, find one with video to inherit
                orphaned = get_transitions_involving(project_dir, kf_id)
                inherited_tr_id = None
                for tr in orphaned:
                    sel = tr.get("selected")
                    if sel is not None and sel != [None]:
                        inherited_tr_id = tr["id"]
                        break

                for tr in orphaned:
                    db_del_tr(project_dir, tr["id"], now)

                # Soft-delete the keyframe
                db_del_kf(project_dir, kf_id, now)

                # Bridge: find timeline neighbors by timestamp
                removed_time = parse_ts(kf["timestamp"])
                all_kfs = db_get_kfs(project_dir)
                sorted_kfs = sorted(all_kfs, key=lambda k: parse_ts(k["timestamp"]))
                prev_kf = None
                next_kf = None
                for k in sorted_kfs:
                    t = parse_ts(k["timestamp"])
                    if t < removed_time:
                        prev_kf = k
                    elif t > removed_time and next_kf is None:
                        next_kf = k

                if prev_kf and next_kf:
                    # Check if bridge already exists
                    from beatlab.db import get_transitions as db_get_trs
                    existing_bridge = next((t for t in db_get_trs(project_dir) if t["from"] == prev_kf["id"] and t["to"] == next_kf["id"]), None)
                    if existing_bridge:
                        _log(f"[delete-kf] {kf_id}: bridge {prev_kf['id']} -> {next_kf['id']} already exists as {existing_bridge['id']}")
                    else:
                        new_tr_id = next_transition_id(project_dir)
                        pt = parse_ts(prev_kf["timestamp"])
                        nt = parse_ts(next_kf["timestamp"])
                        dur = round(nt - pt, 2)

                        selected = None
                        if inherited_tr_id:
                            old_sel = project_dir / "selected_transitions" / f"{inherited_tr_id}_slot_0.mp4"
                            old_cand_dir = project_dir / "transition_candidates" / inherited_tr_id / "slot_0"
                            if old_sel.exists():
                                new_sel = project_dir / "selected_transitions" / f"{new_tr_id}_slot_0.mp4"
                                _shutil.copy2(str(old_sel), str(new_sel))
                                selected = 1
                                _log(f"[delete-kf] {kf_id}: inherited video from {inherited_tr_id}")
                            if old_cand_dir.exists():
                                new_cand_dir = project_dir / "transition_candidates" / new_tr_id / "slot_0"
                                new_cand_dir.mkdir(parents=True, exist_ok=True)
                                for f in old_cand_dir.iterdir():
                                    _shutil.copy2(str(f), str(new_cand_dir / f.name))

                        db_add_tr(project_dir, {
                            "id": new_tr_id, "from": prev_kf["id"], "to": next_kf["id"],
                            "duration_seconds": dur, "slots": 1,
                            "action": "", "use_global_prompt": False, "selected": selected,
                            "remap": {"method": "linear", "target_duration": dur},
                        })
                        _log(f"[delete-kf] {kf_id}: bridged {prev_kf['id']} -> {next_kf['id']} as {new_tr_id}")

                _log(f"[delete-kf] {kf_id}: done")
                self._json_response({"success": True, "binned": {"id": kf_id, "deleted_at": now}})
            except Exception as e:
                _log(f"[delete-kf] {kf_id}: ERROR {e}")
                import traceback
                traceback.print_exc()
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_restore_keyframe(self, project_name: str):
            """POST /api/projects/:name/restore-keyframe — restore a keyframe from bin."""
            body = self._read_json_body()
            if body is None:
                return

            kf_id = body.get("keyframeId")
            if not kf_id:
                return self._error(400, "BAD_REQUEST", "Missing 'keyframeId'")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                from beatlab.db import restore_keyframe as db_restore_kf
                db_restore_kf(project_dir, kf_id)
                self._json_response({"success": True, "keyframe": {"id": kf_id}})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_set_base_image(self, project_name: str):
            """POST /api/projects/:name/set-base-image — copy a still as the selected keyframe image."""
            body = self._read_json_body()
            if body is None:
                return

            kf_id = body.get("keyframeId")
            still_name = body.get("stillName")
            if not kf_id or not still_name:
                return self._error(400, "BAD_REQUEST", "Missing 'keyframeId' or 'stillName'")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                import shutil
                source = project_dir / "assets" / "stills" / still_name
                if not source.exists():
                    return self._error(404, "NOT_FOUND", f"Still not found: {still_name}")

                dest_dir = project_dir / "selected_keyframes"
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(source), str(dest_dir / f"{kf_id}.png"))

                from beatlab.db import update_keyframe
                update_keyframe(project_dir, kf_id, source=f"assets/stills/{still_name}")

                self._json_response({"success": True, "keyframeId": kf_id, "still": still_name})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_assign_pool_video(self, project_name: str):
            """POST /api/projects/:name/assign-pool-video — assign a pool video to an existing transition."""
            body = self._read_json_body()
            if body is None:
                return

            tr_id = body.get("transitionId")
            pool_path = body.get("poolPath")
            if not tr_id or not pool_path:
                return self._error(400, "BAD_REQUEST", "Missing 'transitionId' or 'poolPath'")

            _log(f"assign-pool-video: {project_name} {tr_id} <- {pool_path}")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                import shutil
                from beatlab.db import get_transition, update_transition

                source = project_dir / pool_path
                if not source.exists():
                    return self._error(404, "NOT_FOUND", f"Pool video not found: {pool_path}")

                tr = get_transition(project_dir, tr_id)
                if not tr:
                    return self._error(404, "NOT_FOUND", f"Transition {tr_id} not found")

                # Copy as candidate v(N+1)
                cand_dir = project_dir / "transition_candidates" / tr_id / "slot_0"
                cand_dir.mkdir(parents=True, exist_ok=True)
                existing = len(list(cand_dir.glob("v*.mp4")))
                variant = existing + 1
                shutil.copy2(str(source), str(cand_dir / f"v{variant}.mp4"))

                # Set as selected
                sel_dir = project_dir / "selected_transitions"
                sel_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(source), str(sel_dir / f"{tr_id}_slot_0.mp4"))
                update_transition(project_dir, tr_id, selected=variant)

                # Extract first frame as keyframe image (non-blocking)
                import subprocess as sp
                import threading
                from_kf_id = tr.get("from")
                if from_kf_id:
                    def _extract():
                        try:
                            sel_kf_dir = project_dir / "selected_keyframes"
                            sel_kf_dir.mkdir(parents=True, exist_ok=True)
                            sp.run(["ffmpeg", "-y", "-i", str(source), "-vframes", "1", "-q:v", "2",
                                    str(sel_kf_dir / f"{from_kf_id}.png")], capture_output=True, timeout=10)
                        except Exception:
                            pass
                    threading.Thread(target=_extract, daemon=True).start()

                _log(f"  Assigned v{variant} to {tr_id}")
                self._json_response({"success": True, "transitionId": tr_id, "variant": variant})
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_get_bench(self, project_name: str):
            """GET /api/projects/:name/bench — list benched items with usage tracking."""
            project_dir = self._get_project_dir(project_name)
            if project_dir is None:
                return self._json_response({"items": []})
            try:
                from beatlab.db import get_bench
                items = get_bench(project_dir)
                self._json_response({"items": items})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_bench_add(self, project_name: str):
            """POST /api/projects/:name/bench/add — add an item to the bench."""
            body = self._read_json_body()
            if body is None:
                return

            bench_type = body.get("type")  # "keyframe" or "transition"
            entity_id = body.get("entityId")  # the kf/tr id on the timeline
            source_path = body.get("sourcePath")  # direct path (from pool)
            label = body.get("label", "")

            if not bench_type or (not entity_id and not source_path):
                return self._error(400, "BAD_REQUEST", "Missing 'type' and ('entityId' or 'sourcePath')")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                from beatlab.db import add_to_bench, get_transition, get_keyframe

                # Resolve source path from entity if not provided directly
                if not source_path and entity_id:
                    if bench_type == "transition":
                        tr = get_transition(project_dir, entity_id)
                        if tr:
                            source_path = f"selected_transitions/{entity_id}_slot_0.mp4"
                            if not label:
                                label = f"{entity_id} ({tr['from']}→{tr['to']})"
                    elif bench_type == "keyframe":
                        kf = get_keyframe(project_dir, entity_id)
                        if kf:
                            source_path = f"selected_keyframes/{entity_id}.png"
                            if not label:
                                label = f"{entity_id} @ {kf['timestamp']}"

                if not source_path:
                    return self._error(404, "NOT_FOUND", f"Entity {entity_id} not found")

                bench_id = add_to_bench(project_dir, bench_type, source_path, label)
                _log(f"bench-add: {project_name} {bench_type} {source_path} -> {bench_id}")
                self._json_response({"success": True, "benchId": bench_id})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_bench_remove(self, project_name: str):
            """POST /api/projects/:name/bench/remove — remove an item from the bench."""
            body = self._read_json_body()
            if body is None:
                return

            bench_id = body.get("benchId")
            if not bench_id:
                return self._error(400, "BAD_REQUEST", "Missing 'benchId'")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                from beatlab.db import remove_from_bench
                remove_from_bench(project_dir, bench_id)
                self._json_response({"success": True})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_split_transition(self, project_name: str):
            """POST /api/projects/:name/split-transition — split a transition at the playhead."""
            body = self._read_json_body()
            if body is None:
                return

            tr_id = body.get("transitionId")
            split_time = body.get("splitTime")
            if not tr_id or split_time is None:
                return self._error(400, "BAD_REQUEST", "Missing 'transitionId' or 'splitTime'")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                from beatlab.db import (
                    get_transition, get_keyframe, delete_transition as db_del_tr,
                    add_keyframe as db_add_kf, add_transition as db_add_tr,
                    next_keyframe_id, next_transition_id,
                )
                from datetime import datetime, timezone
                import subprocess as sp
                import shutil
                import threading

                def parse_ts(ts):
                    parts = str(ts).split(":")
                    if len(parts) == 2:
                        return int(parts[0]) * 60 + float(parts[1])
                    return float(ts) if isinstance(ts, (int, float)) else 0

                def to_ts(s):
                    m = int(s) // 60
                    return f"{m}:{s - m*60:05.2f}"

                tr = get_transition(project_dir, tr_id)
                if not tr:
                    return self._error(404, "NOT_FOUND", f"Transition {tr_id} not found")

                from_kf = get_keyframe(project_dir, tr["from"])
                to_kf = get_keyframe(project_dir, tr["to"])
                if not from_kf or not to_kf:
                    return self._error(400, "BAD_REQUEST", "Keyframes not found")

                from_time = parse_ts(from_kf["timestamp"])
                to_time = parse_ts(to_kf["timestamp"])

                if split_time <= from_time or split_time >= to_time:
                    return self._error(400, "BAD_REQUEST", "Split time must be within transition range")

                _log(f"split-transition: {tr_id} at {split_time:.2f}s (range {from_time:.2f}-{to_time:.2f})")

                split_progress = (split_time - from_time) / (to_time - from_time)
                dur1 = round(split_time - from_time, 2)
                dur2 = round(to_time - split_time, 2)

                # Create new keyframe at split point
                new_kf_id = next_keyframe_id(project_dir)
                db_add_kf(project_dir, {
                    "id": new_kf_id, "timestamp": to_ts(split_time), "section": "",
                    "source": f"selected_keyframes/{new_kf_id}.png", "prompt": "",
                    "candidates": [], "selected": None,
                })

                # Soft-delete original transition
                now = datetime.now(timezone.utc).isoformat()
                db_del_tr(project_dir, tr_id, now)

                # Create two new transitions
                tr1_id = next_transition_id(project_dir)
                tr2_id = next_transition_id(project_dir)
                # Temp: need tr1 committed before tr2 id generation works
                db_add_tr(project_dir, {
                    "id": tr1_id, "from": tr["from"], "to": new_kf_id,
                    "duration_seconds": dur1, "slots": 1, "action": tr.get("action", ""),
                    "use_global_prompt": tr.get("use_global_prompt", False), "selected": None,
                    "remap": {"method": "linear", "target_duration": dur1},
                })
                # Re-get next ID after tr1 is committed
                tr2_id = next_transition_id(project_dir)
                db_add_tr(project_dir, {
                    "id": tr2_id, "from": new_kf_id, "to": tr["to"],
                    "duration_seconds": dur2, "slots": 1, "action": tr.get("action", ""),
                    "use_global_prompt": tr.get("use_global_prompt", False), "selected": None,
                    "remap": {"method": "linear", "target_duration": dur2},
                })

                _log(f"  Created {new_kf_id}, {tr1_id} ({dur1}s), {tr2_id} ({dur2}s)")

                # If original had a selected video, extract keyframe frame synchronously
                # then split video in background
                sel_video = project_dir / "selected_transitions" / f"{tr_id}_slot_0.mp4"
                if sel_video.exists():
                    # Probe duration and compute split point
                    probe = sp.run(
                        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(sel_video)],
                        capture_output=True, text=True,
                    )
                    video_dur = float(probe.stdout.strip()) if probe.stdout.strip() else (to_time - from_time)
                    split_at = split_progress * video_dur

                    # Extract frame at split point as new keyframe image (synchronous — fast)
                    sel_kf_dir = project_dir / "selected_keyframes"
                    sel_kf_dir.mkdir(parents=True, exist_ok=True)
                    sp.run(["ffmpeg", "-y", "-ss", str(split_at), "-i", str(sel_video),
                            "-vframes", "1", "-q:v", "2",
                            str(sel_kf_dir / f"{new_kf_id}.png")], capture_output=True, timeout=10)
                    _log(f"  Extracted keyframe frame at {split_at:.2f}s -> {new_kf_id}.png")

                    def _split_video():
                        try:
                            # Split part 1
                            cand1_dir = project_dir / "transition_candidates" / tr1_id / "slot_0"
                            cand1_dir.mkdir(parents=True, exist_ok=True)
                            part1 = cand1_dir / "v1.mp4"
                            sp.run(["ffmpeg", "-y", "-i", str(sel_video), "-t", str(split_at),
                                    "-c", "copy", str(part1)], capture_output=True, timeout=30)
                            sel1 = project_dir / "selected_transitions" / f"{tr1_id}_slot_0.mp4"
                            shutil.copy2(str(part1), str(sel1))

                            # Split part 2
                            cand2_dir = project_dir / "transition_candidates" / tr2_id / "slot_0"
                            cand2_dir.mkdir(parents=True, exist_ok=True)
                            part2 = cand2_dir / "v1.mp4"
                            sp.run(["ffmpeg", "-y", "-i", str(sel_video), "-ss", str(split_at),
                                    "-c", "copy", str(part2)], capture_output=True, timeout=30)
                            sel2 = project_dir / "selected_transitions" / f"{tr2_id}_slot_0.mp4"
                            shutil.copy2(str(part2), str(sel2))

                            # Update DB with selected variants
                            from beatlab.db import update_transition
                            update_transition(project_dir, tr1_id, selected=1)
                            update_transition(project_dir, tr2_id, selected=1)

                            _log(f"  Video split complete: {tr1_id} + {tr2_id}")
                        except Exception as e:
                            _log(f"  Warning: video split failed: {e}")

                    threading.Thread(target=_split_video, daemon=True).start()

                self._json_response({
                    "success": True, "keyframeId": new_kf_id,
                    "transition1": tr1_id, "transition2": tr2_id,
                })
            except Exception as e:
                _log(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_insert_pool_item(self, project_name: str):
            """POST /api/projects/:name/insert-pool-item — insert a pool keyframe at a given time."""
            body = self._read_json_body()
            if body is None:
                return

            item_type = body.get("type")
            pool_path = body.get("poolPath")
            at_time = body.get("atTime", 0)
            if not item_type or not pool_path:
                return self._error(400, "BAD_REQUEST", "Missing 'type' or 'poolPath'")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                import shutil
                from beatlab.db import (
                    add_keyframe as db_add_kf, get_keyframes as db_get_kfs,
                    next_keyframe_id, next_transition_id,
                    add_transition as db_add_tr, delete_transition as db_del_tr,
                    get_transitions as db_get_trs,
                )
                from datetime import datetime, timezone

                source = project_dir / pool_path
                if not source.exists():
                    return self._error(404, "NOT_FOUND", f"Pool item not found: {pool_path}")

                def parse_ts(ts):
                    parts = str(ts).split(":")
                    return int(parts[0]) * 60 + float(parts[1]) if len(parts) == 2 else (float(ts) if isinstance(ts, (int, float)) else 0)

                def to_ts(s):
                    m = int(s) // 60
                    return f"{m}:{s - m*60:05.2f}"

                _log(f"insert-pool-item: type={item_type} path={pool_path} atTime={at_time}")

                if item_type == "keyframe":
                    kf_id = next_keyframe_id(project_dir)
                    dest = project_dir / "selected_keyframes" / f"{kf_id}.png"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(source), str(dest))
                    cand_dir = project_dir / "keyframe_candidates" / "candidates" / f"section_{kf_id}"
                    cand_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(source), str(cand_dir / "v1.png"))

                    db_add_kf(project_dir, {
                        "id": kf_id, "timestamp": to_ts(at_time), "section": "",
                        "source": pool_path, "prompt": f"Inserted from pool: {source.name}",
                        "candidates": [f"keyframe_candidates/candidates/section_{kf_id}/v1.png"],
                        "selected": 1,
                    })

                    # Find neighbors and split spanning transition
                    all_kfs = db_get_kfs(project_dir)
                    sorted_kfs = sorted(all_kfs, key=lambda k: parse_ts(k["timestamp"]))
                    new_idx = next((i for i, k in enumerate(sorted_kfs) if k["id"] == kf_id), -1)
                    prev_kf = sorted_kfs[new_idx - 1] if new_idx > 0 else None
                    next_kf = sorted_kfs[new_idx + 1] if new_idx < len(sorted_kfs) - 1 else None

                    if prev_kf and next_kf:
                        all_trs = db_get_trs(project_dir)
                        old_tr = next((t for t in all_trs if t["from"] == prev_kf["id"] and t["to"] == next_kf["id"]), None)
                        if old_tr:
                            now = datetime.now(timezone.utc).isoformat()
                            db_del_tr(project_dir, old_tr["id"], now)
                            pt = parse_ts(prev_kf["timestamp"])
                            nt = parse_ts(next_kf["timestamp"])
                            d1, d2 = round(at_time - pt, 2), round(nt - at_time, 2)
                            tr1_id = next_transition_id(project_dir)
                            db_add_tr(project_dir, {"id": tr1_id, "from": prev_kf["id"], "to": kf_id,
                                "duration_seconds": d1, "slots": 1, "action": "", "use_global_prompt": False,
                                "selected": None, "remap": {"method": "linear", "target_duration": d1}})
                            tr2_id = next_transition_id(project_dir)
                            db_add_tr(project_dir, {"id": tr2_id, "from": kf_id, "to": next_kf["id"],
                                "duration_seconds": d2, "slots": 1, "action": "", "use_global_prompt": False,
                                "selected": None, "remap": {"method": "linear", "target_duration": d2}})

                    self._json_response({"success": True, "type": "keyframe", "id": kf_id})
                else:
                    return self._error(400, "BAD_REQUEST", f"Use 'Assign to TR' for video segments")

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_delete_transition(self, project_name: str):
            """POST /api/projects/:name/delete-transition — soft-delete a transition to bin."""
            body = self._read_json_body()
            if body is None:
                return

            tr_id = body.get("transitionId")
            if not tr_id:
                return self._error(400, "BAD_REQUEST", "Missing 'transitionId'")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                from beatlab.db import delete_transition as db_del_tr
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc).isoformat()
                db_del_tr(project_dir, tr_id, now)
                self._json_response({"success": True, "binned": {"id": tr_id, "deleted_at": now}})
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

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                from beatlab.db import restore_transition as db_restore_tr
                db_restore_tr(project_dir, tr_id)
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

            _log(f"update-transition-action: {project_name} {tr_id} action={repr(action[:50] if action else None)}")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                from beatlab.db import update_transition, get_transition
                tr = get_transition(project_dir, tr_id)
                if not tr:
                    return self._error(404, "NOT_FOUND", f"Transition {tr_id} not found")

                updates = {}
                if action is not None:
                    updates["action"] = action
                if use_global is not None:
                    updates["use_global_prompt"] = use_global
                if updates:
                    update_transition(project_dir, tr_id, **updates)

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
            curve_points = body.get("curvePoints")
            if not tr_id:
                return self._error(400, "BAD_REQUEST", "Missing 'transitionId'")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                from beatlab.db import get_transition, update_transition
                tr = get_transition(project_dir, tr_id)
                if not tr:
                    return self._error(404, "NOT_FOUND", f"Transition {tr_id} not found")

                remap = tr.get("remap", {"method": "linear", "target_duration": 0})
                if target_duration is not None:
                    remap["target_duration"] = target_duration
                if method is not None:
                    remap["method"] = method
                if curve_points is not None:
                    remap["curve_points"] = curve_points
                elif method == "linear" and "curve_points" in remap:
                    del remap["curve_points"]

                update_transition(project_dir, tr_id, remap=remap)
                self._json_response({"success": True, "transitionId": tr_id, "remap": remap})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_generate_transition_action(self, project_name: str):
            """POST /api/projects/:name/generate-transition-action — LLM-generate action for a single transition."""
            body = self._read_json_body()
            if body is None:
                return

            tr_id = body.get("transitionId")
            section_context = body.get("sectionContext")
            if not tr_id:
                return self._error(400, "BAD_REQUEST", "Missing 'transitionId'")

            _log(f"generate-transition-action: {project_name} {tr_id} (section context: {'yes' if section_context else 'no'})")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                import base64
                import os
                from beatlab.db import get_transition, get_keyframe, get_meta, update_transition

                tr = get_transition(project_dir, tr_id)
                if not tr:
                    return self._error(404, "NOT_FOUND", f"Transition {tr_id} not found")

                from_kf = get_keyframe(project_dir, tr["from"])
                to_kf = get_keyframe(project_dir, tr["to"])
                if not from_kf or not to_kf:
                    return self._error(400, "BAD_REQUEST", f"Keyframes {tr['from']} or {tr['to']} not found")
                selected_dir = project_dir / "selected_keyframes"
                from_img = selected_dir / f"{tr['from']}.png"
                to_img = selected_dir / f"{tr['to']}.png"

                if not from_img.exists() or not to_img.exists():
                    _log(f"  Missing images: from={from_img.exists()} to={to_img.exists()}")
                    return self._error(400, "BAD_REQUEST", f"Selected keyframe images not found — from:{from_img.exists()} to:{to_img.exists()}")

                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    return self._error(500, "INTERNAL_ERROR", "ANTHROPIC_API_KEY not set")

                _log(f"  Calling Claude for {tr_id} ({tr['from']} -> {tr['to']})...")
                from anthropic import Anthropic
                client = Anthropic(api_key=api_key)

                from_b64 = base64.b64encode(from_img.read_bytes()).decode()
                to_b64 = base64.b64encode(to_img.read_bytes()).decode()
                from_ctx = from_kf.get("context") or {}
                to_ctx = to_kf.get("context") or {}
                meta = get_meta(project_dir)
                master_prompt = meta.get("prompt", "")
                master_context = f"Overall creative direction: {master_prompt}\n\n" if master_prompt else ""

                n_slots = tr.get("slots", 1)
                selected_slot_kf_dir = project_dir / "selected_slot_keyframes"

                section_text = f"\n\nMusical context for this section:\n{section_context}\n" if section_context else ""

                if n_slots <= 1:
                    # Single-slot: generate one action from the two keyframes
                    user_content = [
                        {"type": "text", "text": f"You are a visual effects director for a music video. {master_context}Describe the ideal visual transition between these two keyframes.{section_text}\n\n"},
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
                    _log(f"  Generated action: {action[:80]}...")
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

                update_transition(project_dir, tr_id, action=tr.get("action", ""))

                _log(f"  Saved action for {tr_id}")
                self._json_response({"success": True, "action": tr.get("action", ""), "slotActions": tr.get("slot_actions", [])})
            except Exception as e:
                _log(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_enhance_transition_action(self, project_name: str):
            """POST /api/projects/:name/enhance-transition-action — enhance an existing action prompt to be more descriptive."""
            body = self._read_json_body()
            if body is None:
                return

            tr_id = body.get("transitionId")
            current_action = body.get("action", "")
            section_context = body.get("sectionContext")
            if not tr_id or not current_action:
                return self._error(400, "BAD_REQUEST", "Missing 'transitionId' or 'action'")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                import base64
                import os
                from beatlab.db import get_transition

                tr = get_transition(project_dir, tr_id)
                if not tr:
                    return self._error(404, "NOT_FOUND", f"Transition {tr_id} not found")

                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    return self._error(500, "INTERNAL_ERROR", "ANTHROPIC_API_KEY not set")

                from_img = project_dir / "selected_keyframes" / f"{tr['from']}.png"
                to_img = project_dir / "selected_keyframes" / f"{tr['to']}.png"

                section_text = f"\n\nMusical context for this section:\n{section_context}\n" if section_context else ""
                user_content = [
                    {"type": "text", "text":
                        "You are a visual effects director enhancing a transition prompt for Veo video generation. "
                        "Take the user's existing prompt and make it more vivid, specific, and cinematic. "
                        "Add details about camera movement, lighting, particle effects, color shifts, and timing. "
                        "Keep the core intent but make it significantly more descriptive for AI video generation.\n\n"
                        f"Current prompt: \"{current_action}\"\n\n"
                        f"{section_text}"},
                ]

                # Include keyframe images if available for visual context
                if from_img.exists() and to_img.exists():
                    from_b64 = base64.b64encode(from_img.read_bytes()).decode()
                    to_b64 = base64.b64encode(to_img.read_bytes()).decode()
                    user_content.extend([
                        {"type": "text", "text": "FROM keyframe:\n"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": from_b64}},
                        {"type": "text", "text": "\nTO keyframe:\n"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": to_b64}},
                        {"type": "text", "text": "\nUse these images to inform your enhancement — reference specific visual elements you see.\n\n"},
                    ])

                user_content.append({"type": "text", "text":
                    "Reply with ONLY the enhanced prompt, no preamble or explanation. "
                    "Keep it to 2-4 sentences."})

                from anthropic import Anthropic
                client = Anthropic(api_key=api_key)
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=400,
                    messages=[{"role": "user", "content": user_content}],
                )

                enhanced = response.content[0].text.strip()
                self._json_response({"success": True, "action": enhanced})
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

            project_dir = work_dir / project_name

            # Export DB to YAML before generation (narrative.py still reads YAML)
            if (project_dir / "project.db").exists():
                try:
                    from beatlab.db import export_to_yaml
                    export_to_yaml(project_dir)
                except Exception as ex:
                    _log(f"  Warning: DB->YAML export failed: {ex}")

            yaml_path = self._require_yaml_path(project_name)
            if yaml_path is None:
                return

            # Count existing candidates so we generate beyond them (v5, v6, etc.)
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

                    # Persist candidates to DB
                    from beatlab.db import update_keyframe
                    update_keyframe(project_dir, kf_id, candidates=candidates)

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
            duration = body.get("duration")  # optional: 4, 6, or 8 seconds
            _log(f"[generate-transition-candidates] tr={tr_id} count={count} duration={duration} (body keys: {list(body.keys())})")
            if not tr_id:
                return self._error(400, "BAD_REQUEST", "Missing 'transitionId'")

            project_dir = work_dir / project_name

            # Export DB to YAML before generation (narrative.py still reads YAML)
            if (project_dir / "project.db").exists():
                try:
                    from beatlab.db import export_to_yaml
                    export_to_yaml(project_dir)
                    _log(f"  Exported DB->YAML for {project_name}")
                except Exception as ex:
                    _log(f"  Warning: DB->YAML export failed: {ex}")

            yaml_path = self._require_yaml_path(project_name)
            if yaml_path is None:
                return

            # Count existing candidates so we generate beyond them (v5, v6, etc.)
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
                    _log(f"[job {job_id}] Starting transition candidate generation for {tr_id}...")
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
                            duration_seconds=duration,
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
                    _log(f"[job {job_id}] FAILED: {e}")
                    import traceback
                    traceback.print_exc()
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

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                from beatlab.db import get_meta, set_meta
                meta = get_meta(project_dir)
                for key in ("motion_prompt", "default_transition_prompt"):
                    if key in body:
                        set_meta(project_dir, key, body[key])
                        meta[key] = body[key]

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
                    "rules": rules,
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
            project_dir = self._get_project_dir(project_name)
            if project_dir is None:
                return self._json_response({"watchedFolders": []})
            try:
                from beatlab.db import get_meta
                import json
                meta = get_meta(project_dir)
                wf = meta.get("watched_folders", [])
                if isinstance(wf, str):
                    wf = json.loads(wf)
                self._json_response({"watchedFolders": wf})
            except Exception:
                self._json_response({"watchedFolders": []})

        def _handle_get_effects(self, project_name: str):
            """GET /api/projects/:name/effects — load user-authored effects."""
            project_dir = self._get_project_dir(project_name)
            if project_dir is None:
                return self._json_response({"effects": [], "suppressions": []})
            try:
                from beatlab.db import get_effects, get_suppressions
                self._json_response({
                    "effects": get_effects(project_dir),
                    "suppressions": get_suppressions(project_dir),
                })
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_update_effects(self, project_name: str):
            """POST /api/projects/:name/effects — update user-authored effects.

            Body: { "effects": [...], "suppressions": [...] }
            """
            body = self._read_json_body()
            if body is None:
                return

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                from beatlab.db import save_effects
                effects = body.get("effects", [])
                suppressions = body.get("suppressions", [])
                save_effects(project_dir, effects, suppressions)
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

            # Lock per-project to prevent concurrent git/YAML operations
            lock = _get_project_lock(project_name)
            with lock:
                return self._do_version_commit(project_dir, message)

        def _do_version_commit(self, project_dir, message):
            import subprocess as sp
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

        def _get_project_dir(self, project_name: str) -> Path | None:
            """Get project directory, auto-importing YAML to SQLite if needed."""
            project_dir = work_dir / project_name
            if not project_dir.is_dir():
                return None
            db_path = project_dir / "project.db"
            if not db_path.exists():
                # Auto-import from YAML on first access
                yaml_exists = (project_dir / "timeline.yaml").exists() or (project_dir / "narrative_keyframes.yaml").exists()
                if yaml_exists:
                    from beatlab.db import import_from_yaml
                    _log(f"Auto-importing {project_name} from YAML to SQLite...")
                    import_from_yaml(project_dir)
                    _log(f"  Import complete")
            return project_dir

        def _require_project_dir(self, project_name: str) -> Path | None:
            """Get project directory or send 404."""
            d = self._get_project_dir(project_name)
            if d is None:
                self._error(404, "NOT_FOUND", f"Project not found: {project_name}")
                return None
            return d

        def _get_yaml_path(self, project_name: str) -> Path | None:
            """Get the YAML path for a project — prefers timeline.yaml (split format), falls back to legacy."""
            project_dir = work_dir / project_name
            split = project_dir / "timeline.yaml"
            if split.exists():
                return split
            legacy = project_dir / "narrative_keyframes.yaml"
            if legacy.exists():
                return legacy
            return None

        def _load_timeline_data(self, parsed: dict) -> dict:
            """Extract the active timeline's data from either split or legacy format.
            Returns a dict with 'keyframes', 'transitions', 'bin', 'transition_bin' keys.
            Also sets '_split' flag and '_active' name if split format.
            """
            if "timelines" in parsed:
                active = parsed.get("active_timeline", "default")
                tl = parsed.get("timelines", {}).get(active, {})
                return {
                    "keyframes": tl.get("keyframes", []),
                    "transitions": tl.get("transitions", []),
                    "bin": tl.get("bin", []),
                    "transition_bin": tl.get("transition_bin", []),
                    "_split": True,
                    "_active": active,
                }
            return {
                "keyframes": parsed.get("keyframes", []),
                "transitions": parsed.get("transitions", []),
                "bin": parsed.get("bin", []),
                "transition_bin": parsed.get("transition_bin", []),
                "_split": False,
            }

        def _save_timeline_data(self, parsed: dict, tl_data: dict):
            """Write timeline data back into the parsed YAML structure."""
            if tl_data.get("_split"):
                active = tl_data.get("_active", "default")
                if "timelines" not in parsed:
                    parsed["timelines"] = {}
                parsed["timelines"][active] = {
                    "keyframes": tl_data["keyframes"],
                    "transitions": tl_data["transitions"],
                    "bin": tl_data["bin"],
                    "transition_bin": tl_data["transition_bin"],
                }
            else:
                parsed["keyframes"] = tl_data["keyframes"]
                parsed["transitions"] = tl_data["transitions"]
                parsed["bin"] = tl_data["bin"]
                parsed["transition_bin"] = tl_data["transition_bin"]

        def _require_yaml_path(self, project_name: str) -> Path | None:
            """Get YAML path or send 404 error. Returns None if error was sent."""
            path = self._get_yaml_path(project_name)
            if path is None:
                self._error(404, "NOT_FOUND", "No narrative_keyframes.yaml found")
                return None
            return path

        def _ruamel_load(self, yaml_path):
            """Load YAML with ruamel for round-trip editing."""
            from ruamel.yaml import YAML
            ryaml = YAML()
            ryaml.width = 1000
            ryaml.preserve_quotes = True
            with open(yaml_path) as f:
                return ryaml, ryaml.load(f)

        def _ruamel_save(self, ryaml, parsed, yaml_path):
            """Save YAML with ruamel, preserving formatting."""
            with open(yaml_path, "w") as f:
                ryaml.dump(parsed, f)

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

        def _handle_get_section_settings(self, project_name: str):
            """GET /api/projects/:name/section-settings?section=label — get persisted settings for a section."""
            from urllib.parse import parse_qs
            qs = parse_qs(urlparse(self.path).query)
            section_label = qs.get("section", [""])[0]

            project_dir = self._get_project_dir(project_name)
            if project_dir is None:
                return self._json_response({})

            try:
                from beatlab.db import get_meta
                import json
                meta = get_meta(project_dir)
                still = meta.get(f"section_still:{section_label}", None)
                suggestions_raw = meta.get(f"section_suggestions:{section_label}", None)
                suggestions = json.loads(suggestions_raw) if isinstance(suggestions_raw, str) else suggestions_raw
                self._json_response({
                    "still": still,
                    "suggestions": suggestions,
                })
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_section_settings(self, project_name: str):
            """POST /api/projects/:name/section-settings — persist section settings (still, suggestions)."""
            body = self._read_json_body()
            if body is None:
                return

            section_label = body.get("sectionLabel")
            if not section_label:
                return self._error(400, "BAD_REQUEST", "Missing 'sectionLabel'")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                from beatlab.db import set_meta
                import json

                if "still" in body:
                    set_meta(project_dir, f"section_still:{section_label}", body["still"])
                if "suggestions" in body:
                    set_meta(project_dir, f"section_suggestions:{section_label}",
                             json.dumps(body["suggestions"]))

                self._json_response({"success": True})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_promote_staged_candidate(self, project_name: str):
            """POST /api/projects/:name/promote-staged-candidate — copy a staged candidate as a keyframe's selected image.

            Body: { "keyframeId": "kf_XXX", "stagingId": "evt_0_1234", "variant": 1 }
            """
            body = self._read_json_body()
            if body is None:
                return

            kf_id = body.get("keyframeId")
            staging_id = body.get("stagingId")
            variant = body.get("variant", 1)
            if not kf_id or not staging_id:
                return self._error(400, "BAD_REQUEST", "Missing 'keyframeId' or 'stagingId'")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                import shutil
                from beatlab.db import update_keyframe

                staging_file = project_dir / "staging" / staging_id / f"v{variant}.png"
                if not staging_file.exists():
                    return self._error(404, "NOT_FOUND", f"Staged candidate not found: {staging_file}")

                # Copy to selected_keyframes
                sel_dir = project_dir / "selected_keyframes"
                sel_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(staging_file), str(sel_dir / f"{kf_id}.png"))

                # Also copy to keyframe candidates dir
                cand_dir = project_dir / "keyframe_candidates" / "candidates" / f"section_{kf_id}"
                cand_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(staging_file), str(cand_dir / f"v{variant}.png"))

                update_keyframe(project_dir, kf_id, selected=variant,
                                candidates=[f"keyframe_candidates/candidates/section_{kf_id}/v{variant}.png"])

                _log(f"promote-staged: {staging_id}/v{variant} -> {kf_id}")
                self._json_response({"success": True, "keyframeId": kf_id})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_generate_staged_candidate(self, project_name: str):
            """POST /api/projects/:name/generate-staged-candidate — generate a keyframe image without creating a timeline keyframe.

            Body: { "prompt": "...", "stillName": "dark.png", "stagingId": "evt_0_1234" }
            Returns: { "success": true, "path": "staging/evt_0_1234/v1.png" }
            """
            body = self._read_json_body()
            if body is None:
                return

            prompt = body.get("prompt")
            still_name = body.get("stillName")
            staging_id = body.get("stagingId")
            count = body.get("count", 1)
            if not prompt or not still_name or not staging_id:
                return self._error(400, "BAD_REQUEST", "Missing 'prompt', 'stillName', or 'stagingId'")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            _log(f"generate-staged-candidate: {project_name} id={staging_id} still={still_name} count={count}")

            source = project_dir / "assets" / "stills" / still_name
            if not source.exists():
                return self._error(404, "NOT_FOUND", f"Still not found: {still_name}")

            from beatlab.ws_server import job_manager
            job_id = job_manager.create_job("staged_candidate", total=count, meta={"stagingId": staging_id, "project": project_name})

            def _run():
                try:
                    from beatlab.render.google_video import GoogleVideoClient
                    client = GoogleVideoClient(vertex=True)

                    staging_dir = project_dir / "staging" / staging_id
                    staging_dir.mkdir(parents=True, exist_ok=True)

                    paths = []
                    existing = len(list(staging_dir.glob("v*.png")))
                    for i in range(count):
                        v = existing + i + 1
                        out_path = str(staging_dir / f"v{v}.png")
                        if Path(out_path).exists():
                            paths.append(f"staging/{staging_id}/v{v}.png")
                            continue
                        varied = f"{prompt}, variation {v}" if v > 1 else prompt
                        try:
                            client.stylize_image(str(source), varied, out_path)
                            paths.append(f"staging/{staging_id}/v{v}.png")
                            job_manager.update_progress(job_id, i + 1, f"v{v} done")
                        except Exception as e:
                            _log(f"  v{v} FAILED: {e}")
                            job_manager.update_progress(job_id, i + 1, f"v{v} failed")

                    # Return ALL candidates in staging dir (not just newly generated)
                    all_paths = sorted([
                        f"staging/{staging_id}/{f.name}"
                        for f in staging_dir.glob("v*.png")
                    ])
                    job_manager.complete_job(job_id, {"stagingId": staging_id, "candidates": all_paths})
                except Exception as e:
                    _log(f"  staged generation FAILED: {e}")
                    job_manager.fail_job(job_id, str(e))

            import threading
            threading.Thread(target=_run, daemon=True).start()
            self._json_response({"jobId": job_id, "stagingId": staging_id})

        def _handle_enhance_keyframe_prompt(self, project_name: str):
            """POST /api/projects/:name/enhance-keyframe-prompt — enhance an existing keyframe prompt to be more vivid and cinematic."""
            body = self._read_json_body()
            if body is None:
                return

            current_prompt = body.get("prompt", "")
            section_content = body.get("sectionContent", "")
            event = body.get("event", {})

            if not current_prompt:
                return self._error(400, "BAD_REQUEST", "Missing 'prompt'")

            _log(f"enhance-keyframe-prompt: {project_name} prompt={current_prompt[:60]!r}")

            try:
                import os
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    return self._error(500, "INTERNAL_ERROR", "ANTHROPIC_API_KEY not set")

                from anthropic import Anthropic
                client = Anthropic(api_key=api_key)

                event_context = ""
                if event:
                    event_context = (
                        f"\n\nAudio event context:\n"
                        f"  Time: {event.get('time', 0):.2f}s\n"
                        f"  Stem: {event.get('stem_source', '?')}\n"
                        f"  Effect: {event.get('effect', '?')}\n"
                        f"  Intensity: {event.get('intensity', 0) * 100:.0f}%\n"
                    )
                    if event.get("rationale"):
                        event_context += f"  Rationale: {event['rationale']}\n"

                section_text = f"\n\nMusical context for this section:\n{section_content}\n" if section_content else ""

                prompt_text = (
                    "You are a visionary art director enhancing a keyframe image prompt for Imagen style transfer. "
                    "Take the user's existing prompt and make it more vivid, specific, and cinematic. "
                    "Add details about materials, textures, lighting quality, atmosphere, scale, and spatial depth. "
                    "Keep the core scene and intent but make it significantly more descriptive and tangible.\n\n"
                    f"Current prompt: \"{current_prompt}\"\n"
                    f"{section_text}"
                    f"{event_context}\n"
                    "Reply with ONLY the enhanced prompt, no preamble or explanation. "
                    "Keep it to 2-4 sentences. Describe a CONCRETE, FILMABLE scene."
                )

                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt_text}],
                )

                enhanced = response.content[0].text.strip()
                _log(f"  Enhanced prompt: {enhanced[:80]}...")
                self._json_response({"success": True, "prompt": enhanced})

            except Exception as e:
                _log(f"  enhance-keyframe-prompt error: {e}")
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_suggest_keyframe_prompts(self, project_name: str):
            """POST /api/projects/:name/suggest-keyframe-prompts — LLM-generate style prompts for audio events."""
            body = self._read_json_body()
            if body is None:
                return

            section_label = body.get("sectionLabel", "")
            section_content = body.get("sectionContent", "")
            events = body.get("events", [])
            base_still = body.get("baseStillName", "")

            if not events:
                return self._error(400, "BAD_REQUEST", "Missing 'events'")

            _log(f"suggest-keyframe-prompts: {project_name} section={section_label!r} events={len(events)} still={base_still!r}")

            try:
                import os
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    return self._error(500, "INTERNAL_ERROR", "ANTHROPIC_API_KEY not set")

                _log(f"  Calling Claude for {len(events)} event prompts...")
                from anthropic import Anthropic
                client = Anthropic(api_key=api_key)

                event_list = "\n".join(
                    f"  {i}: t={ev.get('time', 0):.2f}s, stem={ev.get('stem_source', '?')}, "
                    f"effect={ev.get('effect', '?')}, intensity={ev.get('intensity', 0) * 100:.0f}%"
                    for i, ev in enumerate(events)
                )

                prompt_text = (
                    f"You are a visionary art director creating keyframe images for a cinematic music video. "
                    f"Each prompt will transform a base photograph (\"{base_still}\") into a vivid, "
                    f"tangible scene through Imagen style transfer.\n\n"
                    f"CRITICAL: Each prompt MUST describe a SPECIFIC, CONCRETE scene, place, or action. "
                    f"NOT abstract descriptions of mood or color. Think like a film location scout crossed with a painter:\n"
                    f"- A mist-shrouded ancient forest with bioluminescent fungi pulsing on twisted bark\n"
                    f"- Tiny porcelain dancers frozen mid-leap inside a cracked music box\n"
                    f"- A haunting gothic cathedral where stained glass bleeds liquid color onto stone floors\n"
                    f"- A dark menacing techno underworld of chrome corridors and neon-dripping pipework\n"
                    f"- An underwater ballroom where jellyfish chandeliers illuminate drowned aristocrats\n"
                    f"- A burning library where pages become flocks of fire-birds spiraling upward\n\n"
                    f"Each prompt should conjure a PLACE you could walk into, a SCENE you could film, "
                    f"an ACTION caught mid-frame. Be wildly creative. Be specific. Be cinematic.\n\n"
                    f"Section: \"{section_label}\"\n"
                    f"Musical description:\n{section_content}\n\n"
                    f"Audio events in this section:\n{event_list}\n\n"
                    f"For each event, write a prompt (2-3 sentences) that:\n"
                    f"- Describes a TANGIBLE scene, location, or frozen moment (not abstract feelings)\n"
                    f"- Matches the musical energy: quiet = intimate/delicate/mysterious, loud = explosive/overwhelming/massive\n"
                    f"- Includes specific visual details: materials, lighting quality, weather, scale, texture\n"
                    f"- Varies WILDLY across events — each should feel like a different world or dimension\n"
                    f"- Treats the base image as the subject placed INTO this new world\n\n"
                    f"Respond with ONLY a JSON array, no markdown fences: [{{\"eventIndex\": 0, \"prompt\": \"...\"}}, ...]"
                )

                import json as _json
                import re as _re
                suggestions = None
                max_retries = 2

                for attempt in range(max_retries + 1):
                    response = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=16384,
                        messages=[{"role": "user", "content": prompt_text}],
                    )

                    text = response.content[0].text if response.content else ""
                    _log(f"  Attempt {attempt + 1}: response length={len(text)}, stop={response.stop_reason}")

                    json_match = _re.search(r"\[[\s\S]*\]", text)
                    if json_match:
                        try:
                            suggestions = _json.loads(json_match.group(0))
                            break
                        except _json.JSONDecodeError:
                            _log(f"  Attempt {attempt + 1}: JSON parse error, retrying...")
                    else:
                        _log(f"  Attempt {attempt + 1}: no JSON array found in response: {text[:200]}")

                if suggestions is None:
                    return self._error(500, "INTERNAL_ERROR", "Failed to parse prompt suggestions after retries")

                _log(f"  Generated {len(suggestions)} prompt suggestions")

                # Auto-persist suggestions to DB
                try:
                    from beatlab.db import set_meta
                    import json as _json2
                    project_dir = self._get_project_dir(project_name)
                    if project_dir:
                        set_meta(project_dir, f"section_suggestions:{section_label}", _json2.dumps(suggestions))
                        if base_still:
                            set_meta(project_dir, f"section_still:{section_label}", base_still)
                except Exception:
                    pass  # non-critical

                self._json_response({"suggestions": suggestions})

            except Exception as e:
                _log(f"  suggest-keyframe-prompts error: {e}")
                self._error(500, "INTERNAL_ERROR", str(e))

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
    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer((host, port), handler)

    # Start WebSocket server for real-time job progress
    ws_port = port + 1
    from beatlab.ws_server import start_ws_server, FolderWatcher
    import beatlab.ws_server as _ws_mod
    _ws_mod.folder_watcher = FolderWatcher(wd)
    start_ws_server(host, ws_port)

    # Folder watches are lazy — activated when frontend opens a project and calls watch-folder,
    # NOT restored on server boot. This avoids inotify overhead for projects not being viewed.

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
