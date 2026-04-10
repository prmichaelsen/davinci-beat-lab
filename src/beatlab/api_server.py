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


def _log(msg: str, level: str = "info"):
    from datetime import datetime
    now = datetime.now()
    ts = now.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)
    # Broadcast to WebSocket clients
    try:
        from beatlab.ws_server import job_manager
        job_manager._broadcast({"type": "log", "message": msg, "timestamp": now.isoformat(), "level": level})
    except Exception:
        pass


def _next_variant(directory: Path, ext: str = ".png") -> int:
    """Find the next available variant number in a directory (max existing + 1)."""
    import re as _re
    max_v = 0
    for f in directory.glob(f"v*{ext}"):
        m = _re.match(r"v(\d+)", f.stem)
        if m:
            max_v = max(max_v, int(m.group(1)))
    return max_v + 1


def _get_project_settings(project_dir: Path) -> dict:
    """Read settings.yaml for a project."""
    import yaml
    settings_path = project_dir / "settings.yaml"
    if settings_path.exists():
        with open(settings_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _get_image_backend(project_dir: Path) -> str:
    """Get image generation backend from project settings."""
    return _get_project_settings(project_dir).get("image_backend", "vertex")


def _get_video_backend(project_dir: Path) -> str:
    """Get video generation backend from project settings."""
    return _get_project_settings(project_dir).get("video_backend", "vertex")


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

            # GET /api/projects/:name/workspace-views
            m = re.match(r"^/api/projects/([^/]+)/workspace-views$", path)
            if m:
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import get_meta
                meta = get_meta(project_dir)
                views = {k.replace("workspace_view:", ""): v for k, v in meta.items() if k.startswith("workspace_view:")}
                return self._json_response({"views": views})

            # GET /api/projects/:name/workspace-views/:name
            m = re.match(r"^/api/projects/([^/]+)/workspace-views/([^/]+)$", path)
            if m:
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import get_meta
                meta = get_meta(project_dir)
                layout = meta.get(f"workspace_view:{m.group(2)}")
                if layout is None:
                    return self._error(404, "NOT_FOUND", f"Workspace view not found: {m.group(2)}")
                return self._json_response({"layout": layout})

            # GET /api/projects/:name/checkpoints
            m = re.match(r"^/api/projects/([^/]+)/checkpoints$", path)
            if m:
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from datetime import datetime as _dt
                import yaml as _yaml
                # Load manifest for names
                manifest_path = project_dir / "checkpoints.yaml"
                name_map = {}
                if manifest_path.exists():
                    with open(manifest_path) as _mf:
                        for entry in (_yaml.safe_load(_mf) or []):
                            if isinstance(entry, dict):
                                name_map[entry.get("filename", "")] = entry.get("name", "")
                checkpoints = []
                for f in sorted(project_dir.glob("project.db.checkpoint-*"), reverse=True):
                    stat = f.stat()
                    checkpoints.append({
                        "filename": f.name,
                        "name": name_map.get(f.name, ""),
                        "created": _dt.fromtimestamp(stat.st_mtime).isoformat(),
                        "size_bytes": stat.st_size,
                    })
                return self._json_response({"checkpoints": checkpoints, "active": "project.db"})

            # GET /api/projects/:name/undo-history
            m = re.match(r"^/api/projects/([^/]+)/undo-history$", path)
            if m:
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import undo_history
                return self._json_response({"history": undo_history(project_dir)})

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

            # GET /api/projects/:name/tracks
            m = re.match(r"^/api/projects/([^/]+)/tracks$", path)
            if m:
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None:
                    return
                from beatlab.db import get_tracks, get_opacity_keyframes
                tracks = get_tracks(project_dir)
                for t in tracks:
                    t["opacityKeyframes"] = get_opacity_keyframes(project_dir, t["id"])
                return self._json_response({"tracks": tracks})

            # GET /api/projects/:name/unselected-candidates
            m = re.match(r"^/api/projects/([^/]+)/unselected-candidates$", path)
            if m:
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import get_keyframes
                kfs = get_keyframes(project_dir)
                candidates = []
                for kf in kfs:
                    kf_id = kf["id"]
                    selected = kf.get("selected")
                    cand_dir = project_dir / "keyframe_candidates" / "candidates" / f"section_{kf_id}"
                    if not cand_dir.is_dir(): continue
                    for f in sorted(cand_dir.glob("v*.png"), key=lambda p: int(p.stem.replace("v", ""))):
                        vnum = int(f.stem.replace("v", ""))
                        if vnum != selected:
                            candidates.append({
                                "keyframeId": kf_id,
                                "variant": vnum,
                                "path": f"keyframe_candidates/candidates/section_{kf_id}/{f.name}",
                            })
                return self._json_response({"candidates": candidates})

            # GET /api/projects/:name/markers
            m = re.match(r"^/api/projects/([^/]+)/markers$", path)
            if m:
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None:
                    return
                from beatlab.db import get_markers
                return self._json_response({"markers": get_markers(project_dir)})

            # GET /api/projects/:name/prompt-roster
            m = re.match(r"^/api/projects/([^/]+)/prompt-roster$", path)
            if m:
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import get_prompt_roster
                return self._json_response({"prompts": get_prompt_roster(project_dir)})

            # GET /api/projects/:name/pool
            m = re.match(r"^/api/projects/([^/]+)/pool$", path)
            if m:
                return self._handle_get_pool(m.group(1))

            # GET /api/projects/:name/version/history (deprecated — git removed)
            m = re.match(r"^/api/projects/([^/]+)/version/history$", path)
            if m:
                return self._json_response({"commits": [], "branch": "", "branches": []})

            # GET /api/projects/:name/version/diff (deprecated — git removed)
            m = re.match(r"^/api/projects/([^/]+)/version/diff$", path)
            if m:
                return self._json_response({"changes": []})

            # GET /api/projects/:name/effects
            m = re.match(r"^/api/projects/([^/]+)/effects$", path)
            if m:
                return self._handle_get_effects(m.group(1))

            # GET /api/projects/:name/thumb/(.*)  — resized image thumbnail (cached to disk)
            m = re.match(r"^/api/projects/([^/]+)/thumb/(.+)$", path)
            if m:
                return self._handle_image_thumb(m.group(1), m.group(2))

            # GET /api/projects/:name/thumbnail/(.*)  — first-frame JPEG for video files
            m = re.match(r"^/api/projects/([^/]+)/thumbnail/(.+)$", path)
            if m:
                return self._handle_video_thumbnail(m.group(1), m.group(2))

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
                "add-keyframe", "duplicate-keyframe", "delete-keyframe", "batch-delete-keyframes", "restore-keyframe",
                "delete-transition", "restore-transition",
                "split-transition", "insert-pool-item", "paste-group",
                "checkpoint",
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

            # POST /api/projects/:name/paste-group
            m = re.match(r"^/api/projects/([^/]+)/paste-group$", path)
            if m:
                return self._handle_paste_group(m.group(1))

            # POST /api/projects/:name/delete-keyframe
            m = re.match(r"^/api/projects/([^/]+)/delete-keyframe$", path)
            if m:
                return self._handle_delete_keyframe(m.group(1))

            # POST /api/projects/:name/batch-delete-keyframes
            m = re.match(r"^/api/projects/([^/]+)/batch-delete-keyframes$", path)
            if m:
                return self._handle_batch_delete_keyframes(m.group(1))

            # POST /api/projects/:name/restore-keyframe
            m = re.match(r"^/api/projects/([^/]+)/restore-keyframe$", path)
            if m:
                return self._handle_restore_keyframe(m.group(1))

            # POST /api/projects/:name/batch-set-base-image
            m = re.match(r"^/api/projects/([^/]+)/batch-set-base-image$", path)
            if m:
                return self._handle_batch_set_base_image(m.group(1))

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

            # POST /api/projects/:name/unlink-keyframe
            m = re.match(r"^/api/projects/([^/]+)/unlink-keyframe$", path)
            if m:
                return self._handle_unlink_keyframe(m.group(1))

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

            # POST /api/projects/:name/undo
            m = re.match(r"^/api/projects/([^/]+)/undo$", path)
            if m:
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import undo_execute
                result = undo_execute(project_dir)
                if result:
                    _log(f"undo: {result['description']}")
                    return self._json_response({"success": True, **result})
                return self._json_response({"success": False, "message": "Nothing to undo"})

            # POST /api/projects/:name/redo
            m = re.match(r"^/api/projects/([^/]+)/redo$", path)
            if m:
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import redo_execute
                result = redo_execute(project_dir)
                if result:
                    _log(f"redo: {result['description']}")
                    return self._json_response({"success": True, **result})
                return self._json_response({"success": False, "message": "Nothing to redo"})

            # POST /api/projects/:name/workspace-views/:viewName
            m = re.match(r"^/api/projects/([^/]+)/workspace-views/([^/]+)$", path)
            if m:
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                body = self._read_json_body()
                if body is None: return
                from beatlab.db import set_meta
                set_meta(project_dir, f"workspace_view:{m.group(2)}", body.get("layout", {}))
                _log(f"workspace-view saved: {m.group(1)} / {m.group(2)}")
                return self._json_response({"success": True})

            # POST /api/projects/:name/workspace-views/:viewName/delete
            m = re.match(r"^/api/projects/([^/]+)/workspace-views/([^/]+)/delete$", path)
            if m:
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import get_db
                conn = get_db(project_dir)
                conn.execute("DELETE FROM meta WHERE key = ?", (f"workspace_view:{m.group(2)}",))
                conn.commit()
                _log(f"workspace-view deleted: {m.group(1)} / {m.group(2)}")
                return self._json_response({"success": True})

            # POST /api/projects/:name/checkpoint
            m = re.match(r"^/api/projects/([^/]+)/checkpoint$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                import sqlite3 as _sqlite3
                from datetime import datetime
                db_path = project_dir / "project.db"
                if not db_path.exists():
                    return self._error(404, "NOT_FOUND", "No project.db found")
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                name = body.get("name", "")
                filename = f"project.db.checkpoint-{ts}"
                dst = project_dir / filename
                # Use SQLite backup API — safe for WAL-mode databases
                src_conn = _sqlite3.connect(str(db_path))
                dst_conn = _sqlite3.connect(str(dst))
                try:
                    src_conn.backup(dst_conn)
                finally:
                    dst_conn.close()
                    src_conn.close()
                # Save metadata to checkpoints.yaml
                import yaml as _yaml
                manifest_path = project_dir / "checkpoints.yaml"
                manifest = []
                if manifest_path.exists():
                    with open(manifest_path) as _mf:
                        manifest = _yaml.safe_load(_mf) or []
                manifest.append({"filename": filename, "name": name or "", "created": datetime.now().isoformat()})
                with open(manifest_path, "w") as _mf:
                    _yaml.dump(manifest, _mf, default_flow_style=False)

                _log(f"checkpoint: {m.group(1)} -> {filename}{' (' + name + ')' if name else ''}")
                return self._json_response({"success": True, "filename": filename, "name": name})

            # POST /api/projects/:name/checkpoint/restore
            m = re.match(r"^/api/projects/([^/]+)/checkpoint/restore$", path)
            if m:
                project_name = m.group(1)
                project_dir = self._require_project_dir(project_name)
                if project_dir is None: return
                body = self._read_json_body()
                if body is None: return
                filename = body.get("filename", "")
                checkpoint_path = project_dir / filename
                if not filename.startswith("project.db.checkpoint-") or not checkpoint_path.exists():
                    return self._error(404, "NOT_FOUND", f"Checkpoint not found: {filename}")
                db_path = project_dir / "project.db"
                # Close all existing connections to this project's DB
                from beatlab.db import close_db
                close_db(project_dir)
                # Use SQLite backup API to restore checkpoint over the active DB
                import sqlite3 as _sqlite3
                src_conn = _sqlite3.connect(str(checkpoint_path))
                dst_conn = _sqlite3.connect(str(db_path))
                try:
                    src_conn.backup(dst_conn)
                finally:
                    dst_conn.close()
                    src_conn.close()
                _log(f"checkpoint restore: {project_name} <- {filename}")
                return self._json_response({"success": True, "message": f"Restored from {filename}"})

            # POST /api/projects/:name/checkpoint/delete
            m = re.match(r"^/api/projects/([^/]+)/checkpoint/delete$", path)
            if m:
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                body = self._read_json_body()
                if body is None: return
                filename = body.get("filename", "")
                checkpoint_path = project_dir / filename
                if not filename.startswith("project.db.checkpoint-") or not checkpoint_path.exists():
                    return self._error(404, "NOT_FOUND", f"Checkpoint not found: {filename}")
                checkpoint_path.unlink()
                _log(f"checkpoint deleted: {m.group(1)} / {filename}")
                return self._json_response({"success": True})

            # POST /api/projects/:name/bench/capture
            m = re.match(r"^/api/projects/([^/]+)/bench/capture$", path)
            if m:
                return self._handle_bench_capture(m.group(1))

            # POST /api/projects/:name/bench/upload
            m = re.match(r"^/api/projects/([^/]+)/bench/upload$", path)
            if m:
                return self._handle_bench_upload(m.group(1))

            # POST /api/projects/:name/bench/add
            m = re.match(r"^/api/projects/([^/]+)/bench/add$", path)
            if m:
                return self._handle_bench_add(m.group(1))

            # POST /api/projects/:name/bench/remove
            m = re.match(r"^/api/projects/([^/]+)/bench/remove$", path)
            if m:
                return self._handle_bench_remove(m.group(1))

            # POST /api/projects/:name/tracks/add
            m = re.match(r"^/api/projects/([^/]+)/tracks/add$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import add_track as db_add_track, get_tracks as db_get_tracks
                existing = db_get_tracks(project_dir)
                max_num = max((int(t["id"].replace("track_", "")) for t in existing if t["id"].startswith("track_")), default=0)
                track_id = f"track_{max_num + 1}"
                # Add at top (highest z_order = rendered on top in compositor)
                z_order = max((t["z_order"] for t in existing), default=-1) + 1
                db_add_track(project_dir, {"id": track_id, "name": body.get("name", f"Track {len(existing) + 1}"), "z_order": z_order, **{k: v for k, v in body.items() if k in ("blend_mode", "base_opacity", "enabled")}})
                _log(f"tracks/add: {m.group(1)} -> {track_id} (z_order={z_order})")
                return self._json_response({"success": True, "id": track_id})

            # POST /api/projects/:name/tracks/update
            m = re.match(r"^/api/projects/([^/]+)/tracks/update$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import update_track as db_update_track
                track_id = body.pop("id", None)
                if not track_id: return self._error(400, "BAD_REQUEST", "Missing 'id'")
                field_map = {"blendMode": "blend_mode", "baseOpacity": "base_opacity", "chromaKey": "chroma_key"}
                mapped = {field_map.get(k, k): v for k, v in body.items() if field_map.get(k, k) in ("name", "blend_mode", "base_opacity", "enabled", "z_order", "chroma_key", "hidden")}
                _log(f"tracks/update: {track_id} {mapped}")
                db_update_track(project_dir, track_id, **mapped)
                return self._json_response({"success": True})

            # POST /api/projects/:name/tracks/delete
            m = re.match(r"^/api/projects/([^/]+)/tracks/delete$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import delete_track as db_delete_track
                del_id = body.get("id", "")
                _log(f"tracks/delete: {del_id}")
                db_delete_track(project_dir, del_id)
                return self._json_response({"success": True})

            # POST /api/projects/:name/tracks/reorder
            m = re.match(r"^/api/projects/([^/]+)/tracks/reorder$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import reorder_tracks as db_reorder_tracks
                track_ids = body.get("trackIds", [])
                _log(f"tracks/reorder: {track_ids}")
                db_reorder_tracks(project_dir, track_ids)
                return self._json_response({"success": True})

            # POST /api/projects/:name/update-rules
            m = re.match(r"^/api/projects/([^/]+)/update-rules$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                rules = body.get("rules", [])
                # Update the active audio intelligence file with new rules
                import yaml as pyyaml
                settings_path = project_dir / "settings.yaml"
                ai_file = None
                if settings_path.exists():
                    with open(settings_path) as f:
                        s = pyyaml.safe_load(f) or {}
                    ai_file = s.get("audio_intelligence_file")
                if not ai_file:
                    candidates = sorted(project_dir.glob("audio_intelligence*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if candidates:
                        ai_file = candidates[0].name
                if ai_file and (project_dir / ai_file).exists():
                    import json as _json
                    ai_path = project_dir / ai_file
                    with open(ai_path) as f:
                        data = _json.load(f)
                    data["layer3_rules"] = rules
                    with open(ai_path, "w") as f:
                        _json.dump(data, f, indent=2)
                    _log(f"update-rules: saved {len(rules)} rules to {ai_file}")
                    return self._json_response({"success": True, "count": len(rules)})
                return self._error(404, "NOT_FOUND", "No audio intelligence file found")

            # POST /api/projects/:name/reapply-rules
            m = re.match(r"^/api/projects/([^/]+)/reapply-rules$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return

                import yaml as pyyaml
                settings_path = project_dir / "settings.yaml"
                ai_file = None
                if settings_path.exists():
                    with open(settings_path) as f:
                        s = pyyaml.safe_load(f) or {}
                    ai_file = s.get("audio_intelligence_file")
                if not ai_file:
                    candidates = sorted(project_dir.glob("audio_intelligence*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if candidates:
                        ai_file = candidates[0].name
                if not ai_file or not (project_dir / ai_file).exists():
                    return self._error(404, "NOT_FOUND", "No audio intelligence file found")

                import json as _json
                ai_path = project_dir / ai_file
                with open(ai_path) as f:
                    data = _json.load(f)

                # Save rules immediately if provided
                if "rules" in body:
                    data["layer3_rules"] = body["rules"]
                    with open(ai_path, "w") as f:
                        _json.dump(data, f, indent=2)
                    _log(f"reapply-rules: saved {len(body['rules'])} rules")

                layer1 = data.get("layer1", {})
                all_rules = data.get("layer3_rules", [])
                if not layer1:
                    return self._error(400, "BAD_REQUEST", "No layer1 DSP data in audio intelligence file")

                section_start = body.get("sectionStart")
                section_end = body.get("sectionEnd")

                # If section range provided, only reapply rules for that section
                if section_start is not None and section_end is not None:
                    section_rules = [r for r in all_rules if r.get("_group_start") == section_start and r.get("_group_end") == section_end]
                    _log(f"reapply-rules: targeted {len(section_rules)} rules for {section_start}s-{section_end}s")

                    from beatlab.audio_intelligence import apply_rules as _apply
                    new_events = _apply(layer1, section_rules)

                    # Splice: remove old events in this range, add new ones
                    old_events = data.get("layer3_events", [])
                    kept = [e for e in old_events if not (section_start <= e["time"] <= section_end)]
                    merged = sorted(kept + new_events, key=lambda e: e["time"])
                    data["layer3_events"] = merged

                    with open(ai_path, "w") as f:
                        _json.dump(data, f, indent=2)

                    _log(f"  Section: {len(new_events)} new events, {len(merged)} total")
                    return self._json_response({"success": True, "eventCount": len(merged)})

                # Full reapply in background thread
                from beatlab.ws_server import job_manager
                job_id = job_manager.create_job("reapply_rules", total=1, meta={"project": m.group(1)})
                rules = all_rules

                def _run():
                    try:
                        from beatlab.audio_intelligence import apply_rules as _apply
                        _log(f"[job {job_id}] Applying {len(rules)} rules...")
                        events = _apply(layer1, rules)
                        import json as _j
                        with open(ai_path) as f:
                            fresh = _j.load(f)
                        fresh["layer3_events"] = events
                        with open(ai_path, "w") as f:
                            _j.dump(fresh, f, indent=2)
                        _log(f"[job {job_id}] Generated {len(events)} events")
                        job_manager.complete_job(job_id, {"eventCount": len(events)})
                    except Exception as e:
                        _log(f"[job {job_id}] FAILED: {e}")
                        job_manager.fail_job(job_id, str(e))

                import threading
                threading.Thread(target=_run, daemon=True).start()
                return self._json_response({"success": True, "jobId": job_id, "ruleCount": len(rules)})

            # POST /api/projects/:name/generate-keyframe-variations
            m = re.match(r"^/api/projects/([^/]+)/generate-keyframe-variations$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                kf_id = body.get("keyframeId")
                count = body.get("count", 4)
                if not kf_id: return self._error(400, "BAD_REQUEST", "Missing 'keyframeId'")
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return

                source_img = project_dir / "selected_keyframes" / f"{kf_id}.png"
                if not source_img.exists():
                    # Fall back to base stills
                    for still in sorted((project_dir / "assets" / "stills").glob("*.png")) if (project_dir / "assets" / "stills").is_dir() else []:
                        source_img = still
                        break
                if not source_img.exists():
                    return self._error(400, "BAD_REQUEST", f"No source image found for {kf_id}")

                from beatlab.db import get_keyframe
                kf = get_keyframe(project_dir, kf_id)
                kf_prompt = kf.get("prompt", "") if kf else ""

                from beatlab.ws_server import job_manager
                job_id = job_manager.create_job("keyframe_variations", total=count, meta={"keyframeId": kf_id, "project": m.group(1)})

                def _run_variations():
                    try:
                        import os
                        from anthropic import Anthropic
                        api_key = os.environ.get("ANTHROPIC_API_KEY")
                        client_llm = Anthropic(api_key=api_key)

                        _log(f"[job {job_id}] Generating {count} variation prompts for {kf_id}...")
                        job_manager.update_progress(job_id, 0, "Generating prompts with Claude...")

                        response = client_llm.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=4096,
                            messages=[{"role": "user", "content": (
                                f"Generate {count} wildly different style transformation prompts for a keyframe image. "
                                f"Each prompt should create a dramatically different visual world from the same source image.\n\n"
                                f"Original keyframe context: {kf_prompt[:200] if kf_prompt else 'No prompt'}\n\n"
                                f"Create {count} prompts that span the full spectrum:\n"
                                f"- One grounded/realistic transformation (different location, weather, time of day)\n"
                                f"- One surreal/dreamlike (impossible physics, melting reality, dream logic)\n"
                                f"- One cosmic/abstract (celestial energies, particle dissolution, void spaces)\n"
                                f"- One dark/dramatic (gothic, industrial, underwater, fire)\n\n"
                                f"Each prompt should be 2-3 sentences with specific visual details.\n\n"
                                f"Respond with ONLY a JSON array: [\"prompt 1\", \"prompt 2\", ...]"
                            )}],
                        )
                        import json as _json, re as _re
                        text = response.content[0].text if response.content else "[]"
                        json_match = _re.search(r"\[[\s\S]*\]", text)
                        prompts = _json.loads(json_match.group(0)) if json_match else []
                        _log(f"[job {job_id}] Got {len(prompts)} prompts")

                        # Generate images
                        from beatlab.render.google_video import GoogleVideoClient
                        from beatlab.db import get_meta as _get_meta
                        img_client = GoogleVideoClient(vertex=True)
                        _image_model = _get_meta(project_dir).get("image_model", "replicate/nano-banana-2")
                        candidates_dir = project_dir / "keyframe_candidates" / "candidates" / f"section_{kf_id}"
                        candidates_dir.mkdir(parents=True, exist_ok=True)
                        existing = _next_variant(candidates_dir, ".png") - 1

                        from beatlab.db import update_keyframe
                        paths = []
                        for i, prompt in enumerate(prompts[:count]):
                            v = existing + i + 1
                            out_path = str(candidates_dir / f"v{v}.png")
                            job_manager.update_progress(job_id, i, f"Generating v{v}: {prompt[:50]}...")
                            try:
                                img_client.stylize_image(str(source_img), prompt, out_path, image_model=_image_model)
                                paths.append(f"keyframe_candidates/candidates/section_{kf_id}/v{v}.png")
                                all_cands = sorted([
                                    f"keyframe_candidates/candidates/section_{kf_id}/{f.name}"
                                    for f in candidates_dir.glob("v*.png")
                                ], key=lambda p: int(p.rsplit("v", 1)[-1].split(".")[0]))
                                update_keyframe(project_dir, kf_id, candidates=all_cands)
                            except Exception as e:
                                _log(f"  v{v} failed: {e}")
                                job_manager.update_progress(job_id, i + 1, f"v{v} failed")

                        job_manager.complete_job(job_id, {"keyframeId": kf_id, "candidates": all_cands, "prompts": prompts[:count]})
                    except Exception as e:
                        _log(f"[job {job_id}] FAILED: {e}")
                        job_manager.fail_job(job_id, str(e))

                import threading
                threading.Thread(target=_run_variations, daemon=True).start()
                return self._json_response({"jobId": job_id, "keyframeId": kf_id})

            # POST /api/projects/:name/escalate-keyframe
            m = re.match(r"^/api/projects/([^/]+)/escalate-keyframe$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                kf_id = body.get("keyframeId")
                count = body.get("count", 2)
                if not kf_id: return self._error(400, "BAD_REQUEST", "Missing 'keyframeId'")
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return

                source_img = project_dir / "selected_keyframes" / f"{kf_id}.png"
                if not source_img.exists():
                    return self._error(400, "BAD_REQUEST", f"No source image found for {kf_id}")

                from beatlab.db import get_keyframe
                kf = get_keyframe(project_dir, kf_id)
                kf_prompt = kf.get("prompt", "") if kf else ""

                from beatlab.ws_server import job_manager
                job_id = job_manager.create_job("escalate_keyframe", total=count, meta={"keyframeId": kf_id, "project": m.group(1)})

                def _run_escalate():
                    try:
                        import os
                        from anthropic import Anthropic
                        client_llm = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

                        _log(f"[job {job_id}] Escalating {kf_id} ({count} variants)...")
                        job_manager.update_progress(job_id, 0, "Generating escalated prompts with Claude...")

                        escalate_instruction = (
                            f"Take this keyframe and INTENSIFY it. "
                            f"Push every element further — bolder colors, more dramatic lighting, "
                            f"stronger contrast, more extreme angles, more vivid details. "
                            f"Don't change the subject or concept, just amplify what's already there.\n\n"
                            f"{'Original prompt: ' + kf_prompt[:300] + chr(10) + chr(10) if kf_prompt else ''}"
                            f"Generate {count} escalation prompts, each more intense than the last:\n"
                            f"1. Moderate escalation — same scene, pushed 30% more dramatic\n"
                            f"2. Heavy escalation — same scene, pushed to cinematic extremes\n"
                            f"{'3. Maximum escalation — same scene at its absolute visual peak' + chr(10) if count >= 3 else ''}"
                            f"{'4. Beyond — transcendent version, almost abstract in its intensity' + chr(10) if count >= 4 else ''}\n"
                            f"Each prompt should be 2-3 sentences with specific visual details. "
                            f"Keep the same subject/composition but push every visual property to its extreme.\n\n"
                            f"Respond with ONLY a JSON array: [\"prompt 1\", \"prompt 2\", ...]"
                        )

                        # Always send the image so Claude can see what to intensify
                        import base64 as _b64
                        with open(str(source_img), "rb") as _imgf:
                            img_b64 = _b64.b64encode(_imgf.read()).decode()
                        img_ext = source_img.suffix.lower()
                        img_media = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(img_ext.lstrip("."), "image/png")

                        response = client_llm.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=4096,
                            messages=[{"role": "user", "content": [
                                {"type": "image", "source": {"type": "base64", "media_type": img_media, "data": img_b64}},
                                {"type": "text", "text": escalate_instruction},
                            ]}],
                        )
                        import json as _json, re as _re
                        text = response.content[0].text if response.content else "[]"
                        json_match = _re.search(r"\[[\s\S]*\]", text)
                        prompts = _json.loads(json_match.group(0)) if json_match else []
                        _log(f"[job {job_id}] Got {len(prompts)} escalation prompts")

                        from beatlab.render.google_video import GoogleVideoClient
                        from beatlab.db import get_meta as _get_meta2
                        img_client = GoogleVideoClient(vertex=True)
                        _image_model = _get_meta2(project_dir).get("image_model", "replicate/nano-banana-2")
                        candidates_dir = project_dir / "keyframe_candidates" / "candidates" / f"section_{kf_id}"
                        candidates_dir.mkdir(parents=True, exist_ok=True)
                        existing = _next_variant(candidates_dir, ".png") - 1

                        from beatlab.db import update_keyframe
                        for i, prompt in enumerate(prompts[:count]):
                            v = existing + i + 1
                            out_path = str(candidates_dir / f"v{v}.png")
                            job_manager.update_progress(job_id, i, f"Escalating v{v}: {prompt[:50]}...")
                            try:
                                img_client.stylize_image(str(source_img), prompt, out_path, image_model=_image_model)
                                # Update DB after each image so UI can show it immediately
                                all_cands = sorted([
                                    f"keyframe_candidates/candidates/section_{kf_id}/{f.name}"
                                    for f in candidates_dir.glob("v*.png")
                                ], key=lambda p: int(p.rsplit("v", 1)[-1].split(".")[0]))
                                update_keyframe(project_dir, kf_id, candidates=all_cands)
                            except Exception as e:
                                import traceback as _tb
                                _log(f"  v{v} failed: {type(e).__name__}: {e}")
                                _tb.print_exc()
                                job_manager.update_progress(job_id, i + 1, f"v{v} failed")

                        job_manager.complete_job(job_id, {"keyframeId": kf_id, "candidates": all_cands, "prompts": prompts[:count]})
                    except Exception as e:
                        _log(f"[job {job_id}] FAILED: {e}")
                        job_manager.fail_job(job_id, str(e))

                import threading
                threading.Thread(target=_run_escalate, daemon=True).start()
                _log(f"escalate-keyframe: {kf_id} count={count}")
                return self._json_response({"jobId": job_id, "keyframeId": kf_id})

            # POST /api/projects/:name/copy-transition-style
            m = re.match(r"^/api/projects/([^/]+)/copy-transition-style$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                source_id = body.get("sourceId")
                target_id = body.get("targetId")
                if not source_id or not target_id: return self._error(400, "BAD_REQUEST", "Missing sourceId or targetId")
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import get_transition, update_transition, get_transition_effects, add_transition_effect, delete_transition_effect
                src = get_transition(project_dir, source_id)
                if not src: return self._error(404, "NOT_FOUND", f"Source {source_id} not found")

                # Copy style fields
                style_fields = {}
                for key in ("blend_mode", "opacity", "opacity_curve", "red_curve", "green_curve", "blue_curve", "black_curve", "hue_shift_curve", "saturation_curve", "invert_curve", "chroma_key", "is_adjustment", "hidden", "mask_center_x", "mask_center_y", "mask_radius", "mask_feather", "transform_x", "transform_y", "transform_x_curve", "transform_y_curve", "transform_z_curve", "anchor_x", "anchor_y"):
                    # Copy all style fields including None (clears target's old values)
                    style_fields[key] = src.get(key)
                if style_fields:
                    update_transition(project_dir, target_id, **style_fields)

                # Copy effects: clear existing, then add source's
                existing_fx = get_transition_effects(project_dir, target_id)
                for fx in existing_fx:
                    delete_transition_effect(project_dir, fx["id"])
                for fx in get_transition_effects(project_dir, source_id):
                    add_transition_effect(project_dir, target_id, fx["type"], fx.get("params"))

                _log(f"copy-transition-style: {source_id} -> {target_id} ({len(style_fields)} fields, {len(get_transition_effects(project_dir, source_id))} effects)")
                return self._json_response({"success": True})

            # POST /api/projects/:name/duplicate-transition-video
            m = re.match(r"^/api/projects/([^/]+)/duplicate-transition-video$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                source_id = body.get("sourceId")
                target_id = body.get("targetId")
                if not source_id or not target_id: return self._error(400, "BAD_REQUEST", "Missing sourceId or targetId")
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                import shutil
                src_sel = project_dir / "selected_transitions" / f"{source_id}_slot_0.mp4"
                if src_sel.exists():
                    dst_sel = project_dir / "selected_transitions" / f"{target_id}_slot_0.mp4"
                    dst_sel.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src_sel), str(dst_sel))
                src_cands = project_dir / "transition_candidates" / source_id
                if src_cands.is_dir():
                    dst_cands = project_dir / "transition_candidates" / target_id
                    dst_cands.mkdir(parents=True, exist_ok=True)
                    for slot_dir in src_cands.iterdir():
                        if slot_dir.is_dir():
                            dst_slot = dst_cands / slot_dir.name
                            dst_slot.mkdir(parents=True, exist_ok=True)
                            for f in slot_dir.iterdir():
                                if not (dst_slot / f.name).exists():
                                    shutil.copy2(str(f), str(dst_slot / f.name))
                from beatlab.db import get_transition, update_transition
                src_tr = get_transition(project_dir, source_id)
                if src_tr:
                    updates = {}
                    if src_tr.get("selected"):
                        updates["selected"] = src_tr["selected"]
                    if src_tr.get("action"):
                        updates["action"] = src_tr["action"]
                    if updates:
                        update_transition(project_dir, target_id, **updates)

                # Extract first frame as keyframe image for the target's from-kf
                dst_tr = get_transition(project_dir, target_id)
                if dst_tr and dst_tr.get("from"):
                    from_kf_id = dst_tr["from"]
                    sel_video = project_dir / "selected_transitions" / f"{target_id}_slot_0.mp4"
                    if sel_video.exists():
                        import subprocess as sp
                        import threading
                        def _extract():
                            try:
                                sel_kf_dir = project_dir / "selected_keyframes"
                                sel_kf_dir.mkdir(parents=True, exist_ok=True)
                                sp.run(["ffmpeg", "-y", "-i", str(sel_video), "-vframes", "1", "-q:v", "2",
                                        str(sel_kf_dir / f"{from_kf_id}.png")], capture_output=True, timeout=10)
                            except Exception:
                                pass
                        threading.Thread(target=_extract, daemon=True).start()

                _log(f"duplicate-transition-video: {source_id} -> {target_id}")
                return self._json_response({"success": True})

            # POST /api/projects/:name/update-keyframe-label
            m = re.match(r"^/api/projects/([^/]+)/update-keyframe-label$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import update_keyframe
                kf_id = body["keyframeId"]
                _log(f"update-keyframe-label: {kf_id} label={body.get('label', '')!r}")
                update_keyframe(project_dir, kf_id, label=body.get("label", ""), label_color=body.get("labelColor", ""))
                return self._json_response({"success": True})

            # POST /api/projects/:name/update-transition-label
            m = re.match(r"^/api/projects/([^/]+)/update-transition-label$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import update_transition
                tr_id = body["transitionId"]
                fields = {"label": body.get("label", ""), "label_color": body.get("labelColor", "")}
                if "tags" in body:
                    fields["tags"] = body["tags"]
                _log(f"update-transition-label: {tr_id} label={body.get('label', '')!r} tags={body.get('tags')}")
                update_transition(project_dir, tr_id, **fields)
                return self._json_response({"success": True})

            # POST /api/projects/:name/update-keyframe-style
            m = re.match(r"^/api/projects/([^/]+)/update-keyframe-style$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import update_keyframe
                from beatlab.db import undo_begin as _ub
                _ub(project_dir, f"Update keyframe style {body.get('keyframeId', '')}")
                fields = {}
                if "blendMode" in body:
                    fields["blend_mode"] = body["blendMode"]
                if "opacity" in body:
                    fields["opacity"] = body["opacity"]
                if "refinementPrompt" in body:
                    fields["refinement_prompt"] = body["refinementPrompt"]
                kf_id = body["keyframeId"]
                _log(f"update-keyframe-style: {kf_id} {fields}")
                update_keyframe(project_dir, kf_id, **fields)
                return self._json_response({"success": True})

            # POST /api/projects/:name/update-transition-style
            m = re.match(r"^/api/projects/([^/]+)/update-transition-style$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import update_transition
                from beatlab.db import undo_begin as _ub
                _ub(project_dir, f"Update transition style {body.get('transitionId', '')}")
                fields = {}
                if "blendMode" in body:
                    fields["blend_mode"] = body["blendMode"]
                if "opacity" in body:
                    fields["opacity"] = body["opacity"]
                if "opacityCurve" in body:
                    fields["opacity_curve"] = body["opacityCurve"]
                if "redCurve" in body:
                    fields["red_curve"] = body["redCurve"]
                if "greenCurve" in body:
                    fields["green_curve"] = body["greenCurve"]
                if "blueCurve" in body:
                    fields["blue_curve"] = body["blueCurve"]
                if "blackCurve" in body:
                    fields["black_curve"] = body["blackCurve"]
                if "hueShiftCurve" in body:
                    fields["hue_shift_curve"] = body["hueShiftCurve"]
                if "saturationCurve" in body:
                    fields["saturation_curve"] = body["saturationCurve"]
                if "invertCurve" in body:
                    fields["invert_curve"] = body["invertCurve"]
                if "maskCenterX" in body:
                    fields["mask_center_x"] = body["maskCenterX"]
                if "maskCenterY" in body:
                    fields["mask_center_y"] = body["maskCenterY"]
                if "maskRadius" in body:
                    fields["mask_radius"] = body["maskRadius"]
                if "maskFeather" in body:
                    fields["mask_feather"] = body["maskFeather"]
                if "transformX" in body:
                    fields["transform_x"] = body["transformX"]
                if "transformY" in body:
                    fields["transform_y"] = body["transformY"]
                if "transformXCurve" in body:
                    fields["transform_x_curve"] = body["transformXCurve"]
                if "transformYCurve" in body:
                    fields["transform_y_curve"] = body["transformYCurve"]
                if "transformZCurve" in body:
                    fields["transform_z_curve"] = body["transformZCurve"]
                if "chromaKey" in body:
                    fields["chroma_key"] = body["chromaKey"]
                if "isAdjustment" in body:
                    fields["is_adjustment"] = int(body["isAdjustment"])
                if "hidden" in body:
                    fields["hidden"] = body["hidden"]
                if "anchorX" in body:
                    fields["anchor_x"] = body["anchorX"]
                if "anchorY" in body:
                    fields["anchor_y"] = body["anchorY"]
                tr_id = body["transitionId"]
                _log(f"update-transition-style: {tr_id} {fields}")
                update_transition(project_dir, tr_id, **fields)
                return self._json_response({"success": True})

            # POST /api/projects/:name/assign-keyframe-image
            m = re.match(r"^/api/projects/([^/]+)/assign-keyframe-image$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                kf_id = body.get("keyframeId")
                source_path = body.get("sourcePath")
                if not kf_id or not source_path: return self._error(400, "BAD_REQUEST", "Missing keyframeId or sourcePath")
                import shutil
                src = project_dir / source_path
                if not src.exists(): return self._error(404, "NOT_FOUND", f"Source not found: {source_path}")
                dst = project_dir / "selected_keyframes" / f"{kf_id}.png"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
                # Also create as v1 candidate so it appears in the candidates panel
                cand_dir = project_dir / "keyframe_candidates" / "candidates" / f"section_{kf_id}"
                cand_dir.mkdir(parents=True, exist_ok=True)
                existing = _next_variant(cand_dir, ".png") - 1
                v = existing + 1
                shutil.copy2(str(src), str(cand_dir / f"v{v}.png"))
                from beatlab.db import update_keyframe
                import time as _t
                cache_bust = int(_t.time())
                all_cands = sorted([
                    f"keyframe_candidates/candidates/section_{kf_id}/{f.name}"
                    for f in cand_dir.glob("v*.png")
                ], key=lambda p: int(p.rsplit("v", 1)[-1].split(".")[0]))
                update_keyframe(project_dir, kf_id, selected=v, candidates=all_cands)
                _log(f"assign-keyframe-image: {source_path} -> {kf_id} as v{v} (selected={v})")
                return self._json_response({"success": True, "selected": v})

            # POST /api/projects/:name/transition-effects/add
            m = re.match(r"^/api/projects/([^/]+)/transition-effects/add$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import add_transition_effect
                tr_id = body.get("transitionId")
                etype = body.get("type")
                params = body.get("params", {})
                if not tr_id or not etype: return self._error(400, "BAD_REQUEST", "Missing transitionId or type")
                effect_id = add_transition_effect(project_dir, tr_id, etype, params)
                _log(f"transition-effects/add: {tr_id} type={etype} -> {effect_id}")
                return self._json_response({"success": True, "id": effect_id})

            # POST /api/projects/:name/transition-effects/update
            m = re.match(r"^/api/projects/([^/]+)/transition-effects/update$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import update_transition_effect
                effect_id = body.pop("id", None)
                if not effect_id: return self._error(400, "BAD_REQUEST", "Missing id")
                _log(f"transition-effects/update: {effect_id} {body}")
                update_transition_effect(project_dir, effect_id, **body)
                return self._json_response({"success": True})

            # POST /api/projects/:name/transition-effects/delete
            m = re.match(r"^/api/projects/([^/]+)/transition-effects/delete$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import delete_transition_effect
                fx_id = body.get("id", "")
                _log(f"transition-effects/delete: {fx_id}")
                delete_transition_effect(project_dir, fx_id)
                return self._json_response({"success": True})

            # POST /api/projects/:name/save-as-still
            m = re.match(r"^/api/projects/([^/]+)/save-as-still$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                source_path = body.get("sourcePath")
                name = body.get("name")
                if not source_path: return self._error(400, "BAD_REQUEST", "Missing 'sourcePath'")
                import shutil
                src = project_dir / source_path
                if not src.exists(): return self._error(404, "NOT_FOUND", f"Source not found: {source_path}")
                stills_dir = project_dir / "assets" / "stills"
                stills_dir.mkdir(parents=True, exist_ok=True)
                # Auto-name if not provided
                if not name:
                    name = src.stem + src.suffix
                # Avoid overwriting
                dest = stills_dir / name
                counter = 1
                while dest.exists():
                    dest = stills_dir / f"{src.stem}_{counter}{src.suffix}"
                    counter += 1
                shutil.copy2(str(src), str(dest))
                _log(f"save-as-still: {source_path} -> assets/stills/{dest.name}")
                return self._json_response({"success": True, "name": dest.name, "path": f"assets/stills/{dest.name}"})

            # POST /api/projects/:name/markers/add
            m = re.match(r"^/api/projects/([^/]+)/markers/add$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import add_marker
                marker_id = body.get("id", f"m_{int(__import__('time').time() * 1000)}")
                _log(f"markers/add: {marker_id} time={body.get('time', 0)} label={body.get('label', '')!r}")
                add_marker(project_dir, marker_id, body.get("time", 0), body.get("label", ""), body.get("type", "note"))
                return self._json_response({"success": True, "id": marker_id})

            # POST /api/projects/:name/markers/update
            m = re.match(r"^/api/projects/([^/]+)/markers/update$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import update_marker
                marker_id = body.pop("id", None)
                if not marker_id: return self._error(400, "BAD_REQUEST", "Missing 'id'")
                updates = {k: v for k, v in body.items() if k in ("time", "label", "type")}
                _log(f"markers/update: {marker_id} {updates}")
                update_marker(project_dir, marker_id, **updates)
                return self._json_response({"success": True})

            # POST /api/projects/:name/markers/remove
            m = re.match(r"^/api/projects/([^/]+)/markers/remove$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import delete_marker
                rm_id = body.get("id", "")
                _log(f"markers/remove: {rm_id}")
                delete_marker(project_dir, rm_id)
                return self._json_response({"success": True})

            # POST /api/projects/:name/prompt-roster/add
            m = re.match(r"^/api/projects/([^/]+)/prompt-roster/add$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import add_prompt_roster
                pid = body.get("id", f"pr_{int(__import__('time').time() * 1000)}")
                add_prompt_roster(project_dir, pid, body.get("name", ""), body.get("template", ""), body.get("category", "general"))
                return self._json_response({"success": True, "id": pid})

            # POST /api/projects/:name/prompt-roster/update
            m = re.match(r"^/api/projects/([^/]+)/prompt-roster/update$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import update_prompt_roster
                pid = body.pop("id", None)
                if not pid: return self._error(400, "BAD_REQUEST", "Missing 'id'")
                updates = {k: v for k, v in body.items() if k in ("name", "template", "category")}
                update_prompt_roster(project_dir, pid, **updates)
                return self._json_response({"success": True})

            # POST /api/projects/:name/prompt-roster/remove
            m = re.match(r"^/api/projects/([^/]+)/prompt-roster/remove$", path)
            if m:
                body = self._read_json_body()
                if body is None: return
                project_dir = self._require_project_dir(m.group(1))
                if project_dir is None: return
                from beatlab.db import delete_prompt_roster
                delete_prompt_roster(project_dir, body.get("id", ""))
                return self._json_response({"success": True})

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

            # POST /api/projects/:name/version/commit (deprecated — git removed, no-op)
            m = re.match(r"^/api/projects/([^/]+)/version/commit$", path)
            if m:
                return self._json_response({"success": True, "noChanges": True})

            # POST /api/projects/:name/version/checkout (deprecated)
            m = re.match(r"^/api/projects/([^/]+)/version/checkout$", path)
            if m:
                return self._error(410, "GONE", "Git versioning removed — use checkpoint/restore instead")

            # POST /api/projects/:name/version/branch (deprecated)
            m = re.match(r"^/api/projects/([^/]+)/version/branch$", path)
            if m:
                return self._error(410, "GONE", "Git versioning removed — use checkpoint/restore instead")

            # POST /api/projects/:name/version/delete-branch (deprecated)
            m = re.match(r"^/api/projects/([^/]+)/version/delete-branch$", path)
            if m:
                return self._error(410, "GONE", "Git versioning removed — use checkpoint/restore instead")

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
            _log(f"browse: path={subpath or '/'}")
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
            _log(f"list-projects: listing projects")
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
            _log(f"get-keyframes: {project_name}")
            project_dir = self._get_project_dir(project_name)
            if project_dir is None:
                return self._error(404, "NOT_FOUND", f"Project not found: {project_name}")

            from beatlab.db import get_meta, get_keyframes as db_get_keyframes, get_transitions as db_get_transitions, get_tracks

            # Check if DB exists (auto-import happens in _get_project_dir)
            if not (project_dir / "project.db").exists():
                return self._json_response({
                    "meta": {"title": project_name, "fps": 24, "resolution": [1920, 1080]},
                    "keyframes": [], "transitions": [], "audioFile": None, "projectName": project_name,
                    "tracks": [{"id": "track_1", "name": "Track 1", "zOrder": 0, "blendMode": "normal", "baseOpacity": 1.0, "enabled": True}],
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
                    ], key=lambda p: int(p.rsplit("v", 1)[-1].split(".")[0]))

                ctx = kf.get("context")
                keyframes.append({
                    "id": kf_id,
                    "timestamp": kf.get("timestamp", "0:00"),
                    "section": kf.get("section", ""),
                    "prompt": kf.get("prompt", ""),
                    "selected": kf.get("selected"),
                    "hasSelectedImage": has_selected,
                    "trackId": kf.get("track_id", "track_1"),
                    "label": kf.get("label", ""),
                    "labelColor": kf.get("label_color", ""),
                    "blendMode": kf.get("blend_mode", ""),
                    "refinementPrompt": kf.get("refinement_prompt", ""),
                    "opacity": kf.get("opacity"),
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
            from beatlab.db import get_all_transition_effects
            all_tr_effects = get_all_transition_effects(project_dir)
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
                            ], key=lambda p: int(p.rsplit("v", 1)[-1].split(".")[0]))
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
                    "includeSectionDesc": tr.get("include_section_desc", True),
                    "trackId": tr.get("track_id", "track_1"),
                    "label": tr.get("label", ""),
                    "labelColor": tr.get("label_color", ""),
                    "tags": tr.get("tags", []),
                    "blendMode": tr.get("blend_mode", ""),
                    "opacity": tr.get("opacity"),
                    "opacityCurve": tr.get("opacity_curve"),
                    "redCurve": tr.get("red_curve"),
                    "greenCurve": tr.get("green_curve"),
                    "blueCurve": tr.get("blue_curve"),
                    "blackCurve": tr.get("black_curve"),
                    "hueShiftCurve": tr.get("hue_shift_curve"),
                    "saturationCurve": tr.get("saturation_curve"),
                    "invertCurve": tr.get("invert_curve"),
                    "maskCenterX": tr.get("mask_center_x"),
                    "maskCenterY": tr.get("mask_center_y"),
                    "maskRadius": tr.get("mask_radius"),
                    "maskFeather": tr.get("mask_feather"),
                    "transformX": tr.get("transform_x"),
                    "transformY": tr.get("transform_y"),
                    "transformXCurve": tr.get("transform_x_curve"),
                    "transformYCurve": tr.get("transform_y_curve"),
                    "transformZCurve": tr.get("transform_z_curve"),
                    "anchorX": tr.get("anchor_x"),
                    "anchorY": tr.get("anchor_y"),
                    "chromaKey": tr.get("chroma_key"),
                    "isAdjustment": tr.get("is_adjustment", False),
                    "hidden": tr.get("hidden", False),
                    "candidates": slot_candidates,
                    "hasSelectedVideos": has_selected_videos,
                    "selected": selected_list,
                    "remap": tr.get("remap", {"method": "linear", "target_duration": 0}),
                    "slotKeyframeCandidates": slot_kf_candidates,
                    "selectedSlotKeyframes": selected_slot_kfs,
                    "slotActions": tr.get("slot_actions", []),
                    "effects": all_tr_effects.get(tr_id, []),
                })

            self._json_response({
                "meta": result_meta,
                "keyframes": keyframes,
                "transitions": transitions,
                "audioFile": audio_file,
                "projectName": project_name,
                "tracks": [{
                    "id": t["id"], "name": t["name"], "zOrder": t["z_order"],
                    "blendMode": t["blend_mode"], "baseOpacity": t["base_opacity"],
                    "enabled": t["enabled"], "chromaKey": t.get("chroma_key"), "hidden": t.get("hidden", False),
                } for t in (get_tracks(project_dir) if (project_dir / "project.db").exists() else [{"id": "track_1", "name": "Track 1", "z_order": 0, "blend_mode": "normal", "base_opacity": 1.0, "enabled": True}])],
            })

        def _handle_get_beats(self, project_name: str):
            """GET /api/projects/:name/beats — load beats.json."""
            _log(f"get-beats: {project_name}")
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
                _log(f"select-slot-keyframes: {len(selections)} selections")
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

            _log(f"select-transitions: {len(selections)} selections")
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

                    if variant is not None:
                        source = tr_candidates_root / tr_id / f"slot_{slot_idx}" / f"v{variant}.mp4"
                        dest = selected_dir / f"{tr_id}_slot_{slot_idx}.mp4"
                        if source.exists():
                            shutil.copy2(str(source), str(dest))
                    else:
                        # Deselect — remove the selected video
                        dest = selected_dir / f"{tr_id}_slot_{slot_idx}.mp4"
                        if dest.exists():
                            dest.unlink()

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

            from beatlab.db import undo_begin as _ub
            _ub(project_dir, f"Update timestamp {kf_id} to {new_timestamp}")

            try:
                _log(f"update-timestamp: {kf_id} -> {new_timestamp}")
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
            _log(f"get-bin: {project_name}")
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
                # Only include transitions that have actual video files
                has_video = (project_dir / "selected_transitions" / f"{tr['id']}_slot_0.mp4").exists()
                if not has_video:
                    continue
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
            _log(f"get-descriptions: {project_name}")
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
            _log(f"get-pool: {project_name}")
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
            track_id = body.get("trackId", "track_1")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            from beatlab.db import undo_begin
            undo_begin(project_dir, f"Add keyframe at {timestamp}")

            try:
                from beatlab.db import (
                    add_keyframe as db_add_kf, get_keyframes as db_get_kfs,
                    next_keyframe_id, next_transition_id,
                    add_transition as db_add_tr, update_transition as db_update_tr,
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
                    "candidates": [], "selected": None, "track_id": track_id,
                }
                db_add_kf(project_dir, new_kf)

                # Find timeline neighbors on the same track
                all_kfs = [k for k in db_get_kfs(project_dir) if k.get("track_id", "track_1") == track_id]
                sorted_kfs = sorted(all_kfs, key=lambda k: parse_ts(k["timestamp"]))
                new_idx = next((i for i, k in enumerate(sorted_kfs) if k["id"] == new_id), -1)
                prev_kf = sorted_kfs[new_idx - 1] if new_idx > 0 else None
                next_kf = sorted_kfs[new_idx + 1] if new_idx < len(sorted_kfs) - 1 else None

                # Wire transitions: relink existing spanning transition + create new one
                prev_time = parse_ts(prev_kf["timestamp"]) if prev_kf else None
                next_time = parse_ts(next_kf["timestamp"]) if next_kf else None

                old_tr = None
                if prev_kf and next_kf:
                    all_trs = db_get_trs(project_dir)
                    old_tr = next((t for t in all_trs if t["from"] == prev_kf["id"] and t["to"] == next_kf["id"]), None)

                if old_tr:
                    # Relink: old_tr keeps its video, just points to new_kf now
                    dur_before = round(new_time - prev_time, 2)
                    db_update_tr(project_dir, old_tr["id"], to=new_id, duration_seconds=dur_before,
                                 remap={"method": "linear", "target_duration": dur_before})
                    _log(f"  Relinked {old_tr['id']}: {prev_kf['id']} -> {new_id} (was -> {next_kf['id']})")

                    # Blank transition from new_kf -> next_kf
                    dur_after = round(next_time - new_time, 2)
                    tr2_id = next_transition_id(project_dir)
                    db_add_tr(project_dir, {
                        "id": tr2_id, "from": new_id, "to": next_kf["id"],
                        "duration_seconds": dur_after, "slots": 1,
                        "action": "", "use_global_prompt": False, "selected": None,
                        "remap": {"method": "linear", "target_duration": dur_after},
                        "track_id": track_id,
                    })
                else:
                    # No spanning transition — create new transitions to neighbors
                    if prev_kf:
                        dur_before = round(new_time - prev_time, 2)
                        tr1_id = next_transition_id(project_dir)
                        db_add_tr(project_dir, {
                            "id": tr1_id, "from": prev_kf["id"], "to": new_id,
                            "duration_seconds": dur_before, "slots": 1,
                            "action": "", "use_global_prompt": False, "selected": None,
                            "remap": {"method": "linear", "target_duration": dur_before},
                            "track_id": track_id,
                        })
                    if next_kf:
                        dur_after = round(next_time - new_time, 2)
                        tr2_id = next_transition_id(project_dir)
                        db_add_tr(project_dir, {
                            "id": tr2_id, "from": new_id, "to": next_kf["id"],
                            "duration_seconds": dur_after, "slots": 1,
                            "action": "", "use_global_prompt": False, "selected": None,
                            "remap": {"method": "linear", "target_duration": dur_after},
                            "track_id": track_id,
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

            from beatlab.db import undo_begin
            undo_begin(project_dir, f"Duplicate keyframe {source_id}")

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

                track_id = source_kf.get("track_id", "track_1")
                new_kf = {
                    "id": new_id, "timestamp": timestamp,
                    "section": source_kf.get("section", ""),
                    "source": f"selected_keyframes/{new_id}.png",
                    "prompt": source_kf.get("prompt", ""),
                    "candidates": new_candidates,
                    "selected": source_kf.get("selected"),
                    "track_id": track_id,
                }
                db_add_kf(project_dir, new_kf)

                # Wire up transitions — filter by track, check overlaps, copy properties from spanning tr
                all_kfs = [k for k in db_get_kfs(project_dir)
                           if k.get("track_id", "track_1") == track_id and not k.get("deleted_at")]
                sorted_kfs = sorted(all_kfs, key=lambda k: parse_ts(k["timestamp"]))
                new_idx = next((i for i, k in enumerate(sorted_kfs) if k["id"] == new_id), -1)
                prev_kf = sorted_kfs[new_idx - 1] if new_idx > 0 else None
                next_kf = sorted_kfs[new_idx + 1] if new_idx < len(sorted_kfs) - 1 else None

                from datetime import datetime, timezone
                all_trs = [t for t in db_get_trs(project_dir)
                           if t.get("track_id", "track_1") == track_id and not t.get("deleted_at")]

                # Find and remove ALL transitions that span across the new keyframe's position.
                # This handles cases where multiple transitions cover the insertion point
                # (e.g., a normal tr from kf_B→kf_C AND a bad spanning tr from kf_A→kf_D).
                kf_time_map = {k["id"]: parse_ts(k["timestamp"]) for k in sorted_kfs}
                spanning_trs = []
                for t in all_trs:
                    from_time = kf_time_map.get(t["from"])
                    to_time = kf_time_map.get(t["to"])
                    if from_time is not None and to_time is not None:
                        if from_time < new_time < to_time:
                            spanning_trs.append(t)
                # Use the first one for property inheritance
                old_tr = spanning_trs[0] if spanning_trs else None
                for t in spanning_trs:
                    db_del_tr(project_dir, t["id"], datetime.now(timezone.utc).isoformat())

                # Check for existing transitions to avoid duplicates
                existing_from_prev = any(t["from"] == prev_kf["id"] and t["to"] == new_id for t in all_trs) if prev_kf else False
                existing_to_next = any(t["from"] == new_id and t["to"] == next_kf["id"] for t in all_trs) if next_kf else False

                prev_time = parse_ts(prev_kf["timestamp"]) if prev_kf else None
                next_time = parse_ts(next_kf["timestamp"]) if next_kf else None

                # Build base properties from old spanning transition (if any)
                tr_props = {}
                if old_tr:
                    for prop in ("action", "use_global_prompt", "blend_mode", "opacity",
                                 "opacity_curve", "red_curve", "green_curve", "blue_curve",
                                 "black_curve", "saturation_curve", "hue_shift_curve",
                                 "invert_curve", "chroma_key", "is_adjustment",
                                 "mask_center_x", "mask_center_y", "mask_radius", "mask_feather",
                                 "transform_x", "transform_y", "hidden",
                                 "label", "label_color", "tags"):
                        if old_tr.get(prop) is not None:
                            tr_props[prop] = old_tr[prop]

                if prev_kf and not existing_from_prev:
                    dur_before = round(new_time - prev_time, 2)
                    if dur_before > 0.05:
                        tr1_id = next_transition_id(project_dir)
                        # Copy video candidates and selected from old_tr if it exists
                        tr1_data = {
                            "id": tr1_id, "from": prev_kf["id"], "to": new_id,
                            "duration_seconds": dur_before, "slots": 1,
                            "selected": None,
                            "remap": {"method": "linear", "target_duration": dur_before},
                            "track_id": track_id,
                            **tr_props,
                        }
                        db_add_tr(project_dir, tr1_data)

                        # Copy selected video from old spanning transition
                        if old_tr:
                            old_sel = project_dir / "selected_transitions" / f"{old_tr['id']}_slot_0.mp4"
                            if old_sel.exists():
                                new_sel = project_dir / "selected_transitions" / f"{tr1_id}_slot_0.mp4"
                                new_sel.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(str(old_sel), str(new_sel))
                                from beatlab.db import update_transition
                                update_transition(project_dir, tr1_id, selected=[1])

                            # Copy transition candidates
                            old_cand = project_dir / "transition_candidates" / old_tr["id"]
                            if old_cand.is_dir():
                                new_cand = project_dir / "transition_candidates" / tr1_id
                                new_cand.mkdir(parents=True, exist_ok=True)
                                for slot_dir in old_cand.iterdir():
                                    if slot_dir.is_dir():
                                        dst_slot = new_cand / slot_dir.name
                                        dst_slot.mkdir(parents=True, exist_ok=True)
                                        for f in slot_dir.iterdir():
                                            shutil.copy2(str(f), str(dst_slot / f.name))

                            # Copy transition effects
                            from beatlab.db import get_transition_effects, add_transition_effect
                            for fx in get_transition_effects(project_dir, old_tr["id"]):
                                add_transition_effect(project_dir, tr1_id, fx["type"], fx.get("params"))

                if next_kf and not existing_to_next:
                    dur_after = round(next_time - new_time, 2)
                    if dur_after > 0.05:
                        tr2_id = next_transition_id(project_dir)
                        tr2_data = {
                            "id": tr2_id, "from": new_id, "to": next_kf["id"],
                            "duration_seconds": dur_after, "slots": 1,
                            "selected": None,
                            "remap": {"method": "linear", "target_duration": dur_after},
                            "track_id": track_id,
                            **tr_props,
                        }
                        db_add_tr(project_dir, tr2_data)

                        # Copy video + candidates to second transition too
                        if old_tr:
                            old_sel = project_dir / "selected_transitions" / f"{old_tr['id']}_slot_0.mp4"
                            if old_sel.exists():
                                new_sel = project_dir / "selected_transitions" / f"{tr2_id}_slot_0.mp4"
                                shutil.copy2(str(old_sel), str(new_sel))
                                from beatlab.db import update_transition
                                update_transition(project_dir, tr2_id, selected=[1])

                            old_cand = project_dir / "transition_candidates" / old_tr["id"]
                            if old_cand.is_dir():
                                new_cand = project_dir / "transition_candidates" / tr2_id
                                new_cand.mkdir(parents=True, exist_ok=True)
                                for slot_dir in old_cand.iterdir():
                                    if slot_dir.is_dir():
                                        dst_slot = new_cand / slot_dir.name
                                        dst_slot.mkdir(parents=True, exist_ok=True)
                                        for f in slot_dir.iterdir():
                                            shutil.copy2(str(f), str(dst_slot / f.name))

                            from beatlab.db import get_transition_effects, add_transition_effect
                            for fx in get_transition_effects(project_dir, old_tr["id"]):
                                add_transition_effect(project_dir, tr2_id, fx["type"], fx.get("params"))

                _log(f"  Duplicated {source_id} -> {new_id} at {timestamp} ({len(new_candidates)} candidates copied)")
                self._json_response({"success": True, "keyframe": {"id": new_id, "timestamp": timestamp}})
            except Exception as e:
                _log(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_paste_group(self, project_name: str):
            """POST /api/projects/:name/paste-group — duplicate a group of keyframes+transitions to a new position/track.

            Body: { "keyframeIds": ["kf_001", ...], "targetTime": "0:30.00", "targetTrackId": "track_2" }
            """
            body = self._read_json_body()
            if body is None:
                return

            kf_ids = body.get("keyframeIds", [])
            target_time_str = body.get("targetTime")
            target_track = body.get("targetTrackId", "track_1")
            if not kf_ids or not target_time_str:
                return self._error(400, "BAD_REQUEST", "Missing 'keyframeIds' or 'targetTime'")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            from beatlab.db import undo_begin
            undo_begin(project_dir, f"Paste {len(kf_ids)} keyframes")

            try:
                from beatlab.db import (
                    get_keyframe as db_get_kf, add_keyframe as db_add_kf,
                    get_transitions as db_get_trs, add_transition as db_add_tr,
                    next_keyframe_id, next_transition_id, update_transition,
                )
                import shutil

                def parse_ts(ts):
                    parts = str(ts).split(":")
                    if len(parts) == 2:
                        return int(parts[0]) * 60 + float(parts[1])
                    return float(ts) if isinstance(ts, (int, float)) else 0

                def secs_to_ts(s):
                    m = int(s) // 60
                    sec = s - m * 60
                    return f"{m}:{sec:05.2f}"

                target_time = parse_ts(target_time_str)

                # 1. Read source keyframes, sort by time
                src_kfs = []
                for kid in kf_ids:
                    kf = db_get_kf(project_dir, kid)
                    if kf and not kf.get("deleted_at"):
                        src_kfs.append(kf)
                if not src_kfs:
                    return self._error(404, "NOT_FOUND", "No valid keyframes found")

                src_kfs.sort(key=lambda k: parse_ts(k["timestamp"]))
                min_time = parse_ts(src_kfs[0]["timestamp"])

                # 2. Create new keyframes with offset times
                id_map = {}  # old_kf_id -> new_kf_id
                created_kfs = []
                for src in src_kfs:
                    offset = parse_ts(src["timestamp"]) - min_time
                    new_time = target_time + offset
                    new_ts = secs_to_ts(new_time)
                    new_id = next_keyframe_id(project_dir)
                    id_map[src["id"]] = new_id

                    # Copy candidate files
                    src_cand_dir = project_dir / "keyframe_candidates" / "candidates" / f"section_{src['id']}"
                    dst_cand_dir = project_dir / "keyframe_candidates" / "candidates" / f"section_{new_id}"
                    new_candidates = []
                    if src_cand_dir.exists():
                        dst_cand_dir.mkdir(parents=True, exist_ok=True)
                        for f in sorted(src_cand_dir.iterdir()):
                            if f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                                shutil.copy2(str(f), str(dst_cand_dir / f.name))
                                new_candidates.append(f"keyframe_candidates/candidates/section_{new_id}/{f.name}")

                    # Copy selected keyframe image
                    src_sel = project_dir / "selected_keyframes" / f"{src['id']}.png"
                    if src_sel.exists():
                        dst_sel = project_dir / "selected_keyframes" / f"{new_id}.png"
                        dst_sel.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(src_sel), str(dst_sel))

                    db_add_kf(project_dir, {
                        "id": new_id, "timestamp": new_ts,
                        "section": src.get("section", ""),
                        "source": f"selected_keyframes/{new_id}.png",
                        "prompt": src.get("prompt", ""),
                        "candidates": new_candidates,
                        "selected": src.get("selected"),
                        "track_id": target_track,
                        "label": src.get("label", ""),
                        "label_color": src.get("label_color", ""),
                        "blend_mode": src.get("blend_mode", ""),
                        "opacity": src.get("opacity"),
                    })
                    created_kfs.append({"id": new_id, "timestamp": new_ts})

                # 3. Find transitions between source kfs and duplicate them
                src_kf_set = set(kf_ids)
                all_trs = db_get_trs(project_dir)
                internal_trs = [t for t in all_trs
                                if t["from"] in src_kf_set and t["to"] in src_kf_set
                                and not t.get("deleted_at")]

                # Build existing time ranges on target track for overlap check
                from beatlab.db import get_keyframes as db_get_kfs_paste
                all_kfs_paste = {k["id"]: k for k in db_get_kfs_paste(project_dir) if not k.get("deleted_at")}
                target_trs = [t for t in all_trs
                              if t.get("track_id") == target_track and not t.get("deleted_at")]
                existing_ranges = []
                for t in target_trs:
                    fk = all_kfs_paste.get(t["from"])
                    tk = all_kfs_paste.get(t["to"])
                    if fk and tk:
                        existing_ranges.append((parse_ts(fk["timestamp"]), parse_ts(tk["timestamp"])))

                created_trs = []
                for src_tr in internal_trs:
                    new_from = id_map.get(src_tr["from"])
                    new_to = id_map.get(src_tr["to"])
                    if not new_from or not new_to:
                        continue

                    # Skip zero-length transitions
                    from_ts = parse_ts(next((k["timestamp"] for k in created_kfs if k["id"] == new_from), "0"))
                    to_ts = parse_ts(next((k["timestamp"] for k in created_kfs if k["id"] == new_to), "0"))
                    if to_ts - from_ts <= 0.05:
                        continue

                    # Skip if overlaps an existing transition on the target track
                    overlaps = any(ef < to_ts and et > from_ts for ef, et in existing_ranges)
                    if overlaps:
                        continue

                    new_tr_id = next_transition_id(project_dir)

                    # Copy transition video candidates
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

                    # Copy selected transition video
                    src_sel = project_dir / "selected_transitions" / f"{src_tr['id']}_slot_0.mp4"
                    if src_sel.exists():
                        dst_sel = project_dir / "selected_transitions" / f"{new_tr_id}_slot_0.mp4"
                        dst_sel.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(src_sel), str(dst_sel))

                    db_add_tr(project_dir, {
                        "id": new_tr_id, "from": new_from, "to": new_to,
                        "duration_seconds": src_tr.get("duration_seconds", 0),
                        "slots": src_tr.get("slots", 1),
                        "action": src_tr.get("action", ""),
                        "use_global_prompt": src_tr.get("use_global_prompt", False),
                        "selected": src_tr.get("selected"),
                        "remap": src_tr.get("remap", {"method": "linear", "target_duration": 0}),
                        "track_id": target_track,
                        "blend_mode": src_tr.get("blend_mode", ""),
                        "opacity": src_tr.get("opacity"),
                        "opacity_curve": src_tr.get("opacity_curve"),
                        "red_curve": src_tr.get("red_curve"),
                        "green_curve": src_tr.get("green_curve"),
                        "blue_curve": src_tr.get("blue_curve"),
                        "black_curve": src_tr.get("black_curve"),
                        "hue_shift_curve": src_tr.get("hue_shift_curve"),
                        "saturation_curve": src_tr.get("saturation_curve"),
                        "invert_curve": src_tr.get("invert_curve"),
                        "chroma_key": src_tr.get("chroma_key"),
                        "is_adjustment": src_tr.get("is_adjustment", False),
                        "hidden": src_tr.get("hidden", False),
                        "mask_center_x": src_tr.get("mask_center_x"),
                        "mask_center_y": src_tr.get("mask_center_y"),
                        "mask_radius": src_tr.get("mask_radius"),
                        "mask_feather": src_tr.get("mask_feather"),
                        "transform_x": src_tr.get("transform_x"),
                        "transform_y": src_tr.get("transform_y"),
                        "transform_x_curve": src_tr.get("transform_x_curve"),
                        "transform_y_curve": src_tr.get("transform_y_curve"),
                        "transform_z_curve": src_tr.get("transform_z_curve"),
                        "anchor_x": src_tr.get("anchor_x"),
                        "anchor_y": src_tr.get("anchor_y"),
                        "label": src_tr.get("label", ""),
                        "label_color": src_tr.get("label_color", ""),
                        "tags": src_tr.get("tags", []),
                    })

                    # Copy transition effects
                    from beatlab.db import get_transition_effects, add_transition_effect
                    for fx in get_transition_effects(project_dir, src_tr["id"]):
                        add_transition_effect(project_dir, new_tr_id, fx["type"], fx.get("params"))

                    created_trs.append({"id": new_tr_id, "from": new_from, "to": new_to})

                _log(f"paste-group: {len(created_kfs)} kfs, {len(created_trs)} trs pasted at {target_time_str} on {target_track}")
                self._json_response({
                    "success": True,
                    "keyframes": created_kfs,
                    "transitions": created_trs,
                })
            except Exception as e:
                _log(f"paste-group FAILED: {e}")
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

            from beatlab.db import undo_begin
            undo_begin(project_dir, f"Delete keyframe {kf_id}")

            try:
                from beatlab.db import (
                    get_keyframe, delete_keyframe as db_del_kf,
                    get_transitions_involving, delete_transition as db_del_tr,
                )
                from datetime import datetime, timezone

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

                # Respond immediately — bridging happens in the background
                self._json_response({"success": True, "binned": {"id": kf_id, "deleted_at": now}})

                # Bridge neighbors in a background thread (file I/O can be slow)
                import threading
                def _bridge():
                    try:
                        from beatlab.db import (
                            get_keyframes as db_get_kfs, get_transitions as db_get_trs,
                            next_transition_id, add_transition as db_add_tr,
                        )
                        import os

                        def parse_ts(ts):
                            parts = str(ts).split(":")
                            if len(parts) == 2:
                                return int(parts[0]) * 60 + float(parts[1])
                            return float(ts) if isinstance(ts, (int, float)) else 0

                        removed_time = parse_ts(kf["timestamp"])
                        kf_track = kf.get("track_id", "track_1")
                        all_kfs = [k for k in db_get_kfs(project_dir) if k.get("track_id", "track_1") == kf_track]
                        sorted_kfs = sorted(all_kfs, key=lambda k: parse_ts(k["timestamp"]))
                        prev_kf = None
                        next_kf = None
                        for k in sorted_kfs:
                            t = parse_ts(k["timestamp"])
                            if t < removed_time:
                                prev_kf = k
                            elif t > removed_time and next_kf is None:
                                next_kf = k

                        if not prev_kf or not next_kf:
                            return

                        existing_bridge = next((t for t in db_get_trs(project_dir) if t["from"] == prev_kf["id"] and t["to"] == next_kf["id"]), None)
                        if existing_bridge:
                            _log(f"[delete-kf] {kf_id}: bridge {prev_kf['id']} -> {next_kf['id']} already exists as {existing_bridge['id']}")
                            return

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
                                # Hard link instead of copy — instant, no I/O
                                try:
                                    os.link(str(old_sel), str(new_sel))
                                except OSError:
                                    import shutil
                                    shutil.copy2(str(old_sel), str(new_sel))
                                selected = 1
                                _log(f"[delete-kf] {kf_id}: inherited video from {inherited_tr_id}")
                            if old_cand_dir.exists():
                                new_cand_dir = project_dir / "transition_candidates" / new_tr_id / "slot_0"
                                new_cand_dir.mkdir(parents=True, exist_ok=True)
                                for f in old_cand_dir.iterdir():
                                    try:
                                        os.link(str(f), str(new_cand_dir / f.name))
                                    except OSError:
                                        import shutil
                                        shutil.copy2(str(f), str(new_cand_dir / f.name))

                        db_add_tr(project_dir, {
                            "id": new_tr_id, "from": prev_kf["id"], "to": next_kf["id"],
                            "duration_seconds": dur, "slots": 1,
                            "action": "", "use_global_prompt": False, "selected": selected,
                            "remap": {"method": "linear", "target_duration": dur},
                            "track_id": kf_track,
                        })
                        _log(f"[delete-kf] {kf_id}: bridged {prev_kf['id']} -> {next_kf['id']} as {new_tr_id}")
                    except Exception as e:
                        _log(f"[delete-kf] {kf_id}: bridge ERROR {e}")

                threading.Thread(target=_bridge, daemon=True).start()

            except Exception as e:
                _log(f"[delete-kf] {kf_id}: ERROR {e}")
                import traceback
                traceback.print_exc()
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_batch_delete_keyframes(self, project_name: str):
            """POST /api/projects/:name/batch-delete-keyframes — soft-delete multiple keyframes."""
            body = self._read_json_body()
            if body is None:
                return
            kf_ids = body.get("keyframeIds", [])
            if not kf_ids:
                return self._error(400, "BAD_REQUEST", "Missing 'keyframeIds'")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                from beatlab.db import (
                    get_keyframe, delete_keyframe as db_del_kf, get_keyframes as db_get_kfs,
                    get_transitions_involving, delete_transition as db_del_tr,
                    next_transition_id, add_transition as db_add_tr, get_transitions as db_get_trs,
                )
                from datetime import datetime, timezone

                def parse_ts(ts):
                    parts = str(ts).split(":")
                    if len(parts) == 2:
                        return int(parts[0]) * 60 + float(parts[1])
                    return float(ts) if isinstance(ts, (int, float)) else 0

                now = datetime.now(timezone.utc).isoformat()
                deleted = []
                for kf_id in kf_ids:
                    kf = get_keyframe(project_dir, kf_id)
                    if not kf:
                        continue
                    # Delete orphaned transitions
                    for tr in get_transitions_involving(project_dir, kf_id):
                        db_del_tr(project_dir, tr["id"], now)
                    db_del_kf(project_dir, kf_id, now)
                    deleted.append(kf_id)

                # Bridge gaps: rebuild transitions between remaining neighbors
                all_kfs = db_get_kfs(project_dir)
                sorted_kfs = sorted(all_kfs, key=lambda k: parse_ts(k["timestamp"]))
                all_trs = db_get_trs(project_dir)
                existing_pairs = set((t["from"], t["to"]) for t in all_trs)

                for i in range(len(sorted_kfs) - 1):
                    a = sorted_kfs[i]
                    b = sorted_kfs[i + 1]
                    if (a["id"], b["id"]) not in existing_pairs:
                        dur = round(parse_ts(b["timestamp"]) - parse_ts(a["timestamp"]), 2)
                        tr_id = next_transition_id(project_dir)
                        db_add_tr(project_dir, {
                            "id": tr_id, "from": a["id"], "to": b["id"],
                            "duration_seconds": dur, "slots": 1,
                            "action": "", "use_global_prompt": False, "selected": None,
                            "remap": {"method": "linear", "target_duration": dur},
                        })

                _log(f"[batch-delete-kf] {project_name}: deleted {len(deleted)} keyframes")
                self._json_response({"success": True, "deleted": deleted})
            except Exception as e:
                _log(f"[batch-delete-kf] ERROR: {e}")
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
                _log(f"restore-keyframe: {kf_id}")
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
                _log(f"set-base-image: {kf_id} from {still_name}")
                import shutil
                source = project_dir / "assets" / "stills" / still_name
                if not source.exists():
                    return self._error(404, "NOT_FOUND", f"Still not found: {still_name}")

                dest_dir = project_dir / "selected_keyframes"
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(source), str(dest_dir / f"{kf_id}.png"))

                # Add to candidates so it appears in the candidates panel
                cand_dir = project_dir / "keyframe_candidates" / "candidates" / f"section_{kf_id}"
                cand_dir.mkdir(parents=True, exist_ok=True)
                existing = _next_variant(cand_dir, ".png") - 1
                v = existing + 1
                shutil.copy2(str(source), str(cand_dir / f"v{v}.png"))
                all_cands = sorted([
                    f"keyframe_candidates/candidates/section_{kf_id}/{f.name}"
                    for f in cand_dir.glob("v*.png")
                ], key=lambda p: int(p.rsplit("v", 1)[-1].split(".")[0]))

                from beatlab.db import update_keyframe
                import time as _t
                update_keyframe(project_dir, kf_id, source=f"assets/stills/{still_name}", selected=v, candidates=all_cands)

                self._json_response({"success": True, "keyframeId": kf_id, "still": still_name})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_batch_set_base_image(self, project_name: str):
            """POST /api/projects/:name/batch-set-base-image — set base image for multiple keyframes at once."""
            body = self._read_json_body()
            if body is None:
                return

            items = body.get("items", [])
            if not items:
                return self._error(400, "BAD_REQUEST", "Missing 'items' array of {keyframeId, stillName}")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                import shutil
                from beatlab.db import update_keyframe

                dest_dir = project_dir / "selected_keyframes"
                dest_dir.mkdir(parents=True, exist_ok=True)

                results = []
                for item in items:
                    kf_id = item.get("keyframeId")
                    still_name = item.get("stillName")
                    if not kf_id or not still_name:
                        continue

                    source = project_dir / "assets" / "stills" / still_name
                    if not source.exists():
                        # Also check pool/keyframes
                        source = project_dir / "pool" / "keyframes" / still_name
                    if not source.exists():
                        results.append({"keyframeId": kf_id, "error": f"Still not found: {still_name}"})
                        continue

                    shutil.copy2(str(source), str(dest_dir / f"{kf_id}.png"))

                    cand_dir = project_dir / "keyframe_candidates" / "candidates" / f"section_{kf_id}"
                    cand_dir.mkdir(parents=True, exist_ok=True)
                    existing = _next_variant(cand_dir, ".png") - 1
                    v = existing + 1
                    shutil.copy2(str(source), str(cand_dir / f"v{v}.png"))
                    all_cands = sorted([
                        f"keyframe_candidates/candidates/section_{kf_id}/{f.name}"
                        for f in cand_dir.glob("v*.png")
                    ], key=lambda p: int(p.rsplit("v", 1)[-1].split(".")[0]))

                    update_keyframe(project_dir, kf_id, source=still_name, selected=v, candidates=all_cands)
                    results.append({"keyframeId": kf_id, "success": True})

                _log(f"batch-set-base-image: {len(results)} keyframes updated")
                self._json_response({"success": True, "results": results})
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
                existing = _next_variant(cand_dir, ".mp4") - 1
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
            _log(f"get-bench: {project_name}")
            project_dir = self._get_project_dir(project_name)
            if project_dir is None:
                return self._json_response({"items": []})
            try:
                from beatlab.db import get_bench
                items = get_bench(project_dir)
                self._json_response({"items": items})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_bench_capture(self, project_name: str):
            """POST /api/projects/:name/bench/capture — capture full-res frame at a timeline time and add to bench."""
            body = self._read_json_body()
            if body is None:
                return

            time_sec = body.get("time")
            if time_sec is None:
                return self._error(400, "BAD_REQUEST", "Missing 'time'")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                import subprocess as sp
                import traceback as _tb
                from beatlab.db import get_keyframes, get_transitions, add_to_bench

                track_id = body.get("trackId", "track_1")
                _log(f"bench-capture: {project_name} time={time_sec} track={track_id}")

                def parse_ts(ts):
                    parts = str(ts).split(":")
                    if len(parts) == 2:
                        return int(parts[0]) * 60 + float(parts[1])
                    return float(ts) if isinstance(ts, (int, float)) else 0

                # Find keyframe at or before this time on the selected track
                all_kfs = [kf for kf in get_keyframes(project_dir) if kf.get("deleted_at") is None and kf.get("track_id", "track_1") == track_id]
                sorted_kfs = sorted(all_kfs, key=lambda k: parse_ts(k["timestamp"]))
                current_kf = None
                for k in sorted_kfs:
                    if parse_ts(k["timestamp"]) <= time_sec:
                        current_kf = k
                    else:
                        break

                # Find active transition at this time on the selected track
                all_trs = [tr for tr in get_transitions(project_dir) if tr.get("deleted_at") is None and tr.get("track_id", "track_1") == track_id]
                active_tr = None
                tr_from_time = 0.0
                tr_to_time = 0.0
                kf_map = {k["id"]: k for k in sorted_kfs}
                for tr in all_trs:
                    from_kf = kf_map.get(tr["from"])
                    to_kf = kf_map.get(tr["to"])
                    if not from_kf or not to_kf:
                        continue
                    ft = parse_ts(from_kf["timestamp"])
                    tt = parse_ts(to_kf["timestamp"])
                    sel = tr.get("selected")
                    has_video = sel is not None and sel not in (0, "null", "none", "None")
                    # Also verify the video file actually exists on disk
                    if has_video and ft <= time_sec < tt:
                        video_file = project_dir / "selected_transitions" / f"{tr['id']}_slot_0.mp4"
                        if not video_file.exists():
                            continue
                        active_tr = tr
                        tr_from_time = ft
                        tr_to_time = tt
                        break

                snap_dir = project_dir / "bench_snapshots"
                snap_dir.mkdir(parents=True, exist_ok=True)
                import time as _t
                snap_name = f"bench_{int(_t.time() * 1000)}.png"
                snap_path = snap_dir / snap_name

                _log(f"  current_kf={current_kf['id'] if current_kf else None}, active_tr={active_tr['id'] if active_tr else None}")

                if active_tr:
                    video_path = project_dir / "selected_transitions" / f"{active_tr['id']}_slot_0.mp4"
                    _log(f"  transition path: {video_path} exists={video_path.exists()}")
                    if not video_path.exists():
                        return self._error(404, "NOT_FOUND", f"Transition video not found: {active_tr['id']}")

                    timeline_dur = tr_to_time - tr_from_time
                    progress = (time_sec - tr_from_time) / timeline_dur if timeline_dur > 0 else 0
                    video_dur = active_tr.get("duration_seconds", timeline_dur)
                    video_time = progress * video_dur
                    _log(f"  ffmpeg: -ss {video_time:.3f} from {video_path.name} -> {snap_path}")

                    result = sp.run(
                        ["ffmpeg", "-y", "-ss", str(video_time), "-i", str(video_path),
                         "-vframes", "1", "-q:v", "2", str(snap_path)],
                        capture_output=True, text=True, timeout=10,
                    )
                    if result.returncode != 0:
                        _log(f"  ffmpeg failed: {result.stderr[:300]}")
                    label = f"frame @ {int(time_sec // 60)}:{time_sec % 60:05.2f} ({active_tr['id']})"
                elif current_kf:
                    import shutil
                    kf_img = project_dir / "selected_keyframes" / f"{current_kf['id']}.png"
                    _log(f"  keyframe path: {kf_img} exists={kf_img.exists()}")
                    if kf_img.exists():
                        shutil.copy2(str(kf_img), str(snap_path))
                    else:
                        return self._error(404, "NOT_FOUND", f"No image for {current_kf['id']}")
                    label = f"frame @ {int(time_sec // 60)}:{time_sec % 60:05.2f} ({current_kf['id']})"
                else:
                    _log(f"  no keyframe or transition found at t={time_sec}")
                    return self._error(404, "NOT_FOUND", "No keyframe or transition at this time")

                _log(f"  snap_path exists={snap_path.exists()}")
                if not snap_path.exists():
                    return self._error(500, "INTERNAL_ERROR", "Failed to capture frame")

                source_path = f"bench_snapshots/{snap_name}"
                bench_id = add_to_bench(project_dir, "keyframe", source_path, label)
                _log(f"  success: {source_path} ({bench_id})")
                self._json_response({"success": True, "benchId": bench_id, "sourcePath": source_path})
            except Exception as e:
                _log(f"bench-capture error: {e}")
                _tb.print_exc()
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_bench_upload(self, project_name: str):
            """POST /api/projects/:name/bench/upload — upload a frame snapshot and add to bench."""
            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            try:
                import cgi
                content_type = self.headers.get('Content-Type', '')
                if 'multipart/form-data' not in content_type:
                    return self._error(400, "BAD_REQUEST", "Expected multipart/form-data")

                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)

                # Parse multipart form data
                boundary = content_type.split('boundary=')[-1].encode()
                parts = body.split(b'--' + boundary)
                file_data = None
                file_name = None
                label = ''

                for part in parts:
                    if b'Content-Disposition' not in part:
                        continue
                    header_end = part.find(b'\r\n\r\n')
                    if header_end < 0:
                        continue
                    header = part[:header_end].decode('utf-8', errors='replace')
                    payload = part[header_end + 4:]
                    # Strip trailing \r\n-- boundary marker
                    if payload.endswith(b'\r\n'):
                        payload = payload[:-2]

                    if 'name="file"' in header:
                        file_data = payload
                        # Extract filename
                        for h in header.split('\r\n'):
                            if 'filename=' in h:
                                file_name = h.split('filename=')[-1].strip('"').strip("'")
                    elif 'name="label"' in header:
                        label = payload.decode('utf-8', errors='replace').strip()

                if not file_data or not file_name:
                    return self._error(400, "BAD_REQUEST", "Missing file upload")

                # Save to bench_snapshots directory
                snap_dir = project_dir / "bench_snapshots"
                snap_dir.mkdir(parents=True, exist_ok=True)
                out_path = snap_dir / file_name
                out_path.write_bytes(file_data)

                # Add to bench DB
                from beatlab.db import add_to_bench
                source_path = f"bench_snapshots/{file_name}"
                bench_id = add_to_bench(project_dir, "keyframe", source_path, label or file_name)
                _log(f"bench-upload: {project_name} {source_path} -> {bench_id}")
                self._json_response({"success": True, "benchId": bench_id, "sourcePath": source_path})
            except Exception as e:
                _log(f"bench-upload error: {e}")
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
                _log(f"bench-remove: {bench_id}")
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

                # Inherit track_id from the original transition
                tr_track = tr.get("track_id", "track_1")

                # Create new keyframe at split point
                new_kf_id = next_keyframe_id(project_dir)
                db_add_kf(project_dir, {
                    "id": new_kf_id, "timestamp": to_ts(split_time), "section": "",
                    "source": f"selected_keyframes/{new_kf_id}.png", "prompt": "",
                    "candidates": [], "selected": None, "track_id": tr_track,
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
                    "track_id": tr_track,
                })
                # Re-get next ID after tr1 is committed
                tr2_id = next_transition_id(project_dir)
                db_add_tr(project_dir, {
                    "id": tr2_id, "from": new_kf_id, "to": tr["to"],
                    "duration_seconds": dur2, "slots": 1, "action": tr.get("action", ""),
                    "use_global_prompt": tr.get("use_global_prompt", False), "selected": None,
                    "remap": {"method": "linear", "target_duration": dur2},
                    "track_id": tr_track,
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
                    # Add to candidates and mark as selected
                    import time as _time
                    cand_dir = project_dir / "keyframe_candidates" / "candidates" / f"section_{new_kf_id}"
                    cand_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(sel_kf_dir / f"{new_kf_id}.png"), str(cand_dir / "v1.png"))
                    from beatlab.db import update_keyframe as _upd_kf
                    _upd_kf(project_dir, new_kf_id, selected=1,
                            candidates=[f"keyframe_candidates/candidates/section_{new_kf_id}/v1.png"])

                    def _split_video():
                        try:
                            # Split part 1 — re-encode for precise cut
                            cand1_dir = project_dir / "transition_candidates" / tr1_id / "slot_0"
                            cand1_dir.mkdir(parents=True, exist_ok=True)
                            part1 = cand1_dir / "v1.mp4"
                            sp.run(["ffmpeg", "-y", "-i", str(sel_video), "-t", str(split_at),
                                    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                                    "-c:a", "aac", "-b:a", "192k",
                                    str(part1)], capture_output=True, timeout=60)
                            sel1 = project_dir / "selected_transitions" / f"{tr1_id}_slot_0.mp4"
                            shutil.copy2(str(part1), str(sel1))

                            # Split part 2 — re-encode to get a clean cut (stream copy can't seek precisely)
                            cand2_dir = project_dir / "transition_candidates" / tr2_id / "slot_0"
                            cand2_dir.mkdir(parents=True, exist_ok=True)
                            part2 = cand2_dir / "v1.mp4"
                            sp.run(["ffmpeg", "-y", "-ss", str(split_at), "-i", str(sel_video),
                                    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                                    "-c:a", "aac", "-b:a", "192k",
                                    str(part2)], capture_output=True, timeout=60)
                            sel2 = project_dir / "selected_transitions" / f"{tr2_id}_slot_0.mp4"
                            shutil.copy2(str(part2), str(sel2))

                            # Update DB with selected variants
                            from beatlab.db import update_transition
                            update_transition(project_dir, tr1_id, selected=1)
                            update_transition(project_dir, tr2_id, selected=1)

                            _log(f"  Video split complete: {tr1_id} + {tr2_id}")
                        except Exception as e:
                            _log(f"  Warning: video split failed: {e}")

                    _split_video()  # synchronous — ensures files exist before response

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

                    # Find neighbors ON THE SAME TRACK and split spanning transition
                    track_id = body.get("trackId", "track_1")
                    all_kfs = [k for k in db_get_kfs(project_dir)
                               if k.get("track_id", "track_1") == track_id and not k.get("deleted_at")]
                    sorted_kfs = sorted(all_kfs, key=lambda k: parse_ts(k["timestamp"]))
                    new_idx = next((i for i, k in enumerate(sorted_kfs) if k["id"] == kf_id), -1)
                    prev_kf = sorted_kfs[new_idx - 1] if new_idx > 0 else None
                    next_kf = sorted_kfs[new_idx + 1] if new_idx < len(sorted_kfs) - 1 else None

                    if prev_kf and next_kf:
                        all_trs = [t for t in db_get_trs(project_dir)
                                   if t.get("track_id", "track_1") == track_id and not t.get("deleted_at")]
                        # Find spanning transition between neighbors
                        old_tr = next((t for t in all_trs if t["from"] == prev_kf["id"] and t["to"] == next_kf["id"]), None)

                        # Check no transition already exists from prev/next to new kf
                        existing_from_prev = any(t["from"] == prev_kf["id"] and t["to"] == kf_id for t in all_trs)
                        existing_to_next = any(t["from"] == kf_id and t["to"] == next_kf["id"] for t in all_trs)

                        if old_tr:
                            now = datetime.now(timezone.utc).isoformat()
                            db_del_tr(project_dir, old_tr["id"], now)

                        pt = parse_ts(prev_kf["timestamp"])
                        nt = parse_ts(next_kf["timestamp"])
                        d1, d2 = round(at_time - pt, 2), round(nt - at_time, 2)

                        # Only create transitions if they don't already exist and have positive duration
                        if not existing_from_prev and d1 > 0.05:
                            tr1_id = next_transition_id(project_dir)
                            db_add_tr(project_dir, {"id": tr1_id, "from": prev_kf["id"], "to": kf_id,
                                "duration_seconds": d1, "slots": 1, "action": "", "use_global_prompt": False,
                                "selected": None, "remap": {"method": "linear", "target_duration": d1},
                                "track_id": track_id})
                        if not existing_to_next and d2 > 0.05:
                            tr2_id = next_transition_id(project_dir)
                            db_add_tr(project_dir, {"id": tr2_id, "from": kf_id, "to": next_kf["id"],
                                "duration_seconds": d2, "slots": 1, "action": "", "use_global_prompt": False,
                                "selected": None, "remap": {"method": "linear", "target_duration": d2},
                                "track_id": track_id})

                    self._json_response({"success": True, "type": "keyframe", "id": kf_id})
                else:
                    return self._error(400, "BAD_REQUEST", f"Use 'Assign to TR' for video segments")

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_unlink_keyframe(self, project_name: str):
            """POST /api/projects/:name/unlink-keyframe — remove all transitions touching a keyframe.

            Body: { "keyframeId": "kf_XXX", "side": "both" | "left" | "right" }
            side defaults to "both".
            """
            body = self._read_json_body()
            if body is None:
                return

            kf_id = body.get("keyframeId")
            if not kf_id:
                return self._error(400, "BAD_REQUEST", "Missing 'keyframeId'")

            side = body.get("side", "both")

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            from beatlab.db import undo_begin
            undo_begin(project_dir, f"Unlink keyframe {kf_id}")

            try:
                from beatlab.db import get_transitions_involving, delete_transition as db_del_tr
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc).isoformat()

                orphaned = get_transitions_involving(project_dir, kf_id)
                deleted = []
                for tr in orphaned:
                    if side == "left" and tr["to"] != kf_id:
                        continue
                    if side == "right" and tr["from"] != kf_id:
                        continue
                    db_del_tr(project_dir, tr["id"], now)
                    deleted.append(tr["id"])

                _log(f"unlink-keyframe: {kf_id} side={side} deleted={deleted}")
                self._json_response({"success": True, "deleted": deleted})
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

            project_dir = self._require_project_dir(project_name)
            if project_dir is None:
                return

            from beatlab.db import undo_begin as _ub
            _ub(project_dir, f"Delete transition {tr_id}")

            try:
                _log(f"delete-transition: {tr_id}")
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
                _log(f"restore-transition: {tr_id}")
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

                include_section_desc = body.get("includeSectionDesc")
                updates = {}
                if action is not None:
                    updates["action"] = action
                if use_global is not None:
                    updates["use_global_prompt"] = use_global
                if include_section_desc is not None:
                    updates["include_section_desc"] = include_section_desc
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
                from beatlab.db import undo_begin as _ub_remap
                _ub_remap(project_dir, f"Update transition remap {tr_id}")
                _log(f"update-transition-remap: {tr_id} method={method}")
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
                _log(f"enhance-transition-action: {tr_id}")
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

            _log(f"generate-slot-keyframe-candidates: tr_id={tr_id or 'all'}")
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
            count = body.get("count", 4)
            refinement_prompt = body.get("refinementPrompt")  # optional: refine from selected image
            freeform = body.get("freeform", False)  # optional: generate from prompt only, no base image
            _log(f"generate-keyframe-candidates: {project_name} kf={kf_id} count={count} freeform={freeform} refinement={bool(refinement_prompt)} body_keys={list(body.keys())}")
            if not kf_id:
                return self._error(400, "BAD_REQUEST", "Missing 'keyframeId'")

            project_dir = work_dir / project_name

            # Freeform: generate from prompt text only, no source image needed
            if freeform:
                from beatlab.db import get_keyframe, update_keyframe, get_meta
                kf = get_keyframe(project_dir, kf_id)
                if not kf:
                    return self._error(404, "NOT_FOUND", f"Keyframe {kf_id} not found")
                prompt = kf.get("prompt", "")
                if not prompt:
                    return self._error(400, "BAD_REQUEST", f"Keyframe {kf_id} has no prompt for freeform generation")

                # Determine aspect ratio from project resolution
                meta = get_meta(project_dir)
                resolution = meta.get("resolution", [1920, 1080])
                if isinstance(resolution, list) and len(resolution) == 2:
                    w, h = int(resolution[0]), int(resolution[1])
                else:
                    w, h = 1920, 1080
                from math import gcd
                g = gcd(w, h)
                aspect_ratio = f"{w // g}:{h // g}"
                _log(f"  freeform: resolution={w}x{h} aspect={aspect_ratio}")

                candidates_dir = project_dir / "keyframe_candidates" / "candidates" / f"section_{kf_id}"
                candidates_dir.mkdir(parents=True, exist_ok=True)
                existing_count = _next_variant(candidates_dir, ".png") - 1

                from beatlab.ws_server import job_manager
                job_id = job_manager.create_job("keyframe_candidates", total=count, meta={"keyframeId": kf_id, "project": project_name})

                img_backend = _get_image_backend(project_dir)

                def _run_freeform():
                    try:
                        from beatlab.render.google_video import GoogleVideoClient
                        client = GoogleVideoClient(vertex=True)
                        import time as _time
                        for i in range(count):
                            v = existing_count + i + 1
                            out_path = str(candidates_dir / f"v{v}.png")
                            varied = f"{prompt}, variation {v}" if v > 1 else prompt
                            while True:
                                try:
                                    client.generate_image(varied, out_path, aspect_ratio=aspect_ratio, image_backend=img_backend)
                                    _log(f"  freeform v{v} done")
                                    break
                                except Exception as e:
                                    _log(f"  freeform v{v} failed: {e} — retrying in 60s")
                                    job_manager.update_progress(job_id, i + 1, f"v{v} failed, retrying in 60s...")
                                    _time.sleep(60)
                            job_manager.update_progress(job_id, i + 1, f"v{v} done")

                        all_cands = sorted([
                            f"keyframe_candidates/candidates/section_{kf_id}/{f.name}"
                            for f in candidates_dir.glob("v*.png")
                        ], key=lambda p: int(p.rsplit("v", 1)[-1].split(".")[0]))
                        update_keyframe(project_dir, kf_id, candidates=all_cands)
                        job_manager.complete_job(job_id, {"keyframeId": kf_id, "candidates": all_cands})
                    except Exception as e:
                        job_manager.fail_job(job_id, str(e))

                import threading
                threading.Thread(target=_run_freeform, daemon=True).start()
                return self._json_response({"jobId": job_id, "keyframeId": kf_id})

            # If refinement prompt provided, generate directly from the selected keyframe image
            if refinement_prompt:
                source_img = project_dir / "selected_keyframes" / f"{kf_id}.png"
                if not source_img.exists():
                    return self._error(400, "BAD_REQUEST", f"No selected image for {kf_id} to refine")

                candidates_dir = project_dir / "keyframe_candidates" / "candidates" / f"section_{kf_id}"
                candidates_dir.mkdir(parents=True, exist_ok=True)
                existing_count = _next_variant(candidates_dir, ".png") - 1

                from beatlab.ws_server import job_manager
                job_id = job_manager.create_job("keyframe_candidates", total=count, meta={"keyframeId": kf_id, "project": project_name})

                img_backend = _get_image_backend(project_dir)

                def _run_refine():
                    try:
                        from beatlab.render.google_video import GoogleVideoClient
                        client = GoogleVideoClient(vertex=True)
                        paths = []
                        import time as _time
                        for i in range(count):
                            v = existing_count + i + 1
                            out_path = str(candidates_dir / f"v{v}.png")
                            varied = f"{refinement_prompt}, variation {v}" if v > 1 else refinement_prompt
                            while True:
                                try:
                                    client.transform_image(str(source_img), varied, out_path, image_backend=img_backend)
                                    paths.append(f"keyframe_candidates/candidates/section_{kf_id}/v{v}.png")
                                    break
                                except Exception as e:
                                    _log(f"  v{v} failed: {e} — retrying in 60s")
                                    job_manager.update_progress(job_id, i + 1, f"v{v} failed, retrying in 60s...")
                                    _time.sleep(60)
                            job_manager.update_progress(job_id, i + 1, f"v{v} done")

                        # Persist candidates to DB
                        from beatlab.db import update_keyframe
                        all_cands = sorted([
                            f"keyframe_candidates/candidates/section_{kf_id}/{f.name}"
                            for f in candidates_dir.glob("v*.png")
                        ], key=lambda p: int(p.rsplit("v", 1)[-1].split(".")[0]))
                        update_keyframe(project_dir, kf_id, candidates=all_cands)

                        job_manager.complete_job(job_id, {"keyframeId": kf_id, "candidates": all_cands})
                    except Exception as e:
                        job_manager.fail_job(job_id, str(e))

                import threading
                threading.Thread(target=_run_refine, daemon=True).start()
                return self._json_response({"jobId": job_id, "keyframeId": kf_id})

            # Read keyframe directly from DB — no YAML export needed
            from beatlab.db import get_keyframe, update_keyframe
            kf = get_keyframe(project_dir, kf_id)
            if not kf:
                return self._error(404, "NOT_FOUND", f"Keyframe {kf_id} not found")

            source = kf.get("source", f"selected_keyframes/{kf_id}.png")
            source_path = project_dir / source
            if not source_path.exists():
                # Fallback: try selected_keyframes
                source_path = project_dir / "selected_keyframes" / f"{kf_id}.png"
            if not source_path.exists():
                return self._error(400, "BAD_REQUEST", f"No source image for {kf_id}")

            prompt = kf.get("prompt", "")
            if not prompt:
                return self._error(400, "BAD_REQUEST", f"Keyframe {kf_id} has no prompt")

            candidates_dir = project_dir / "keyframe_candidates" / "candidates" / f"section_{kf_id}"
            candidates_dir.mkdir(parents=True, exist_ok=True)
            existing_count = _next_variant(candidates_dir, ".png") - 1

            from beatlab.ws_server import job_manager
            job_id = job_manager.create_job("keyframe_candidates", total=count, meta={"keyframeId": kf_id, "project": project_name})

            _log(f"  stylize: {kf_id} source={source_path.name} prompt={prompt[:60]!r} count={count} existing={existing_count}")

            def _run():
                try:
                    from beatlab.render.google_video import GoogleVideoClient
                    from beatlab.db import get_meta as _get_meta_gen
                    from concurrent.futures import ThreadPoolExecutor
                    client = GoogleVideoClient(vertex=True)
                    _img_model = _get_meta_gen(project_dir).get("image_model", "replicate/nano-banana-2")

                    def _gen_one(v):
                        import time as _time
                        out_path = str(candidates_dir / f"v{v}.png")
                        if Path(out_path).exists():
                            job_manager.update_progress(job_id, v - existing_count, f"v{v} cached")
                            return
                        varied = f"{prompt}, variation {v}" if v > 1 else prompt
                        while True:
                            try:
                                client.stylize_image(str(source_path), varied, out_path, image_model=_img_model)
                                _log(f"    {kf_id} v{v} done")
                                break
                            except Exception as e:
                                _log(f"    {kf_id} v{v} FAILED: {e} — retrying in 60s")
                                job_manager.update_progress(job_id, v - existing_count, f"v{v} failed, retrying in 60s...")
                                _time.sleep(60)
                        job_manager.update_progress(job_id, v - existing_count, f"v{v}")

                    variants = list(range(existing_count + 1, existing_count + count + 1))
                    with ThreadPoolExecutor(max_workers=count) as pool:
                        pool.map(_gen_one, variants)

                    all_cands = sorted([
                        f"keyframe_candidates/candidates/section_{kf_id}/{f.name}"
                        for f in candidates_dir.glob("v*.png")
                    ], key=lambda p: int(p.rsplit("v", 1)[-1].split(".")[0]))
                    update_keyframe(project_dir, kf_id, candidates=all_cands)

                    job_manager.complete_job(job_id, {"keyframeId": kf_id, "candidates": all_cands})
                except Exception as e:
                    job_manager.fail_job(job_id, str(e))

            import threading
            from pathlib import Path
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
            use_next_tr_frame = body.get("useNextTransitionFrame", False)  # use first frame of next transition's video as end frame
            no_end_frame = body.get("noEndFrame", False)  # generate from start image only, no end frame conditioning
            _log(f"[generate-transition-candidates] tr={tr_id} count={count} duration={duration} useNextTrFrame={use_next_tr_frame} noEndFrame={no_end_frame} (body keys: {list(body.keys())})")
            if not tr_id:
                return self._error(400, "BAD_REQUEST", "Missing 'transitionId'")

            project_dir = work_dir / project_name

            # Read transition directly from DB — no YAML export needed
            from beatlab.db import get_transition, get_meta
            tr = get_transition(project_dir, tr_id)
            if not tr:
                return self._error(404, "NOT_FOUND", f"Transition {tr_id} not found")

            meta = get_meta(project_dir)
            motion_prompt = meta.get("motionPrompt") or meta.get("motion_prompt") or ""
            max_seconds = duration or meta.get("transition_max_seconds") or 8

            from_kf_id = tr["from"]
            to_kf_id = tr["to"]
            n_slots = tr.get("slots", 1)
            tr_duration = tr.get("duration_seconds", 0)
            action = tr.get("action") or "Smooth cinematic transition"

            selected_kf_dir = project_dir / "selected_keyframes"
            start_img = str(selected_kf_dir / f"{from_kf_id}.png")
            end_img = str(selected_kf_dir / f"{to_kf_id}.png")

            from pathlib import Path as _Path
            if not _Path(start_img).exists():
                return self._error(400, "BAD_REQUEST", f"Start keyframe image not found: {from_kf_id}")

            # If useNextTransitionFrame: extract first frame from the next transition's selected video as end frame
            if use_next_tr_frame:
                from beatlab.db import get_transitions as _get_all_trs
                all_trs = _get_all_trs(project_dir)
                # Find transitions starting from to_kf on the same track
                next_tr = next((t for t in all_trs if t["from"] == to_kf_id and t.get("track_id") == tr.get("track_id")), None)
                if next_tr:
                    next_sel_video = project_dir / "selected_transitions" / f"{next_tr['id']}_slot_0.mp4"
                    if next_sel_video.exists():
                        # Extract first frame
                        import subprocess as _sp
                        extracted = project_dir / "selected_keyframes" / f"_next_tr_start_{tr_id}.png"
                        extracted.parent.mkdir(parents=True, exist_ok=True)
                        _sp.run(["ffmpeg", "-y", "-i", str(next_sel_video), "-vframes", "1", "-q:v", "2", str(extracted)],
                                capture_output=True, timeout=10)
                        if extracted.exists():
                            end_img = str(extracted)
                            _log(f"  useNextTransitionFrame: using first frame of {next_tr['id']} as end image")
                        else:
                            _log(f"  useNextTransitionFrame: ffmpeg extraction failed, falling back to keyframe")
                    else:
                        _log(f"  useNextTransitionFrame: next transition {next_tr['id']} has no selected video, falling back to keyframe")
                else:
                    _log(f"  useNextTransitionFrame: no next transition from {to_kf_id}, falling back to keyframe")

            if not no_end_frame and not _Path(end_img).exists():
                return self._error(400, "BAD_REQUEST", f"End keyframe image not found: {to_kf_id}")

            # Count existing candidates
            tr_candidates_dir = project_dir / "transition_candidates" / tr_id
            existing_count = 0
            if tr_candidates_dir.exists():
                if slot_index is not None:
                    slot_dir = tr_candidates_dir / f"slot_{slot_index}"
                    if slot_dir.exists():
                        existing_count = _next_variant(slot_dir, ".mp4") - 1
                else:
                    for sd in tr_candidates_dir.iterdir():
                        if sd.is_dir():
                            existing_count = max(existing_count, len(list(sd.glob("v*.mp4"))))

            prompt = f"{action}. Camera and motion: {motion_prompt}" if motion_prompt else action
            if duration:
                slot_duration = duration
            else:
                slot_duration = min(max_seconds, tr_duration / n_slots) if tr_duration > 0 else max_seconds

            _log(f"  veo: {tr_id} {from_kf_id}→{to_kf_id} prompt={prompt[:60]!r} dur={slot_duration}s count={count} existing={existing_count}")

            from beatlab.ws_server import job_manager
            job_id = job_manager.create_job("transition_candidates", total=count, meta={"transitionId": tr_id, "project": project_name})

            vid_backend = _get_video_backend(project_dir)

            def _run():
                try:
                    from beatlab.render.google_video import GoogleVideoClient, PromptRejectedError
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    from pathlib import Path as _Path

                    if vid_backend.startswith("runway"):
                        from beatlab.render.google_video import RunwayVideoClient
                        _, _, runway_model = vid_backend.partition("/")
                        client = RunwayVideoClient(model=runway_model or "veo3.1_fast")
                        _log(f"  Using Runway backend: {runway_model or 'veo3.1_fast'}")
                    else:
                        client = GoogleVideoClient(vertex=True)
                    job_manager.update_progress(job_id, 0, f"Starting video generation ({vid_backend})...")

                    # Build jobs for each slot + variant
                    gen_jobs = []
                    for si in range(n_slots):
                        if slot_index is not None and si != slot_index:
                            continue
                        slot_dir = project_dir / "transition_candidates" / tr_id / f"slot_{si}"
                        slot_dir.mkdir(parents=True, exist_ok=True)

                        # For multi-slot, use slot keyframes if available
                        s_img = start_img if si == 0 else str(project_dir / "selected_slot_keyframes" / f"{tr_id}_slot_{si - 1}.png")
                        e_img = end_img if si == n_slots - 1 else str(project_dir / "selected_slot_keyframes" / f"{tr_id}_slot_{si}.png")
                        if not _Path(s_img).exists():
                            s_img = start_img
                        if not _Path(e_img).exists():
                            e_img = end_img

                        slot_existing = _next_variant(slot_dir, ".mp4") - 1
                        for v in range(slot_existing + 1, slot_existing + count + 1):
                            output = str(slot_dir / f"v{v}.mp4")
                            if _Path(output).exists():
                                continue
                            gen_jobs.append({"slot": si, "variant": v, "start": s_img, "end": e_img, "output": output})

                    if not gen_jobs:
                        job_manager.complete_job(job_id, {"transitionId": tr_id, "candidates": {}})
                        return

                    _log(f"[job {job_id}] Generating {len(gen_jobs)} Veo clips for {tr_id}...")
                    completed_count = [0]
                    rejected = []

                    def _gen(j):
                        try:
                            if no_end_frame:
                                client.generate_video_from_image(
                                    image_path=j["start"],
                                    prompt=prompt,
                                    output_path=j["output"],
                                    duration_seconds=int(slot_duration),
                                )
                            else:
                                client.generate_video_transition(
                                    start_frame_path=j["start"],
                                    end_frame_path=j["end"],
                                    prompt=prompt,
                                    output_path=j["output"],
                                    duration_seconds=int(slot_duration),
                                    on_status=lambda msg: job_manager.update_progress(job_id, completed_count[0], msg),
                                )
                            completed_count[0] += 1
                            _log(f"    {tr_id} slot_{j['slot']} v{j['variant']} done")
                        except PromptRejectedError as e:
                            # Retry prompt rejections up to 5 times (content filters can be flaky)
                            max_rejection_retries = 5
                            succeeded = False
                            for retry_i in range(max_rejection_retries):
                                _log(f"    ⚠ PROMPT REJECTED (attempt {retry_i + 1}/{max_rejection_retries}): {tr_id} — {e}")
                                job_manager.update_progress(job_id, completed_count[0], f"Prompt rejected, retrying ({retry_i + 1}/{max_rejection_retries})...")
                                import time as _time
                                _time.sleep(5 * (retry_i + 1))
                                try:
                                    if no_end_frame:
                                        client.generate_video_from_image(
                                            image_path=j["start"], prompt=prompt,
                                            output_path=j["output"], duration_seconds=int(slot_duration),
                                        )
                                    else:
                                        client.generate_video_transition(
                                            start_frame_path=j["start"], end_frame_path=j["end"],
                                            prompt=prompt, output_path=j["output"],
                                            duration_seconds=int(slot_duration),
                                            on_status=lambda msg: job_manager.update_progress(job_id, completed_count[0], msg),
                                        )
                                    completed_count[0] += 1
                                    _log(f"    {tr_id} slot_{j['slot']} v{j['variant']} succeeded on retry {retry_i + 1}")
                                    succeeded = True
                                    break
                                except PromptRejectedError:
                                    continue
                                except Exception:
                                    break
                            if not succeeded:
                                _log(f"    ⚠ PROMPT REJECTED after {max_rejection_retries} retries: {tr_id}")
                                rejected.append(tr_id)
                                job_manager.update_progress(job_id, completed_count[0], f"⚠ {tr_id}: prompt rejected after {max_rejection_retries} attempts")
                        except Exception as e:
                            _log(f"    ⚠ {tr_id} slot_{j['slot']} v{j['variant']} FAILED: {e}")

                    with ThreadPoolExecutor(max_workers=min(len(gen_jobs), 4)) as pool:
                        futures = [pool.submit(_gen, j) for j in gen_jobs]
                        for f in as_completed(futures):
                            try:
                                f.result()
                            except Exception:
                                pass

                    # Collect results
                    candidates = {}
                    tr_cand_dir = project_dir / "transition_candidates" / tr_id
                    if tr_cand_dir.exists():
                        for sd in sorted(tr_cand_dir.iterdir()):
                            if sd.is_dir():
                                videos = sorted([
                                    f"transition_candidates/{tr_id}/{sd.name}/{f.name}"
                                    for f in sd.glob("v*.mp4")
                                ])
                                candidates[sd.name] = videos

                    result = {"transitionId": tr_id, "candidates": candidates}
                    if rejected:
                        result["rejected"] = rejected
                        result["rejectionMessage"] = f"Prompt rejected for {len(rejected)} variant(s) after retries"
                    job_manager.complete_job(job_id, result)
                except Exception as e:
                    _log(f"[job {job_id}] FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    err = str(e)
                    if "transient" in err.lower() or "None" in err:
                        job_manager.fail_job(job_id, f"Veo returned empty results after retries — try again. ({err[:80]})")
                    else:
                        job_manager.fail_job(job_id, err)

            import threading
            from pathlib import Path
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
                _log(f"update-meta: {list(k for k in ('motion_prompt', 'default_transition_prompt', 'image_model') if k in body)}")
                from beatlab.db import get_meta, set_meta
                meta = get_meta(project_dir)
                for key in ("motion_prompt", "default_transition_prompt", "image_model"):
                    if key in body:
                        set_meta(project_dir, key, body[key])
                        meta[key] = body[key]

                self._json_response({"success": True, "meta": meta})
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_get_audio_intelligence(self, project_name: str):
            """GET /api/projects/:name/audio-intelligence — return processed beat events from audio intelligence file."""
            _log(f"get-audio-intelligence: {project_name}")
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

                # Extract onsets-only from layer1 for client-side rule preview (~1.5MB)
                layer1 = data.get("layer1", {})
                onsets = {}
                for stem, bands in layer1.items():
                    onsets[stem] = {}
                    for band, bdata in bands.items():
                        onsets[stem][band] = bdata.get("onsets", [])

                self._json_response({
                    "activeFile": ai_file,
                    "availableFiles": available,
                    "events": events,
                    "sections": sections,
                    "rules": rules,
                    "ruleCount": len(rules),
                    "onsets": onsets,
                })
            except Exception as e:
                self._error(500, "INTERNAL_ERROR", str(e))

        def _handle_get_settings(self, project_name: str):
            """GET /api/projects/:name/settings — read project settings from settings.yaml."""
            _log(f"get-settings: {project_name}")
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

            _log(f"update-settings: settings updated")
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
            _log(f"get-watched-folders: {project_name}")
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
            _log(f"get-effects: {project_name}")
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
                from beatlab.db import save_effects, get_suppressions
                effects = body.get("effects", [])
                # Only update suppressions if explicitly provided in the request
                if "suppressions" in body:
                    suppressions = body["suppressions"]
                else:
                    suppressions = get_suppressions(project_dir)
                _log(f"update-effects: {len(effects)} effects, {len(suppressions)} suppressions (suppressions_in_body={'suppressions' in body})")
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
                _log(f"watch-folder: {folder_path}")
                result = folder_watcher.add_watch(project_name, folder_path)

                # Persist to DB
                from beatlab.db import get_meta, set_meta
                project_dir = work_dir / project_name
                meta = get_meta(project_dir)
                watched = meta.get("watched_folders", [])
                if not isinstance(watched, list):
                    watched = []
                if folder_path not in watched:
                    watched.append(folder_path)
                set_meta(project_dir, "watched_folders", watched)

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

            _log(f"unwatch-folder: {folder_path}")
            from beatlab.ws_server import folder_watcher
            if folder_watcher:
                folder_watcher.remove_watch(project_name, folder_path)

            # Remove from DB
            from beatlab.db import get_meta, set_meta
            project_dir = work_dir / project_name
            if project_dir.is_dir():
                meta = get_meta(project_dir)
                watched = meta.get("watched_folders", [])
                if not isinstance(watched, list):
                    watched = []
                if folder_path in watched:
                    watched.remove(folder_path)
                set_meta(project_dir, "watched_folders", watched)

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
                _log(f"import: source={source_path}")
                from beatlab.db import get_db, add_keyframe, add_transition, next_keyframe_id, next_transition_id
                import shutil
                from datetime import datetime, timezone

                project_dir = work_dir / project_name
                get_db(project_dir)  # ensure DB exists

                # Get starting IDs
                kf_num = int(next_keyframe_id(project_dir).replace("kf_", ""))
                tr_num = int(next_transition_id(project_dir).replace("tr_", ""))

                # Collect files
                IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}
                VIDEO_EXTS = {'.mp4', '.webm', '.mov'}

                files = sorted(source.iterdir()) if source.is_dir() else [source]

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
                        kf_id = f"kf_{kf_num:03d}"
                        kf_num += 1
                        dest = selected_kf_dir / f"{kf_id}.png"
                        shutil.copy2(str(f), str(dest))

                        add_keyframe(project_dir, {
                            "id": kf_id,
                            "timestamp": format_ts(current_ts),
                            "section": "",
                            "source": str(f),
                            "prompt": f"Imported from {f.name}",
                            "context": None,
                            "candidates": [],
                            "selected": 1,
                            "deleted_at": now,
                        })
                        imported_kf.append(kf_id)
                        current_ts += 1.0

                    elif ext in VIDEO_EXTS:
                        tr_id = f"tr_{tr_num:03d}"
                        tr_num += 1
                        dest = selected_tr_dir / f"{tr_id}_slot_0{ext}"
                        shutil.copy2(str(f), str(dest))

                        add_transition(project_dir, {
                            "id": tr_id,
                            "from": "",
                            "to": "",
                            "duration_seconds": 0,
                            "slots": 1,
                            "action": f"Imported from {f.name}",
                            "selected": [],
                            "remap": {"method": "linear", "target_duration": 0},
                            "deleted_at": now,
                        })
                        imported_tr.append(tr_id)

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
            _log(f"ls: {project_name} path={subpath or '/'}")
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

        def _handle_image_thumb(self, project_name: str, file_path: str):
            """GET /api/projects/:name/thumb/{path} — serve a resized thumbnail of an image, cached to .thumbs/."""
            project_dir = self._get_project_dir(project_name)
            if project_dir is None:
                return self._error(404, "NOT_FOUND", "Project not found")

            source = project_dir / file_path
            if not source.exists() or source.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp'):
                return self._error(404, "NOT_FOUND", f"Image not found: {file_path}")

            # Cache dir: .thumbs/ mirrors the source path
            thumb_dir = project_dir / ".thumbs" / Path(file_path).parent
            thumb_dir.mkdir(parents=True, exist_ok=True)
            thumb_path = thumb_dir / source.name

            if not thumb_path.exists() or thumb_path.stat().st_mtime < source.stat().st_mtime:
                from PIL import Image as _PILImage
                with _PILImage.open(str(source)) as img:
                    img.thumbnail((256, 256), _PILImage.LANCZOS)
                    img.save(str(thumb_path), "JPEG", quality=80)

            with open(str(thumb_path), "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(data)

        def _handle_video_thumbnail(self, project_name: str, file_path: str):
            """GET /api/projects/:name/thumbnail/* — extract and serve first frame of a video as JPEG."""
            _log(f"video-thumbnail: {file_path}")
            import subprocess as sp
            import tempfile

            full_path = (work_dir / project_name / file_path).resolve()
            if not str(full_path).startswith(str(work_dir.resolve())):
                return self._error(403, "FORBIDDEN", "Path traversal denied")
            if not full_path.exists():
                return self._error(404, "NOT_FOUND", f"File not found: {file_path}")

            # Check for cached thumbnail next to the video
            thumb_path = full_path.with_suffix(".thumb.jpg")
            if not thumb_path.exists():
                try:
                    sp.run(
                        ["ffmpeg", "-y", "-i", str(full_path), "-vframes", "1",
                         "-vf", "scale=320:-1", "-q:v", "4", str(thumb_path)],
                        capture_output=True, timeout=10,
                    )
                except Exception:
                    pass

            if thumb_path.exists():
                data = thumb_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "public, max-age=86400")
                self._cors_headers()
                self.end_headers()
                self.wfile.write(data)
                return

            # Fallback: couldn't generate thumbnail
            return self._error(500, "INTERNAL_ERROR", "Failed to generate thumbnail")

        def _handle_serve_file(self, project_name: str, file_path: str):
            """GET /api/projects/:name/files/* — serve project files with Range support and caching."""
            _log(f"serve-file: {file_path}")
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
            """GET /api/projects/:name/narrative — return sections from DB."""
            _log(f"get-narrative: {project_name}")
            project_dir = self._require_project_dir(project_name)
            if project_dir is None: return
            from beatlab.db import get_sections
            self._json_response({"sections": get_sections(project_dir)})

        def _handle_update_narrative(self, project_name: str):
            """POST /api/projects/:name/narrative — update sections in DB."""
            body = self._read_json_body()
            if body is None:
                return
            project_dir = self._require_project_dir(project_name)
            if project_dir is None: return
            from beatlab.db import get_sections, set_sections
            sections = body.get("sections")
            if sections is not None:
                _log(f"update-narrative: {len(sections)} sections")
                set_sections(project_dir, sections)
            result = get_sections(project_dir)
            self._json_response({"success": True, "sections": len(result)})

        def _handle_get_timelines(self, project_name: str):
            """GET /api/projects/:name/timelines — list available timelines."""
            _log(f"get-timelines: {project_name}")
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
                _log(f"timeline-switch: {name}")
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
                _log(f"timeline-import: source={source_path}")
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
                _log(f"timeline-create: {name}")
                create_timeline(project_dir, name, copy_from)
                self._json_response({"success": True, "name": name})
            except ValueError as e:
                self._error(400, "BAD_REQUEST", str(e))

        # ── Git Version Handlers ─────────────────────────────────

        # Git versioning removed — replaced by checkpoint-based backups.
        # version/* endpoints return no-ops or 410 GONE above.

        def _handle_version_history(self, project_name: str):
            """GET /api/projects/:name/version/history — DEPRECATED (git removed)"""
            _log(f"version-history: {project_name}")
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
            _log(f"version-checkout: {sha}")

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
            _log(f"version-branch: {name} create={create}")

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
            _log(f"version-diff: {project_name}")
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
            _log(f"version-delete-branch: {name}")

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
            _log(f"get-section-settings: section={section_label}")

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
                _log(f"section-settings: {section_label}")
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
                    from beatlab.db import get_meta as _get_meta_stg
                    client = GoogleVideoClient(vertex=True)
                    _img_model = _get_meta_stg(project_dir).get("image_model", "replicate/nano-banana-2")

                    staging_dir = project_dir / "staging" / staging_id
                    staging_dir.mkdir(parents=True, exist_ok=True)

                    paths = []
                    existing = _next_variant(staging_dir, ".png") - 1
                    for i in range(count):
                        v = existing + i + 1
                        out_path = str(staging_dir / f"v{v}.png")
                        if Path(out_path).exists():
                            paths.append(f"staging/{staging_id}/v{v}.png")
                            continue
                        varied = f"{prompt}, variation {v}" if v > 1 else prompt
                        try:
                            client.stylize_image(str(source), varied, out_path, image_model=_img_model)
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
                import json as _json
                import re as _re

                BATCH_SIZE = 30  # max events per Claude call

                system_prompt = (
                    f"You are a visionary art director creating keyframe images for a cinematic music video. "
                    f"Each prompt will transform a base photograph (\"{base_still}\") into a vivid scene "
                    f"through Imagen style transfer.\n\n"
                    f"Your prompts should span the full spectrum from concrete to abstract. Mix freely between:\n\n"
                    f"CONCRETE — tangible places and scenes:\n"
                    f"- A mist-shrouded ancient forest with bioluminescent fungi pulsing on twisted bark\n"
                    f"- A haunting gothic cathedral where stained glass bleeds liquid color onto stone floors\n"
                    f"- An underwater ballroom where jellyfish chandeliers illuminate drowned aristocrats\n\n"
                    f"ABSTRACT — celestial, cosmic, and ethereal:\n"
                    f"- Entities of pure light floating in infinite black space, trailing ribbons of golden plasma\n"
                    f"- Celestial energies carved into the sky like cracks in reality, violet and amber fire bleeding through\n"
                    f"- A figure dissolving into thousands of luminous particles drifting upward like inverse rain\n"
                    f"- Geometric mandalas of living crystal rotating in a void of deep indigo, humming with color\n"
                    f"- The subject's silhouette filled with a galaxy, stars spilling from their edges like sand\n\n"
                    f"Match the prompt style to the musical energy:\n"
                    f"- Quiet/intimate → dreamlike, ethereal, delicate abstractions or whispered landscapes\n"
                    f"- Building/rising → transformative, things becoming other things, reality bending\n"
                    f"- Loud/climactic → explosive cosmic events, overwhelming scale, sensory overload\n"
                    f"- Descending/fading → dissolution, particles scattering, light dimming into beautiful darkness\n\n"
                    f"Section: \"{section_label}\"\n"
                    f"Musical description:\n{section_content}\n\n"
                    f"For each event, write a prompt (2-3 sentences) that:\n"
                    f"- Creates a SPECIFIC visual — whether a real place, an impossible space, or a cosmic abstraction\n"
                    f"- Includes concrete visual details even for abstract scenes: what material, what light, what color, what texture\n"
                    f"- Varies WILDLY across events — alternate between grounded and transcendent\n"
                    f"- Treats the base image as the subject transformed by or placed within this vision\n\n"
                    f"Respond with ONLY a JSON array, no markdown fences: [{{\"eventIndex\": N, \"prompt\": \"...\"}}, ...]"
                )

                all_suggestions = []
                batches = [events[i:i + BATCH_SIZE] for i in range(0, len(events), BATCH_SIZE)]
                _log(f"  Processing {len(batches)} batch(es) of up to {BATCH_SIZE} events each")

                for batch_idx, batch in enumerate(batches):
                    event_list = "\n".join(
                        f"  {ev.get('_originalIndex', i)}: t={ev.get('time', 0):.2f}s, stem={ev.get('stem_source', '?')}, "
                        f"effect={ev.get('effect', '?')}, intensity={ev.get('intensity', 0) * 100:.0f}%"
                        for i, ev in enumerate(batch)
                    )
                    # Tag events with their original index for multi-batch
                    for i, ev in enumerate(batch):
                        if "_originalIndex" not in ev:
                            ev["_originalIndex"] = batch_idx * BATCH_SIZE + i

                    batch_prompt = f"{system_prompt}\n\nAudio events:\n{event_list}"

                    batch_suggestions = None
                    for attempt in range(3):
                        response = client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=16384,
                            messages=[{"role": "user", "content": batch_prompt}],
                        )
                        text = response.content[0].text if response.content else ""
                        _log(f"  Batch {batch_idx + 1}/{len(batches)} attempt {attempt + 1}: len={len(text)}, stop={response.stop_reason}")

                        json_match = _re.search(r"\[[\s\S]*\]", text)
                        if json_match:
                            try:
                                batch_suggestions = _json.loads(json_match.group(0))
                                break
                            except _json.JSONDecodeError:
                                _log(f"  Batch {batch_idx + 1} attempt {attempt + 1}: JSON parse error")
                        else:
                            _log(f"  Batch {batch_idx + 1} attempt {attempt + 1}: no JSON array found")

                    if batch_suggestions:
                        all_suggestions.extend(batch_suggestions)
                    else:
                        _log(f"  Batch {batch_idx + 1} failed after retries")

                suggestions = all_suggestions
                if not suggestions:
                    return self._error(500, "INTERNAL_ERROR", "Failed to parse prompt suggestions after retries")

                _log(f"  Generated {len(suggestions)} prompt suggestions across {len(batches)} batch(es)")

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
