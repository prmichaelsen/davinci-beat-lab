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

            # POST /api/projects/:name/update-timestamp
            m = re.match(r"^/api/projects/([^/]+)/update-timestamp$", path)
            if m:
                return self._handle_update_timestamp(m.group(1))

            self._error(404, "NOT_FOUND", f"No route: POST {path}")

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors_headers()
            self.end_headers()

        # ── Handlers ─────────────────────────────────────────────

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

            self._json_response({
                "meta": result_meta,
                "keyframes": keyframes,
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

                    with open(full_path, "rb") as f:
                        f.seek(start)
                        self.wfile.write(f.read(length))
                    return

            # Full file response
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self._cors_headers()
            self.end_headers()

            with open(full_path, "rb") as f:
                # Stream in 64KB chunks to avoid loading large files into memory
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

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

    _log(f"SceneCraft API server running at http://{host}:{port}")
    _log(f"  Work dir: {wd}")
    _log(f"  Projects: {len([d for d in wd.iterdir() if d.is_dir()])}")
    _log("")
    _log("Endpoints:")
    _log("  GET  /api/projects                          List projects")
    _log("  GET  /api/projects/:name/keyframes          Keyframe data for editor")
    _log("  GET  /api/projects/:name/beats              Beat analysis data")
    _log("  GET  /api/projects/:name/ls?path=             List directory contents")
    _log("  GET  /api/projects/:name/files/*             Serve project files (audio/video/images)")
    _log("  POST /api/projects/:name/select-keyframes   Apply keyframe selections")
    _log("  POST /api/projects/:name/select-slot-keyframes  Apply slot selections")
    _log("  POST /api/projects/:name/update-timestamp   Update keyframe timestamp")
    _log("")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("Shutting down.")
        server.shutdown()
