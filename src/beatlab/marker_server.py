"""Marker UI web server — serves waveform editor for manual hit marker placement."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Use stdlib http.server to avoid adding FastAPI/Flask dependency for this simple use case.


def start_server(
    audio_path: str,
    beats_path: str | None = None,
    hits_path: str = "hits.json",
    fps: float = 30.0,
    port: int = 8080,
) -> None:
    """Start the marker UI web server."""
    audio_file = Path(audio_path).resolve()
    beats_file = Path(beats_path).resolve() if beats_path else None
    hits_file = Path(hits_path).resolve()

    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    html_path = Path(__file__).parent / "marker_ui.html"
    if not html_path.exists():
        raise FileNotFoundError(f"UI template not found: {html_path}")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/" or path == "":
                self._serve_file(html_path, "text/html")
            elif path == "/audio":
                ct = mimetypes.guess_type(str(audio_file))[0] or "audio/wav"
                self._serve_file(audio_file, ct)
            elif path == "/beats":
                if beats_file and beats_file.exists():
                    self._serve_file(beats_file, "application/json")
                else:
                    self._json_response({"beats": [], "sections": []})
            elif path == "/hits":
                if hits_file.exists():
                    self._serve_file(hits_file, "application/json")
                else:
                    self._json_response({"fps": fps, "hits": []})
            elif path == "/config":
                self._json_response({"fps": fps, "audio_name": audio_file.name})
            else:
                self.send_error(404)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path == "/hits":
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    data = json.loads(body)
                    hits_file.parent.mkdir(parents=True, exist_ok=True)
                    hits_file.write_text(json.dumps(data, indent=2))
                    self._json_response({"ok": True})
                except (json.JSONDecodeError, OSError) as e:
                    self.send_error(400, str(e))
            else:
                self.send_error(404)

        def _serve_file(self, fpath: Path, content_type: str):
            data = fpath.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def _json_response(self, obj: dict):
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            # Quiet logging — only errors
            pass

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Marker UI running at http://localhost:{port}")
    print(f"  Audio: {audio_file.name}")
    if beats_file:
        print(f"  Beats: {beats_file.name}")
    print(f"  Hits:  {hits_file}")
    print(f"  FPS:   {fps}")
    print()
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()
