# Beatlab Server

**Concept**: `beatlab server` command — HTTP server exposing narrative pipeline functions for the beatlab-synthesizer web frontend
**Created**: 2026-03-26
**Status**: Design Specification

---

## Overview

Add a `beatlab server` command that starts an HTTP server inside davinci-beat-lab, exposing narrative pipeline operations as REST endpoints. The beatlab-synthesizer frontend (a TanStack Start web app at `../beatlab-synthesizer`) calls these endpoints instead of manipulating `.beatlab_work/` files directly, ensuring all operations go through the same Python functions the CLI uses.

**Deployment model:** Both the beatlab server and synthesizer run on a **provisioned cloud desktop instance per customer** with a mounted volume holding `.beatlab_work/`. No GCS, no database, no multi-tenant complexity. YAML files on the mounted volume are the primary storage — treated as project documents (like a `.docx`). GPU-heavy operations (generation, rendering) shell out to separate machines (Vast.ai) as beatlab already does, keeping the desktop lightweight and avoiding irreversible scale-up traps.

---

## Problem Statement

- The synthesizer frontend needs to invoke pipeline operations (keyframe selection, timestamp editing, candidate browsing) but directly editing YAML or copying files from Node.js risks diverging from beatlab's actual behavior (which includes cache invalidation, downstream file deletion, specific field update logic).
- Shelling out to CLI commands is fragile — argument escaping, no structured errors, no progress streaming.
- We own both codebases, so the cleanest approach is to expose internal functions via HTTP within this project.

---

## Solution

A new `api_server.py` module following the existing `marker_server.py` pattern (stdlib `http.server`, no new dependencies). The server exposes REST endpoints that call the same internal functions as the CLI commands.

**Architecture:**

```
beatlab-synthesizer (React/TanStack)
       │
       │ HTTP (proxied through TanStack server fns)
       ▼
beatlab server (Python, localhost:8888)
       │
       │ direct function calls
       ▼
render/narrative.py, cli.py internals
       │
       ▼
.beatlab_work/ filesystem + narrative_keyframes.yaml
```

---

## Implementation

### Component 1: CLI command

Add to `cli.py`:

```python
@main.command()
@click.option("--port", default=8888, help="Server port")
@click.option("--host", default="0.0.0.0", help="Bind address")
def server(port, host):
    """Start REST API server for beatlab-synthesizer."""
    from beatlab.api_server import run_server
    run_server(host, port)
```

### Component 2: `src/beatlab/api_server.py`

Follows `marker_server.py` conventions: `BaseHTTPRequestHandler` subclass, `_json_response()` helper, CORS headers, suppressed logging.

#### Route mapping

**Phase 1 — Core editor operations** (replaces synthesizer server fns):

| Method | Path | Handler | Calls |
|---|---|---|---|
| GET | `/api/projects` | `handle_list_projects` | `os.listdir(WORK_DIR)` + metadata scan |
| GET | `/api/projects/:name/keyframes` | `handle_get_keyframes` | `yaml.safe_load()` on `narrative_keyframes.yaml` |
| POST | `/api/projects/:name/select-keyframes` | `handle_select_keyframes` | `narrative.apply_keyframe_selection()` |
| POST | `/api/projects/:name/select-slot-keyframes` | `handle_select_slot_keyframes` | `narrative.apply_slot_keyframe_selection()` |
| POST | `/api/projects/:name/select-transitions` | `handle_select_transitions` | (future) |
| POST | `/api/projects/:name/update-timestamp` | `handle_update_timestamp` | YAML field update + save |
| GET | `/api/projects/:name/files/*` | `handle_serve_file` | Stream file with Range support |

**Phase 2 — Analysis & audio** (exposes CLI analyze/stems commands):

| Method | Path | Handler | Calls |
|---|---|---|---|
| GET | `/api/projects/:name/beats` | `handle_get_beats` | Load `beats.json` — beat data, sections, stem analysis |
| POST | `/api/projects/:name/analyze` | `handle_analyze` | `analyzer.analyze_audio()` + `create_beat_map()` (async job) |
| POST | `/api/projects/:name/stems` | `handle_stems` | `stems.separate_stems_remote()` + `analyze_all_stems()` (async job) |
| GET | `/api/projects/:name/stems` | `handle_get_stems` | Return stem analysis summary from beats.json |

**Phase 3 — Rendering & effects** (exposes CLI render/effects commands):

| Method | Path | Handler | Calls |
|---|---|---|---|
| POST | `/api/projects/:name/render` | `handle_render` | `render_google_pipeline()` (async job) |
| POST | `/api/projects/:name/effects` | `handle_effects` | `effects_opencv.apply_effects()` (async job) |
| POST | `/api/projects/:name/assemble` | `handle_assemble` | `narrative.assemble_final()` (async job) |
| POST | `/api/projects/:name/ingredients` | `handle_upload_ingredients` | Save reference images to `ingredients/` dir |
| GET | `/api/projects/:name/ingredients` | `handle_list_ingredients` | List files in `ingredients/` dir |
| DELETE | `/api/projects/:name/ingredients/:filename` | `handle_delete_ingredient` | Remove ingredient file |

**Phase 4 — Platform management** (cloud desktop ops):

| Method | Path | Handler | Calls |
|---|---|---|---|
| POST | `/api/auth/login` | `handle_login` | Token-based auth (cloud deployment) |
| GET | `/api/billing/credits` | `handle_get_credits` | Read `.scenecraft/billing.yaml` |
| POST | `/api/billing/purchase` | `handle_purchase_credits` | Stripe integration + update billing.yaml |
| GET | `/api/billing/usage` | `handle_get_usage` | Usage history from billing.yaml |
| GET | `/api/gpu/status` | `handle_gpu_status` | `VastAIManager` instance state |
| POST | `/api/gpu/destroy` | `handle_gpu_destroy` | `VastAIManager.destroy_instance()` |
| GET | `/api/storage/usage` | `handle_storage_usage` | Volume disk usage |

**Async job tracking** (shared across all long-running operations):

| Method | Path | Handler | Calls |
|---|---|---|---|
| GET | `/api/jobs/:id` | `handle_get_job` | Check job status, return progress/result |
| GET | `/api/jobs` | `handle_list_jobs` | List active/recent jobs |
| POST | `/api/jobs/:id/cancel` | `handle_cancel_job` | Cancel a running job |
| WS | `/api/projects/:name/progress` | `handle_progress_ws` | WebSocket stream for real-time progress |

Long-running operations (analyze, stems, render, effects, assemble) return immediately with a job ID. The client polls `GET /api/jobs/:id` or connects to the WebSocket for real-time progress updates.

#### Internal function mapping

**`apply_keyframe_selection()`** (render/narrative.py:644):
- Input: `work_dir: Path`, `yaml_data: dict`, `selections: dict[str, int]`
- For each `kf_id → variant`: copies `keyframe_candidates/candidates/section_{kf_id}/v{variant}.png` → `selected_keyframes/{kf_id}.png`
- Updates `kf["selected"] = variant` in YAML data
- Saves YAML
- The endpoint parses the POST body, loads the YAML, calls this function, returns result

**`apply_slot_keyframe_selection()`** (render/narrative.py:920):
- Input: `work_dir: Path`, `yaml_data: dict`, `selections: dict[str, int]`
- Copies `slot_keyframe_candidates/{slot_key}/v{variant}.png` → `selected_slot_keyframes/{slot_key}.png`
- Saves YAML

**Timestamp update** (new helper, not currently a standalone function):
- Load YAML, find keyframe by ID, update `timestamp` field, save
- Simple enough to implement inline in the handler, but should be a shared utility function in `render/narrative.py` so the CLI can also call it if needed

#### File serving

The `/api/projects/:name/files/*` endpoint replaces the synthesizer's current direct filesystem access. Supports:
- `Range` header for audio/video streaming (206 Partial Content)
- MIME type detection from extension
- Path traversal prevention (resolved path must start with work dir)

#### Request/response format

All POST endpoints accept JSON body. All responses are JSON with `Content-Type: application/json`.

Error format:
```json
{ "error": "Keyframe kf_999 not found", "code": "NOT_FOUND" }
```

#### GET `/api/projects/:name/keyframes` response

Returns the full keyframe data the editor needs, including candidate file paths:

```json
{
  "meta": { "title": "...", "fps": 24, "resolution": [1920, 1080] },
  "keyframes": [
    {
      "id": "kf_001",
      "timestamp": "0:00",
      "section": "1A",
      "prompt": "...",
      "selected": 1,
      "candidates": [
        "keyframe_candidates/candidates/section_kf_001/v1.png",
        "keyframe_candidates/candidates/section_kf_001/v2.png",
        "keyframe_candidates/candidates/section_kf_001/v3.png",
        "keyframe_candidates/candidates/section_kf_001/v4.png"
      ],
      "hasSelectedImage": true,
      "context": {
        "mood": "dreamy, serene",
        "energy": "low",
        "instruments": ["soothing vocals", "ethereal pads"],
        "motifs": ["PAD-VERSE-1A"],
        "events": [],
        "visual_direction": "Slow, gentle, ethereal.",
        "details": "..."
      }
    }
  ]
}
```

### Component 3: Server startup and shutdown

```python
def run_server(host: str, port: int):
    work_dir = Path.cwd() / ".beatlab_work"
    if not work_dir.exists():
        click.echo(f"No .beatlab_work directory found in {Path.cwd()}")
        raise SystemExit(1)

    handler = make_handler(work_dir)
    server = HTTPServer((host, port), handler)
    click.echo(f"Beatlab API server running at http://{host}:{port}")
    click.echo(f"Serving projects from {work_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nShutting down.")
        server.shutdown()
```

The `make_handler(work_dir)` factory creates a handler class with the work_dir baked in (same pattern as `marker_server.py`).

---

## Benefits

- **No drift**: Endpoints call the exact same functions as `beatlab narrative select-keyframes`, `select-slot-keyframes`, etc.
- **Zero new dependencies**: stdlib `http.server` only, same as `marker_server.py`
- **Incremental**: Ship with 3 endpoints (keyframes, select-keyframes, update-timestamp), add more as the synthesizer grows
- **Existing pattern**: Follows the exact same HTTP server pattern already established in `marker_server.py`

---

## Trade-offs

- **Two processes**: User runs `beatlab server` alongside the synthesizer dev server. Could be mitigated with a launch script or `Procfile`.
- **No async framework**: stdlib HTTP server is synchronous. Long-running ops (assemble) need threading or a simple job queue. Acceptable for single-user local dev.
- **Verbose handler code**: No automatic routing or JSON parsing from stdlib. Manageable at <10 endpoints.
- **No auth**: Localhost-only by design. Would need token auth if ever networked.

---

## Dependencies

- **Internal**: `render/narrative.py` (apply_keyframe_selection, apply_slot_keyframe_selection, assemble_final), `yaml`, `pathlib`, `shutil`
- **External**: None (stdlib only)
- **Consumer**: beatlab-synthesizer frontend calls these endpoints via a TypeScript proxy client (`src/lib/beatlab-client.ts`)

---

## Testing Strategy

- **Unit tests**: Test handler functions with fixture YAML data and temp directories
- **Integration test**: Start server in subprocess, make HTTP calls, verify YAML changes and file copies match CLI behavior
- **Parity test**: Run the same selection via CLI (`beatlab narrative select-keyframes`) and via HTTP endpoint, compare resulting files byte-for-byte

---

## Migration Path

1. **Phase 1**: Core editor — `api_server.py` with projects, keyframes, selections, timestamp updates, file serving. Replaces synthesizer's direct filesystem server fns.
2. **Phase 2**: Analysis & audio — beats, analyze, stems endpoints. Async job tracking infrastructure (job IDs, polling, WebSocket progress).
3. **Phase 3**: Rendering & effects — render, effects, assemble, ingredients upload. All long-running ops use async jobs.
4. **Phase 4**: Platform — auth, billing/credits, GPU management, storage. Cloud desktop deployment.
5. **Phase 5** (optional): Migrate to FastAPI if endpoint count or async complexity warrants it.

---

## Key Design Decisions

### Architecture

| Decision | Choice | Rationale |
|---|---|---|
| HTTP framework | stdlib `http.server` | Matches `marker_server.py`, zero deps, adequate for local dev |
| Server location | Inside beatlab as `beatlab server` command | Direct access to internal functions, no wrapper drift |
| Work dir discovery | `Path.cwd() / ".beatlab_work"` | Consistent with all other beatlab commands |
| Routing | Path prefix matching in `do_GET`/`do_POST` | Simple, no framework needed for <10 routes |
| Storage | YAML on mounted volume, no database | YAML-as-document — the file is the project. Provisioned desktop + volume per customer. No GCS, no D1. |
| Deployment | Provisioned cloud desktop per customer | Lightweight I/O instance, GPU ops to Vast.ai. Customer isolation by design. Avoids irreversible scale-up. |

### Data Flow

| Decision | Choice | Rationale |
|---|---|---|
| Keyframe selection | Call `apply_keyframe_selection()` directly | Same behavior as CLI — file copy + YAML update atomically |
| Timestamp updates | New utility function in `render/narrative.py` | Keep YAML editing logic centralized, usable by both CLI and server |
| File serving | Serve from `.beatlab_work/` with Range support | Synthesizer doesn't need filesystem access; audio/video streaming works |
| Response format | JSON for all endpoints, consistent error shape | Simple for the TypeScript client to consume |

---

## Future Considerations

- **File watching**: Push YAML changes to frontend when files are edited externally (e.g., by CLI)
- **Candidate generation trigger**: Expose `narrative keyframes` as an endpoint to generate new candidates from the UI
- **`scenecraft archive`**: Backup project to object storage from mounted volume for disaster recovery
- **Batch operations**: Bulk select keyframes, bulk render segments
- **Project import/export**: Zip/download a project, upload to a new desktop
- **Collaboration hooks**: Notify other sessions when project files change

---

**Status**: Design Specification
**Recommendation**: Implement Phase 1 — `beatlab server` command with keyframes, select-keyframes, update-timestamp, and file serving endpoints
**Related Documents**: [beatlab-synthesizer design](../../beatlab-synthesizer/agent/design/local.beatlab-server.md), [marker_server.py](../src/beatlab/marker_server.py), [render/narrative.py](../src/beatlab/render/narrative.py)
