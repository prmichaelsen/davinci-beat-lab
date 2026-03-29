"""Narrative keyframe pipeline — YAML-driven keyframe generation and Veo transition pipeline."""

from __future__ import annotations

import math
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


# ── Timestamp Parsing ──────────────────────────────────────────────


def _extract_frame(video_path: str, output_path: str, position: str = "first") -> str:
    """Extract first or last frame from a video as PNG.

    Args:
        video_path: Path to the video file.
        output_path: Where to save the extracted frame.
        position: "first" or "last".

    Returns:
        output_path
    """
    import subprocess
    if position == "first":
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-frames:v", "1", "-q:v", "2", output_path,
        ], capture_output=True, check=True)
    else:
        # Get last frame: seek to near end, grab last frame
        # First get duration
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
            capture_output=True, text=True,
        )
        import json
        duration = float(json.loads(probe.stdout)["format"]["duration"])
        # Seek to 0.5s before end to ensure we get the last frame
        seek = max(0, duration - 0.5)
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(seek), "-i", video_path,
            "-update", "1", "-q:v", "2", output_path,
        ], capture_output=True, check=True)
    return output_path


def _parse_timestamp(ts: str) -> float:
    """Parse M:SS or M:SS.mmm to seconds."""
    ts = str(ts).strip("'\"")
    match = re.match(r"^(\d+):(\d{2})(?:\.(\d+))?$", ts)
    if not match:
        raise ValueError(f"Invalid timestamp format: {ts!r} (expected M:SS or M:SS.mmm)")
    minutes = int(match.group(1))
    seconds = int(match.group(2))
    millis = int(match.group(3)) if match.group(3) else 0
    return minutes * 60 + seconds + millis / (10 ** len(str(millis)))


# ── YAML Loading & Validation ──────────────────────────────────────


def load_narrative(yaml_path: str) -> dict:
    """Load and validate a narrative_keyframes.yaml file.

    Returns the parsed dict with timestamps converted to seconds and
    all paths resolved relative to the YAML file's parent directory.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Narrative YAML not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Validate top-level structure
    for key in ("meta", "keyframes", "transitions"):
        if key not in data:
            raise ValueError(f"Missing required top-level key: {key}")

    meta = data["meta"]
    for key in ("title", "audio", "fps", "resolution", "candidates_per_slot", "transition_max_seconds"):
        if key not in meta:
            raise ValueError(f"Missing required meta key: {key}")

    # Build keyframe index
    kf_ids = set()
    for kf in data["keyframes"]:
        for key in ("id", "timestamp", "source", "prompt"):
            if key not in kf:
                raise ValueError(f"Keyframe missing required key: {key}")
        if kf["id"] in kf_ids:
            raise ValueError(f"Duplicate keyframe ID: {kf['id']}")
        kf_ids.add(kf["id"])
        # Parse timestamp to seconds
        kf["_timestamp_seconds"] = _parse_timestamp(kf["timestamp"])

    # Validate transitions
    for tr in data["transitions"]:
        for key in ("id", "from", "to", "duration_seconds", "slots"):
            if key not in tr:
                raise ValueError(f"Transition {tr.get('id', '?')} missing required key: {key}")
        if tr["from"] not in kf_ids:
            raise ValueError(f"Transition {tr['id']}: 'from' references unknown keyframe {tr['from']}")
        if tr["to"] not in kf_ids:
            raise ValueError(f"Transition {tr['id']}: 'to' references unknown keyframe {tr['to']}")

    # Resolve paths relative to YAML parent
    base_dir = yaml_path.parent
    for kf in data["keyframes"]:
        source = Path(kf["source"])
        if not source.is_absolute():
            kf["_source_resolved"] = str(base_dir / source)
        else:
            kf["_source_resolved"] = str(source)
        # Resolve existing_keyframe if present
        if kf.get("existing_keyframe"):
            ekf = Path(kf["existing_keyframe"])
            if not ekf.is_absolute():
                kf["_existing_keyframe_resolved"] = str(base_dir / ekf)
            else:
                kf["_existing_keyframe_resolved"] = str(ekf)

    # Resolve existing_segment paths on transitions
    for tr in data["transitions"]:
        if tr.get("existing_segment"):
            segs = tr["existing_segment"]
            if isinstance(segs, str):
                segs = [segs]
            resolved = []
            for s in segs:
                p = Path(s)
                if not p.is_absolute():
                    resolved.append(str(base_dir / p))
                else:
                    resolved.append(str(p))
            tr["_existing_segment_resolved"] = resolved

    audio = Path(meta["audio"])
    if not audio.is_absolute():
        meta["_audio_resolved"] = str(base_dir / audio)
    else:
        meta["_audio_resolved"] = str(audio)

    data["_yaml_path"] = str(yaml_path)
    data["_work_dir"] = str(yaml_path.parent)

    return data


def save_narrative(data: dict, yaml_path: str | None = None) -> None:
    """Write the narrative data back to YAML, stripping internal fields."""
    yaml_path = yaml_path or data.get("_yaml_path")
    if not yaml_path:
        raise ValueError("No yaml_path provided and none stored in data")

    # Deep copy and strip internal fields
    import copy
    out = copy.deepcopy(data)
    for key in list(out.keys()):
        if key.startswith("_"):
            del out[key]
    for kf in out.get("keyframes", []):
        for key in list(kf.keys()):
            if key.startswith("_"):
                del kf[key]
    for tr in out.get("transitions", []):
        for key in list(tr.keys()):
            if key.startswith("_"):
                del tr[key]
    if "_audio_resolved" in out.get("meta", {}):
        del out["meta"]["_audio_resolved"]

    with open(yaml_path, "w") as f:
        yaml.dump(out, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def narrative_stats(data: dict) -> dict:
    """Return summary stats for a loaded narrative."""
    keyframes = data["keyframes"]
    transitions = data["transitions"]
    total_slots = sum(tr["slots"] for tr in transitions)
    multi_slot = [tr for tr in transitions if tr["slots"] > 1]
    intermediate_kfs = sum(tr["slots"] - 1 for tr in multi_slot)

    selected_kf = sum(1 for kf in keyframes if kf.get("selected") is not None)
    has_candidates_kf = sum(1 for kf in keyframes if kf.get("candidates"))
    existing_kf = sum(1 for kf in keyframes if kf.get("existing_keyframe"))
    existing_tr = sum(1 for tr in transitions if tr.get("existing_segment"))

    return {
        "keyframes": len(keyframes),
        "transitions": len(transitions),
        "total_slots": total_slots,
        "multi_slot_transitions": len(multi_slot),
        "intermediate_keyframes_needed": intermediate_kfs,
        "keyframes_with_candidates": has_candidates_kf,
        "keyframes_selected": selected_kf,
        "existing_keyframes": existing_kf,
        "existing_transitions": existing_tr,
    }


# ── Grid Generation ────────────────────────────────────────────────


def make_slot_grid(
    slot_images: list[list[str]],
    output_path: str,
    title: str,
    slot_labels: list[str] | None = None,
) -> str:
    """Create a grid image with rows=slots, columns=variants.

    Args:
        slot_images: List of lists — slot_images[slot_idx] = [v1.png, v2.png, ...].
        output_path: Where to save the grid PNG.
        title: Title text for the grid.
        slot_labels: Optional label per row (default: "slot 0", "slot 1", ...).
    """
    from PIL import Image, ImageDraw, ImageFont

    n_slots = len(slot_images)
    n_variants = max(len(row) for row in slot_images) if slot_images else 0
    if n_slots == 0 or n_variants == 0:
        return output_path

    # Load first image to get dimensions
    sample = Image.open(slot_images[0][0])
    w, h = sample.size
    sample.close()

    padding = 4
    label_w = 120  # left column for slot labels
    label_h = 30   # bottom label per image
    title_h = 50

    sheet_w = label_w + n_variants * (w + padding) + padding
    sheet_h = title_h + n_slots * (h + label_h + padding) + padding

    sheet = Image.new("RGB", (sheet_w, sheet_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_slot = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (OSError, IOError):
        font_title = ImageFont.load_default()
        font_label = font_title
        font_slot = font_title

    # Title
    draw.text((padding, 10), title, fill="white", font=font_title)

    if not slot_labels:
        slot_labels = [f"slot {i}" for i in range(n_slots)]

    for row_idx, (row_paths, slot_label) in enumerate(zip(slot_images, slot_labels)):
        y_base = title_h + padding + row_idx * (h + label_h + padding)

        # Slot label on the left
        draw.text((padding, y_base + h // 2 - 10), slot_label, fill="#aaaaaa", font=font_slot)

        for col_idx, img_path in enumerate(row_paths):
            x = label_w + col_idx * (w + padding)
            y = y_base

            img = Image.open(img_path)
            if img.size != (w, h):
                img = img.resize((w, h))
            sheet.paste(img, (x, y))
            img.close()

            # Variant label
            draw.text((x + 5, y + h + 2), f"v{col_idx + 1}", fill="white", font=font_label)

            # Border
            draw.rectangle([x - 1, y - 1, x + w, y + h], outline="white", width=2)

    sheet.save(output_path)
    return output_path


# ── Keyframe Candidate Generation ─────────────────────────────────


def _make_replicate_stylize_fn():
    """Create a stylize function that calls Nano Banana 2 via Replicate REST API (no SDK)."""
    import base64
    import json
    import os
    import time
    import urllib.request

    REPLICATE_API = "https://api.replicate.com/v1"
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise ValueError(
            "REPLICATE_API_TOKEN environment variable is required.\n"
            "Get a token at: https://replicate.com/account/api-tokens"
        )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    def _post(url: str, data: dict) -> dict:
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())

    def _get(url: str) -> dict:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())

    def _image_to_data_uri(image_path: str) -> str:
        ext = Path(image_path).suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "image/png")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:{mime};base64,{b64}"

    def stylize_fn(source_path: str, style_prompt: str, output_path: str) -> str:
        image_uri = _image_to_data_uri(source_path)

        prediction = _post(
            f"{REPLICATE_API}/models/google/nano-banana-2/predictions",
            {
                "input": {
                    "prompt": style_prompt,
                    "image_input": [image_uri],
                    "aspect_ratio": "16:9",
                    "output_format": "png",
                },
            },
        )

        # Poll until complete
        url = prediction["urls"]["get"]
        start = time.time()
        while time.time() - start < 300:
            result = _get(url)
            status = result.get("status")
            if status == "succeeded":
                break
            elif status in ("failed", "canceled"):
                error = result.get("error", "Unknown error")
                raise RuntimeError(f"Nano Banana prediction failed: {error}")
            time.sleep(3)
        else:
            raise TimeoutError("Nano Banana prediction timed out after 300s")

        # Download output image
        output = result.get("output")
        if isinstance(output, str):
            img_url = output
        elif isinstance(output, list) and len(output) > 0:
            img_url = str(output[0])
        elif isinstance(output, dict) and "url" in output:
            img_url = output["url"]
        else:
            raise RuntimeError(f"Unexpected Replicate output format: {output}")

        urllib.request.urlretrieve(img_url, output_path)
        return output_path

    return stylize_fn


def resolve_existing_boundary_frames(yaml_path: str) -> None:
    """Extract first/last frames from existing transition segments and place them
    in selected_keyframes/ so adjacent Veo transitions can use them.

    For each transition with existing_segment:
    - Extract first frame of the first segment -> selected_keyframes/{from_kf_id}.png
      (only if that keyframe doesn't already have a selected image)
    - Extract last frame of the last segment -> selected_keyframes/{to_kf_id}.png
      (only if that keyframe doesn't already have a selected image)
    """
    data = load_narrative(yaml_path)
    work_dir = Path(data["_work_dir"])
    selected_dir = work_dir / "selected_keyframes"
    selected_dir.mkdir(parents=True, exist_ok=True)

    for tr in data["transitions"]:
        existing = tr.get("_existing_segment_resolved")
        if not existing:
            continue

        from_kf_id = tr["from"]
        to_kf_id = tr["to"]

        # First frame of first segment -> from keyframe
        first_seg = existing[0]
        from_dest = selected_dir / f"{from_kf_id}.png"
        if not from_dest.exists() and Path(first_seg).exists():
            _extract_frame(first_seg, str(from_dest), "first")
            _log(f"  {tr['id']}: extracted first frame of {Path(first_seg).name} -> {from_kf_id}.png")

        # Last frame of last segment -> to keyframe
        last_seg = existing[-1]
        to_dest = selected_dir / f"{to_kf_id}.png"
        if not to_dest.exists() and Path(last_seg).exists():
            _extract_frame(last_seg, str(to_dest), "last")
            _log(f"  {tr['id']}: extracted last frame of {Path(last_seg).name} -> {to_kf_id}.png")

    _log("Existing segment boundary frames resolved.")


def generate_keyframe_candidates(
    yaml_path: str,
    vertex: bool = False,
    candidates_per_slot: int | None = None,
    segment_filter: set[str] | None = None,
    use_replicate: bool = False,
    regen: dict[str, set[str]] | None = None,
) -> None:
    """Generate styled image candidates for each keyframe using Nano Banana.

    Args:
        regen: If set, maps keyframe IDs to variant sets to regenerate.
               e.g. {"kf_005": {"v1", "v2"}, "kf_007": set()} where empty set = all variants.
               Targeted keyframes are automatically included even without segment_filter.
    """
    data = load_narrative(yaml_path)
    work_dir = Path(data["_work_dir"])
    n_candidates = candidates_per_slot or data["meta"]["candidates_per_slot"]

    from beatlab.render.candidates import generate_image_candidates, make_contact_sheet

    if use_replicate:
        stylize_fn = _make_replicate_stylize_fn()
    else:
        from beatlab.render.google_video import GoogleVideoClient
        client = GoogleVideoClient(vertex=vertex)

        def stylize_fn(source_path: str, style_prompt: str, output_path: str) -> str:
            return client.stylize_image(source_path, style_prompt, output_path)

    keyframes = data["keyframes"]
    if segment_filter:
        keyframes = [kf for kf in keyframes if kf["id"] in segment_filter]

    kf_candidates_dir = work_dir / "keyframe_candidates"
    kf_candidates_dir.mkdir(parents=True, exist_ok=True)

    # Handle --regen: delete targeted variants so generate_image_candidates recreates them
    # Also ensure regen keyframes are included in the keyframes list
    if regen is not None:
        kf_by_id = {kf["id"]: kf for kf in data["keyframes"]}
        for kf_id, variants in regen.items():
            if kf_id not in kf_by_id:
                _log(f"  WARNING: --regen references unknown keyframe {kf_id}")
                continue
            # Ensure this keyframe is in our working list
            if not any(kf["id"] == kf_id for kf in keyframes):
                keyframes.append(kf_by_id[kf_id])

            cand_dir = kf_candidates_dir / "candidates" / f"section_{kf_id}"
            if not cand_dir.exists():
                continue
            if len(variants) == 0:
                # Regen all variants
                for f in cand_dir.glob("v*.png"):
                    f.unlink()
                    _log(f"  {kf_id}: deleted {f.name} for regen")
                grid = cand_dir / "grid.png"
                if grid.exists():
                    grid.unlink()
            else:
                # Regen specific variants
                for v in variants:
                    v_name = v if v.endswith(".png") else f"{v}.png"
                    target = cand_dir / v_name
                    if target.exists():
                        target.unlink()
                        _log(f"  {kf_id}: deleted {v_name} for regen")

    # Auto-select keyframes with existing_keyframe — copy to selected_keyframes, skip generation
    selected_dir = work_dir / "selected_keyframes"
    selected_dir.mkdir(parents=True, exist_ok=True)
    for kf in keyframes:
        ekf_path = kf.get("_existing_keyframe_resolved")
        if ekf_path and Path(ekf_path).exists():
            dest = selected_dir / f"{kf['id']}.png"
            if not dest.exists():
                shutil.copy2(ekf_path, str(dest))
                _log(f"  {kf['id']}: using existing keyframe {ekf_path}")
            kf["selected"] = "existing"

    # Build jobs list, skipping already-generated and existing keyframes
    # generate_image_candidates stores in: {work_dir}/candidates/section_{key}/v*.png
    jobs = []
    regen_grid_only = []
    for kf in keyframes:
        kf_id = kf["id"]
        # Skip if existing keyframe is set
        if kf.get("_existing_keyframe_resolved") and Path(kf["_existing_keyframe_resolved"]).exists():
            continue
        cand_dir = kf_candidates_dir / "candidates" / f"section_{kf_id}"
        existing = list(cand_dir.glob("v*.png")) if cand_dir.exists() else []
        if len(existing) >= n_candidates:
            if regen is not None and kf_id in regen and len(regen[kf_id]) > 0:
                # Specific variants were regened but all slots filled — just rebuild grid
                regen_grid_only.append(kf)
            else:
                _log(f"  {kf_id}: {len(existing)} candidates exist, skipping")
            continue
        jobs.append(kf)

    # Rebuild grids for keyframes that had specific variants regened but are fully populated
    if regen_grid_only:
        for kf in regen_grid_only:
            kf_id = kf["id"]
            cand_dir = kf_candidates_dir / "candidates" / f"section_{kf_id}"
            paths = sorted(str(p) for p in cand_dir.glob("v*.png"))
            grid_path = str(cand_dir / "grid.png")
            make_contact_sheet(paths, grid_path, kf_id)
            kf["candidates"] = paths
            _log(f"  {kf_id}: rebuilt grid after regen")

    if not jobs:
        _log("All keyframe candidates already generated.")
        save_narrative(data, yaml_path)
        return

    # When 4 or fewer keyframes targeted, parallelize candidates within each keyframe
    # Otherwise, parallelize across keyframes (each generates candidates sequentially)
    parallelize_within = len(jobs) <= 4

    if parallelize_within:
        _log(f"Generating candidates for {len(jobs)} keyframes ({n_candidates} each, parallelizing within each keyframe)...")
    else:
        _log(f"Generating candidates for {len(jobs)} keyframes ({n_candidates} each, max 10 parallel)...")

    import threading
    lock = threading.Lock()
    completed = [0]

    def _generate_single_candidate(kf, variant_idx):
        """Generate a single candidate variant for a keyframe."""
        kf_id = kf["id"]
        source_path = kf["_source_resolved"]
        prompt = kf["prompt"]
        varied_prompt = f"{prompt}, variation {variant_idx}" if variant_idx > 1 else prompt

        cand_dir = kf_candidates_dir / "candidates" / f"section_{kf_id}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(cand_dir / f"v{variant_idx}.png")

        if Path(out_path).exists():
            return out_path

        stylize_fn(source_path, varied_prompt, out_path)
        return out_path

    def _generate_kf(kf):
        kf_id = kf["id"]
        source_path = kf["_source_resolved"]
        prompt = kf["prompt"]

        paths = generate_image_candidates(
            section_idx=kf_id,
            source_image_path=source_path,
            style_prompt=prompt,
            count=n_candidates,
            work_dir=str(kf_candidates_dir),
            stylize_fn=stylize_fn,
        )

        # Generate contact sheet in the same dir as candidates
        cand_dir = kf_candidates_dir / "candidates" / f"section_{kf_id}"
        grid_path = str(cand_dir / "grid.png")
        make_contact_sheet(paths, grid_path, kf_id)

        with lock:
            kf["candidates"] = [str(p) for p in paths]
            completed[0] += 1
            _log(f"  [{completed[0]}/{len(jobs)}] {kf_id}: done ({grid_path})")

    if parallelize_within:
        # Parallelize candidates within each keyframe
        for kf in jobs:
            kf_id = kf["id"]
            _log(f"  {kf_id}: generating {n_candidates} candidates in parallel...")

            with ThreadPoolExecutor(max_workers=n_candidates) as pool:
                futures = {
                    pool.submit(_generate_single_candidate, kf, v + 1): v + 1
                    for v in range(n_candidates)
                }
                paths = [None] * n_candidates
                for f in as_completed(futures):
                    v = futures[f]
                    try:
                        paths[v - 1] = f.result()
                    except Exception as e:
                        _log(f"    v{v} FAILED: {e}")

            paths = [p for p in paths if p]

            # Generate contact sheet
            cand_dir = kf_candidates_dir / "candidates" / f"section_{kf_id}"
            grid_path = str(cand_dir / "grid.png")
            make_contact_sheet(paths, grid_path, kf_id)

            kf["candidates"] = paths
            completed[0] += 1
            _log(f"  [{completed[0]}/{len(jobs)}] {kf_id}: done ({grid_path})")
    else:
        # Parallelize across keyframes
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(_generate_kf, kf) for kf in jobs]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    _log(f"  FAILED: {e}")

    # Save updated YAML
    save_narrative(data, yaml_path)
    _log("Keyframe candidate generation complete.")


# ── Keyframe Selection ─────────────────────────────────────────────


def apply_keyframe_selection(yaml_path: str, selections: dict[str, int]) -> None:
    """Apply keyframe selections: {kf_id: variant_index (1-based)}."""
    data = load_narrative(yaml_path)
    work_dir = Path(data["_work_dir"])
    selected_dir = work_dir / "selected_keyframes"
    selected_dir.mkdir(parents=True, exist_ok=True)

    kf_by_id = {kf["id"]: kf for kf in data["keyframes"]}

    for kf_id, variant in selections.items():
        if kf_id not in kf_by_id:
            _log(f"  WARNING: Unknown keyframe ID: {kf_id}")
            continue

        kf = kf_by_id[kf_id]
        candidates_dir = work_dir / "keyframe_candidates" / "candidates" / f"section_{kf_id}"
        source = candidates_dir / f"v{variant}.png"

        if not source.exists():
            _log(f"  WARNING: Candidate not found: {source}")
            continue

        dest = selected_dir / f"{kf_id}.png"
        shutil.copy2(str(source), str(dest))
        kf["selected"] = variant
        _log(f"  {kf_id}: selected v{variant} -> {dest}")

    save_narrative(data, yaml_path)
    _log("Keyframe selections applied.")


# ── Transition Action Generation ───────────────────────────────────


def generate_transition_actions(yaml_path: str) -> None:
    """Generate LLM transition actions for transitions with empty/null action.

    Sends the actual selected keyframe images to Claude along with text context
    so the LLM can see what was selected and describe the ideal visual journey.
    """
    import base64
    import os

    data = load_narrative(yaml_path)
    work_dir = Path(data["_work_dir"])
    selected_dir = work_dir / "selected_keyframes"

    kf_by_id = {kf["id"]: kf for kf in data["keyframes"]}

    # Find transitions needing actions
    needs_action = [
        tr for tr in data["transitions"]
        if not tr.get("action") or tr["action"].strip() == ""
    ]

    if not needs_action:
        _log("All transitions already have actions.")
        return

    _log(f"Generating actions for {len(needs_action)} transitions...")

    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY required for transition action generation")

    client = Anthropic(api_key=api_key)

    for i, tr in enumerate(needs_action):
        from_kf = kf_by_id[tr["from"]]
        to_kf = kf_by_id[tr["to"]]

        from_img = selected_dir / f"{tr['from']}.png"
        to_img = selected_dir / f"{tr['to']}.png"

        if not from_img.exists() or not to_img.exists():
            _log(f"  {tr['id']}: skipping — selected keyframes not found ({from_img.exists()}, {to_img.exists()})")
            continue

        # Build multimodal message
        from_b64 = base64.b64encode(from_img.read_bytes()).decode()
        to_b64 = base64.b64encode(to_img.read_bytes()).decode()

        from_ctx = from_kf.get("context", {})
        to_ctx = to_kf.get("context", {})

        master_prompt = data.get("meta", {}).get("prompt", "")
        master_context = f"Overall creative direction: {master_prompt}\n\n" if master_prompt else ""

        user_content = [
            {"type": "text", "text": f"You are a visual effects director for a music video. {master_context}Describe the ideal visual transition between these two keyframes.\n\n"},
            {"type": "text", "text": f"FROM keyframe ({tr['from']}):\n"
                f"  Timestamp: {from_kf['timestamp']}\n"
                f"  Mood: {from_ctx.get('mood', 'unknown')}\n"
                f"  Energy: {from_ctx.get('energy', 'unknown')}\n"
                f"  Instruments: {', '.join(from_ctx.get('instruments', []))}\n"
                f"  Motifs: {', '.join(from_ctx.get('motifs', []))}\n"
                f"  Visual direction: {from_ctx.get('visual_direction', '')}\n\n"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": from_b64}},
            {"type": "text", "text": f"\nTO keyframe ({tr['to']}):\n"
                f"  Timestamp: {to_kf['timestamp']}\n"
                f"  Mood: {to_ctx.get('mood', 'unknown')}\n"
                f"  Energy: {to_ctx.get('energy', 'unknown')}\n"
                f"  Instruments: {', '.join(to_ctx.get('instruments', []))}\n"
                f"  Motifs: {', '.join(to_ctx.get('motifs', []))}\n"
                f"  Visual direction: {to_ctx.get('visual_direction', '')}\n\n"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": to_b64}},
            {"type": "text", "text": f"\nTransition duration: {tr['duration_seconds']}s, {tr['slots']} slot(s).\n\n"
                "Write a concise cinematic transition description (1-3 sentences) that describes the visual journey "
                "from the first image to the second, considering the musical context. "
                "Focus on motion, transformation, and mood shift. "
                "This will be used as a prompt for Veo video generation.\n\n"
                "Reply with ONLY the transition description, no preamble."},
        ]

        _log(f"  [{i+1}/{len(needs_action)}] {tr['id']}: {tr['from']} -> {tr['to']} ({tr['duration_seconds']}s)...")

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": user_content}],
        )

        action = response.content[0].text.strip()
        tr["action"] = action
        _log(f"    Action: {action[:80]}...")

    save_narrative(data, yaml_path)
    _log("Transition action generation complete.")


# ── Slot Keyframe Generation (multi-slot transitions) ──────────────


def generate_slot_keyframe_candidates(
    yaml_path: str,
    vertex: bool = False,
    candidates_per_slot: int | None = None,
    use_replicate: bool = False,
) -> None:
    """Generate intermediate keyframe candidates for multi-slot transitions.

    Produces a combined grid per transition: rows=intermediate slots, columns=variants.
    """
    data = load_narrative(yaml_path)
    work_dir = Path(data["_work_dir"])
    n_candidates = candidates_per_slot or data["meta"]["candidates_per_slot"]

    kf_by_id = {kf["id"]: kf for kf in data["keyframes"]}
    multi_slot = [tr for tr in data["transitions"] if tr["slots"] > 1]

    if not multi_slot:
        _log("No multi-slot transitions. Skipping.")
        return

    from beatlab.render.candidates import generate_image_candidates

    if use_replicate:
        stylize_fn = _make_replicate_stylize_fn()
    else:
        from beatlab.render.google_video import GoogleVideoClient
        client = GoogleVideoClient(vertex=vertex)
        def stylize_fn(source_path: str, style_prompt: str, output_path: str) -> str:
            return client.stylize_image(source_path, style_prompt, output_path)

    slot_kf_dir = work_dir / "slot_keyframe_candidates"
    slot_kf_dir.mkdir(parents=True, exist_ok=True)

    # Generate intermediate keyframe prompts via LLM
    import base64
    import os
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY required for slot keyframe generation")

    anthropic_client = Anthropic(api_key=api_key)

    total_intermediates = sum(tr["slots"] - 1 for tr in multi_slot)
    _log(f"Generating slot keyframe candidates for {len(multi_slot)} multi-slot transitions ({total_intermediates} intermediates)...")

    for tr in multi_slot:
        n_intermediates = tr["slots"] - 1
        from_kf = kf_by_id[tr["from"]]
        to_kf = kf_by_id[tr["to"]]

        selected_dir = work_dir / "selected_keyframes"
        from_img = selected_dir / f"{tr['from']}.png"
        to_img = selected_dir / f"{tr['to']}.png"

        if not from_img.exists() or not to_img.exists():
            _log(f"  {tr['id']}: skipping — selected keyframes not found")
            continue

        from_b64 = base64.b64encode(from_img.read_bytes()).decode()
        to_b64 = base64.b64encode(to_img.read_bytes()).decode()

        from_ctx = from_kf.get("context", {})
        to_ctx = to_kf.get("context", {})

        # Ask LLM for intermediate keyframe prompts
        user_content = [
            {"type": "text", "text": f"You are a visual effects director. This transition has {tr['slots']} slots spanning {tr['duration_seconds']}s.\n\n"},
            {"type": "text", "text": f"FROM: {from_kf['prompt']}\n  Mood: {from_ctx.get('mood')}, Energy: {from_ctx.get('energy')}\n"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": from_b64}},
            {"type": "text", "text": f"\nTO: {to_kf['prompt']}\n  Mood: {to_ctx.get('mood')}, Energy: {to_ctx.get('energy')}\n"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": to_b64}},
            {"type": "text", "text": f"\nTransition action: {tr.get('action', 'smooth transition')}\n\n"
                f"Generate {n_intermediates} intermediate keyframe prompt(s) — one for each boundary between the {tr['slots']} slots. "
                f"These should describe evenly-spaced visual states between the FROM and TO images.\n\n"
                f"Reply with one prompt per line, numbered 1-{n_intermediates}. Each prompt should be a concise image description (1-2 sentences). No preamble."},
        ]

        _log(f"  {tr['id']}: generating {n_intermediates} intermediate keyframe prompts...")

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": user_content}],
        )

        # Parse numbered prompts
        lines = [l.strip() for l in response.content[0].text.strip().split("\n") if l.strip()]
        prompts = []
        for line in lines:
            cleaned = re.sub(r"^\d+[\.\)\-:\s]+", "", line).strip()
            if cleaned:
                prompts.append(cleaned)

        if len(prompts) < n_intermediates:
            _log(f"    WARNING: Expected {n_intermediates} prompts, got {len(prompts)}. Padding with generic prompts.")
            while len(prompts) < n_intermediates:
                prompts.append(f"Smooth visual transition between {from_kf['prompt'][:50]} and {to_kf['prompt'][:50]}")

        tr["_slot_keyframe_prompts"] = prompts[:n_intermediates]

        # Generate candidates for each intermediate keyframe
        source_img = str(from_img)
        all_slot_images = []  # for combined grid: list of lists

        for slot_idx in range(n_intermediates):
            slot_key = f"{tr['id']}_slot_{slot_idx}"

            # generate_image_candidates stores in: {work_dir}/candidates/section_{key}/
            cand_dir = slot_kf_dir / "candidates" / f"section_{slot_key}"
            existing = list(cand_dir.glob("v*.png")) if cand_dir.exists() else []
            existing_count = len(existing)

            # Always generate at least 1 more candidate (additive, not replace)
            target_count = max(n_candidates, existing_count + 1)

            _log(f"    {slot_key}: {existing_count} existing, generating up to {target_count} total...")

            paths = generate_image_candidates(
                section_idx=slot_key,
                source_image_path=source_img,
                style_prompt=prompts[slot_idx],
                count=target_count,
                work_dir=str(slot_kf_dir),
                stylize_fn=stylize_fn,
            )
            all_slot_images.append(paths)

        # Generate combined grid: rows=slots, columns=variants
        if all_slot_images:
            grid_path = str(slot_kf_dir / f"{tr['id']}_grid.png")
            slot_labels = [f"slot {i}" for i in range(n_intermediates)]
            make_slot_grid(all_slot_images, grid_path, f"{tr['id']} — {n_intermediates} intermediate keyframes", slot_labels)
            _log(f"    Combined grid: {grid_path}")

    save_narrative(data, yaml_path)
    _log("Slot keyframe candidate generation complete.")


def apply_slot_keyframe_selection(yaml_path: str, selections: dict[str, int]) -> None:
    """Apply slot keyframe selections: {tr_id_slot_N: variant_index (1-based)}."""
    data = load_narrative(yaml_path)
    work_dir = Path(data["_work_dir"])
    selected_dir = work_dir / "selected_slot_keyframes"
    selected_dir.mkdir(parents=True, exist_ok=True)

    slot_kf_dir = work_dir / "slot_keyframe_candidates"

    # Build lookup: tr_id -> transition dict
    tr_by_id: dict[str, dict] = {}
    for tr in data.get("transitions", []):
        tr_by_id[tr.get("id", "")] = tr

    for slot_key, variant in selections.items():
        source = slot_kf_dir / "candidates" / f"section_{slot_key}" / f"v{variant}.png"
        if not source.exists():
            _log(f"  WARNING: Candidate not found: {source}")
            continue

        dest = selected_dir / f"{slot_key}.png"
        shutil.copy2(str(source), str(dest))
        _log(f"  {slot_key}: selected v{variant} -> {dest}")

        # Persist selection index in the transition YAML data
        # slot_key format: "tr_006_slot_0" -> tr_id = "tr_006"
        parts = slot_key.rsplit("_slot_", 1)
        if len(parts) == 2:
            tr_id = parts[0]
            tr = tr_by_id.get(tr_id)
            if tr is not None:
                if "selected_slot_keyframes" not in tr or not isinstance(tr.get("selected_slot_keyframes"), dict):
                    tr["selected_slot_keyframes"] = {}
                tr["selected_slot_keyframes"][slot_key] = variant

    save_narrative(data, yaml_path)
    _log("Slot keyframe selections applied.")


# ── Transition Video Generation ────────────────────────────────────


def generate_transition_candidates(
    yaml_path: str,
    vertex: bool = False,
    candidates_per_slot: int | None = None,
    segment_filter: set[str] | None = None,
    slot_filter: set[int] | None = None,
    on_status=None,
) -> None:
    """Generate Veo transition video candidates for each slot."""
    # First resolve boundary frames from any existing segments
    resolve_existing_boundary_frames(yaml_path)

    data = load_narrative(yaml_path)
    work_dir = Path(data["_work_dir"])
    n_candidates = candidates_per_slot or 1
    max_seconds = data["meta"]["transition_max_seconds"]

    kf_by_id = {kf["id"]: kf for kf in data["keyframes"]}
    selected_kf_dir = work_dir / "selected_keyframes"
    selected_slot_kf_dir = work_dir / "selected_slot_keyframes"
    tr_candidates_dir = work_dir / "transition_candidates"
    tr_candidates_dir.mkdir(parents=True, exist_ok=True)

    from beatlab.render.google_video import GoogleVideoClient
    client = GoogleVideoClient(vertex=vertex)

    transitions = data["transitions"]
    if segment_filter:
        transitions = [tr for tr in transitions if tr["id"] in segment_filter]

    # Auto-handle transitions with existing_segment — copy to selected_transitions, skip generation
    selected_tr_dir = work_dir / "selected_transitions"
    selected_tr_dir.mkdir(parents=True, exist_ok=True)
    for tr in transitions:
        existing = tr.get("_existing_segment_resolved")
        if existing:
            # For single-segment: copy as slot_0
            # For multi-segment chains: copy as slot_0, slot_1, etc.
            for i, seg_path in enumerate(existing):
                if Path(seg_path).exists():
                    dest = selected_tr_dir / f"{tr['id']}_slot_{i}.mp4"
                    if not dest.exists():
                        shutil.copy2(seg_path, str(dest))
                        _log(f"  {tr['id']} slot_{i}: using existing segment {Path(seg_path).name}")

    # Build all jobs, skipping transitions with existing_segment
    jobs = []
    for tr in transitions:
        if tr.get("_existing_segment_resolved"):
            continue
        n_slots = tr["slots"]
        slot_duration = min(max_seconds, tr["duration_seconds"] / n_slots)

        for slot_idx in range(n_slots):
            if slot_filter is not None and slot_idx not in slot_filter:
                continue
            # Determine start/end images for this slot
            if n_slots == 1:
                start_img = str(selected_kf_dir / f"{tr['from']}.png")
                end_img = str(selected_kf_dir / f"{tr['to']}.png")
            else:
                # Multi-slot: chain through intermediate keyframes
                if slot_idx == 0:
                    start_img = str(selected_kf_dir / f"{tr['from']}.png")
                else:
                    start_img = str(selected_slot_kf_dir / f"{tr['id']}_slot_{slot_idx - 1}.png")

                if slot_idx == n_slots - 1:
                    end_img = str(selected_kf_dir / f"{tr['to']}.png")
                else:
                    end_img = str(selected_slot_kf_dir / f"{tr['id']}_slot_{slot_idx}.png")

            if not Path(start_img).exists() or not Path(end_img).exists():
                _log(f"  {tr['id']} slot {slot_idx}: skipping — keyframe images not found")
                continue

            slot_dir = tr_candidates_dir / tr["id"] / f"slot_{slot_idx}"
            slot_dir.mkdir(parents=True, exist_ok=True)

            slot_actions = tr.get("slot_actions", [])
            action = slot_actions[slot_idx] if slot_idx < len(slot_actions) else (tr.get("action") or "Smooth cinematic transition")
            motion_prompt = data.get("meta", {}).get("motion_prompt", "")
            prompt = f"{action}. Camera and motion: {motion_prompt}" if motion_prompt else action

            for v in range(n_candidates):
                output = str(slot_dir / f"v{v + 1}.mp4")
                if Path(output).exists():
                    continue
                jobs.append({
                    "tr_id": tr["id"],
                    "slot_idx": slot_idx,
                    "variant": v + 1,
                    "start_img": start_img,
                    "end_img": end_img,
                    "prompt": prompt,
                    "output": output,
                    "duration": slot_duration,
                })

    if not jobs:
        _log("All transition candidates already generated.")
        return

    _log(f"Generating {len(jobs)} Veo transition clips (sequential)...")

    from beatlab.render.google_video import PromptRejectedError
    rejected = []

    for i, job in enumerate(jobs):
        if job['tr_id'] in rejected:
            _log(f"    [{i + 1}/{len(jobs)}] {job['tr_id']} — skipping (prompt rejected)")
            continue
        _log(f"    [{i + 1}/{len(jobs)}] {job['tr_id']} slot_{job['slot_idx']} v{job['variant']}...")
        _log(f"    Prompt: {job['prompt'][:150]}...")
        try:
            client.generate_video_transition(
                start_frame_path=job["start_img"],
                end_frame_path=job["end_img"],
                prompt=job["prompt"],
                output_path=job["output"],
                duration_seconds=int(job["duration"]),
                on_status=on_status,
            )
            _log(f"    [{i + 1}/{len(jobs)}] {job['tr_id']} slot_{job['slot_idx']} v{job['variant']} done")
        except PromptRejectedError as e:
            _log(f"    ⚠ PROMPT REJECTED: {job['tr_id']} — {e}")
            rejected.append(job['tr_id'])

    if rejected:
        _log(f"⚠ {len(set(rejected))} transitions had prompts rejected. Edit their actions and retry: {', '.join(set(rejected))}")

    _log("Transition candidate generation complete.")


def apply_transition_selection(yaml_path: str, selections: dict[str, int]) -> None:
    """Apply transition selections: {tr_id_slot_N: variant_index (1-based)}."""
    data = load_narrative(yaml_path)
    work_dir = Path(data["_work_dir"])
    selected_dir = work_dir / "selected_transitions"
    selected_dir.mkdir(parents=True, exist_ok=True)

    tr_candidates_dir = work_dir / "transition_candidates"
    tr_by_id = {tr["id"]: tr for tr in data["transitions"]}

    for key, variant in selections.items():
        # key is like "tr_001_slot_0" or "tr_001" (shorthand for slot_0)
        if "_slot_" in key:
            tr_id, slot_part = key.rsplit("_slot_", 1)
            slot_idx = int(slot_part)
        else:
            tr_id = key
            slot_idx = 0

        source = tr_candidates_dir / tr_id / f"slot_{slot_idx}" / f"v{variant}.mp4"
        if not source.exists():
            _log(f"  WARNING: Candidate not found: {source}")
            continue

        dest = selected_dir / f"{tr_id}_slot_{slot_idx}.mp4"
        shutil.copy2(str(source), str(dest))
        _log(f"  {key}: selected v{variant} -> {dest}")

        # Record selection in the transition's selected list
        # Each slot entry is either:
        #   - an integer (variant number from generated candidates, e.g. 2 = v2.mp4)
        #   - a string (path to an imported/external file)
        tr = tr_by_id.get(tr_id)
        if tr:
            n_slots = tr.get("slots", 1)
            if not isinstance(tr.get("selected"), list) or len(tr["selected"]) != n_slots:
                tr["selected"] = [None] * n_slots
            tr["selected"][slot_idx] = variant

    save_narrative(data, yaml_path)
    _log("Transition selections applied.")


# ── Assembly ───────────────────────────────────────────────────────


def assemble_final(yaml_path: str, output_path: str) -> str:
    """Time-remap selected transitions, concatenate, and mux audio."""
    import subprocess

    data = load_narrative(yaml_path)
    work_dir = Path(data["_work_dir"])
    meta = data["meta"]
    selected_tr_dir = work_dir / "selected_transitions"
    remapped_dir = work_dir / "remapped"
    remapped_dir.mkdir(parents=True, exist_ok=True)

    transitions = data["transitions"]
    max_seconds = meta["transition_max_seconds"]

    # Phase 1: Time-remap each transition
    remapped_clips = []
    for tr in transitions:
        n_slots = tr["slots"]
        target_per_slot = tr["duration_seconds"] / n_slots

        slot_clips = []
        for slot_idx in range(n_slots):
            selected = selected_tr_dir / f"{tr['id']}_slot_{slot_idx}.mp4"
            if not selected.exists():
                _log(f"  WARNING: Missing selected transition: {selected}")
                continue

            # Probe actual duration
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(selected)],
                capture_output=True, text=True,
            )
            import json
            probe_data = json.loads(probe.stdout)
            actual_duration = float(probe_data["format"]["duration"])

            remapped = remapped_dir / f"{tr['id']}_slot_{slot_idx}.mp4"
            speed_factor = actual_duration / target_per_slot

            if abs(speed_factor - 1.0) > 0.05:
                _log(f"  {tr['id']} slot_{slot_idx}: remap {actual_duration:.1f}s -> {target_per_slot:.1f}s ({speed_factor:.2f}x)")
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(selected),
                    "-filter:v", f"setpts={1/speed_factor}*PTS",
                    "-an", str(remapped),
                ], capture_output=True, check=True)
            else:
                shutil.copy2(str(selected), str(remapped))

            slot_clips.append(str(remapped))

        if not slot_clips:
            continue

        # Concatenate slots within transition if multi-slot
        if len(slot_clips) == 1:
            tr_clip = slot_clips[0]
        else:
            concat_list = remapped_dir / f"{tr['id']}_concat.txt"
            with open(concat_list, "w") as f:
                for clip in slot_clips:
                    f.write(f"file '{clip}'\n")
            tr_clip = str(remapped_dir / f"{tr['id']}.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_list), "-c", "copy", tr_clip,
            ], capture_output=True, check=True)

        remapped_clips.append(tr_clip)

    if not remapped_clips:
        raise RuntimeError("No remapped clips to assemble")

    # Phase 2: Concatenate all transitions
    _log(f"Concatenating {len(remapped_clips)} transition clips...")
    final_concat_list = remapped_dir / "final_concat.txt"
    with open(final_concat_list, "w") as f:
        for clip in remapped_clips:
            f.write(f"file '{clip}'\n")

    no_audio = str(work_dir / "narrative_noaudio.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(final_concat_list), "-c", "copy", no_audio,
    ], capture_output=True, check=True)

    # Phase 3: Mux audio
    audio_path = meta["_audio_resolved"]
    _log(f"Muxing audio from {audio_path}...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", no_audio,
        "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        output_path,
    ], capture_output=True, check=True)

    _log(f"Final output: {output_path}")
    return output_path
