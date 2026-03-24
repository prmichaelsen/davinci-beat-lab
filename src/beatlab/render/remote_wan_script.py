"""Remote Wan2.1 + FILM render script — runs ON the cloud GPU instance via SSH.

Usage (on remote):
    python3 /workspace/render_wan.py /workspace/wan_input /workspace/wan_output [model]

Expects:
    /workspace/wan_input/clips/section_NNN_chunk_NNN.mp4  — source video clips
    /workspace/wan_input/plan.json — render plan with per-clip style/denoise/transitions
    /workspace/wan_input/audio.wav — original audio (for final mux)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.parse
import uuid


COMFYUI_URL = "http://127.0.0.1:8188"
CLIENT_ID = str(uuid.uuid4())


# ── ComfyUI ──────────────────────────────────────────────────────────────────

def ensure_comfyui_running(timeout=900):
    """Ensure ComfyUI is running — start it if not."""
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=5)
        print("ComfyUI already running.", flush=True)
        return True
    except Exception:
        pass

    print("ComfyUI not running. Attempting to start...", flush=True)

    comfy_dirs = [
        "/workspace/ComfyUI_clean",
        "/workspace/ComfyUI",
        "/opt/workspace-internal/ComfyUI",
    ]
    comfy_dir = None
    for d in comfy_dirs:
        if os.path.exists(os.path.join(d, "main.py")):
            comfy_dir = d
            break

    if comfy_dir is None:
        print("  No ComfyUI found. Installing...", flush=True)
        comfy_dir = "/workspace/ComfyUI_clean"
        subprocess.run(
            ["git", "clone", "https://github.com/comfyanonymous/ComfyUI.git", comfy_dir],
            capture_output=True, timeout=120,
        )
        subprocess.run(
            ["pip", "install", "-q", "-r", f"{comfy_dir}/requirements.txt"],
            capture_output=True, timeout=300,
        )

    # Ensure VHS (Video Helper Suite) custom nodes are installed
    vhs_dir = os.path.join(comfy_dir, "custom_nodes", "ComfyUI-VideoHelperSuite")
    if not os.path.exists(vhs_dir):
        print("  Installing VHS custom nodes...", flush=True)
        subprocess.run(
            ["git", "clone", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git", vhs_dir],
            capture_output=True, timeout=60,
        )
        vhs_reqs = os.path.join(vhs_dir, "requirements.txt")
        if os.path.exists(vhs_reqs):
            subprocess.run(["pip", "install", "-q", "-r", vhs_reqs], capture_output=True, timeout=120)

    # Ensure models are linked
    models_dir = os.path.join(comfy_dir, "models")
    internal_models = "/opt/workspace-internal/ComfyUI/models"
    if os.path.exists(internal_models) and not os.path.islink(models_dir):
        print("  Linking models directory...", flush=True)
        if os.path.exists(models_dir):
            shutil.rmtree(models_dir)
        os.symlink(internal_models, models_dir)

    print(f"  Starting ComfyUI from {comfy_dir}...", flush=True)
    subprocess.Popen(
        ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"],
        cwd=comfy_dir,
        stdout=open("/tmp/comfyui.log", "w"),
        stderr=subprocess.STDOUT,
    )

    print("  Waiting for ComfyUI to be ready...", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=5)
            print("  ComfyUI ready.", flush=True)
            return True
        except Exception:
            time.sleep(3)

    print("  ERROR: ComfyUI failed to start.", flush=True)
    return False


# ── Upload / Download ────────────────────────────────────────────────────────

def upload_file(file_path):
    """Upload a file (image or video) to ComfyUI."""
    boundary = uuid.uuid4().hex
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    content_type = "video/mp4" if ext in (".mp4", ".mov", ".avi") else "image/png"

    with open(file_path, "rb") as f:
        file_data = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read()).get("name", filename)


def queue_and_wait(workflow, timeout=600):
    """Queue a workflow and wait for completion."""
    data = json.dumps({"prompt": workflow, "client_id": CLIENT_ID}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        prompt_id = json.loads(resp.read())["prompt_id"]
    start = time.time()
    while time.time() - start < timeout:
        with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10) as resp:
            history = json.loads(resp.read())
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(2)
    raise TimeoutError(f"Prompt {prompt_id} timed out after {timeout}s")


def download_output(filename, subfolder, output_path):
    """Download a generated file from ComfyUI output."""
    params = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": "output"})
    urllib.request.urlretrieve(f"{COMFYUI_URL}/view?{params}", output_path)


def check_vhs_available():
    """Check if VHS (Video Helper Suite) nodes are actually installed."""
    try:
        with urllib.request.urlopen(f"{COMFYUI_URL}/object_info/VHS_LoadVideo", timeout=10) as resp:
            data = json.loads(resp.read())
            # If the node exists, the response will have the node info keyed by class name
            return "VHS_LoadVideo" in data
    except Exception:
        return False


# ── Workflows ────────────────────────────────────────────────────────────────

def build_video_workflow(video_name, prompt, negative_prompt, denoise, seed, model):
    """VHS video-to-video workflow using SD img2img on video frames."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model}},
        "2": {"class_type": "VHS_LoadVideo", "inputs": {
            "video": video_name,
            "force_rate": 0,
            "custom_width": 0,
            "custom_height": 0,
            "frame_load_cap": 0,
            "skip_first_frames": 0,
            "select_every_nth": 1,
        }},
        "3": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["1", 1]}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
            "latent_image": ["3", 0], "seed": seed, "steps": 20, "cfg": 7.0,
            "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": denoise}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["7", 0], "frame_rate": 24, "loop_count": 0,
            "filename_prefix": "beatlab_wan", "format": "video/h264-mp4",
            "pingpong": False, "save_output": True}},
    }


# ── Rendering ────────────────────────────────────────────────────────────────

def render_clip_vhs(clip_path, output_path, prompt, negative_prompt, denoise, seed, model):
    """Render a video clip using VHS video workflow."""
    uploaded = upload_file(clip_path)
    workflow = build_video_workflow(uploaded, prompt, negative_prompt, denoise, seed, model)
    result = queue_and_wait(workflow)
    for node_output in result.get("outputs", {}).values():
        for key in ("gifs", "videos", "images"):
            items = node_output.get(key, [])
            if items:
                download_output(items[0]["filename"], items[0].get("subfolder", ""), output_path)
                return
    raise RuntimeError("No output from VHS workflow")


# ── FILM Transitions ─────────────────────────────────────────────────────────

def extract_clip_frames(clip_path, output_dir):
    """Extract frames from a video clip. Returns list of frame paths."""
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", clip_path, f"{output_dir}/frame_%06d.png"],
        capture_output=True, check=True,
    )
    return sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith("frame_") and f.endswith(".png")
    )


def film_interpolate(frame_a, frame_b, num_frames, output_dir, prefix="interp"):
    """Generate interpolated frames between two images using ffmpeg minterpolate."""
    os.makedirs(output_dir, exist_ok=True)
    if num_frames <= 0:
        return []

    with tempfile.TemporaryDirectory() as tmpdir:
        # 2-frame video
        concat_file = os.path.join(tmpdir, "concat.txt")
        with open(concat_file, "w") as f:
            f.write(f"file '{os.path.abspath(frame_a)}'\nduration 1.0\n")
            f.write(f"file '{os.path.abspath(frame_b)}'\nduration 1.0\n")

        input_video = os.path.join(tmpdir, "input.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
             "-vsync", "vfr", "-pix_fmt", "yuv420p", input_video],
            capture_output=True, check=True,
        )

        target_fps = num_frames + 2
        interp_video = os.path.join(tmpdir, "interp.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_video,
             "-filter:v", f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1",
             "-pix_fmt", "yuv420p", interp_video],
            capture_output=True, check=True,
        )

        frame_pattern = os.path.join(tmpdir, f"{prefix}_%04d.png")
        subprocess.run(
            ["ffmpeg", "-y", "-i", interp_video, frame_pattern],
            capture_output=True, check=True,
        )

        all_frames = sorted(
            os.path.join(tmpdir, f) for f in os.listdir(tmpdir)
            if f.startswith(prefix) and f.endswith(".png")
        )
        intermediate = all_frames[1:-1] if len(all_frames) > 2 else all_frames

        results = []
        for i, src in enumerate(intermediate[:num_frames]):
            dst = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
            shutil.copy2(src, dst)
            results.append(dst)
        return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    input_dir = sys.argv[1]   # /workspace/wan_input
    output_dir = sys.argv[2]  # /workspace/wan_output
    model = sys.argv[3] if len(sys.argv) > 3 else "v1-5-pruned-emaonly-fp16.safetensors"
    negative_prompt = "blurry, low quality, distorted, deformed, static, frozen"

    os.makedirs(output_dir, exist_ok=True)

    # Load render plan
    plan_path = os.path.join(input_dir, "plan.json")
    if not os.path.exists(plan_path):
        print("ERROR: plan.json not found", flush=True)
        sys.exit(1)

    with open(plan_path) as f:
        plan = json.load(f)

    clips = plan.get("clips", [])
    transitions = plan.get("transitions", [])
    fps = plan.get("fps", 24.0)
    audio_path = os.path.join(input_dir, "audio.wav")

    print(f"Plan: {len(clips)} clips, {len(transitions)} transitions, {fps} fps", flush=True)

    # ── Phase 1: Render clips through ComfyUI ──
    print(f"\nPhase 1: Rendering {len(clips)} clips...", flush=True)

    if not ensure_comfyui_running():
        print("ERROR: ComfyUI not responding", flush=True)
        sys.exit(1)

    # Check model
    try:
        with urllib.request.urlopen(f"{COMFYUI_URL}/object_info/CheckpointLoaderSimple", timeout=10) as resp:
            info = json.loads(resp.read())
            available = info.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
            if available and model not in available:
                print(f"WARNING: {model} not found, using {available[0]}", flush=True)
                model = available[0]
    except Exception as e:
        print(f"Could not check models: {e}", flush=True)

    if not check_vhs_available():
        print("FATAL: VHS nodes not available. Install ComfyUI-VideoHelperSuite.", flush=True)
        sys.exit(1)
    print("VHS nodes: available", flush=True)

    styled_clips_dir = os.path.join(output_dir, "styled_clips")
    os.makedirs(styled_clips_dir, exist_ok=True)

    total = len(clips)
    start_time = time.time()

    for idx, clip_info in enumerate(clips):
        clip_name = clip_info["clip"]
        style = clip_info.get("style_prompt", "artistic stylized")
        denoise = clip_info.get("denoise", 0.5)
        seed = clip_info.get("seed", 42 + idx)

        input_clip = os.path.join(input_dir, "clips", clip_name)
        output_clip = os.path.join(styled_clips_dir, clip_name)

        if os.path.exists(output_clip):
            print(f"  [{idx+1}/{total}] {clip_name} (cached)", flush=True)
            continue

        if not os.path.exists(input_clip):
            print(f"  [{idx+1}/{total}] {clip_name} SKIPPED (not found)", flush=True)
            continue

        try:
            render_clip_vhs(input_clip, output_clip, style, negative_prompt, denoise, seed, model)
        except Exception as e:
            print(f"  [{idx+1}/{total}] {clip_name} FAILED: {e}", flush=True)
            sys.exit(1)

        elapsed = time.time() - start_time
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        eta = (total - idx - 1) / rate if rate > 0 else 0
        print(f"  [{idx+1}/{total}] {clip_name} done ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

    print(f"\nPhase 1 complete: {total} clips rendered", flush=True)

    # ── Phase 2: FILM transitions + assembly ──
    print(f"\nPhase 2: FILM transitions ({len(transitions)} boundaries)...", flush=True)

    # Extract frames from each styled clip
    all_section_frames = []
    frames_base = os.path.join(output_dir, "section_frames")

    for idx, clip_info in enumerate(clips):
        clip_name = clip_info["clip"]
        styled_clip = os.path.join(styled_clips_dir, clip_name)
        frame_dir = os.path.join(frames_base, f"clip_{idx:03d}")

        if not os.path.exists(styled_clip):
            print(f"  WARNING: {clip_name} missing, skipping", flush=True)
            all_section_frames.append([])
            continue

        frames = extract_clip_frames(styled_clip, frame_dir)
        all_section_frames.append(frames)

    # Generate FILM transitions and assemble
    transitions_dir = os.path.join(output_dir, "transitions")
    os.makedirs(transitions_dir, exist_ok=True)

    final_frames = []
    window = 3

    for i, section_frames in enumerate(all_section_frames):
        if not section_frames:
            continue

        # Add section frames (trim tail if transition follows)
        if i < len(all_section_frames) - 1 and len(section_frames) > window:
            final_frames.extend(section_frames[:-window])
        else:
            final_frames.extend(section_frames)

        # FILM transition to next section
        if i < len(transitions) and i + 1 < len(all_section_frames):
            num_trans = transitions[i].get("frames", 8)
            next_frames = all_section_frames[i + 1]

            if num_trans > 0 and section_frames and next_frames:
                tail = section_frames[-window:]
                head = next_frames[:window]
                trans_dir = os.path.join(transitions_dir, f"trans_{i:03d}")
                trans_frames = film_interpolate(
                    tail[-1], head[0], num_trans, trans_dir, prefix=f"t{i:03d}",
                )
                final_frames.extend(trans_frames)
                print(f"  Transition {i}→{i+1}: {len(trans_frames)} frames", flush=True)

    print(f"  Total frames after assembly: {len(final_frames)}", flush=True)

    # ── Phase 3: Reassemble final video ──
    print(f"\nPhase 3: Reassembling final video...", flush=True)

    final_frames_dir = os.path.join(output_dir, "final_frames")
    os.makedirs(final_frames_dir, exist_ok=True)

    for i, src in enumerate(final_frames):
        dst = os.path.join(final_frames_dir, f"frame_{i:06d}.png")
        if src != dst:
            shutil.copy2(src, dst)

    output_video = os.path.join(output_dir, "output.mp4")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", f"{final_frames_dir}/frame_%06d.png",
    ]

    # Mux audio if available
    if os.path.exists(audio_path):
        ffmpeg_cmd.extend(["-i", audio_path, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-shortest"])

    ffmpeg_cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", output_video])

    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    print(f"\nDone! Output: {output_video}", flush=True)
    print(f"Total frames: {len(final_frames)}", flush=True)


if __name__ == "__main__":
    main()
