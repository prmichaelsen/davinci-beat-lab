"""Remote render script v2 — keyframe SD render + EbSynth propagation.

Runs ON the cloud GPU instance via SSH.

Usage (on remote):
    python3 /workspace/render_v2.py /workspace/input /workspace/output [model]

Expects:
    /workspace/input/frame_NNNNNN.png  — source frames
    /workspace/input/keyframes.json    — keyframe list with frame, denoise, prompt, seed
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.parse
import uuid


COMFYUI_URL = "http://127.0.0.1:8188"
CLIENT_ID = str(uuid.uuid4())


# ── EbSynth ──────────────────────────────────────────────────────────────────

def ensure_ebsynth() -> str:
    """Find or build EbSynth binary."""
    # Check if already built (also check the test build we did manually)
    for p in ("/workspace/ebsynth_build/bin/ebsynth", "/workspace/ebsynth_check/bin/ebsynth"):
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p

    # Check PATH
    for name in ("ebsynth",):
        try:
            r = subprocess.run(["which", name], capture_output=True, text=True)
            if r.returncode == 0:
                return r.stdout.strip()
        except Exception:
            pass

    # Build from source
    print("Building EbSynth from source...", flush=True)
    build_dir = "/workspace/ebsynth_build"
    binary = f"{build_dir}/bin/ebsynth"
    try:
        if not os.path.exists(build_dir):
            print("  Cloning ebsynth repo...", flush=True)
            subprocess.run(
                ["git", "clone", "https://github.com/jamriska/ebsynth.git", build_dir],
                capture_output=True, check=True, timeout=60,
            )

        print("  Compiling (CPU only)...", flush=True)
        subprocess.run(
            ["bash", "build-linux-cpu_only.sh"],
            cwd=build_dir,
            capture_output=True, check=True, timeout=120,
        )

        if os.path.exists(binary) and os.access(binary, os.X_OK):
            print(f"  EbSynth built successfully", flush=True)
            return binary

        print("WARNING: EbSynth built but binary not found at bin/ebsynth", flush=True)
        return ""

    except Exception as e:
        print(f"WARNING: Could not build EbSynth: {e}", flush=True)
        print("Falling back to keyframe-only output (no temporal coherence)", flush=True)
        return ""


def propagate_frame(ebsynth_bin, style_img, source_at_style, target_source, output):
    """Propagate style from keyframe to target using EbSynth."""
    try:
        r = subprocess.run(
            [ebsynth_bin, "-style", style_img, "-guide", source_at_style, target_source,
             "-output", output, "-weight", "1.0"],
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode == 0 and os.path.exists(output)
    except Exception:
        return False


def propagate_all(source_dir, styled_keyframes, output_dir, ebsynth_bin, blend_width=4):
    """Propagate style from keyframes to all intermediate frames."""
    kf_frames = sorted(styled_keyframes.keys())
    if not kf_frames:
        return

    all_frames = sorted(
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(source_dir)
        if f.startswith("frame_") and f.endswith(".png")
    )
    total = len(all_frames)
    done = 0
    start_time = time.time()

    def src(n):
        return os.path.join(source_dir, f"frame_{n:06d}.png")

    def out(n):
        return os.path.join(output_dir, f"frame_{n:06d}.png")

    # Copy keyframes
    for kf in kf_frames:
        if not os.path.exists(out(kf)):
            shutil.copy2(styled_keyframes[kf], out(kf))
        done += 1

    print(f"Propagating from {len(kf_frames)} keyframes to {total} frames...", flush=True)

    for i in range(len(kf_frames)):
        kf_a = kf_frames[i]
        kf_b = kf_frames[i + 1] if i + 1 < len(kf_frames) else None

        if kf_b is None:
            # Last keyframe — forward to end
            for f in range(kf_a + 1, all_frames[-1] + 1):
                if not os.path.exists(out(f)):
                    if not propagate_frame(ebsynth_bin, styled_keyframes[kf_a], src(kf_a), src(f), out(f)):
                        shutil.copy2(styled_keyframes[kf_a], out(f))
                done += 1
                _progress(done, total, start_time)
            continue

        mid = (kf_a + kf_b) // 2

        # Forward propagate from A
        for f in range(kf_a + 1, mid + 1):
            if not os.path.exists(out(f)):
                if not propagate_frame(ebsynth_bin, styled_keyframes[kf_a], src(kf_a), src(f), out(f)):
                    shutil.copy2(styled_keyframes[kf_a], out(f))
            done += 1
            _progress(done, total, start_time)

        # Backward propagate from B
        for f in range(kf_b - 1, mid, -1):
            if not os.path.exists(out(f)):
                if not propagate_frame(ebsynth_bin, styled_keyframes[kf_b], src(kf_b), src(f), out(f)):
                    shutil.copy2(styled_keyframes[kf_b], out(f))
            done += 1
            _progress(done, total, start_time)

    print(f"\nPropagation complete: {done}/{total} frames in {time.time() - start_time:.0f}s", flush=True)


def _progress(done, total, start_time):
    if done % 50 == 0 or done == total:
        elapsed = time.time() - start_time
        fps = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / fps if fps > 0 else 0
        print(f"  [propagate {done}/{total}] {fps:.1f} fps, ETA {eta:.0f}s", flush=True)


# ── ComfyUI ──────────────────────────────────────────────────────────────────

def ensure_comfyui_running(timeout=900):
    """Ensure ComfyUI is running — start it if not."""
    # Check if already running
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=5)
        print("ComfyUI already running.", flush=True)
        return True
    except Exception:
        pass

    # Try to start it
    print("ComfyUI not running. Attempting to start...", flush=True)

    # Find ComfyUI installation
    comfy_dirs = [
        "/workspace/ComfyUI_clean",
        "/workspace/ComfyUI",
        "/opt/workspace-internal/ComfyUI",
    ]
    comfy_dir = None
    for d in comfy_dirs:
        main_py = os.path.join(d, "main.py")
        if os.path.exists(main_py):
            comfy_dir = d
            break

    if comfy_dir is None:
        # Clone fresh
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

    # Ensure models are linked
    models_dir = os.path.join(comfy_dir, "models")
    internal_models = "/opt/workspace-internal/ComfyUI/models"
    if os.path.exists(internal_models) and not os.path.islink(models_dir):
        print("  Linking models directory...", flush=True)
        if os.path.exists(models_dir):
            shutil.rmtree(models_dir)
        os.symlink(internal_models, models_dir)

    # Start ComfyUI
    print(f"  Starting ComfyUI from {comfy_dir}...", flush=True)
    subprocess.Popen(
        ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"],
        cwd=comfy_dir,
        stdout=open("/tmp/comfyui.log", "w"),
        stderr=subprocess.STDOUT,
    )

    # Wait for it
    print("  Waiting for ComfyUI to be ready...", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=5)
            print("  ComfyUI ready.", flush=True)
            return True
        except Exception:
            time.sleep(3)

    print("  ERROR: ComfyUI failed to start. Check /tmp/comfyui.log", flush=True)
    return False


def upload_image(image_path):
    boundary = uuid.uuid4().hex
    filename = os.path.basename(image_path)
    with open(image_path, "rb") as f:
        file_data = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read()).get("name", filename)


def queue_and_wait(workflow, timeout=120):
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
        time.sleep(0.5)
    raise TimeoutError(f"Prompt {prompt_id} timed out")


def download_output(filename, subfolder, output_path):
    params = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": "output"})
    urllib.request.urlretrieve(f"{COMFYUI_URL}/view?{params}", output_path)


def build_workflow(image_name, prompt, negative_prompt, denoise, seed, model):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "3": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["1", 1]}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
            "latent_image": ["3", 0], "seed": seed, "steps": 20, "cfg": 7.0,
            "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": denoise}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "beatlab_render"}},
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "v1-5-pruned-emaonly-fp16.safetensors"
    negative_prompt = "blurry, low quality, distorted, deformed"

    os.makedirs(output_dir, exist_ok=True)

    # Load keyframes
    keyframes_path = os.path.join(input_dir, "keyframes.json")
    if not os.path.exists(keyframes_path):
        print("ERROR: keyframes.json not found — falling back to frame_params.json", flush=True)
        # Fall back to old per-frame behavior
        params_path = os.path.join(input_dir, "frame_params.json")
        if not os.path.exists(params_path):
            print("ERROR: No keyframes.json or frame_params.json found", flush=True)
            sys.exit(1)
        # Import and run old per-frame logic
        _run_per_frame(input_dir, output_dir, model, negative_prompt, params_path)
        return

    with open(keyframes_path) as f:
        keyframes = json.load(f)

    print(f"Loaded {len(keyframes)} keyframes", flush=True)

    # ── Phase 1: Render keyframes through SD ──
    print(f"\nPhase 1: Rendering {len(keyframes)} keyframes through SD...", flush=True)

    print("Waiting for ComfyUI...", flush=True)
    if not ensure_comfyui_running():
        print("ERROR: ComfyUI not responding", flush=True)
        sys.exit(1)
    print("ComfyUI ready.", flush=True)

    # Check models
    try:
        with urllib.request.urlopen(f"{COMFYUI_URL}/object_info/CheckpointLoaderSimple", timeout=10) as resp:
            info = json.loads(resp.read())
            available = info.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
            if available and model not in available:
                print(f"WARNING: {model} not found, using {available[0]}", flush=True)
                model = available[0]
    except Exception as e:
        print(f"Could not check models: {e}", flush=True)

    styled_keyframes_dir = os.path.join(output_dir, "_keyframes")
    os.makedirs(styled_keyframes_dir, exist_ok=True)

    styled_map = {}  # frame_num → styled path
    total_kf = len(keyframes)
    start_time = time.time()

    for idx, kf in enumerate(keyframes):
        frame_num = kf["frame"]
        input_path = os.path.join(input_dir, f"frame_{frame_num:06d}.png")
        styled_path = os.path.join(styled_keyframes_dir, f"frame_{frame_num:06d}.png")

        if os.path.exists(styled_path):
            styled_map[frame_num] = styled_path
            print(f"  [{idx+1}/{total_kf}] frame {frame_num} (cached)", flush=True)
            continue

        if not os.path.exists(input_path):
            print(f"  [{idx+1}/{total_kf}] frame {frame_num} SKIPPED (not found)", flush=True)
            continue

        try:
            uploaded = upload_image(input_path)
            workflow = build_workflow(uploaded, kf["prompt"], negative_prompt, kf["denoise"], kf["seed"], model)
            result = queue_and_wait(workflow)
            outputs = result.get("outputs", {})
            for node_output in outputs.values():
                images = node_output.get("images", [])
                if images:
                    download_output(images[0]["filename"], images[0].get("subfolder", ""), styled_path)
                    styled_map[frame_num] = styled_path
                    break
        except Exception as e:
            print(f"  [{idx+1}/{total_kf}] frame {frame_num} FAILED: {e}", flush=True)
            continue

        elapsed = time.time() - start_time
        fps = (idx + 1) / elapsed if elapsed > 0 else 0
        eta = (total_kf - idx - 1) / fps if fps > 0 else 0
        print(f"  [{idx+1}/{total_kf}] frame {frame_num} done ({fps:.1f} fps, ETA {eta:.0f}s)", flush=True)

    print(f"\nPhase 1 complete: {len(styled_map)}/{total_kf} keyframes rendered", flush=True)

    if not styled_map:
        print("ERROR: No keyframes rendered successfully", flush=True)
        sys.exit(1)

    # ── Phase 2: EbSynth propagation ──
    ebsynth_bin = ensure_ebsynth()
    if ebsynth_bin:
        print(f"\nPhase 2: EbSynth propagation...", flush=True)
        propagate_all(input_dir, styled_map, output_dir, ebsynth_bin)
    else:
        print("\nPhase 2: No EbSynth — copying keyframes only (no propagation)", flush=True)
        for frame_num, styled_path in styled_map.items():
            out = os.path.join(output_dir, f"frame_{frame_num:06d}.png")
            if not os.path.exists(out):
                shutil.copy2(styled_path, out)

    total_output = len([f for f in os.listdir(output_dir) if f.startswith("frame_") and f.endswith(".png")])
    print(f"\nRender complete: {total_output} frames in output", flush=True)


def _run_per_frame(input_dir, output_dir, model, negative_prompt, params_path):
    """Fallback: old per-frame rendering."""
    with open(params_path) as f:
        frame_params = json.load(f)

    print(f"Per-frame mode: {len(frame_params)} frames", flush=True)
    print("Waiting for ComfyUI...", flush=True)
    if not ensure_comfyui_running():
        print("ERROR: ComfyUI not responding", flush=True)
        sys.exit(1)

    done = 0
    total = len(frame_params)
    start_time = time.time()

    for fp in frame_params:
        frame_num = fp["frame"]
        input_path = os.path.join(input_dir, f"frame_{frame_num:06d}.png")
        output_path = os.path.join(output_dir, f"frame_{frame_num:06d}.png")

        if os.path.exists(output_path):
            done += 1
            continue
        if not os.path.exists(input_path):
            done += 1
            continue

        try:
            uploaded = upload_image(input_path)
            workflow = build_workflow(uploaded, fp["prompt"], negative_prompt, fp["denoise"], fp["seed"], model)
            result = queue_and_wait(workflow)
            for node_output in result.get("outputs", {}).values():
                images = node_output.get("images", [])
                if images:
                    download_output(images[0]["filename"], images[0].get("subfolder", ""), output_path)
                    break
        except Exception as e:
            print(f"  Frame {frame_num} FAILED: {e}", flush=True)

        done += 1
        if done % 10 == 0:
            elapsed = time.time() - start_time
            fps = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / fps if fps > 0 else 0
            print(f"  [{done}/{total}] {fps:.1f} fps, ETA {eta:.0f}s", flush=True)

    print(f"\nRender complete: {done}/{total} frames", flush=True)


if __name__ == "__main__":
    main()
