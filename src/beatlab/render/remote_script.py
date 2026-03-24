"""Remote render script — runs ON the cloud GPU instance via SSH.

This script is uploaded to the instance and executed there.
It reads frame_params.json, processes each frame through ComfyUI's API,
and saves results to the output directory.

Usage (on remote):
    python3 /workspace/render.py /workspace/input /workspace/output
"""

import json
import os
import sys
import time
import urllib.request
import urllib.parse
import uuid


COMFYUI_URL = "http://127.0.0.1:8188"
CLIENT_ID = str(uuid.uuid4())


def wait_for_comfyui(timeout=300):
    """Wait for ComfyUI to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=5)
            return True
        except Exception:
            time.sleep(2)
    return False


def upload_image(image_path):
    """Upload an image to ComfyUI's input directory."""
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
        f"{COMFYUI_URL}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    return result.get("name", filename)


def queue_and_wait(workflow, timeout=120):
    """Queue a workflow and wait for completion."""
    data = json.dumps({"prompt": workflow, "client_id": CLIENT_ID}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    prompt_id = result["prompt_id"]

    # Poll for completion
    start = time.time()
    while time.time() - start < timeout:
        with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10) as resp:
            history = json.loads(resp.read())
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(0.5)
    raise TimeoutError(f"Prompt {prompt_id} timed out after {timeout}s")


def download_output(filename, subfolder, output_path):
    """Download a generated image from ComfyUI."""
    params = urllib.parse.urlencode({
        "filename": filename, "subfolder": subfolder, "type": "output"
    })
    urllib.request.urlretrieve(f"{COMFYUI_URL}/view?{params}", output_path)


def build_workflow(image_name, prompt, negative_prompt, denoise, seed, model,
                   controlnet_model=None):
    """Build img2img workflow."""
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": model}},
        "2": {"class_type": "LoadImage",
              "inputs": {"image": image_name}},
        "3": {"class_type": "VAEEncode",
              "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["1", 1]}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative_prompt, "clip": ["1", 1]}},
        "6": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0],
                         "negative": ["5", 0], "latent_image": ["3", 0],
                         "seed": seed, "steps": 20, "cfg": 7.0,
                         "sampler_name": "euler_ancestral",
                         "scheduler": "normal", "denoise": denoise}},
        "7": {"class_type": "VAEDecode",
              "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage",
              "inputs": {"images": ["7", 0],
                         "filename_prefix": "beatlab_render"}},
    }

    if controlnet_model:
        wf["10"] = {"class_type": "ControlNetLoader",
                     "inputs": {"control_net_name": controlnet_model}}
        wf["11"] = {"class_type": "Canny",
                     "inputs": {"image": ["2", 0],
                                "low_threshold": 100, "high_threshold": 200}}
        wf["12"] = {"class_type": "ControlNetApplyAdvanced",
                     "inputs": {"positive": ["4", 0], "negative": ["5", 0],
                                "control_net": ["10", 0], "image": ["11", 0],
                                "strength": 0.7, "start_percent": 0.0,
                                "end_percent": 1.0}}
        wf["6"]["inputs"]["positive"] = ["12", 0]
        wf["6"]["inputs"]["negative"] = ["12", 1]

    return wf


def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "sd_xl_base_1.0.safetensors"
    negative_prompt = "blurry, low quality, distorted, deformed"

    os.makedirs(output_dir, exist_ok=True)

    # Load frame params
    params_path = os.path.join(input_dir, "frame_params.json")
    with open(params_path) as f:
        frame_params = json.load(f)

    print(f"Waiting for ComfyUI...", flush=True)
    if not wait_for_comfyui():
        print("ERROR: ComfyUI not responding", flush=True)
        sys.exit(1)
    print(f"ComfyUI ready. Rendering {len(frame_params)} frames...", flush=True)

    # Check available models
    try:
        with urllib.request.urlopen(f"{COMFYUI_URL}/object_info/CheckpointLoaderSimple", timeout=10) as resp:
            info = json.loads(resp.read())
            available_models = info.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
            if available_models:
                print(f"Available models: {available_models}", flush=True)
                if model not in available_models:
                    print(f"WARNING: {model} not found, using {available_models[0]}", flush=True)
                    model = available_models[0]
    except Exception as e:
        print(f"Could not check models: {e}", flush=True)

    done = 0
    total = len(frame_params)
    start_time = time.time()

    for fp in frame_params:
        frame_num = fp["frame"]
        input_path = os.path.join(input_dir, f"frame_{frame_num:06d}.png")
        output_path = os.path.join(output_dir, f"frame_{frame_num:06d}.png")

        # Skip already rendered
        if os.path.exists(output_path):
            done += 1
            continue

        if not os.path.exists(input_path):
            done += 1
            continue

        try:
            uploaded_name = upload_image(input_path)
            workflow = build_workflow(
                uploaded_name, fp["prompt"], negative_prompt,
                fp["denoise"], fp["seed"], model,
            )
            result = queue_and_wait(workflow)

            # Find output image
            outputs = result.get("outputs", {})
            for node_id, node_output in outputs.items():
                images = node_output.get("images", [])
                if images:
                    download_output(
                        images[0]["filename"],
                        images[0].get("subfolder", ""),
                        output_path,
                    )
                    break
        except Exception as e:
            print(f"  Frame {frame_num} FAILED: {e}", flush=True)
            continue

        done += 1
        elapsed = time.time() - start_time
        fps = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / fps if fps > 0 else 0
        print(f"  [{done}/{total}] frame {frame_num} done ({fps:.1f} fps, ETA {eta:.0f}s)", flush=True)

    print(f"\nRender complete: {done}/{total} frames in {time.time() - start_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
