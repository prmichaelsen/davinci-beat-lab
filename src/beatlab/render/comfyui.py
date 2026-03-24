"""ComfyUI client for remote SD img2img rendering."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Callable

import urllib.request
import urllib.parse


class ComfyUIClient:
    """Connects to a ComfyUI instance and submits img2img render jobs."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8188):
        self.base_url = f"http://{host}:{port}"
        self.client_id = str(uuid.uuid4())

    def _api_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _post_json(self, path: str, data: dict) -> dict:
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            self._api_url(path),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())

    def _get_json(self, path: str) -> dict:
        with urllib.request.urlopen(self._api_url(path), timeout=60) as resp:
            return json.loads(resp.read())

    def upload_image(self, image_path: str, subfolder: str = "input") -> str:
        """Upload an image to ComfyUI's input directory."""
        import mimetypes
        boundary = uuid.uuid4().hex
        filename = Path(image_path).name

        with open(image_path, "rb") as f:
            file_data = f.read()

        content_type = mimetypes.guess_type(image_path)[0] or "image/png"

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            self._api_url("/upload/image"),
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        return result.get("name", filename)

    def queue_prompt(self, workflow: dict) -> str:
        """Queue a workflow for execution. Returns prompt_id."""
        data = {"prompt": workflow, "client_id": self.client_id}
        result = self._post_json("/prompt", data)
        return result["prompt_id"]

    def wait_for_completion(self, prompt_id: str, timeout: int = 120) -> dict:
        """Poll until prompt completes. Returns output info."""
        start = time.time()
        while time.time() - start < timeout:
            history = self._get_json(f"/history/{prompt_id}")
            if prompt_id in history:
                return history[prompt_id]
            time.sleep(1)
        raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout}s")

    def download_output(self, filename: str, subfolder: str, output_path: str) -> None:
        """Download a generated image from ComfyUI."""
        params = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": "output"})
        url = self._api_url(f"/view?{params}")
        urllib.request.urlretrieve(url, output_path)

    def render_frame(
        self,
        image_path: str,
        prompt: str,
        denoise: float,
        seed: int,
        negative_prompt: str = "blurry, low quality, distorted, deformed",
        model: str = "sd_xl_base_1.0.safetensors",
        controlnet_model: str | None = "diffusers_xl_canny_full.safetensors",
    ) -> str:
        """Render a single frame via img2img. Returns path to output on server."""
        uploaded_name = self.upload_image(image_path)

        workflow = build_img2img_workflow(
            image_name=uploaded_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            denoise=denoise,
            seed=seed,
            model=model,
            controlnet_model=controlnet_model,
        )

        prompt_id = self.queue_prompt(workflow)
        result = self.wait_for_completion(prompt_id)

        # Extract output image info
        outputs = result.get("outputs", {})
        for node_id, node_output in outputs.items():
            images = node_output.get("images", [])
            if images:
                return images[0]  # {filename, subfolder, type}

        raise RuntimeError(f"No output images from prompt {prompt_id}")

    def render_batch(
        self,
        frame_params: list[dict],
        frames_dir: str,
        output_dir: str,
        model: str = "sd_xl_base_1.0.safetensors",
        negative_prompt: str = "blurry, low quality, distorted, deformed",
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Render a batch of frames. Skips already-rendered frames for resume support."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        total = len(frame_params)

        for i, fp in enumerate(frame_params):
            frame_num = fp["frame"]
            input_path = f"{frames_dir}/frame_{frame_num:06d}.png"
            output_path = f"{output_dir}/frame_{frame_num:06d}.png"

            # Resume: skip if already rendered
            if Path(output_path).exists():
                if progress_callback:
                    progress_callback(i + 1, total)
                continue

            if not Path(input_path).exists():
                continue

            output_info = self.render_frame(
                image_path=input_path,
                prompt=fp["prompt"],
                denoise=fp["denoise"],
                seed=fp["seed"],
                model=model,
                negative_prompt=negative_prompt,
            )

            # Download the output
            self.download_output(
                filename=output_info["filename"],
                subfolder=output_info.get("subfolder", ""),
                output_path=output_path,
            )

            if progress_callback:
                progress_callback(i + 1, total)


def build_img2img_workflow(
    image_name: str,
    prompt: str,
    negative_prompt: str,
    denoise: float,
    seed: int,
    model: str,
    controlnet_model: str | None = None,
) -> dict:
    """Build a ComfyUI API workflow for img2img with optional ControlNet.

    Returns a workflow dict ready for the /prompt API.
    """
    workflow = {
        # Load checkpoint
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": model},
        },
        # Load input image
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        },
        # VAE Encode (image → latent)
        "3": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["2", 0],
                "vae": ["1", 2],
            },
        },
        # CLIP Text Encode (positive)
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["1", 1],
            },
        },
        # CLIP Text Encode (negative)
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["1", 1],
            },
        },
        # KSampler
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["3", 0],
                "seed": seed,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": denoise,
            },
        },
        # VAE Decode (latent → image)
        "7": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["6", 0],
                "vae": ["1", 2],
            },
        },
        # Save Image
        "8": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["7", 0],
                "filename_prefix": "beatlab_render",
            },
        },
    }

    # Add ControlNet if specified
    if controlnet_model:
        workflow["10"] = {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": controlnet_model},
        }
        # Canny edge detection on input image
        workflow["11"] = {
            "class_type": "Canny",
            "inputs": {
                "image": ["2", 0],
                "low_threshold": 100,
                "high_threshold": 200,
            },
        }
        # Apply ControlNet
        workflow["12"] = {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["4", 0],
                "negative": ["5", 0],
                "control_net": ["10", 0],
                "image": ["11", 0],
                "strength": 0.7,
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        }
        # Rewire KSampler to use ControlNet output
        workflow["6"]["inputs"]["positive"] = ["12", 0]
        workflow["6"]["inputs"]["negative"] = ["12", 1]

    return workflow
