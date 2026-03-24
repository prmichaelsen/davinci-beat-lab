"""Wan2.1 video-to-video rendering via ComfyUI."""

from __future__ import annotations

import json
import math
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Callable


class Wan21Client:
    """Renders video clips through Wan2.1 via ComfyUI API."""

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
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())

    def _get_json(self, path: str) -> dict:
        with urllib.request.urlopen(self._api_url(path), timeout=60) as resp:
            return json.loads(resp.read())

    def upload_video(self, video_path: str) -> str:
        """Upload a video clip to ComfyUI's input directory."""
        boundary = uuid.uuid4().hex
        filename = Path(video_path).name

        with open(video_path, "rb") as f:
            file_data = f.read()

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: video/mp4\r\n\r\n"
        ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            self._api_url("/upload/image"),
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        return result.get("name", filename)

    def queue_prompt(self, workflow: dict) -> str:
        """Queue a workflow for execution. Returns prompt_id."""
        data = {"prompt": workflow, "client_id": self.client_id}
        result = self._post_json("/prompt", data)
        return result["prompt_id"]

    def wait_for_completion(self, prompt_id: str, timeout: int = 600) -> dict:
        """Poll until prompt completes. Returns output info."""
        start = time.time()
        while time.time() - start < timeout:
            history = self._get_json(f"/history/{prompt_id}")
            if prompt_id in history:
                return history[prompt_id]
            time.sleep(2)
        raise TimeoutError(f"Wan2.1 prompt {prompt_id} did not complete within {timeout}s")

    def download_output_video(self, filename: str, subfolder: str, output_path: str) -> None:
        """Download a generated video from ComfyUI."""
        params = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": "output"})
        url = self._api_url(f"/view?{params}")
        urllib.request.urlretrieve(url, output_path)

    def render_clip(
        self,
        video_path: str,
        style_prompt: str,
        denoise: float = 0.5,
        resolution: tuple[int, int] = (1280, 720),
        negative_prompt: str = "blurry, low quality, distorted, deformed, static, frozen",
        model: str = "wan2.1_v2v_720p.safetensors",
        seed: int | None = None,
    ) -> dict:
        """Render a single video clip through Wan2.1 v2v.

        Args:
            video_path: Path to source video clip.
            style_prompt: Style/content prompt for Wan2.1.
            denoise: Denoising strength (0.0-1.0). Higher = more stylization.
            resolution: Output resolution (width, height).
            negative_prompt: Negative prompt.
            model: Wan2.1 model checkpoint name.
            seed: Random seed. None = random.

        Returns:
            Output info dict with filename/subfolder for download.
        """
        uploaded_name = self.upload_video(video_path)

        if seed is None:
            import random
            seed = random.randint(0, 2**32 - 1)

        workflow = build_wan21_v2v_workflow(
            video_name=uploaded_name,
            prompt=style_prompt,
            negative_prompt=negative_prompt,
            denoise=denoise,
            seed=seed,
            model=model,
            width=resolution[0],
            height=resolution[1],
        )

        prompt_id = self.queue_prompt(workflow)
        result = self.wait_for_completion(prompt_id)

        outputs = result.get("outputs", {})
        for node_id, node_output in outputs.items():
            # Wan2.1 outputs video via VHS_VideoCombine or similar
            gifs = node_output.get("gifs", [])
            if gifs:
                return gifs[0]
            videos = node_output.get("videos", [])
            if videos:
                return videos[0]
            images = node_output.get("images", [])
            if images:
                return images[0]

        raise RuntimeError(f"No output from Wan2.1 prompt {prompt_id}")


def build_wan21_v2v_workflow(
    video_name: str,
    prompt: str,
    negative_prompt: str,
    denoise: float,
    seed: int,
    model: str,
    width: int = 1280,
    height: int = 720,
) -> dict:
    """Build a ComfyUI API workflow for Wan2.1 video-to-video.

    This workflow uses the standard ComfyUI Wan2.1 v2v nodes:
    - LoadVideo → Wan2.1 model → KSampler → SaveVideo
    """
    return {
        # Load Wan2.1 checkpoint
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": model},
        },
        # Load input video
        "2": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": video_name,
                "force_rate": 0,
                "force_size": "Disabled",
                "frame_load_cap": 0,
                "skip_first_frames": 0,
            },
        },
        # VAE Encode video frames
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
        # VAE Decode
        "7": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["6", 0],
                "vae": ["1", 2],
            },
        },
        # Save as video
        "8": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["7", 0],
                "frame_rate": 30,
                "loop_count": 0,
                "filename_prefix": "beatlab_wan",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            },
        },
    }


def chunk_section_frames(
    frames_dir: str,
    start_frame: int,
    end_frame: int,
    fps: float,
    max_clip_seconds: float = 8.0,
) -> list[tuple[int, int]]:
    """Split a section's frame range into chunks of max_clip_seconds.

    Args:
        frames_dir: Directory containing extracted frames.
        start_frame: First frame of the section.
        end_frame: Last frame of the section (exclusive).
        fps: Frame rate.
        max_clip_seconds: Maximum clip duration in seconds.

    Returns:
        List of (start_frame, end_frame) tuples for each chunk.
    """
    max_frames = int(max_clip_seconds * fps)
    if max_frames < 1:
        max_frames = 1

    chunks = []
    current = start_frame
    while current < end_frame:
        chunk_end = min(current + max_frames, end_frame)
        chunks.append((current, chunk_end))
        current = chunk_end

    return chunks


def frames_to_clip(
    frames_dir: str,
    start_frame: int,
    end_frame: int,
    fps: float,
    output_path: str,
) -> str:
    """Assemble numbered frames into a video clip using ffmpeg.

    Args:
        frames_dir: Directory containing frame_NNNNNN.png files.
        start_frame: First frame number.
        end_frame: Last frame number (exclusive).
        fps: Frame rate for output.
        output_path: Where to write the clip.

    Returns:
        output_path
    """
    import subprocess

    # Create a temporary file list for ffmpeg concat (absolute paths)
    frame_list = []
    for i in range(start_frame, end_frame):
        p = Path(frames_dir).resolve() / f"frame_{i:06d}.png"
        if p.exists():
            frame_list.append(str(p))

    if not frame_list:
        raise ValueError(f"No frames found in range {start_frame}-{end_frame}")

    # Write concat file
    concat_path = output_path + ".concat.txt"
    with open(concat_path, "w") as f:
        for fp in frame_list:
            f.write(f"file '{fp}'\n")
            f.write(f"duration {1.0/fps:.6f}\n")

    result = subprocess.run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_path,
            "-vsync", "vfr",
            "-pix_fmt", "yuv420p",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg concat failed (exit {result.returncode}):\n{result.stderr[-500:]}"
        )

    Path(concat_path).unlink(missing_ok=True)
    return output_path
