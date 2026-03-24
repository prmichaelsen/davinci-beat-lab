"""Kling 3.0 video generation via Replicate — start/end frame transitions."""

from __future__ import annotations

import base64
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


class KlingClient:
    """Generate video clips with Kling 3.0 via Replicate API."""

    def __init__(self, api_token: str | None = None):
        try:
            import replicate
        except ImportError:
            raise ImportError(
                "The 'replicate' package is required for --engine kling.\n"
                "Install with: pip install replicate"
            )

        token = api_token or os.environ.get("REPLICATE_API_TOKEN")
        if not token:
            raise ValueError(
                "REPLICATE_API_TOKEN environment variable is required.\n"
                "Get a token at: https://replicate.com/account/api-tokens"
            )

        self._replicate = replicate
        self.client = replicate.Client(api_token=token)

    def _image_to_data_uri(self, image_path: str) -> str:
        """Convert a local image to a data URI for Replicate."""
        ext = Path(image_path).suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "image/png")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:{mime};base64,{b64}"

    def generate_segment(
        self,
        start_frame_path: str,
        end_frame_path: str,
        prompt: str,
        output_path: str,
        duration: int = 10,
        model: str = "kwaivgi/kling-v3-omni-video",
    ) -> str:
        """Generate a video segment from start frame to end frame.

        Args:
            start_frame_path: Path to start frame image.
            end_frame_path: Path to end frame image.
            prompt: Video generation prompt.
            output_path: Where to save the output video.
            duration: Clip duration in seconds (5 or 10).
            model: Replicate model identifier.

        Returns:
            output_path
        """
        start_uri = self._image_to_data_uri(start_frame_path)
        end_uri = self._image_to_data_uri(end_frame_path)

        output = self.client.run(
            model,
            input={
                "prompt": prompt,
                "start_image": start_uri,
                "end_image": end_uri,
                "duration": duration,
                "aspect_ratio": "16:9",
            },
        )

        # Replicate returns a URL or FileOutput — download it
        if hasattr(output, "read"):
            with open(output_path, "wb") as f:
                f.write(output.read())
        elif isinstance(output, str):
            import urllib.request
            urllib.request.urlretrieve(output, output_path)
        elif isinstance(output, list) and len(output) > 0:
            url = str(output[0])
            import urllib.request
            urllib.request.urlretrieve(url, output_path)
        else:
            raise RuntimeError(f"Unexpected Replicate output type: {type(output)}")

        return output_path

    def generate_from_image(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        duration: int = 10,
        model: str = "kwaivgi/kling-v3-omni-video",
    ) -> str:
        """Generate a video from a single start image (no end frame).

        Args:
            image_path: Path to start frame image.
            prompt: Video generation prompt.
            output_path: Where to save the output video.
            duration: Clip duration in seconds.
            model: Replicate model identifier.

        Returns:
            output_path
        """
        image_uri = self._image_to_data_uri(image_path)

        output = self.client.run(
            model,
            input={
                "prompt": prompt,
                "start_image": image_uri,
                "duration": duration,
                "aspect_ratio": "16:9",
            },
        )

        if hasattr(output, "read"):
            with open(output_path, "wb") as f:
                f.write(output.read())
        elif isinstance(output, str):
            import urllib.request
            urllib.request.urlretrieve(output, output_path)
        elif isinstance(output, list) and len(output) > 0:
            url = str(output[0])
            import urllib.request
            urllib.request.urlretrieve(url, output_path)
        else:
            raise RuntimeError(f"Unexpected Replicate output type: {type(output)}")

        return output_path
