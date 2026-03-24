"""Google AI video pipeline — Nano Banana (image stylization) + Veo (video generation)."""

from __future__ import annotations

import io
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


class GoogleVideoClient:
    """Stylize images with Nano Banana and generate video clips with Veo."""

    def __init__(self, api_key: str | None = None, vertex: bool = False,
                 project: str | None = None, location: str = "us-central1"):
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "The 'google-genai' package is required.\n"
                "Install with: pip install google-genai"
            )

        import os

        if vertex:
            # Vertex AI auth — uses ADC (gcloud auth) or service account
            proj = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not proj:
                raise ValueError(
                    "GOOGLE_CLOUD_PROJECT environment variable is required for --vertex.\n"
                    "Set it with: export GOOGLE_CLOUD_PROJECT=your-project-id"
                )
            self.client = genai.Client(vertexai=True, project=proj, location=location)
        else:
            # AI Studio auth — API key
            key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise ValueError(
                    "GOOGLE_API_KEY environment variable is required.\n"
                    "Get a key at: https://aistudio.google.com/apikey"
                )
            self.client = genai.Client(api_key=key)

        self._genai = genai
        self._types = types

    def stylize_image(
        self,
        image_path: str,
        style_prompt: str,
        output_path: str,
        model: str = "gemini-2.5-flash-image",
    ) -> str:
        """Stylize an image using Nano Banana (Gemini image generation).

        Args:
            image_path: Path to source image.
            style_prompt: Style description.
            output_path: Where to save the styled image.
            model: Nano Banana model name.

        Returns:
            output_path
        """
        from google.genai import types

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Detect mime type
        ext = Path(image_path).suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "image/png")

        response = self.client.models.generate_content(
            model=model,
            contents=[
                types.Content(parts=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime),
                    types.Part(text=f"Restyle this image in the following style, keeping the composition and subject intact: {style_prompt}"),
                ]),
            ],
            config=types.GenerateContentConfig(
                response_modalities=["image", "text"],
            ),
        )

        # Find the image part in the response
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                with open(output_path, "wb") as f:
                    f.write(part.inline_data.data)
                return output_path

        raise RuntimeError("Nano Banana did not return an image")

    def generate_video_from_image(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        duration_seconds: int = 8,
        model: str = "veo-3.0-generate-001",
        aspect_ratio: str = "16:9",
    ) -> str:
        """Generate a video clip from a reference image using Veo.

        Args:
            image_path: Path to reference/start frame image.
            prompt: Video generation prompt.
            output_path: Where to save the video.
            duration_seconds: Clip duration (max 8).
            model: Veo model name.
            aspect_ratio: Output aspect ratio.

        Returns:
            output_path
        """
        from google.genai import types

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        ext = Path(image_path).suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "image/png")

        img = types.Image(image_bytes=image_bytes, mime_type=mime)

        operation = self.client.models.generate_videos(
            model=model,
            prompt=prompt,
            image=img,
            config=types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                number_of_videos=1,
                duration_seconds=duration_seconds,
                person_generation="allow_adult",
            ),
        )

        # Poll until done
        while not operation.done:
            time.sleep(10)
            operation = self.client.operations.get(operation)

        generated = operation.result.generated_videos[0]
        self.client.files.download(file=generated.video)
        generated.video.save(output_path)
        return output_path

    def generate_video_transition(
        self,
        start_frame_path: str,
        end_frame_path: str,
        prompt: str,
        output_path: str,
        duration_seconds: int = 2,
        model: str = "veo-3.0-generate-001",
    ) -> str:
        """Generate a transition clip between two frames using Veo.

        Uses first/last frame conditioning to morph between styles.

        Args:
            start_frame_path: Last frame of section A.
            end_frame_path: First frame of section B.
            prompt: Transition prompt.
            output_path: Where to save the transition clip.
            duration_seconds: Transition duration.
            model: Veo model name.

        Returns:
            output_path
        """
        from google.genai import types

        with open(start_frame_path, "rb") as f:
            start_bytes = f.read()
        with open(end_frame_path, "rb") as f:
            end_bytes = f.read()

        ext_a = Path(start_frame_path).suffix.lower()
        ext_b = Path(end_frame_path).suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

        start_img = types.Image(image_bytes=start_bytes, mime_type=mime_map.get(ext_a, "image/png"))

        # Try veo-3.1 with last_frame, fall back to veo-3.0 without it
        try:
            end_img = types.Image(image_bytes=end_bytes, mime_type=mime_map.get(ext_b, "image/png"))
            operation = self.client.models.generate_videos(
                model="veo-3.1-generate-preview",
                prompt=prompt,
                image=start_img,
                config=types.GenerateVideosConfig(
                    aspect_ratio="16:9",
                    number_of_videos=1,
                    duration_seconds=duration_seconds,
                    person_generation="allow_adult",
                    last_frame=end_img,
                ),
            )
        except Exception:
            # Fall back to start-frame-only on veo-3.0
            operation = self.client.models.generate_videos(
                model=model,
                prompt=prompt,
                image=start_img,
                config=types.GenerateVideosConfig(
                    aspect_ratio="16:9",
                    number_of_videos=1,
                    duration_seconds=duration_seconds,
                    person_generation="allow_adult",
                ),
            )

        while not operation.done:
            time.sleep(10)
            operation = self.client.operations.get(operation)

        generated = operation.result.generated_videos[0]
        self.client.files.download(file=generated.video)
        generated.video.save(output_path)
        return output_path
