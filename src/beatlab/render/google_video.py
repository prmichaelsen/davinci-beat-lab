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


def _retry_on_429(func, *args, max_retries: int = 5, **kwargs):
    """Retry a function call with exponential backoff on 429 rate limit errors.

    After exhausting max_retries, waits 60s and resets the retry counter indefinitely.
    """
    while True:
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 2 ** (attempt + 1)  # 2, 4, 8, 16, 32 seconds
                    _log(f"  Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise
        _log(f"  Still rate limited after {max_retries} retries. Waiting 60s then resetting...")
        time.sleep(60)


class PromptRejectedError(Exception):
    """Raised when Veo repeatedly returns None — likely a content safety rejection."""
    pass


def _retry_video_generation(generate_fn, client, output_path, max_retries: int = 8, on_status=None):
    """Retry video generation with backoff on NoneType/transient failures.

    Handles: rate limits (429), transient None results (common with Veo preview models),
    and timeouts. Only raises PromptRejectedError after 6+ consecutive None results.

    Args:
        on_status: optional callback(message: str) for progress reporting
    """
    def _status(msg):
        _log(f"    {msg}")
        if on_status:
            on_status(msg)

    for attempt in range(max_retries):
        try:
            _status(f"Submitting to Veo (attempt {attempt + 1}/{max_retries})...")
            operation = generate_fn()
            _status("Veo accepted, waiting for result...")

            # Poll until done (timeout after 10 minutes)
            poll_start = time.time()
            poll_count = 0
            while not operation.done:
                elapsed = time.time() - poll_start
                if elapsed > 600:
                    raise TimeoutError("Veo generation polling timed out after 10 minutes")
                poll_count += 1
                if poll_count % 3 == 0:  # Log every 30s
                    _status(f"Waiting for Veo... ({int(elapsed)}s)")
                time.sleep(10)
                operation = client.operations.get(operation)
            _status(f"Veo complete ({int(time.time() - poll_start)}s)")

            # Check for valid result — fail immediately on None (Vertex charges per request)
            if operation.result is None:
                raise PromptRejectedError(
                    "Veo returned None result. Likely prompt rejection or content filter. "
                    "Edit the transition action and retry. (Not retrying — Vertex charges per attempt.)"
                )
            if not operation.result.generated_videos:
                raise PromptRejectedError(
                    "Veo returned empty video list. Likely prompt rejection or content filter. "
                    "Edit the transition action and retry. (Not retrying — Vertex charges per attempt.)"
                )

            generated = operation.result.generated_videos[0]
            if generated is None:
                raise PromptRejectedError(
                    "Veo generated video is None. Likely prompt rejection or content filter. "
                    "Edit the transition action and retry. (Not retrying — Vertex charges per attempt.)"
                )

            return generated

        except PromptRejectedError:
            raise
        except Exception as e:
            err_str = str(e)

            is_retryable = (
                "429" in err_str
                or "RESOURCE_EXHAUSTED" in err_str
                or "timed out" in err_str.lower()
            )

            if is_retryable:
                wait = min(2 ** (attempt + 1), 60)
                _status(f"Rate limited. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(
        f"Video generation failed after {max_retries} attempts (rate limits or timeouts). "
        f"Try again later."
    )


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
            self._vertex = True
        else:
            # AI Studio auth — API key
            key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise ValueError(
                    "GOOGLE_API_KEY environment variable is required.\n"
                    "Get a key at: https://aistudio.google.com/apikey"
                )
            self.client = genai.Client(api_key=key)
            self._vertex = False

        self._genai = genai
        self._types = types

    def _save_generated_video(self, generated_video, output_path: str) -> None:
        """Download and save a generated video, handling both Vertex and AI Studio."""
        video = generated_video.video

        if self._vertex:
            # Vertex AI: video has a GCS URI — download via gcloud or urllib
            uri = getattr(video, "uri", None)
            if uri and uri.startswith("gs://"):
                import subprocess
                _log(f"  Downloading from GCS: {uri}")
                subprocess.run(
                    ["gcloud", "storage", "cp", uri, output_path],
                    check=True, capture_output=True,
                )
            else:
                # Try direct video data if available
                video_bytes = getattr(video, "video_bytes", None) or getattr(video, "data", None)
                if video_bytes:
                    with open(output_path, "wb") as f:
                        f.write(video_bytes)
                else:
                    raise RuntimeError(
                        f"Vertex AI: cannot download video. URI={uri}, "
                        f"attrs={[a for a in dir(video) if not a.startswith('_')]}"
                    )
        else:
            # AI Studio: use files.download + save
            self.client.files.download(file=video)
            video.save(output_path)

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

        response = _retry_on_429(
            self.client.models.generate_content,
            model=model,
            contents=[
                types.Content(role="user", parts=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime),
                    types.Part(text=f"Restyle this image in the following style, keeping the composition and subject intact. Hyper-realistic, photorealistic quality. Like a still from a big-budget film shot on 35mm. Rich intricate detail, complex natural textures, sophisticated cinematic lighting, depth of field. Every surface has realistic material properties — metal looks like metal, skin looks like skin, glass refracts light. Style: {style_prompt}"),
                ]),
            ],
            config=types.GenerateContentConfig(
                response_modalities=["image", "text"],
            ),
        )

        # Find the image part in the response
        candidates = response.candidates or []
        if not candidates:
            # Content filter likely blocked it — retry with sanitized prompt
            finish_reason = getattr(response, "prompt_feedback", None)
            raise RuntimeError(
                f"Nano Banana returned no candidates (likely content filter). "
                f"Feedback: {finish_reason}. Style prompt was: {style_prompt[:100]}"
            )

        parts = candidates[0].content.parts if candidates[0].content else []
        for part in parts or []:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                with open(output_path, "wb") as f:
                    f.write(part.inline_data.data)
                return output_path

        raise RuntimeError("Nano Banana did not return an image")

    @staticmethod
    def _load_ingredient_images(ingredient_paths: list[str]) -> list:
        """Load ingredient images as VideoGenerationReferenceImage objects.

        Args:
            ingredient_paths: Up to 3 paths to character/object/style reference images.

        Returns:
            List of VideoGenerationReferenceImage objects.
        """
        from google.genai import types
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
        refs = []
        for p in ingredient_paths[:3]:
            with open(p, "rb") as f:
                img_bytes = f.read()
            ext = Path(p).suffix.lower()
            refs.append(
                types.VideoGenerationReferenceImage(
                    image=types.Image(
                        image_bytes=img_bytes,
                        mime_type=mime_map.get(ext, "image/png"),
                    ),
                    reference_type="asset",
                )
            )
        return refs

    def generate_video_from_image(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        duration_seconds: int = 8,
        model: str = "veo-3.0-generate-001",
        aspect_ratio: str = "16:9",
        ingredients: list[str] | None = None,
    ) -> str:
        """Generate a video clip from a reference image using Veo.

        Args:
            image_path: Path to reference/start frame image.
            prompt: Video generation prompt.
            output_path: Where to save the video.
            duration_seconds: Clip duration (max 8).
            model: Veo model name.
            aspect_ratio: Output aspect ratio.
            ingredients: Optional list of up to 3 character/object reference image paths.

        Returns:
            output_path
        """
        from google.genai import types

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        ext = Path(image_path).suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "image/png")

        img = types.Image(image_bytes=image_bytes, mime_type=mime)

        ref_images = self._load_ingredient_images(ingredients) if ingredients else None
        # Ingredients require veo-3.1
        if ref_images:
            model = "veo-3.1-generate-preview"

        def _generate():
            config = types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                number_of_videos=1,
                duration_seconds=8,
                person_generation="allow_adult",
                **({"reference_images": ref_images} if ref_images else {}),
            )
            return _retry_on_429(
                self.client.models.generate_videos,
                model=model,
                prompt=prompt,
                image=img,
                config=config,
            )

        generated = _retry_video_generation(_generate, self.client, output_path)
        self._save_generated_video(generated, output_path)
        return output_path

    def generate_video_transition(
        self,
        start_frame_path: str,
        end_frame_path: str,
        prompt: str,
        output_path: str,
        duration_seconds: int = 2,
        model: str = "veo-3.0-generate-001",
        ingredients: list[str] | None = None,
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
            ingredients: Optional list of up to 3 character/object reference image paths.

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
        end_img = types.Image(image_bytes=end_bytes, mime_type=mime_map.get(ext_b, "image/png"))

        ref_images = self._load_ingredient_images(ingredients) if ingredients else None

        # Veo requires duration 6-8s; clamp to valid range
        # Always request 8s — Veo charges the same regardless of duration,
        # and shorter requests often get rejected. We trim to target duration in post.
        _log(f"    Generating 8s transition (target {duration_seconds}s) | prompt: {prompt[:80]}...")

        def _generate():
            config = types.GenerateVideosConfig(
                aspect_ratio="16:9",
                number_of_videos=1,
                duration_seconds=8,
                person_generation="allow_adult",
                last_frame=end_img,
                **({"reference_images": ref_images} if ref_images else {}),
            )
            return _retry_on_429(
                self.client.models.generate_videos,
                model="veo-3.1-generate-preview",
                prompt=prompt,
                image=start_img,
                config=config,
            )

        generated = _retry_video_generation(_generate, self.client, output_path)
        self._save_generated_video(generated, output_path)
        return output_path
