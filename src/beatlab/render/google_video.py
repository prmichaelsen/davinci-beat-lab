"""Video generation pipeline — supports Vertex AI (Google) and Runway backends."""

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
        image_model: str = "replicate/nano-banana-2",
        # Legacy compat
        model: str | None = None,
        backend: str | None = None,
    ) -> str:
        """Stylize an image.

        Args:
            image_path: Path to source image.
            style_prompt: Style description.
            output_path: Where to save the styled image.
            image_model: "provider/model" string, e.g. "replicate/nano-banana-2" or "vertex/gemini-2.5-flash-image".

        Returns:
            output_path
        """
        # Legacy: if old-style backend/model args are passed, convert
        if backend or model:
            b = backend or "replicate"
            m = model or "nano-banana-2"
            image_model = f"{b}/{m}"

        provider, _, model_name = image_model.partition("/")
        if not model_name:
            model_name = provider
            provider = "replicate"

        if provider == "replicate":
            return self._stylize_replicate(image_path, style_prompt, output_path, model_name)
        return self._stylize_vertex(image_path, style_prompt, output_path, model_name)

    def _stylize_replicate(self, image_path: str, style_prompt: str, output_path: str, model_name: str = "nano-banana-2") -> str:
        """Stylize via Replicate."""
        import replicate
        import urllib.request

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        full_prompt = (
            f"Restyle this image in the following style, keeping the composition and subject intact. "
            f"Hyper-realistic, photorealistic quality. Like a still from a big-budget film shot on 35mm. "
            f"Rich intricate detail, complex natural textures, sophisticated cinematic lighting, depth of field. "
            f"Style: {style_prompt}"
        )

        # Map short names to replicate model IDs
        model_map = {
            "nano-banana-2": "google/nano-banana-2",
            "sdxl": "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        }
        replicate_model = model_map.get(model_name, f"google/{model_name}")

        # Detect source image dimensions for aspect ratio
        from PIL import Image as _PILImage
        with _PILImage.open(image_path) as _img:
            src_w, src_h = _img.size
        aspect = f"{src_w}:{src_h}"
        # Map to common aspect ratios
        ratio = src_w / src_h
        if abs(ratio - 16/9) < 0.1:
            aspect = "16:9"
        elif abs(ratio - 9/16) < 0.1:
            aspect = "9:16"
        elif abs(ratio - 4/3) < 0.1:
            aspect = "4:3"
        elif abs(ratio - 3/4) < 0.1:
            aspect = "3:4"
        elif abs(ratio - 1) < 0.1:
            aspect = "1:1"

        _log(f"    [replicate] Generating image with {replicate_model} (src={src_w}x{src_h})...")

        last_err = None
        for attempt in range(3):
            try:
                input_data = {"prompt": full_prompt, "output_format": "png"}
                if "nano-banana" in replicate_model:
                    input_data["image_input"] = [open(image_path, "rb")]
                else:
                    input_data["aspect_ratio"] = aspect

                output = replicate.run(replicate_model, input=input_data)

                url = str(output[0]) if isinstance(output, list) else str(output)
                urllib.request.urlretrieve(url, output_path)
                _log(f"    [replicate] Saved to {output_path}")
                return output_path
            except Exception as e:
                last_err = e
                _log(f"    [replicate] Attempt {attempt + 1}/3 failed: {type(e).__name__}: {e}")
                if attempt < 2:
                    import time as _t
                    _t.sleep(2 * (attempt + 1))

        raise RuntimeError(f"Replicate failed after 3 attempts: {last_err}")

    def _stylize_vertex(self, image_path: str, style_prompt: str, output_path: str, model: str = "gemini-2.5-flash-image") -> str:
        """Stylize via Vertex AI Nano Banana."""
        from google.genai import types

        with open(image_path, "rb") as f:
            image_bytes = f.read()

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

        import time as _time
        for attempt in range(3):
            if attempt > 0:
                _log(f"    Retrying image generation (attempt {attempt + 1})...")
                _time.sleep(2)
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

            candidates = response.candidates or []
            if not candidates:
                finish_reason = getattr(response, "prompt_feedback", None)
                if attempt < 2:
                    continue
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

            if attempt < 2:
                continue

        raise RuntimeError("Nano Banana did not return an image after 3 attempts")

    def generate_image(
        self,
        prompt: str,
        output_path: str,
        aspect_ratio: str = "16:9",
        model: str = "imagen-3.0-generate-002",
        image_backend: str | None = None,
    ) -> str:
        """Generate an image from text prompt only.

        Args:
            prompt: Text description of the image to generate.
            output_path: Where to save the generated image.
            aspect_ratio: Aspect ratio string (e.g. "16:9", "1:1", "9:16").
            model: Imagen model name (for vertex).
            image_backend: Override backend ("vertex", "replicate/<model>").

        Returns:
            output_path
        """
        backend = image_backend or "vertex"
        if backend.startswith("replicate"):
            return self._generate_image_replicate(prompt, output_path, aspect_ratio, backend)

        from google.genai import types

        full_prompt = (
            f"Hyper-realistic, photorealistic image. Like a still from a big-budget "
            f"film shot on 35mm. Rich intricate detail, complex natural textures, "
            f"sophisticated cinematic lighting, depth of field. Every surface has "
            f"realistic material properties. Scene: {prompt}"
        )

        response = _retry_on_429(
            self.client.models.generate_images,
            model=model,
            prompt=full_prompt,
            config=types.GenerateImagesConfig(
                aspect_ratio=aspect_ratio,
                number_of_images=1,
            ),
        )

        if not response.generated_images:
            raise RuntimeError(f"Image generation returned no images. Prompt: {prompt[:100]}")

        img = response.generated_images[0].image
        img.save(output_path)
        return output_path

    def _generate_image_replicate(self, prompt: str, output_path: str, aspect_ratio: str, backend: str) -> str:
        """Generate image via Replicate."""
        import replicate
        import urllib.request

        _, _, model_name = backend.partition("/")
        model_name = model_name or "nano-banana-2"

        model_map = {
            "nano-banana-2": "google/nano-banana-2",
            "sdxl": "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        }
        replicate_model = model_map.get(model_name, f"google/{model_name}")

        full_prompt = (
            f"Hyper-realistic, photorealistic image. Like a still from a big-budget "
            f"film shot on 35mm. Rich intricate detail, complex natural textures, "
            f"sophisticated cinematic lighting, depth of field. Scene: {prompt}"
        )

        _log(f"    [replicate] Generating image with {replicate_model}...")
        last_err = None
        for attempt in range(3):
            try:
                input_data = {"prompt": full_prompt, "output_format": "png", "aspect_ratio": aspect_ratio}
                output = replicate.run(replicate_model, input=input_data)
                url = str(output[0]) if isinstance(output, list) else str(output)
                urllib.request.urlretrieve(url, output_path)
                _log(f"    [replicate] Saved to {output_path}")
                return output_path
            except Exception as e:
                last_err = e
                _log(f"    [replicate] Attempt {attempt + 1}/3 failed: {type(e).__name__}: {e}")
                if attempt < 2:
                    import time as _t
                    _t.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Replicate generate_image failed after 3 attempts: {last_err}")

    def transform_image(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        model: str = "gemini-2.5-flash-image",
        image_backend: str | None = None,
    ) -> str:
        """Transform an image based on a prompt, allowing significant changes to content and composition.

        Unlike stylize_image which preserves composition, this method encourages
        the model to modify the scene according to the prompt.
        """
        backend = image_backend or "vertex"
        if backend.startswith("replicate"):
            return self._stylize_replicate(image_path, prompt, output_path,
                                           backend.partition("/")[2] or "nano-banana-2")

        from google.genai import types

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        ext = Path(image_path).suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "image/png")

        # Get source dimensions to enforce matching aspect ratio
        from PIL import Image as _PILImage
        with _PILImage.open(image_path) as src_img:
            src_w, src_h = src_img.size

        import time as _time
        for attempt in range(3):
            if attempt > 0:
                _time.sleep(2)

            response = _retry_on_429(
                self.client.models.generate_content,
                model=model,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime),
                        types.Part(text=f"Edit this image according to these instructions. You may significantly change the scene, add or remove elements, alter lighting, composition, and atmosphere. Maintain photorealistic, cinematic quality. Output the image at the same aspect ratio as the input. Instructions: {prompt}"),
                    ]),
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["image", "text"],
                ),
            )

            candidates = response.candidates or []
            if not candidates:
                if attempt < 2:
                    continue
                raise RuntimeError(f"Transform returned no candidates. Prompt: {prompt[:100]}")

            parts = candidates[0].content.parts if candidates[0].content else []
            for part in parts or []:
                if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                    # Resize to match source dimensions
                    import io
                    out_img = _PILImage.open(io.BytesIO(part.inline_data.data))
                    if (out_img.width, out_img.height) != (src_w, src_h):
                        out_img = out_img.resize((src_w, src_h), _PILImage.LANCZOS)
                    out_img.save(output_path)
                    return output_path

            if attempt < 2:
                continue

        raise RuntimeError("Transform did not return an image after 3 attempts")

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
            model = "veo-3.1-generate-001"

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

        generated = _retry_video_generation(_generate, self.client, output_path, on_status=on_status)
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
        on_status=None,
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

        # Veo 3.1 supports 4, 6, or 8 second durations
        veo_duration = max(4, min(8, duration_seconds))
        # Snap to nearest valid value
        veo_duration = min([4, 6, 8], key=lambda x: abs(x - veo_duration))
        _log(f"    Generating {veo_duration}s transition (target {duration_seconds}s) | prompt: {prompt[:80]}...")

        def _generate():
            config = types.GenerateVideosConfig(
                aspect_ratio="16:9",
                number_of_videos=1,
                duration_seconds=veo_duration,
                person_generation="allow_adult",
                last_frame=end_img,
                **({"reference_images": ref_images} if ref_images else {}),
            )
            return _retry_on_429(
                self.client.models.generate_videos,
                model="veo-3.1-generate-001",
                prompt=prompt,
                image=start_img,
                config=config,
            )

        generated = _retry_video_generation(_generate, self.client, output_path)
        self._save_generated_video(generated, output_path)
        return output_path


class RunwayVideoClient:
    """Video generation via Runway API — supports Veo 3.1 and Gen-4.5."""

    def __init__(self, api_key: str | None = None, model: str = "veo3.1_fast"):
        import os
        self.api_key = api_key or os.environ.get("RUNWAY_API_KEY", "")
        self.model = model
        if not self.api_key:
            raise RuntimeError("RUNWAY_API_KEY not set")

    def generate_video_from_image(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        duration_seconds: int = 8,
        on_status=None,
        **kwargs,
    ) -> str:
        """Generate video from image via Runway API."""
        return self._run_image_to_video(
            image_path=image_path,
            prompt=prompt,
            output_path=output_path,
            duration=duration_seconds,
            on_status=on_status,
        )

    def generate_video_transition(
        self,
        start_frame_path: str,
        end_frame_path: str,
        prompt: str,
        output_path: str,
        duration_seconds: int = 8,
        on_status=None,
        **kwargs,
    ) -> str:
        """Generate transition video via Runway API with first+last frame conditioning."""
        return self._run_image_to_video(
            image_path=start_frame_path,
            prompt=prompt,
            output_path=output_path,
            duration=max(4, min(8, duration_seconds)),
            on_status=on_status,
            last_frame_path=end_frame_path,
        )

    def _run_image_to_video(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        duration: int = 8,
        on_status=None,
        last_frame_path: str | None = None,
    ) -> str:
        """Core Runway image-to-video API call with polling."""
        import base64
        import json
        import urllib.request
        import urllib.error

        def _to_data_uri(path):
            ext = Path(path).suffix.lower()
            mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "image/png")
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:{mime};base64,{b64}"

        # Build promptImage — single URI or array of keyframe objects
        if last_frame_path and Path(last_frame_path).exists():
            prompt_image = [
                {"uri": _to_data_uri(image_path), "position": "first"},
                {"uri": _to_data_uri(last_frame_path), "position": "last"},
            ]
        else:
            prompt_image = _to_data_uri(image_path)

        # Submit generation request
        payload = json.dumps({
            "model": self.model,
            "promptImage": prompt_image,
            "promptText": prompt,
            "ratio": "1920:1080",
            "duration": duration,
        }).encode()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-Runway-Version": "2024-11-06",
        }

        _log(f"    [runway] Submitting {self.model} generation ({duration}s)...")
        req = urllib.request.Request(
            "https://api.dev.runwayml.com/v1/image_to_video",
            data=payload,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            raise RuntimeError(f"Runway API error {e.code}: {body}")

        task_id = result.get("id")
        if not task_id:
            raise RuntimeError(f"Runway returned no task ID: {result}")

        _log(f"    [runway] Task {task_id} submitted, polling...")

        # Poll for completion
        poll_url = f"https://api.dev.runwayml.com/v1/tasks/{task_id}"
        poll_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Runway-Version": "2024-11-06",
        }

        while True:
            time.sleep(10)
            poll_req = urllib.request.Request(poll_url, headers=poll_headers)
            try:
                with urllib.request.urlopen(poll_req) as resp:
                    task = json.loads(resp.read())
            except urllib.error.HTTPError as e:
                _log(f"    [runway] Poll error {e.code}, retrying...")
                continue

            status = task.get("status", "")
            if on_status:
                on_status(f"Runway: {status}")

            if status == "SUCCEEDED":
                output_url = None
                output_list = task.get("output", [])
                if isinstance(output_list, list) and output_list:
                    output_url = output_list[0]
                elif isinstance(task.get("output"), str):
                    output_url = task["output"]

                if not output_url:
                    raise RuntimeError(f"Runway task succeeded but no output URL: {task}")

                _log(f"    [runway] Downloading {output_url[:80]}...")
                urllib.request.urlretrieve(output_url, output_path)
                _log(f"    [runway] Saved to {output_path}")
                return output_path

            elif status == "FAILED":
                error = task.get("failure", task.get("error", "unknown"))
                raise RuntimeError(f"Runway generation failed: {error}")

            elif status in ("PENDING", "RUNNING", "THROTTLED"):
                progress = task.get("progress", 0)
                _log(f"    [runway] {status} ({progress:.0%})...")
            else:
                _log(f"    [runway] Unknown status: {status}")
