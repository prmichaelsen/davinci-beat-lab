"""Use Claude to describe creative transitions between two images."""

from __future__ import annotations

import base64
import sys
from datetime import datetime
from pathlib import Path


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def describe_transition(
    start_image_path: str,
    end_image_path: str,
    style_context: str = "",
    motion_prompt: str = "",
) -> str:
    """Use Claude to describe a creative cinematic transition between two images.

    Args:
        start_image_path: Path to the start frame image.
        end_image_path: Path to the end frame image.
        style_context: Optional style description for context.
        motion_prompt: Optional motion/camera direction.

    Returns:
        A Veo-optimized transition prompt (100-200 words).
    """
    import anthropic

    client = anthropic.Anthropic()

    # Read images as base64
    start_data = base64.standard_b64encode(Path(start_image_path).read_bytes()).decode()
    end_data = base64.standard_b64encode(Path(end_image_path).read_bytes()).decode()

    start_ext = Path(start_image_path).suffix.lower()
    end_ext = Path(end_image_path).suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

    system = """You are a cinematic transition director for AI video generation (Google Veo).
Given two frames (start and end), describe a smooth, creative 8-second transition between them.

Rules:
- Describe the JOURNEY between the frames, not the endpoints (Veo already has the images)
- Use cinematic camera terminology: steadicam, dolly, crane, tracking shot
- Describe material transformations: "melting", "crystallizing", "dissolving into liquid glass"
- Include timestamp beats: "0-3s: ..., 3-6s: ..., 6-8s: ..."
- Add negative constraints: "no cuts, no object popping, no sudden scene shifts"
- Keep it 100-150 words — specific and vivid
- Focus on what CHANGES between the two images: colors, textures, shapes, composition
- Describe transitions as organic/dreamlike/fluid, not mechanical
- Output ONLY the transition prompt, no explanation"""

    user_parts = [
        {"type": "text", "text": "START FRAME:"},
        {"type": "image", "source": {"type": "base64", "media_type": mime_map.get(start_ext, "image/png"), "data": start_data}},
        {"type": "text", "text": "END FRAME:"},
        {"type": "image", "source": {"type": "base64", "media_type": mime_map.get(end_ext, "image/png"), "data": end_data}},
    ]

    context_parts = []
    if style_context:
        context_parts.append(f"Visual style context: {style_context}")
    if motion_prompt:
        context_parts.append(f"Camera direction: {motion_prompt}")
    context_parts.append("Describe the 8-second cinematic transition between these two frames.")

    user_parts.append({"type": "text", "text": " ".join(context_parts)})

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system=system,
        messages=[{"role": "user", "content": user_parts}],
    )

    return response.content[0].text.strip()


def describe_transitions_batch(
    image_pairs: list[tuple[str, str]],
    style_contexts: list[str] | None = None,
    motion_prompt: str = "",
    max_workers: int = 5,
) -> list[str]:
    """Describe transitions for multiple image pairs in parallel.

    Args:
        image_pairs: List of (start_path, end_path) tuples.
        style_contexts: Optional per-pair style context.
        motion_prompt: Shared motion/camera direction.
        max_workers: Parallel workers for Claude calls.

    Returns:
        List of transition prompts in order.
    """
    import concurrent.futures

    results = [None] * len(image_pairs)

    def _describe(idx):
        start, end = image_pairs[idx]
        ctx = style_contexts[idx] if style_contexts and idx < len(style_contexts) else ""
        prompt = describe_transition(start, end, style_context=ctx, motion_prompt=motion_prompt)
        results[idx] = prompt
        _log(f"    Transition {idx+1}/{len(image_pairs)}: described ({len(prompt)} chars)")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_describe, i) for i in range(len(image_pairs))]
        for f in concurrent.futures.as_completed(futures):
            f.result()  # raise any exceptions

    return results
