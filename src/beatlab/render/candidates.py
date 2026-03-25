"""Image candidate generation and selection for iterative refinement."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable


def generate_image_candidates(
    section_idx: int | str,
    source_image_path: str,
    style_prompt: str,
    count: int,
    work_dir: str,
    stylize_fn: Callable[[str, str, str], str],
    base_seed: int = 42,
) -> list[str]:
    """Generate N candidate styled images for a section.

    Args:
        section_idx: Section index or file key (e.g. 42 or "042_001").
        source_image_path: Path to the source keyframe image.
        style_prompt: Style prompt for Nano Banana.
        count: Number of candidates to generate.
        work_dir: Work directory path.
        stylize_fn: Function(source_path, style_prompt, output_path) -> output_path.
        base_seed: Starting seed (each candidate gets base_seed + i).

    Returns:
        List of paths to candidate images.
    """
    key = f"{section_idx:03d}" if isinstance(section_idx, int) else str(section_idx)
    cand_dir = Path(work_dir) / "candidates" / f"section_{key}"
    cand_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i in range(count):
        out_path = str(cand_dir / f"v{i+1}.png")
        if Path(out_path).exists():
            paths.append(out_path)
            continue

        # Slight prompt variation for diversity (append seed hint)
        varied_prompt = f"{style_prompt}, variation {i+1}" if i > 0 else style_prompt
        stylize_fn(source_image_path, varied_prompt, out_path)
        paths.append(out_path)

    return paths


def make_contact_sheet(
    image_paths: list[str],
    output_path: str,
    section_idx: int,
    labels: list[str] | None = None,
) -> str:
    """Create a 2x2 (or NxM) contact sheet from candidate images.

    Args:
        image_paths: List of candidate image paths.
        output_path: Where to save the contact sheet.
        section_idx: Section index (for title).
        labels: Optional labels per image (default: v1, v2, ...).

    Returns:
        output_path
    """
    if not labels:
        labels = [f"v{i+1}" for i in range(len(image_paths))]

    try:
        from PIL import Image, ImageDraw, ImageFont
        _make_contact_sheet_pil(image_paths, output_path, section_idx, labels)
    except ImportError:
        # Fallback: use ffmpeg
        _make_contact_sheet_ffmpeg(image_paths, output_path, section_idx, labels)

    return output_path


def _make_contact_sheet_pil(
    image_paths: list[str],
    output_path: str,
    section_idx: int,
    labels: list[str],
) -> None:
    """Create contact sheet using PIL."""
    from PIL import Image, ImageDraw, ImageFont

    images = [Image.open(p) for p in image_paths]
    n = len(images)

    # Grid layout
    cols = 2 if n <= 4 else 3
    rows = (n + cols - 1) // cols

    # Resize all to same size
    w, h = images[0].size
    label_height = 40
    padding = 4

    sheet_w = cols * w + (cols + 1) * padding
    sheet_h = rows * (h + label_height) + (rows + 1) * padding + 50  # 50 for title

    sheet = Image.new("RGB", (sheet_w, sheet_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)

    # Title
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except (OSError, IOError):
        font_title = ImageFont.load_default()
        font_label = font_title

    draw.text((padding, 10), f"Section {section_idx} — Select a variant", fill="white", font=font_title)

    for i, (img, label) in enumerate(zip(images, labels)):
        col = i % cols
        row = i // cols
        x = padding + col * (w + padding)
        y = 50 + padding + row * (h + label_height + padding)

        # Paste image
        if img.size != (w, h):
            img = img.resize((w, h))
        sheet.paste(img, (x, y))

        # Draw label
        draw.text((x + 10, y + h + 5), label, fill="white", font=font_label)

        # Draw border
        draw.rectangle([x - 1, y - 1, x + w, y + h], outline="white", width=2)

    sheet.save(output_path)


def _make_contact_sheet_ffmpeg(
    image_paths: list[str],
    output_path: str,
    section_idx: int,
    labels: list[str],
) -> None:
    """Fallback contact sheet using ffmpeg tile filter."""
    n = len(image_paths)
    cols = 2 if n <= 4 else 3
    rows = (n + cols - 1) // cols

    inputs = []
    for p in image_paths:
        inputs.extend(["-i", p])

    # Pad to fill grid if needed
    filter_parts = []
    for i in range(n):
        filter_parts.append(f"[{i}:v]drawtext=text='{labels[i]}':fontsize=24:fontcolor=white:x=10:y=h-30[l{i}]")

    labeled = ";".join(filter_parts)
    tile_inputs = "".join(f"[l{i}]" for i in range(n))
    filter_str = f"{labeled};{tile_inputs}xstack=inputs={n}:layout={'|'.join(_tile_layout(n, cols))}[v]"

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_str,
        "-map", "[v]",
        "-frames:v", "1",
        output_path,
    ]

    subprocess.run(cmd, capture_output=True)


def _tile_layout(n: int, cols: int) -> list[str]:
    """Generate xstack layout string for N items in a grid."""
    layouts = []
    for i in range(n):
        col = i % cols
        row = i // cols
        x = f"{col}*w0" if col > 0 else "0"
        y = f"{row}*h0" if row > 0 else "0"
        layouts.append(f"{x}_{y}")
    return layouts


def apply_selection(
    section_idx: int | str,
    variant: int,
    work_dir: str,
) -> list[str]:
    """Apply a candidate selection — copy selected variant to styled image, delete stale outputs.

    Args:
        section_idx: Section index (int) or file key (str like "042_001").
        variant: Variant number (1-indexed: v1, v2, v3, v4).
        work_dir: Work directory path.

    Returns:
        List of stale files that were deleted.
    """
    work = Path(work_dir)
    key = f"{section_idx:03d}" if isinstance(section_idx, int) else str(section_idx)

    cand_path = work / "candidates" / f"section_{key}" / f"v{variant}.png"

    if not cand_path.exists():
        raise FileNotFoundError(f"Candidate v{variant} not found for section {key}: {cand_path}")

    # Copy to styled image location
    styled_path = work / "google_styled" / f"styled_{key}.png"
    shutil.copy2(str(cand_path), str(styled_path))

    # Delete stale downstream files — use glob to match file key patterns
    stale = []
    for pattern in [
        f"google_segments/segment_*_{key}.mp4",
        f"google_segments/segment_{key}_*.mp4",
        f"google_remapped/remapped_{key}.mp4",
        f"google_labeled/labeled_{key}.mp4",
        "google_concat.mp4",
        "google_muxed.mp4",
        "google_output.mp4",
    ]:
        for p in work.glob(pattern):
            p.unlink()
            stale.append(str(p))

    return stale


def apply_cross_selection(
    target_idx: int | str,
    source_idx: int | str,
    variant: int,
    work_dir: str,
) -> list[str]:
    """Apply a cross-section selection — use a candidate from one section for another.

    Args:
        target_idx: Target section index or file key (where the image will be used).
        source_idx: Source section index or file key (where to get the candidate from).
        variant: Variant number from the source section (1-indexed).
        work_dir: Work directory path.

    Returns:
        List of stale files that were deleted.
    """
    work = Path(work_dir)
    source_key = f"{source_idx:03d}" if isinstance(source_idx, int) else str(source_idx)
    target_key = f"{target_idx:03d}" if isinstance(target_idx, int) else str(target_idx)

    cand_path = work / "candidates" / f"section_{source_key}" / f"v{variant}.png"

    if not cand_path.exists():
        raise FileNotFoundError(
            f"Candidate v{variant} not found in section {source_key}: {cand_path}"
        )

    # Copy to target's styled image location
    styled_path = work / "google_styled" / f"styled_{target_key}.png"
    shutil.copy2(str(cand_path), str(styled_path))

    # Also copy into target's candidate dir as v1 for reference
    target_cand_dir = work / "candidates" / f"section_{target_key}"
    target_cand_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(cand_path), str(target_cand_dir / f"v1_from_{source_key}_v{variant}.png"))

    # Delete stale downstream files for target section
    stale = []
    for pattern in [
        f"google_segments/segment_*_{target_key}.mp4",
        f"google_segments/segment_{target_key}_*.mp4",
        f"google_remapped/remapped_{target_key}.mp4",
        f"google_labeled/labeled_{target_key}.mp4",
        "google_concat.mp4",
        "google_muxed.mp4",
        "google_output.mp4",
    ]:
        for p in work.glob(pattern):
            p.unlink()
            stale.append(str(p))

    return stale
