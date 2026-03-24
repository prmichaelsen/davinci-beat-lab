"""Crossfade concatenation for video segments using ffmpeg xfade filter."""

from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def _get_duration(seg_path: str) -> float:
    """Get duration of a video file via ffprobe."""
    p = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", seg_path],
        capture_output=True, text=True,
    )
    try:
        return float(p.stdout.strip())
    except ValueError:
        return 8.0


def _xfade_group(
    segment_paths: list[str],
    output_path: str,
    xfade_duration: float,
) -> bool:
    """Crossfade a small group of segments. Returns True on success."""
    if len(segment_paths) == 1:
        shutil.copy2(segment_paths[0], output_path)
        return True

    seg_durations = [_get_duration(p) for p in segment_paths]

    inputs = []
    for seg_path in segment_paths:
        inputs.extend(["-i", str(Path(seg_path).resolve())])

    n = len(segment_paths)
    if n == 2:
        offset = max(0, seg_durations[0] - xfade_duration)
        filter_str = f"[0:v][1:v]xfade=transition=fade:duration={xfade_duration:.4f}:offset={offset:.4f}[v]"
    else:
        filters = []
        prev = "[0:v]"
        cumulative_duration = seg_durations[0]
        for j in range(1, n):
            offset = max(0, cumulative_duration - xfade_duration)
            out = f"[v{j}]" if j < n - 1 else "[v]"
            filters.append(f"{prev}[{j}:v]xfade=transition=fade:duration={xfade_duration:.4f}:offset={offset:.4f}{out}")
            prev = out
            cumulative_duration += seg_durations[j] - xfade_duration
        filter_str = ";".join(filters)

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_str,
        "-map", "[v]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def concat_with_crossfade(
    segment_paths: list[str],
    output_path: str,
    crossfade_frames: int = 8,
    fps: float = 24.0,
    chunk_size: int = 10,
) -> str:
    """Concatenate video segments with crossfade transitions.

    For large numbers of segments (>chunk_size), processes in chunks:
    1. Crossfade each chunk of ~chunk_size segments
    2. Crossfade the chunks together

    Args:
        segment_paths: List of video file paths to concatenate.
        output_path: Where to write the output.
        crossfade_frames: Number of frames for each crossfade transition.
        fps: Frame rate (used to convert frames to seconds).
        chunk_size: Max segments per ffmpeg xfade call (default: 10).

    Returns:
        output_path
    """
    if not segment_paths:
        raise ValueError("No segments to concatenate")

    if len(segment_paths) == 1:
        shutil.copy2(segment_paths[0], output_path)
        return output_path

    xfade_duration = crossfade_frames / fps

    # Small enough to do in one pass
    if len(segment_paths) <= chunk_size:
        if _xfade_group(segment_paths, output_path, xfade_duration):
            return output_path
        _log(f"  xfade failed for {len(segment_paths)} segments, falling back to hard concat")
        _hard_concat(segment_paths, output_path)
        return output_path

    # Chunked processing
    _log(f"  Chunked crossfade: {len(segment_paths)} segments in chunks of {chunk_size}")

    # Create temp dir for chunks
    out_dir = Path(output_path).parent
    chunk_dir = out_dir / "_xfade_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Split into chunks with 1-segment overlap for seamless crossfading
    # Chunk 0: segments 0..9
    # Chunk 1: segments 9..18  (segment 9 overlaps — crossfade handles it)
    # Chunk 2: segments 18..27
    chunk_paths = []
    step = chunk_size - 1  # overlap by 1

    i = 0
    chunk_idx = 0
    while i < len(segment_paths):
        end = min(i + chunk_size, len(segment_paths))
        chunk = segment_paths[i:end]
        chunk_path = str(chunk_dir / f"chunk_{chunk_idx:03d}.mp4")

        if not Path(chunk_path).exists():
            _log(f"  Chunk {chunk_idx}: segments {i}-{end-1} ({len(chunk)} segments)")
            if not _xfade_group(chunk, chunk_path, xfade_duration):
                _log(f"  Chunk {chunk_idx} xfade failed, using hard concat")
                _hard_concat(chunk, chunk_path)

        chunk_paths.append(chunk_path)
        chunk_idx += 1
        i += step

        # If we'd start a chunk with only 1 segment left, include it in the last chunk
        if i >= len(segment_paths) - 1:
            break

    # Now crossfade the chunks together
    if len(chunk_paths) <= chunk_size:
        _log(f"  Final pass: crossfading {len(chunk_paths)} chunks")
        if not _xfade_group(chunk_paths, output_path, xfade_duration):
            _log(f"  Final xfade failed, using hard concat")
            _hard_concat(chunk_paths, output_path)
    else:
        # Recursive chunking (unlikely but handles huge videos)
        _log(f"  Recursive chunking: {len(chunk_paths)} chunks")
        concat_with_crossfade(chunk_paths, output_path, crossfade_frames, fps, chunk_size)

    # Clean up chunk files
    for cp in chunk_paths:
        Path(cp).unlink(missing_ok=True)
    chunk_dir.rmdir()

    return output_path


def burn_section_labels(
    segment_paths: list[str],
    section_indices: list[int],
    output_dir: str,
) -> list[str]:
    """Burn section number overlay onto each segment clip.

    Args:
        segment_paths: List of video segment paths.
        section_indices: Corresponding section index for each segment.
        output_dir: Where to write labeled clips.

    Returns:
        List of paths to labeled clips.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    labeled = []

    for seg_path, idx in zip(segment_paths, section_indices):
        out_path = str(Path(output_dir) / f"labeled_{idx:03d}.mp4")

        # Skip if already labeled and source hasn't changed
        if Path(out_path).exists() and Path(out_path).stat().st_mtime >= Path(seg_path).stat().st_mtime:
            labeled.append(out_path)
            continue

        text = f"Section {idx}"
        drawtext = (
            f"drawtext=text='{text}'"
            f":fontsize=24"
            f":fontcolor=white"
            f":borderw=2"
            f":bordercolor=black"
            f":x=w-tw-10"
            f":y=h-th-10"
        )
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", seg_path,
                "-vf", drawtext,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                out_path,
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            _log(f"  Label burn failed for section {idx}: {result.stderr[-200:]}")
            labeled.append(seg_path)
        else:
            labeled.append(out_path)

    return labeled


def _hard_concat(segment_paths: list[str], output_path: str) -> None:
    """Fallback: simple concat without crossfade."""
    concat_list = output_path + ".concat.txt"
    with open(concat_list, "w") as f:
        for seg_path in segment_paths:
            f.write(f"file '{Path(seg_path).resolve()}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path],
        check=True, capture_output=True,
    )
    Path(concat_list).unlink(missing_ok=True)
