"""Crossfade concatenation for video segments using ffmpeg xfade filter."""

from __future__ import annotations

import math
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
) -> tuple[bool, str]:
    """Crossfade a small group of segments. Returns (success, stderr)."""
    if len(segment_paths) == 1:
        shutil.copy2(segment_paths[0], output_path)
        return True, ""

    seg_durations = [_get_duration(p) for p in segment_paths]

    # Scale all inputs to same resolution and framerate to avoid xfade mismatch
    inputs = []
    scale_filters = []
    for i, seg_path in enumerate(segment_paths):
        inputs.extend(["-i", str(Path(seg_path).resolve())])
        scale_filters.append(
            f"[{i}:v]fps=24,scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[s{i}]"
        )

    n = len(segment_paths)

    # Ensure xfade duration doesn't exceed any segment's duration
    min_dur = min(seg_durations)
    safe_xfade = min(xfade_duration, min_dur * 0.5)  # never more than half the shortest segment

    if n == 2:
        offset = max(0, seg_durations[0] - safe_xfade)
        xfade_str = f"[s0][s1]xfade=transition=fade:duration={safe_xfade:.4f}:offset={offset:.4f}[v]"
    else:
        xfade_parts = []
        prev = "[s0]"
        cumulative_duration = seg_durations[0]
        for j in range(1, n):
            offset = max(0, cumulative_duration - safe_xfade)
            out = f"[v{j}]" if j < n - 1 else "[v]"
            xfade_parts.append(f"{prev}[s{j}]xfade=transition=fade:duration={safe_xfade:.4f}:offset={offset:.4f}{out}")
            prev = out
            cumulative_duration += seg_durations[j] - safe_xfade
        xfade_str = ";".join(xfade_parts)

    filter_str = ";".join(scale_filters) + ";" + xfade_str

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_str,
        "-map", "[v]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr


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
        ok, stderr = _xfade_group(segment_paths, output_path, xfade_duration)
        if ok:
            return output_path
        raise RuntimeError(
            f"Crossfade failed for {len(segment_paths)} segments. "
            f"Segments: {[Path(s).name for s in segment_paths]}\n"
            f"ffmpeg stderr: {stderr[-500:]}"
        )

    # Chunked processing
    import time as _time

    # Calculate total chunks upfront
    step = chunk_size
    total_chunks = math.ceil(len(segment_paths) / chunk_size)
    total_chunks += 1  # final pass to merge chunks

    _log(f"  Chunked crossfade: {len(segment_paths)} segments → {total_chunks - 1} chunks + final merge")

    # Create temp dir for chunks — named after output to avoid collision on recursive calls
    out_dir = Path(output_path).parent
    out_stem = Path(output_path).stem  # e.g. "google_concat"
    chunk_dir = out_dir / f"_xfade_chunks_{out_stem}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths = []
    chunk_times: list[float] = []

    i = 0
    chunk_idx = 0
    chunks_done = 0
    while i < len(segment_paths):
        end = min(i + chunk_size, len(segment_paths))
        chunk = segment_paths[i:end]
        chunk_path = str(chunk_dir / f"chunk_{chunk_idx:03d}.mp4")

        if not Path(chunk_path).exists():
            # Time estimate
            if chunk_times:
                avg_time = sum(chunk_times) / len(chunk_times)
                remaining = (total_chunks - 1 - chunks_done) * avg_time
                eta_min = remaining / 60
                _log(f"  Chunk {chunk_idx + 1}/{total_chunks - 1}: segments {i}-{end-1} ({len(chunk)} segs) — ETA {eta_min:.1f}m")
            else:
                _log(f"  Chunk {chunk_idx + 1}/{total_chunks - 1}: segments {i}-{end-1} ({len(chunk)} segs)")

            chunk_start = _time.time()

            # Validate all segments in chunk before attempting xfade
            for seg in chunk:
                if not Path(seg).exists():
                    raise RuntimeError(f"Crossfade failed: segment missing: {seg}")
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", seg],
                    capture_output=True, text=True,
                )
                if not probe.stdout.strip():
                    raise RuntimeError(f"Crossfade failed: segment corrupt or unreadable: {seg}")

            ok, stderr = _xfade_group(chunk, chunk_path, xfade_duration)
            chunk_elapsed = _time.time() - chunk_start
            chunk_times.append(chunk_elapsed)

            if not ok:
                raise RuntimeError(
                    f"Crossfade failed on chunk {chunk_idx} (segments {i}-{end-1}). "
                    f"Segments: {[Path(s).name for s in chunk]}\n"
                    f"ffmpeg stderr: {stderr[-500:]}"
                )

            _log(f"    Done in {chunk_elapsed:.1f}s")
        else:
            _log(f"  Chunk {chunk_idx + 1}/{total_chunks - 1}: cached")

        chunks_done += 1
        chunk_paths.append(chunk_path)
        chunk_idx += 1
        i += step

    # Iteratively merge chunks until small enough for one pass
    merge_level = 0
    merge_paths = chunk_paths
    while len(merge_paths) > chunk_size:
        merge_level += 1
        _log(f"  Merge level {merge_level}: {len(merge_paths)} chunks → groups of {chunk_size}")
        level_dir = chunk_dir / f"merge_L{merge_level}"
        level_dir.mkdir(parents=True, exist_ok=True)

        next_paths = []
        gi = 0
        for start in range(0, len(merge_paths), chunk_size):
            group = merge_paths[start:start + chunk_size]
            group_path = str(level_dir / f"group_{gi:03d}.mp4")

            if not Path(group_path).exists():
                _log(f"    Group {gi + 1}: merging {len(group)} chunks")
                grp_start = _time.time()
                ok, stderr = _xfade_group(group, group_path, xfade_duration)
                if not ok:
                    raise RuntimeError(
                        f"Merge level {merge_level} group {gi} failed.\n"
                        f"ffmpeg stderr: {stderr[-500:]}"
                    )
                _log(f"      Done in {_time.time() - grp_start:.1f}s")
            else:
                _log(f"    Group {gi + 1}: cached")

            next_paths.append(group_path)
            gi += 1

        merge_paths = next_paths

    # Final pass — small enough for one ffmpeg call
    total_dur = sum(_get_duration(p) for p in merge_paths)
    _log(f"  Final merge: crossfading {len(merge_paths)} chunks ({total_dur:.1f}s total)")
    merge_start = _time.time()
    ok, stderr = _xfade_group(merge_paths, output_path, xfade_duration)
    if not ok:
        raise RuntimeError(
            f"Final crossfade failed on {len(merge_paths)} chunks.\n"
            f"ffmpeg stderr: {stderr[-500:]}"
        )
    elapsed = _time.time() - merge_start
    speed = total_dur / elapsed if elapsed > 0 else 0
    _log(f"    Done in {elapsed:.1f}s ({speed:.1f}x realtime)")

    # Keep chunks cached for reuse — only stale if source segments change

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
