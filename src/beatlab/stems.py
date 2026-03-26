"""Audio stem separation via Demucs and per-stem analysis."""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path


STEM_NAMES = ("drums", "bass", "vocals", "other")

DEMUCS_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def separate_stems_remote(
    audio_path: str,
    output_dir: str,
    vast_manager: object,
) -> dict[str, str]:
    """Run Demucs on a Vast.ai GPU instance to separate audio into stems.

    Args:
        audio_path: Path to input audio file (WAV).
        output_dir: Local directory to store output stems.
        vast_manager: VastAIManager instance.

    Returns:
        Dict mapping stem name to local file path, e.g. {"drums": "/path/drums.wav", ...}.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Check cache
    expected = {s: str(out / f"{s}.wav") for s in STEM_NAMES}
    if all(Path(p).exists() for p in expected.values()):
        _log("Stems: using cached")
        return expected

    _log("Stems: separating audio via Demucs on Vast.ai...")

    # Get or create a stems-specific instance (cheap, 8GB VRAM)
    instance_id, reused = vast_manager.get_or_create_instance(
        instance_key="stems",
        image=DEMUCS_IMAGE,
        min_vram_gb=8,
        max_price_hr=2.0,
        disk_gb=30,
    )

    if not reused:
        _log(f"  Waiting for instance {instance_id} to be ready...")
        vast_manager.wait_until_ready(instance_id)

    _log(f"  Instance {instance_id} ready (reused={reused})")

    # Wait for SSH to be reachable
    import time as _time
    host, port = vast_manager.get_ssh_info(instance_id)
    for attempt in range(12):
        try:
            vast_manager.ssh_run(instance_id, "echo ok", timeout=15)
            break
        except Exception:
            _log(f"  Waiting for SSH... (attempt {attempt + 1}/12)")
            _time.sleep(10)
    else:
        raise RuntimeError(f"SSH not reachable on instance {instance_id} after 2 minutes")

    # Install demucs + deps — reinstall torch matching the GPU's CUDA arch
    _log("  Installing demucs + dependencies...")
    vast_manager.ssh_run(
        instance_id,
        "apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1;"
        " pip install -q demucs soundfile lameenc 2>/dev/null || pip install demucs soundfile lameenc",
        timeout=600,
    )

    # Upload audio file — stage it in a temp dir so upload_files works (it syncs dirs)
    import tempfile
    remote_work = "/workspace/stems_work"
    audio_name = Path(audio_path).name

    with tempfile.TemporaryDirectory() as staging:
        shutil.copy2(audio_path, Path(staging) / audio_name)
        _log(f"  Uploading {audio_name}...")
        vast_manager.upload_files(instance_id, staging, remote_work)

    # Run demucs — use subprocess directly to capture stderr
    # Use --mp3 to avoid torchcodec issues, convert back to wav after download
    _log("  Running Demucs (htdemucs model)...")
    host, port = vast_manager.get_ssh_info(instance_id)
    ssh_opts = vast_manager._ssh_opts(port)
    demucs_cmd = f"cd {remote_work} && python -m demucs -n htdemucs -d cpu --mp3 --mp3-bitrate 320 -o output {audio_name} 2>&1"
    import subprocess
    demucs_result = subprocess.run(
        f'{ssh_opts} root@{host} "{demucs_cmd}"',
        shell=True, capture_output=True, text=True, timeout=1800,
    )
    _log(f"  Demucs stdout: {demucs_result.stdout[-500:] if demucs_result.stdout else '(empty)'}")
    if demucs_result.stderr:
        _log(f"  Demucs stderr: {demucs_result.stderr[-500:]}")

    # Verify stems exist on remote before downloading
    audio_stem = Path(audio_name).stem
    remote_stems = f"{remote_work}/output/htdemucs/{audio_stem}"
    ls_result = vast_manager.ssh_run(instance_id, f"ls -la {remote_stems}/", timeout=15)
    _log(f"  Remote stems: {ls_result.strip() if ls_result else '(empty dir)'}")
    if not ls_result or "drums" not in ls_result:
        raise RuntimeError(
            f"Demucs did not produce stems. Remote dir contents:\n{ls_result}\n"
            f"Demucs output:\n{demucs_result.stdout[-1000:]}"
        )

    # Download stems (may be .mp3 if --mp3 was used)
    _log("  Downloading stems...")
    vast_manager.download_files(instance_id, remote_stems, str(out))

    # Convert mp3 stems to wav if needed
    for stem in STEM_NAMES:
        wav_path = out / f"{stem}.wav"
        mp3_path = out / f"{stem}.mp3"
        if wav_path.exists():
            continue
        if mp3_path.exists():
            _log(f"  Converting {stem}.mp3 → wav...")
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(mp3_path), "-acodec", "pcm_s16le",
                 "-ar", "44100", "-ac", "2", str(wav_path)],
                check=True, capture_output=True,
            )
        else:
            raise RuntimeError(f"Failed to download stem: {stem} (no .wav or .mp3 found)")

    _log("  Stem separation complete")
    return expected


def separate_stems_local(
    audio_path: str,
    output_dir: str,
) -> dict[str, str]:
    """Run Demucs locally (CPU — slow for long files, use for testing only).

    Args:
        audio_path: Path to input audio file.
        output_dir: Local directory to store output stems.

    Returns:
        Dict mapping stem name to local file path.
    """
    import subprocess

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    expected = {s: str(out / f"{s}.wav") for s in STEM_NAMES}
    if all(Path(p).exists() for p in expected.values()):
        _log("Stems: using cached")
        return expected

    _log("Stems: separating audio via Demucs (local CPU — this will be slow)...")
    result = subprocess.run(
        ["python", "-m", "demucs", "-n", "htdemucs", "--two-stems=None",
         "-o", str(out / "demucs_output"), audio_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {result.stderr[-500:]}")

    # Move stems to expected locations
    audio_stem = Path(audio_path).stem
    demucs_dir = out / "demucs_output" / "htdemucs" / audio_stem
    for stem in STEM_NAMES:
        src = demucs_dir / f"{stem}.wav"
        dst = out / f"{stem}.wav"
        if src.exists():
            shutil.move(str(src), str(dst))
        else:
            raise RuntimeError(f"Demucs did not produce {stem}.wav")

    _log("  Stem separation complete")
    return expected


def _detect_onsets(y, sr_out, hop_length=512) -> list[dict]:
    """Detect onsets using librosa directly (no beat_this)."""
    import librosa
    import numpy as np

    onset_env = librosa.onset.onset_strength(y=y, sr=sr_out, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr_out, hop_length=hop_length, onset_envelope=onset_env, backtrack=True,
    )
    max_idx = len(onset_env) - 1
    onset_frames = np.clip(onset_frames, 0, max_idx) if len(onset_frames) > 0 else onset_frames
    onset_times = librosa.frames_to_time(onset_frames, sr=sr_out, hop_length=hop_length)
    onset_strengths = onset_env[onset_frames] if len(onset_frames) > 0 else np.array([])
    if len(onset_strengths) > 0 and onset_strengths.max() > 0:
        onset_strengths = onset_strengths / onset_strengths.max()

    return [
        {"time": float(t), "strength": float(s)}
        for t, s in zip(onset_times, onset_strengths)
    ]


def analyze_stem(path: str, stem_type: str, sr: int = 22050) -> dict:
    """Analyze a single audio stem with strategy appropriate to its type.

    Args:
        path: Path to stem WAV file.
        stem_type: One of "drums", "bass", "vocals", "other".
        sr: Sample rate for analysis.

    Returns:
        Analysis dict with keys appropriate to the stem type.
    """
    from beatlab.analyzer import load_audio, detect_drops, detect_presence, detect_sections
    import librosa
    import numpy as np

    if stem_type == "drums":
        # Onset detection only — no beat_this grid snapping.
        # Isolated drum track gives clean onsets that are the actual hits.
        y, sr_out = load_audio(path, sr=sr)
        hop_length = 512
        onset_env = librosa.onset.onset_strength(y=y, sr=sr_out, hop_length=hop_length)
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr_out, hop_length=hop_length, onset_envelope=onset_env, backtrack=True,
        )
        max_idx = len(onset_env) - 1
        onset_frames = np.clip(onset_frames, 0, max_idx) if len(onset_frames) > 0 else onset_frames
        onset_times = librosa.frames_to_time(onset_frames, sr=sr_out, hop_length=hop_length)
        onset_strengths = onset_env[onset_frames] if len(onset_frames) > 0 else np.array([])
        if len(onset_strengths) > 0 and onset_strengths.max() > 0:
            onset_strengths = onset_strengths / onset_strengths.max()

        onsets = [
            {"time": float(t), "strength": float(s)}
            for t, s in zip(onset_times, onset_strengths)
        ]

        # Estimate tempo from onset intervals
        if len(onset_times) >= 2:
            intervals = np.diff(onset_times)
            median_interval = float(np.median(intervals))
            tempo = 60.0 / median_interval if median_interval > 0 else 120.0
        else:
            tempo = 120.0

        sections = detect_sections(y, sr_out)

        return {
            "tempo": tempo,
            "onsets": onsets,
            "sections": sections,
        }

    elif stem_type == "bass":
        # Onsets + drop detection
        y, sr_out = load_audio(path, sr=sr)
        onsets = _detect_onsets(y, sr_out)
        drops = detect_drops(y, sr_out)
        return {
            "onsets": onsets,
            "drops": drops,
        }

    elif stem_type == "vocals":
        # Onsets + presence detection
        y, sr_out = load_audio(path, sr=sr)
        onsets = _detect_onsets(y, sr_out)
        presence = detect_presence(y, sr_out)
        return {
            "onsets": onsets,
            "presence": presence,
        }

    else:  # "other"
        # Onsets only
        y, sr_out = load_audio(path, sr=sr)
        onsets = _detect_onsets(y, sr_out)
        return {
            "onsets": onsets,
        }


def analyze_all_stems(stem_paths: dict[str, str], sr: int = 22050) -> dict:
    """Analyze all stems and return a dict suitable for beat_map enrichment.

    Args:
        stem_paths: Dict mapping stem name to WAV path.
        sr: Sample rate for analysis.

    Returns:
        Dict of {stem_name: analysis_dict}.
    """
    results = {}
    for stem_name, path in stem_paths.items():
        if not Path(path).exists():
            _log(f"  Warning: stem {stem_name} not found at {path}, skipping")
            continue
        _log(f"  Analyzing {stem_name} stem...")
        results[stem_name] = analyze_stem(path, stem_name, sr=sr)
        # Log summary
        analysis = results[stem_name]
        parts = []
        if "beats" in analysis:
            parts.append(f"{len(analysis['beats'])} beats")
        if "onsets" in analysis:
            parts.append(f"{len(analysis['onsets'])} onsets")
        if "drops" in analysis:
            parts.append(f"{len(analysis['drops'])} drops")
        if "presence" in analysis:
            parts.append(f"{len(analysis['presence'])} vocal regions")
        _log(f"    {stem_name}: {', '.join(parts)}")

    return results
