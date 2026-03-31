#!/usr/bin/env python3
"""Benchmark stem separation models on a test clip.

Compares Demucs htdemucs, Demucs htdemucs_ft, and audio-separator (BS Roformer/MDX23)
on the same audio clip. Measures separation quality by checking bleed at a known
vocals-only section.

Usage:
    python scripts/benchmark_separation.py .beatlab_work/beyond_the_veil/test_clips/mix_9_11m.wav
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import librosa
import numpy as np


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def measure_bleed(stem_dir: str, vocals_only_start: float = 56.0, vocals_only_end: float = 60.0, sr: int = 22050):
    """Measure bleed by checking non-vocal stem energy during a known vocals-only section.

    Lower energy in drums/bass/other during the vocal section = better separation.
    Returns dict of {stem: rms_energy_in_vocal_section}.
    """
    results = {}
    for stem_file in sorted(Path(stem_dir).glob("*.wav")):
        stem_name = stem_file.stem
        y, _ = librosa.load(str(stem_file), sr=sr, mono=True)
        start_sample = int(vocals_only_start * sr)
        end_sample = int(vocals_only_end * sr)
        segment = y[start_sample:end_sample]
        if len(segment) == 0:
            results[stem_name] = 0.0
            continue
        rms = float(np.sqrt(np.mean(segment ** 2)))
        results[stem_name] = rms
    return results


def run_demucs(audio_path: str, output_dir: str, model: str = "htdemucs"):
    """Run Demucs separation."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _log(f"  Running Demucs ({model})...")
    start = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "demucs", "-n", model, "-d", "cpu", "-o", str(out / "raw"), audio_path],
        capture_output=True, text=True,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        _log(f"  Demucs {model} failed: {result.stderr[-200:]}")
        return None, elapsed

    # Move stems to output dir
    audio_stem = Path(audio_path).stem
    raw_dir = out / "raw" / model / audio_stem
    for f in raw_dir.glob("*.wav"):
        dest = out / f.name
        if not dest.exists():
            f.rename(dest)

    return str(out), elapsed


def run_audio_separator(audio_path: str, output_dir: str, model: str = "UVR-MDX-NET-Inst_HQ_3"):
    """Run audio-separator with a given model."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _log(f"  Running audio-separator ({model})...")
    start = time.time()
    try:
        from audio_separator.separator import Separator
        separator = Separator(output_dir=str(out), model_file_dir="/tmp/audio-separator-models/")
        separator.load_model(model)
        stems = separator.separate(audio_path)
        elapsed = time.time() - start
        _log(f"  Produced stems: {stems}")
        return str(out), elapsed
    except Exception as e:
        elapsed = time.time() - start
        _log(f"  audio-separator ({model}) failed: {e}")
        return None, elapsed


def run_bs_roformer(audio_path: str, output_dir: str):
    """Run BS Roformer via bs-roformer-infer."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _log(f"  Running BS Roformer...")
    start = time.time()
    try:
        from bs_roformer_infer import BsRoformer
        separator = BsRoformer()
        result = separator.separate(audio_path, output_dir=str(out))
        elapsed = time.time() - start
        return str(out), elapsed
    except ImportError:
        _log("  bs-roformer-infer not installed, skipping")
        return None, 0
    except Exception as e:
        elapsed = time.time() - start
        _log(f"  BS Roformer failed: {e}")
        return None, elapsed


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/benchmark_separation.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    base_output = Path(audio_path).parent / "separation_benchmark"
    base_output.mkdir(exist_ok=True)

    # Known vocals-only section in the 9-11m clip
    # 10:58 in full track, clip starts at 9:00, so 1:58 into clip = 118s
    vocal_start = 118.0
    vocal_end = 120.0

    _log(f"Benchmarking stem separation on: {audio_path}")
    _log(f"Vocals-only section: {vocal_start}s - {vocal_end}s")
    _log("")

    models = []

    # Skip Demucs — focusing on audio-separator models

    # audio-separator models via CLI
    as_models = [
        ("model_bs_roformer_ep_317_sdr_12.9755.ckpt", "BS-Roformer-1297"),
        ("mel_band_roformer_kim_ft2_bleedless_unwa.ckpt", "MelBand-Bleedless"),
        ("vocals_mel_band_roformer.ckpt", "MelBand-Vocals-KJ"),
        ("MDX23C-8KFFT-InstVoc_HQ.ckpt", "MDX23C-InstVoc-HQ"),
    ]
    for model_file, label in as_models:
        try:
            out_dir, elapsed = run_audio_separator(audio_path, str(base_output / label.lower()), model_file)
            if out_dir:
                bleed = measure_bleed(out_dir, vocal_start, vocal_end)
                models.append({"name": label, "time": elapsed, "bleed": bleed, "dir": out_dir})
                _log(f"  Done in {elapsed:.0f}s")
        except Exception as e:
            _log(f"  {label} crashed: {e}")

    # Report
    _log("")
    _log("=" * 70)
    _log("RESULTS — Lower bleed in non-vocal stems = better separation")
    _log("=" * 70)
    _log(f"{'Model':<30} {'Time':>6} {'vocals':>10} {'drums':>10} {'bass':>10} {'other':>10}")
    _log("-" * 70)

    for m in models:
        bleed = m["bleed"]
        stems = sorted(bleed.keys())
        vocal_rms = bleed.get("vocals", 0)
        # Non-vocal bleed — lower is better
        non_vocal = {k: v for k, v in bleed.items() if k != "vocals"}
        row = f"{m['name']:<30} {m['time']:>5.0f}s"
        row += f" {vocal_rms:>10.6f}"
        for stem in ["drums", "bass", "other"]:
            val = bleed.get(stem, bleed.get("no_vocals", bleed.get("instrumental", 0)))
            row += f" {val:>10.6f}"
        _log(row)

    _log("")
    _log("Vocal RMS = how much vocal content is preserved (higher = better)")
    _log("Drums/Bass/Other RMS = bleed during vocal-only section (lower = better)")

    # Save results
    results_path = str(base_output / "results.json")
    with open(results_path, "w") as f:
        json.dump([{k: v for k, v in m.items() if k != "dir"} for m in models], f, indent=2)
    _log(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
