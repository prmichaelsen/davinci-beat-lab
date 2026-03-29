"""Audio description — describes what's happening musically in each section."""

from __future__ import annotations

import io
import re
import tempfile
from abc import ABC, abstractmethod

import numpy as np


class AudioDescriber(ABC):
    """Abstract base class for audio description models."""

    @abstractmethod
    def describe(self, audio: np.ndarray, sr: int) -> str:
        """Describe a segment of audio in natural language.

        Args:
            audio: Audio samples as numpy array.
            sr: Sample rate.

        Returns:
            Text description of the audio content.
        """
        ...


class GeminiAudioDescriber(AudioDescriber):
    """Audio describer using Google Gemini Flash API."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash"):
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "The 'google-genai' package is required for --describe mode.\n"
                "Install with: pip install google-genai"
            )

        import os
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required for --describe mode.\n"
                "Get a key at: https://aistudio.google.com/apikey"
            )

        self.client = genai.Client(api_key=key)
        self.model = model

    def describe(self, audio: np.ndarray, sr: int, max_chunk_seconds: float = 30.0) -> str:
        """Describe audio content using Gemini, chunking long segments.

        Args:
            audio: Audio samples.
            sr: Sample rate.
            max_chunk_seconds: Max seconds per Gemini API call (default 30s).
        """
        duration = len(audio) / sr
        if duration > max_chunk_seconds:
            # Split into chunks, describe each, concatenate
            chunk_size = int(max_chunk_seconds * sr)
            chunks = []
            for start in range(0, len(audio), chunk_size):
                chunks.append(audio[start:start + chunk_size])

            descriptions = []
            for i, chunk in enumerate(chunks):
                chunk_start = i * max_chunk_seconds
                chunk_end = min(chunk_start + max_chunk_seconds, duration)
                desc = self._describe_chunk(chunk, sr, chunk_start, chunk_end)
                descriptions.append(f"**[{chunk_start:.0f}s - {chunk_end:.0f}s]**\n{desc}")
            return "\n\n".join(descriptions)
        else:
            return self._describe_chunk(audio, sr, 0, duration)

    def _describe_chunk(self, audio: np.ndarray, sr: int,
                         chunk_start: float = 0, chunk_end: float = 0) -> str:
        """Describe a single audio chunk (max 30s)."""
        import soundfile as sf
        from google.genai import types

        # Write audio to a WAV buffer
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Content(parts=[
                    types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                    types.Part(text=(
                        f"You are a professional music producer with perfect pitch and rhythm. "
                        f"This audio spans {chunk_start:.0f}s to {chunk_end:.0f}s in the full track. "
                        f"Use ABSOLUTE timestamps (starting at {chunk_start:.0f}s), not relative to this chunk.\n\n"
                        "Produce a DETAILED account of every audible musical event. "
                        "Every second of audio must be accounted for — no gaps.\n\n"
                        "## 1. EVENT LOG (most important — be exhaustive)\n"
                        "List EVERY distinct audible event with its timestamp [M:SS]. Event types: "
                        "kick, snare, hi-hat, cymbal_crash, tom, percussion_other, bass_note, bass_drop, "
                        "bass_sustain_start, bass_sustain_end, synth_stab, synth_pad_start, synth_pad_end, "
                        "synth_lead, arpeggio, riser_start, riser_peak, drop, breakdown_start, buildup_start, "
                        "vocal_start, vocal_end, vocal_chop, fx_sweep, fx_impact, silence_start, silence_end.\n"
                        "For repeating patterns, describe the pattern AND list first few timestamps with interval.\n"
                        "For sustained sounds, give BOTH start and end timestamps.\n\n"
                        "## 2. RHYTHM ANALYSIS\n"
                        "- BPM estimate\n- Time signature\n- Kick/snare/hi-hat patterns\n\n"
                        "## 3. ENERGY PROFILE\n"
                        "Rate intensity 1-10 at: start, 25%, 50%, 75%, end. Note sudden energy changes with timestamps.\n\n"
                        "## 4. SUSTAINED SOUNDS\n"
                        "Every sustained sound with start time, end time, and character (pads, drones, reverb tails, risers, bass).\n\n"
                        "## 5. KEY MOMENTS\n"
                        "The 3-5 most visually impactful moments — timestamps and why they're impactful.\n\n"
                        "## 6. INSTRUMENTS HEARD\n"
                        "Complete list of instruments/sounds present.\n\n"
                        "## 7. MOOD & TEXTURE\n"
                        "Mood, emotional sensation, and production texture.\n\n"
                        "Be EXHAUSTIVE. Every second must be covered. More detail = better visual sync."
                    )),
                ]),
            ],
        )
        return response.text.strip()


class Qwen2AudioDescriber(AudioDescriber):
    """Local audio describer using Qwen2-Audio (requires GPU with 8GB+ VRAM)."""

    def __init__(self, model_name: str = "Qwen/Qwen2-Audio-7B-Instruct", device: str | None = None):
        try:
            from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError(
                "The 'transformers' and 'torch' packages are required for Qwen2 mode.\n"
                "Install with: pip install transformers torch accelerate"
            )

        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        ).to(device)
        self._device = device
        self._sr = self.processor.feature_extractor.sampling_rate

    def describe(self, audio: np.ndarray, sr: int) -> str:
        """Describe audio content using Qwen2-Audio."""
        import librosa
        import torch

        if sr != self._sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self._sr)

        audio = audio.astype(np.float32)

        prompt = (
            "Describe the musical content of this audio in 1-2 sentences. "
            "Focus on: instruments, rhythm, energy level, mood, and any notable "
            "transitions or changes. Be specific and concise."
        )

        audio_inputs = self.processor.feature_extractor(
            [audio], sampling_rate=self._sr, return_tensors="pt"
        )
        text_input = f"<|audio_bos|><|AUDIO|><|audio_eos|>{prompt}"
        text_inputs = self.processor.tokenizer(text_input, return_tensors="pt")

        inputs = {**audio_inputs, **text_inputs}
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=128)

        input_len = inputs["input_ids"].size(1)
        output_ids = output_ids[:, input_len:]
        return self.processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def _offset_timestamps(text: str, offset_seconds: float) -> str:
    """Shift timestamps like [0:05] or [1:23] by offset_seconds to make them track-relative."""
    def _replace(m: re.Match) -> str:
        mins = int(m.group(1))
        secs = int(m.group(2))
        total = mins * 60 + secs + offset_seconds
        new_mins = int(total // 60)
        new_secs = int(total % 60)
        return f"[{new_mins}:{new_secs:02d}]"

    return re.sub(r"\[(\d+):(\d{2})\]", _replace, text)


def describe_sections(
    describer: AudioDescriber,
    y: np.ndarray,
    sr: int,
    sections: list[dict],
    on_progress: callable | None = None,
) -> list[str]:
    """Run audio description on every section — no caps, no skipping.

    Consecutive sections of the same type are merged into one description
    to reduce API calls while still covering all audio.
    The returned list always has one description per original section.

    Args:
        describer: AudioDescriber instance.
        y: Full audio time series.
        sr: Sample rate.
        sections: Section dicts with start_time/end_time/type.
        on_progress: Optional callback(completed, total, group_indices, description).

    Returns:
        List of description strings, one per original section.
    """
    if not sections:
        return []

    # Group consecutive same-type sections into description blocks
    groups: list[list[int]] = []
    current_group = [0]
    for i in range(1, len(sections)):
        if sections[i].get("type") == sections[i - 1].get("type"):
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
    groups.append(current_group)

    # Describe ALL groups — no sampling, no caps
    sampled_indices = set(range(len(groups)))

    total_calls = len(sampled_indices)
    completed = 0

    # Describe each group (using the merged audio span)
    group_descriptions: list[str] = []
    last_desc = "Continuation of previous section."
    for gi, group in enumerate(groups):
        if gi not in sampled_indices:
            group_descriptions.append(last_desc)
            continue

        start_sample = int(sections[group[0]]["start_time"] * sr)
        end_sample = min(int(sections[group[-1]]["end_time"] * sr), len(y))
        segment = y[start_sample:end_sample]

        # Cap segment length to ~30 seconds
        max_samples = 30 * sr
        if len(segment) > max_samples:
            segment = segment[:max_samples]

        if len(segment) == 0:
            desc = "Silent or empty section."
        else:
            desc = describer.describe(segment, sr)
            # Offset timestamps from section-relative to track-relative
            section_start = sections[group[0]]["start_time"]
            desc = _offset_timestamps(desc, section_start)

        group_descriptions.append(desc)
        last_desc = desc
        completed += 1

        if on_progress:
            on_progress(completed, total_calls, group, desc)

    # Expand group descriptions back to per-section descriptions
    descriptions: list[str] = [""] * len(sections)
    for gi, group in enumerate(groups):
        for si in group:
            descriptions[si] = group_descriptions[gi]

    return descriptions
