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

    def describe(self, audio: np.ndarray, sr: int) -> str:
        """Describe audio content using Gemini."""
        import soundfile as sf
        from google import genai
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
                        "Give a detailed description of this audio segment. Include:\n"
                        "- Instruments heard (drums, synths, bass, vocals, guitar, pads, etc.)\n"
                        "- Rhythm and tempo feel (driving, laid-back, syncopated, four-on-the-floor, etc.)\n"
                        "- Energy level and how it changes through the segment\n"
                        "- Mood and emotional sensation (dark, euphoric, tense, dreamy, aggressive, etc.)\n"
                        "- Intensity — is it building, peaking, dropping, or sustaining?\n"
                        "- Notable transitions, drops, breakdowns, or changes with approximate timestamps\n"
                        "- Texture and production quality (sparse, layered, distorted, clean, reverb-heavy, etc.)\n"
                        "Be thorough and specific. Use timestamps like [0:05] to call out moments."
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
    max_sections: int = 30,
    on_progress: callable | None = None,
) -> list[str]:
    """Run audio description on each section, grouping similar adjacent sections.

    When there are many sections, consecutive sections of the same type are
    merged for description to keep the total number of model calls manageable.
    The returned list always has one description per original section.

    Args:
        describer: AudioDescriber instance.
        y: Full audio time series.
        sr: Sample rate.
        sections: Section dicts with start_time/end_time/type.
        max_sections: Maximum unique descriptions to generate.
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

    # If still too many groups, sample evenly and reuse descriptions for skipped groups
    if len(groups) > max_sections:
        step = len(groups) / max_sections
        sampled_indices = {int(i * step) for i in range(max_sections)}
    else:
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
