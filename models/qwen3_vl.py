"""Qwen 3 VL integration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
from torch import nn
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig


@dataclass
class Qwen3VLConfig:
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    target_layer: int = 14
    load_in_4bit: bool = True
    device_map: str | dict | None = "auto"
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"


def build_bnb_config(config: Qwen3VLConfig) -> BitsAndBytesConfig:
    """Build a BitsAndBytesConfig for 4-bit quantization."""

    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
    )


def freeze_model(model: nn.Module) -> None:
    """Freeze all model parameters."""

    for param in model.parameters():
        param.requires_grad = False


def _normalize_text_batch(text: str | Sequence[str], batch_size: int) -> List[str]:
    if isinstance(text, str):
        return [text] * batch_size
    if len(text) != batch_size:
        raise ValueError(
            "Text batch length must match frames batch length: "
            f"{len(text)} != {batch_size}"
        )
    return list(text)


def _normalize_frame_batch(
    frames: Sequence[Iterable],
) -> List[List]:
    """Normalize frames to batch format: List[List[Image]].

    Input formats:
    - Single image: image -> [[image]]
    - List of images (one per sample): [img1, img2] -> [[img1], [img2]]
    - List of frame lists (batch): [[img1a, img1b], [img2a, img2b]] -> as-is
    """
    if not frames:
        raise ValueError("frames cannot be empty")
    # Check if first element is a list/tuple (batch of frame sequences)
    if isinstance(frames[0], (list, tuple)):
        return [list(frame_set) for frame_set in frames]
    # Otherwise, each element is a single image - wrap each in a list
    return [[frame] for frame in frames]


class Qwen3VLFeatureExtractor(nn.Module):
    """Wrapper that loads Qwen 3 VL and exposes hidden states at a target layer."""

    def __init__(self, config: Qwen3VLConfig) -> None:
        super().__init__()
        self.config = config
        quantization_config = None
        if config.load_in_4bit:
            quantization_config = build_bnb_config(config)
        self.processor = AutoProcessor.from_pretrained(config.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            config.model_name,
            device_map=config.device_map,
            quantization_config=quantization_config,
            torch_dtype=config.bnb_4bit_compute_dtype,
        )
        self.model.eval()
        freeze_model(self.model)

    @property
    def hidden_size(self) -> int:
        if hasattr(self.model.config, "hidden_size"):
            return self.model.config.hidden_size
        if hasattr(self.model.config, "text_config"):
            return self.model.config.text_config.hidden_size
        raise AttributeError("Unable to determine hidden size from model config")

    def _prepare_inputs(
        self, frames: Sequence[Iterable], text: str | Sequence[str]
    ) -> dict:
        frame_batch = _normalize_frame_batch(frames)
        text_batch = _normalize_text_batch(text, len(frame_batch))

        # Build chat messages with proper image placeholders for each sample
        all_texts = []
        all_images = []
        for sample_frames, sample_text in zip(frame_batch, text_batch):
            # Build content list with images followed by text
            content = []
            for frame in sample_frames:
                content.append({"type": "image", "image": frame})
            content.append({"type": "text", "text": sample_text})

            messages = [{"role": "user", "content": content}]
            formatted_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            all_texts.append(formatted_text)
            all_images.append(list(sample_frames))

        inputs = self.processor(
            text=all_texts,
            images=all_images,
            return_tensors="pt",
            padding=True,
        )
        return inputs

    def _resolve_layer_index(self, num_hidden_states: int) -> int:
        target = self.config.target_layer
        if target < 0:
            return target
        return min(target, num_hidden_states - 1)

    def forward(
        self, frames: Sequence[Iterable], text: str | Sequence[str]
    ) -> torch.Tensor:
        """Encode images + text and return hidden states from the target layer.

        Args:
            frames: Either a list of frames (single sample) or a batch of
                frame lists (batch size, num_frames, ...). Multiple images are
                supported per sample by passing a list of frames per sample.
            text: Single instruction string or a list per sample.
        """

        inputs = self._prepare_inputs(frames, text)
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        layer_index = self._resolve_layer_index(len(hidden_states))
        return hidden_states[layer_index]


def load_qwen3_vl(config: Qwen3VLConfig) -> Qwen3VLFeatureExtractor:
    """Convenience loader for the Qwen 3 VL feature extractor."""

    return Qwen3VLFeatureExtractor(config)
