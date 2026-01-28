"""Simple inference script for QwenVLA."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import yaml
from PIL import Image

from models import QwenVLA, QwenVLAConfig


@dataclass
class InferenceConfig:
    num_frames: int = 3
    num_steps: int = 10
    instruction: str = ""


def _update_dataclass(instance: object, updates: Mapping[str, object]) -> None:
    for key, value in updates.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if dataclasses.is_dataclass(current) and isinstance(value, Mapping):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)


def load_model_config(path: str | None) -> QwenVLAConfig:
    config = QwenVLAConfig()
    if not path:
        return config
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    model_payload = payload.get("model", payload)
    if isinstance(model_payload, Mapping):
        _update_dataclass(config, model_payload)
    return config


def load_inference_config(path: str | None) -> InferenceConfig:
    config = InferenceConfig()
    if not path:
        return config
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    inference_payload = payload.get("inference", payload)
    if isinstance(inference_payload, Mapping):
        _update_dataclass(config, inference_payload)
    return config


def _collect_frame_paths(args: argparse.Namespace, config: InferenceConfig) -> list[Path]:
    if args.frames:
        paths = [Path(path) for path in args.frames]
    elif args.frames_dir:
        dir_path = Path(args.frames_dir)
        if not dir_path.exists():
            raise FileNotFoundError(f"Frames directory not found: {dir_path}")
        allowed = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        paths = [
            path for path in sorted(dir_path.iterdir()) if path.suffix.lower() in allowed
        ]
    else:
        raise ValueError("Provide --frames or --frames-dir")

    if not paths:
        raise ValueError("No frames found to load.")

    if len(paths) < config.num_frames:
        raise ValueError(
            f"Need at least {config.num_frames} frames, found {len(paths)}."
        )

    if len(paths) > config.num_frames:
        paths = paths[-config.num_frames :]

    return paths


def _load_frames(paths: Sequence[Path]) -> list[Image.Image]:
    frames = []
    for path in paths:
        with Image.open(path) as img:
            frames.append(img.convert("RGB"))
    return frames


def _print_action_summary(actions: torch.Tensor) -> None:
    actions = actions.detach().cpu()
    print(f"Predicted actions shape: {tuple(actions.shape)}")
    print(f"Action value range: min={actions.min().item():.4f}, max={actions.max().item():.4f}")
    print("First action vector:")
    print(actions[0, 0].tolist())


def run_inference(
    checkpoint: str,
    frames: Sequence[Iterable],
    instruction: str,
    model_config: QwenVLAConfig,
    num_steps: int,
) -> torch.Tensor:
    print("Loading model...")
    model = QwenVLA(model_config)
    print("Loading checkpoint...")
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    state_dict = checkpoint_data["model"] if isinstance(checkpoint_data, dict) else checkpoint_data

    # Filter out VLM quantization keys (expected to not match since VLM is re-quantized)
    # VLM keys can be under qwen_vl. or temporal_encoder.qwen_vl.
    def is_vlm_key(k: str) -> bool:
        return "qwen_vl." in k

    vlm_keys = [k for k in state_dict.keys() if is_vlm_key(k)]
    trainable_state = {k: v for k, v in state_dict.items() if not is_vlm_key(k)}

    missing, unexpected = model.load_state_dict(trainable_state, strict=False)
    # Filter out expected missing keys (VLM weights that we didn't load)
    missing = [k for k in missing if not is_vlm_key(k)]
    unexpected = [k for k in unexpected if not is_vlm_key(k)]

    if missing:
        print(f"Warning: missing keys in checkpoint: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys in checkpoint: {unexpected}")
    print(f"Loaded trainable weights ({len(trainable_state)} keys, skipped {len(vlm_keys)} VLM keys)")

    model.eval()
    with torch.no_grad():
        actions = model.sample([list(frames)], instruction, states=None, num_steps=num_steps)
    return actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QwenVLA inference.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--frames", nargs="+", help="List of frame image paths.")
    parser.add_argument("--frames-dir", help="Directory containing frame images.")
    parser.add_argument("--config", help="Path to YAML config with model/inference settings.")
    parser.add_argument("--num-frames", type=int, default=None, help="Number of frames to use.")
    parser.add_argument("--num-steps", type=int, default=None, help="Sampling steps.")
    parser.add_argument(
        "--instruction",
        default=None,
        help="Instruction text to condition inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = load_model_config(args.config)
    inference_config = load_inference_config(args.config)
    if args.num_frames is not None:
        inference_config.num_frames = args.num_frames
    if args.num_steps is not None:
        inference_config.num_steps = args.num_steps
    if args.instruction is not None:
        inference_config.instruction = args.instruction

    frame_paths = _collect_frame_paths(args, inference_config)
    frames = _load_frames(frame_paths)

    actions = run_inference(
        checkpoint=args.checkpoint,
        frames=frames,
        instruction=inference_config.instruction,
        model_config=model_config,
        num_steps=inference_config.num_steps,
    )
    _print_action_summary(actions)


if __name__ == "__main__":
    main()
