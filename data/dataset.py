"""PyTorch dataset for QwenVLA training."""

from __future__ import annotations

import json
import os
from typing import List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset


class MiniworldVLADataset(Dataset):
    """Dataset for loading processed Miniworld episodes."""

    def __init__(
        self,
        episode_dirs: str | Sequence[str],
        num_frames: int = 3,
        frame_skip: int = 2,
        chunk_size: int = 50,
        instruction: str = "Navigate to the goal.",
    ) -> None:
        """Initialize dataset.

        Args:
            episode_dirs: Path to episode directory or list of episode directories.
            num_frames: Number of frames for temporal context.
            frame_skip: Gap between frames (2 means t, t-2, t-4).
            chunk_size: Number of actions per sample.
            instruction: Text instruction for all samples.
        """
        if isinstance(episode_dirs, str):
            episode_dirs = [episode_dirs]

        self.episode_dirs = list(episode_dirs)
        self.num_frames = num_frames
        self.frame_skip = frame_skip
        self.chunk_size = chunk_size
        self.instruction = instruction

        # Load all episodes
        self.samples: List[dict] = []
        self.frame_paths: List[List[str]] = []
        self.actions: List[List[dict]] = []

        for episode_dir in self.episode_dirs:
            self._load_episode(episode_dir)

    def _load_episode(self, episode_dir: str) -> None:
        """Load a single episode."""
        meta_path = os.path.join(episode_dir, "meta.json")
        actions_path = os.path.join(episode_dir, "actions.jsonl")
        frames_dir = os.path.join(episode_dir, "frames")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        num_frames_total = meta["num_frames"]
        num_keys = len(meta.get("key_vocab", []))

        # Load all frame actions
        frame_actions = []
        with open(actions_path, "r") as f:
            for line in f:
                data = json.loads(line)
                if data.get("type") == "frame":
                    frame_actions.append(data)

        # Create samples with temporal context
        # Need enough frames for temporal context and action chunk
        min_start = (self.num_frames - 1) * self.frame_skip
        max_start = num_frames_total - self.chunk_size

        if max_start <= min_start:
            return  # Episode too short

        episode_idx = len(self.frame_paths)
        frame_paths = [
            os.path.join(frames_dir, f"frame_{i:06d}.jpg")
            for i in range(num_frames_total)
        ]
        self.frame_paths.append(frame_paths)
        self.actions.append(frame_actions)

        for start_idx in range(min_start, max_start, self.chunk_size // 2):
            self.samples.append({
                "episode_idx": episode_idx,
                "start_idx": start_idx,
                "num_keys": num_keys,
            })

    def _action_to_tensor(self, action: dict, num_keys: int) -> torch.Tensor:
        """Convert action dict to tensor."""
        mouse = action["mouse"]
        buttons = action["buttons"]
        key_state = action.get("key_state", [0] * num_keys)

        # Action vector: [mouse_x, mouse_y, mouse_dx, mouse_dy, left, right, middle, keys...]
        values = [
            mouse["x"],
            mouse["y"],
            mouse["dx"],
            mouse["dy"],
            float(buttons["left"]),
            float(buttons["right"]),
            float(buttons["middle"]),
        ] + [float(k) for k in key_state]

        return torch.tensor(values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        episode_idx = sample["episode_idx"]
        start_idx = sample["start_idx"]
        num_keys = sample["num_keys"]

        frame_paths = self.frame_paths[episode_idx]
        frame_actions = self.actions[episode_idx]

        # Get temporal frame indices (t, t-skip, t-2*skip, ...)
        temporal_indices = [
            start_idx - i * self.frame_skip
            for i in range(self.num_frames)
        ][::-1]  # Reverse to get oldest first

        # Load frames
        frames = [Image.open(frame_paths[i]).convert("RGB") for i in temporal_indices]

        # Get action chunk
        action_tensors = [
            self._action_to_tensor(frame_actions[start_idx + i], num_keys)
            for i in range(self.chunk_size)
        ]
        actions = torch.stack(action_tensors)

        return {
            "images": frames,
            "instruction": self.instruction,
            "text": self.instruction,
            "states": None,
            "actions": actions,
        }


def collate_vla_batch(batch: List[dict]) -> dict:
    """Collate function for VLA batches."""
    images = [sample["images"] for sample in batch]
    instructions = [sample["instruction"] for sample in batch]
    actions = torch.stack([sample["actions"] for sample in batch])

    return {
        "images": images,
        "text": instructions[0] if len(set(instructions)) == 1 else instructions,
        "instruction": instructions[0] if len(set(instructions)) == 1 else instructions,
        "states": None,
        "actions": actions,
    }
