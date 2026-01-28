"""Miniworld integration for QwenVLA inference.

This script runs the QwenVLA model in the Miniworld Sign environment,
converting model outputs (mouse/keyboard format) to Miniworld actions.

The Sign environment tasks the agent with finding a sign that displays a color
(BLUE, RED, or GREEN), then navigating to touch an object of that color.

Usage:
    python inference/miniworld_runner.py --checkpoint checkpoints/test_run/best.pt

    # With visualization window
    python inference/miniworld_runner.py --checkpoint checkpoints/test_run/best.pt --render

    # Run multiple episodes
    python inference/miniworld_runner.py --checkpoint checkpoints/test_run/best.pt --episodes 10

    # Record videos of episodes
    python inference/miniworld_runner.py --checkpoint checkpoints/test_run/best.pt --record-video --video-dir videos/
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Mapping, Optional

import numpy as np
import torch
from PIL import Image

try:
    import imageio.v3 as iio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import QwenVLA, QwenVLAConfig

try:
    import gymnasium as gym
    from miniworld.envs import Sign
except ImportError:
    print("Error: Miniworld not found. Please install from ~/Desktop/Miniworld:")
    print("  cd ~/Desktop/Miniworld && pip install -e .")
    sys.exit(1)


class VideoRecorder:
    """Records frames and saves them as video files."""

    def __init__(self, output_dir: str, fps: int = 10):
        """Initialize video recorder.

        Args:
            output_dir: Directory to save video files
            fps: Frames per second for output video
        """
        if not HAS_IMAGEIO:
            raise RuntimeError(
                "imageio is required for video recording. "
                "Install with: pip install imageio[ffmpeg]"
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frames: List[np.ndarray] = []
        self.episode_count = 0

    def add_frame(self, frame: np.ndarray) -> None:
        """Add a frame to the current episode recording.

        Args:
            frame: RGB numpy array (H, W, 3), uint8
        """
        self.frames.append(frame.copy())

    def save_episode(self, suffix: str = "") -> Optional[str]:
        """Save recorded frames as video and reset buffer.

        Args:
            suffix: Optional suffix for filename (e.g., "_success" or "_fail")

        Returns:
            Path to saved video file, or None if no frames
        """
        if not self.frames:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"episode_{self.episode_count:04d}_{timestamp}{suffix}.mp4"
        filepath = self.output_dir / filename

        # Stack frames into array and write video
        frames_array = np.stack(self.frames, axis=0)
        iio.imwrite(
            str(filepath),
            frames_array,
            fps=self.fps,
        )

        num_frames = len(self.frames)
        self.frames = []
        self.episode_count += 1

        print(f"  Saved video: {filepath} ({num_frames} frames)")
        return str(filepath)

    def reset(self) -> None:
        """Clear frame buffer without saving."""
        self.frames = []


@dataclass
class RunnerConfig:
    """Configuration for Miniworld runner."""

    # Frame capture settings
    num_frames: int = 3
    frame_skip: int = 2  # Gap between temporal frames
    target_size: tuple[int, int] = (384, 384)  # Resize frames to this size

    # Action execution settings
    actions_per_chunk: int = 50  # Actions per inference call
    action_execution_rate: int = 1  # Execute every Nth action from chunk

    # Action conversion settings (model output -> Miniworld action)
    # These map the model's mouse/keyboard output to Miniworld's action space
    # Training data has normalized dx in [-0.23, 0.16] range, scale up for meaningful turns
    mouse_dx_scale: float = 3.0  # Scale factor for mouse_dx -> turn_delta
    mouse_dy_scale: float = 3.0  # Scale factor for mouse_dy -> pitch_delta

    # Key indices in model output (after mouse/button values)
    # Based on key_vocab: [13(Enter), 27(Esc), 38(Up), 65(A), 83(S), 87(W), 120(F9)]
    # These indices are relative to the start of key_state in the action vector
    key_w_index: int = 5  # Forward (W=87 at vocab index 5)
    key_s_index: int = 4  # Backward (S=83 at vocab index 4)
    key_a_index: int = 3  # Strafe left (A=65 at vocab index 3)
    key_d_index: int = -1  # Strafe right (D not in vocab, disabled)

    # Sampling settings
    num_sampling_steps: int = 10


def load_model_config(path: Optional[str]) -> QwenVLAConfig:
    """Load model config from YAML file."""
    import yaml
    import dataclasses

    config = QwenVLAConfig()
    if not path:
        return config

    def update_dataclass(instance: object, updates: Mapping[str, object]) -> None:
        for key, value in updates.items():
            if not hasattr(instance, key):
                continue
            current = getattr(instance, key)
            if dataclasses.is_dataclass(current) and isinstance(value, Mapping):
                update_dataclass(current, value)
            else:
                setattr(instance, key, value)

    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    model_payload = payload.get("model", payload)
    if isinstance(model_payload, Mapping):
        update_dataclass(config, model_payload)
    return config


def load_model(checkpoint_path: str, config: QwenVLAConfig) -> QwenVLA:
    """Load QwenVLA model from checkpoint."""
    print("Loading model...")
    model = QwenVLA(config)

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
    state_dict = (
        checkpoint_data["model"]
        if isinstance(checkpoint_data, dict) and "model" in checkpoint_data
        else checkpoint_data
    )

    # Filter out VLM keys (they're re-quantized on load)
    def is_vlm_key(k: str) -> bool:
        return "qwen_vl." in k

    vlm_keys = [k for k in state_dict.keys() if is_vlm_key(k)]
    trainable_state = {k: v for k, v in state_dict.items() if not is_vlm_key(k)}

    missing, unexpected = model.load_state_dict(trainable_state, strict=False)
    missing = [k for k in missing if not is_vlm_key(k)]
    unexpected = [k for k in unexpected if not is_vlm_key(k)]

    if missing:
        print(f"Warning: missing keys: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected}")

    print(
        f"Loaded {len(trainable_state)} trainable weights "
        f"(skipped {len(vlm_keys)} VLM keys)"
    )
    model.eval()
    return model


def preprocess_frame(
    frame: np.ndarray,
    target_size: tuple[int, int] = (384, 384),
) -> Image.Image:
    """Convert Miniworld observation to PIL Image for model input.

    Args:
        frame: RGB numpy array from Miniworld (H, W, 3), uint8
        target_size: Target size (width, height) for the image

    Returns:
        PIL Image resized to target size
    """
    img = Image.fromarray(frame)
    if img.size != target_size:
        img = img.resize(target_size, Image.Resampling.BILINEAR)
    return img


def convert_model_action_to_miniworld(
    action: np.ndarray,
    config: RunnerConfig,
) -> np.ndarray:
    """Convert QwenVLA model output to Miniworld action format.

    Model output format (13+ dims):
        [0] mouse_x     - Absolute mouse X position (unused for movement)
        [1] mouse_y     - Absolute mouse Y position (unused for movement)
        [2] mouse_dx    - Mouse X delta -> turn_delta
        [3] mouse_dy    - Mouse Y delta -> pitch_delta
        [4] left_button - Left click -> pickup action
        [5] right_button - Right click -> drop action
        [6] middle_button - Middle click (unused)
        [7:] key_states - Keyboard states (WASD for movement)

    Miniworld action format (6 dims):
        [0] forward_speed  - Forward/backward movement [-1, 1]
        [1] strafe_speed   - Left/right strafing [-1, 1]
        [2] turn_delta     - Yaw rotation [-1, 1]
        [3] pitch_delta    - Pitch rotation [-1, 1]
        [4] pickup         - Pickup action [0, 1]
        [5] drop           - Drop action [0, 1]

    Args:
        action: Model output array of shape [action_dim]
        config: Runner configuration with scaling factors

    Returns:
        Miniworld action array of shape [6]
    """
    miniworld_action = np.zeros(6, dtype=np.float32)

    # Extract model outputs
    mouse_dx = action[2] if len(action) > 2 else 0.0
    mouse_dy = action[3] if len(action) > 3 else 0.0
    left_button = action[4] if len(action) > 4 else 0.0
    right_button = action[5] if len(action) > 5 else 0.0

    # Key states start at index 7
    key_offset = 7
    num_keys = len(action) - key_offset if len(action) > key_offset else 0

    # Extract raw key values (model outputs small floats around 0, not binary 0/1)
    def get_key_raw(idx: int) -> float:
        if idx < 0 or idx >= num_keys:
            return 0.0
        return float(action[key_offset + idx])

    key_w = get_key_raw(config.key_w_index)
    key_s = get_key_raw(config.key_s_index)
    key_a = get_key_raw(config.key_a_index)
    key_d = get_key_raw(config.key_d_index)

    # Map to Miniworld actions
    # Forward/backward: use W-S difference, scaled up since model outputs small values
    # Scale by 10x so a difference of 0.1 becomes full speed
    forward_diff = key_w - key_s
    miniworld_action[0] = np.clip(forward_diff * 10.0, -1.0, 1.0)  # forward_speed

    # Strafe from A/D keys (D-A for right positive)
    strafe_diff = key_d - key_a
    miniworld_action[1] = np.clip(strafe_diff * 10.0, -1.0, 1.0)  # strafe_speed

    # Turn from mouse_dx (negative because mouse right = turn left visually)
    miniworld_action[2] = np.clip(
        -mouse_dx * config.mouse_dx_scale, -1.0, 1.0
    )  # turn_delta

    # Pitch from mouse_dy
    miniworld_action[3] = np.clip(
        mouse_dy * config.mouse_dy_scale, -1.0, 1.0
    )  # pitch_delta

    # Pickup/drop from mouse buttons (threshold at 0.5)
    miniworld_action[4] = 1.0 if left_button > 0.5 else 0.0  # pickup
    miniworld_action[5] = 1.0 if right_button > 0.5 else 0.0  # drop

    return miniworld_action


class MiniworldRunner:
    """Runs QwenVLA model in Miniworld Sign environment."""

    def __init__(
        self,
        model: QwenVLA,
        config: RunnerConfig,
        render: bool = False,
        seed: Optional[int] = None,
        verbose: bool = True,
        record_video: bool = False,
        video_dir: str = "videos",
        video_fps: int = 10,
    ):
        """Initialize the runner.

        Args:
            model: Loaded QwenVLA model
            config: Runner configuration
            render: Whether to render the environment window
            seed: Random seed for reproducibility
            verbose: Print debug info during execution
            record_video: Whether to record episode videos
            video_dir: Directory to save videos
            video_fps: Frames per second for recorded videos
        """
        self.model = model
        self.config = config
        self.render = render
        self.seed = seed
        self.verbose = verbose
        self.record_video = record_video

        # Video recorder
        self.video_recorder: Optional[VideoRecorder] = None
        if record_video:
            self.video_recorder = VideoRecorder(video_dir, fps=video_fps)

        # Determine device
        self.device = next(model.parameters()).device

        # Frame buffer for temporal context
        self.frame_buffer: deque[Image.Image] = deque(maxlen=config.num_frames)

        # Create environment with larger observation for better visual input
        # Sign environment default is 80x60, we may want larger for the model
        self.env = gym.make(
            "MiniWorld-Sign-v0",
            obs_width=160,  # Larger observations
            obs_height=120,
        )

        # Task instruction based on environment
        self.instruction = (
            "Navigate to find the sign showing a color, "
            "then touch the object matching that color."
        )

        if self.verbose:
            print(f"Environment: MiniWorld-Sign-v0")
            print(f"Observation size: {self.env.observation_space}")
            print(f"Action space: {self.env.action_space}")
            print(f"Using continuous actions (6-dim Box space)")
            if record_video:
                print(f"Recording videos to: {video_dir}")

    def reset(self) -> dict:
        """Reset the environment and clear frame buffer."""
        self.frame_buffer.clear()
        obs, info = self.env.reset(seed=self.seed)

        # obs is a dict with 'obs' and 'goal' keys
        frame = obs["obs"]

        # Fill frame buffer with initial frame
        processed = preprocess_frame(frame, self.config.target_size)
        for _ in range(self.config.num_frames):
            self.frame_buffer.append(processed)

        # Record initial frame
        if self.video_recorder is not None:
            self.video_recorder.reset()
            self.video_recorder.add_frame(frame)

        if self.render:
            self.env.render()

        return obs

    def get_action_chunk(self) -> np.ndarray:
        """Run model inference to get action chunk.

        Returns:
            Action chunk of shape [chunk_size, action_dim]
        """
        # Prepare frames (oldest first)
        frames = list(self.frame_buffer)

        # Run model inference
        with torch.no_grad():
            actions = self.model.sample(
                images=[frames],  # Batch of 1
                text=self.instruction,
                states=None,
                num_steps=self.config.num_sampling_steps,
            )

        # Convert to numpy: [1, chunk_size, action_dim] -> [chunk_size, action_dim]
        return actions[0].cpu().numpy()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Execute one action in the environment.

        Args:
            action: Miniworld action array [6]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update frame buffer
        frame = obs["obs"]
        processed = preprocess_frame(frame, self.config.target_size)
        self.frame_buffer.append(processed)

        # Record frame for video
        if self.video_recorder is not None:
            self.video_recorder.add_frame(frame)

        if self.render:
            self.env.render()

        return obs, reward, terminated, truncated, info

    def run_episode(self) -> dict:
        """Run a single episode.

        Returns:
            Episode statistics dict
        """
        obs = self.reset()
        total_reward = 0.0
        total_steps = 0
        done = False
        chunk_count = 0

        while not done:
            # Get action chunk from model
            if self.verbose:
                print(f"\n  Generating action chunk {chunk_count + 1}...")

            action_chunk = self.get_action_chunk()
            chunk_count += 1

            if self.verbose:
                print(f"  Action chunk shape: {action_chunk.shape}")
                print(f"  First action (raw model output): {action_chunk[0][:7]}")  # Show first 7 values

            # Execute actions from chunk
            for i in range(0, len(action_chunk), self.config.action_execution_rate):
                model_action = action_chunk[i]
                miniworld_action = convert_model_action_to_miniworld(
                    model_action, self.config
                )

                if self.verbose and i == 0:
                    print(
                        f"  Converted action: fwd={miniworld_action[0]:.2f}, "
                        f"strafe={miniworld_action[1]:.2f}, "
                        f"turn={miniworld_action[2]:.2f}, "
                        f"pitch={miniworld_action[3]:.2f}"
                    )

                obs, reward, terminated, truncated, info = self.step(miniworld_action)
                total_reward += reward
                total_steps += 1
                done = terminated or truncated

                if reward > 0 and self.verbose:
                    print(f"  Got reward: {reward}")

                if done:
                    break

        success = total_reward > 0

        # Save video if recording
        video_path = None
        if self.video_recorder is not None:
            suffix = "_success" if success else "_fail"
            video_path = self.video_recorder.save_episode(suffix=suffix)

        return {
            "success": success,
            "reward": total_reward,
            "steps": total_steps,
            "chunks": chunk_count,
            "video_path": video_path,
        }

    def run(self, num_episodes: int = 1) -> list[dict]:
        """Run multiple episodes.

        Args:
            num_episodes: Number of episodes to run

        Returns:
            List of episode statistics
        """
        results = []

        for ep in range(num_episodes):
            print(f"\n=== Episode {ep + 1}/{num_episodes} ===")
            stats = self.run_episode()
            results.append(stats)

            status = "SUCCESS" if stats["success"] else "FAILED"
            print(
                f"Episode {ep + 1}: {status} | "
                f"Reward: {stats['reward']:.2f} | "
                f"Steps: {stats['steps']} | "
                f"Chunks: {stats['chunks']}"
            )

        # Print summary
        if num_episodes > 1:
            successes = sum(1 for r in results if r["success"])
            avg_reward = np.mean([r["reward"] for r in results])
            avg_steps = np.mean([r["steps"] for r in results])

            print(f"\n=== Summary ({num_episodes} episodes) ===")
            print(f"Success rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
            print(f"Average reward: {avg_reward:.2f}")
            print(f"Average steps: {avg_steps:.1f}")

        return results

    def close(self):
        """Close the environment."""
        self.env.close()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run QwenVLA model in Miniworld Sign environment."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--config",
        help="Path to YAML config file for model settings.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment window.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num-sampling-steps",
        type=int,
        default=10,
        help="Number of ODE integration steps for sampling (default: 10).",
    )
    parser.add_argument(
        "--mouse-dx-scale",
        type=float,
        default=0.1,
        help="Scale factor for mouse_dx -> turn_delta (default: 0.1).",
    )
    parser.add_argument(
        "--mouse-dy-scale",
        type=float,
        default=0.1,
        help="Scale factor for mouse_dy -> pitch_delta (default: 0.1).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output during execution.",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record videos of each episode.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="videos",
        help="Directory to save recorded videos (default: videos/).",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="Frames per second for recorded videos (default: 10).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model
    model_config = load_model_config(args.config)
    model = load_model(args.checkpoint, model_config)

    # Configure runner
    runner_config = RunnerConfig(
        num_sampling_steps=args.num_sampling_steps,
        mouse_dx_scale=args.mouse_dx_scale,
        mouse_dy_scale=args.mouse_dy_scale,
    )

    # Create runner
    runner = MiniworldRunner(
        model=model,
        config=runner_config,
        render=args.render,
        seed=args.seed,
        verbose=not args.quiet,
        record_video=args.record_video,
        video_dir=args.video_dir,
        video_fps=args.video_fps,
    )

    try:
        runner.run(num_episodes=args.episodes)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
