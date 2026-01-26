"""Projection modules for state, action, and vision features."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ProjectorConfig:
    state_dim: int = 0
    action_dim: int = 0
    vlm_hidden_dim: int = 0  # Auto-detected from VLM if 0
    expert_hidden_dim: int = 512
    use_layer_norm: bool = True


class StateProjector(nn.Module):
    """Project state vectors into the expert hidden dimension."""

    def __init__(self, config: ProjectorConfig) -> None:
        super().__init__()
        if config.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        self.proj = nn.Linear(config.state_dim, config.expert_hidden_dim)
        self.norm = nn.LayerNorm(config.expert_hidden_dim) if config.use_layer_norm else None

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        projected = self.proj(states)
        if self.norm is not None:
            projected = self.norm(projected)
        return projected


class ActionProjector(nn.Module):
    """Project action vectors into the expert hidden dimension."""

    def __init__(self, config: ProjectorConfig) -> None:
        super().__init__()
        if config.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        self.proj = nn.Linear(config.action_dim, config.expert_hidden_dim)
        self.norm = nn.LayerNorm(config.expert_hidden_dim) if config.use_layer_norm else None

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        projected = self.proj(actions)
        if self.norm is not None:
            projected = self.norm(projected)
        return projected


class FeatureProjector(nn.Module):
    """Project VLM features into the expert hidden dimension."""

    def __init__(self, config: ProjectorConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(config.vlm_hidden_dim, config.expert_hidden_dim)
        self.norm = nn.LayerNorm(config.expert_hidden_dim) if config.use_layer_norm else None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.proj(features)
        if self.norm is not None:
            projected = self.norm(projected)
        return projected

