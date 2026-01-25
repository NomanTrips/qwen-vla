"""Full QwenVLA model wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

from .action_expert import FlowMatchingConfig, FlowMatchingExpert
from .projectors import FeatureProjector, ProjectorConfig, StateProjector
from .qwen3_vl import Qwen3VLConfig, Qwen3VLFeatureExtractor
from .temporal_encoder import TemporalFeatureConfig, TemporalFeatureExtractor, TokenPooler


@dataclass
class QwenVLAConfig:
    qwen_vl: Qwen3VLConfig = Qwen3VLConfig()
    temporal: TemporalFeatureConfig = TemporalFeatureConfig()
    projectors: ProjectorConfig = ProjectorConfig()
    expert: FlowMatchingConfig = FlowMatchingConfig()
    chunk_size: int = 50
    use_token_pooler: bool = False


class QwenVLA(nn.Module):
    """End-to-end model combining Qwen 3 VL features with action expert."""

    def __init__(
        self,
        config: QwenVLAConfig,
        qwen_vl: Qwen3VLFeatureExtractor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl = qwen_vl or Qwen3VLFeatureExtractor(config.qwen_vl)

        if config.projectors.vlm_hidden_dim <= 0:
            config.projectors.vlm_hidden_dim = self.qwen_vl.hidden_size
        if config.expert.action_dim <= 0:
            config.expert.action_dim = config.projectors.action_dim

        token_pooler = None
        if config.use_token_pooler or config.temporal.use_token_pooler:
            token_pooler = TokenPooler(self.qwen_vl.hidden_size)

        self.temporal_encoder = TemporalFeatureExtractor(
            self.qwen_vl, config.temporal, token_pooler=token_pooler
        )
        self.feature_projector = FeatureProjector(config.projectors)
        self.state_projector = (
            StateProjector(config.projectors)
            if config.projectors.state_dim > 0
            else None
        )
        self.action_expert = FlowMatchingExpert(config.expert)

    def forward(
        self,
        images: Sequence[Iterable],
        text: str | Sequence[str],
        states: torch.Tensor | None,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity for flow matching loss."""

        with torch.no_grad():
            temporal_features = self.temporal_encoder(images, text)
        features = self.feature_projector(temporal_features)

        if self.state_projector is not None and states is not None:
            state_embed = self.state_projector(states)
            state_embed = state_embed[:, None, :]
            features = torch.cat([state_embed, features], dim=1)

        return self.action_expert(actions, features, timesteps)

    @torch.no_grad()
    def sample(
        self,
        images: Sequence[Iterable],
        text: str | Sequence[str],
        states: torch.Tensor | None = None,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Sample action chunk via Euler integration."""

        device = next(self.parameters()).device
        batch_size = len(images) if isinstance(images[0], (list, tuple)) else 1
        actions = torch.randn(
            batch_size, self.config.chunk_size, self.config.expert.action_dim, device=device
        )
        for step in reversed(range(num_steps)):
            t = torch.full((batch_size,), (step + 1) / num_steps, device=device)
            velocity = self.forward(images, text, states, actions, t)
            actions = actions - velocity / num_steps
        return actions

