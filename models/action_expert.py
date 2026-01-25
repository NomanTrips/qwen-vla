"""Flow matching action expert transformer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FlowMatchingConfig:
    action_dim: int = 0
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_actions: int = 128
    timestep_dim: int = 128


def _build_mlp(hidden_dim: int, mlp_ratio: float, dropout: float) -> nn.Module:
    inner_dim = int(hidden_dim * mlp_ratio)
    return nn.Sequential(
        nn.Linear(hidden_dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, hidden_dim),
        nn.Dropout(dropout),
    )


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, device=timesteps.device, dtype=timesteps.dtype)
            * (torch.log(torch.tensor(10000.0, device=timesteps.device)) / (half_dim - 1))
        )
        angles = timesteps[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.embed_dim % 2 == 1:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return self.mlp(embedding)


class ActionTransformerBlock(nn.Module):
    """Single block with cross-attention then causal self-attention."""

    def __init__(self, config: FlowMatchingConfig) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(config.hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(config.hidden_dim)
        self.mlp = _build_mlp(config.hidden_dim, config.mlp_ratio, config.dropout)
        self.mlp_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, actions: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        cross_out, _ = self.cross_attn(actions, features, features, need_weights=False)
        actions = self.cross_norm(actions + cross_out)

        seq_len = actions.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=actions.device, dtype=torch.bool), 1
        )
        self_out, _ = self.self_attn(
            actions, actions, actions, attn_mask=causal_mask, need_weights=False
        )
        actions = self.self_norm(actions + self_out)

        mlp_out = self.mlp(actions)
        return self.mlp_norm(actions + mlp_out)


class FlowMatchingExpert(nn.Module):
    """Transformer that predicts flow matching velocity in action space."""

    def __init__(self, config: FlowMatchingConfig) -> None:
        super().__init__()
        if config.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        self.config = config
        self.action_embed = nn.Linear(config.action_dim, config.hidden_dim)
        self.position_embed = nn.Embedding(config.max_actions, config.hidden_dim)
        self.timestep_embed = TimestepEmbedding(config.timestep_dim)
        self.timestep_proj = nn.Linear(config.timestep_dim, config.hidden_dim)
        self.blocks = nn.ModuleList(
            [ActionTransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.action_dim)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        features: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = noisy_actions.shape
        positions = torch.arange(seq_len, device=noisy_actions.device)
        pos_embed = self.position_embed(positions)

        actions = self.action_embed(noisy_actions) + pos_embed
        timestep_embed = self.timestep_proj(self.timestep_embed(timesteps))
        actions = actions + timestep_embed[:, None, :]

        for block in self.blocks:
            actions = block(actions, features)

        actions = self.final_norm(actions)
        return self.out_proj(actions)

