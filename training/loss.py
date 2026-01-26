"""Loss functions for QwenVLA training."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


LossWeighting = Literal["none", "t", "sqrt_t", "inv_sqrt_t"]


def _sample_timesteps(
    batch_size: int,
    device: torch.device,
    beta_a: float,
    beta_b: float,
) -> torch.Tensor:
    distribution = torch.distributions.Beta(beta_a, beta_b)
    timesteps = distribution.sample((batch_size,)).to(device)
    return timesteps


def flow_matching_loss(
    model: nn.Module,
    images,
    text,
    states: torch.Tensor | None,
    actions: torch.Tensor,
    *,
    beta_a: float = 1.0,
    beta_b: float = 1.0,
    loss_weighting: LossWeighting = "none",
) -> torch.Tensor:
    """Compute flow matching loss for action trajectories.

    Args:
        model: QwenVLA model predicting velocity in action space.
        images: Batch of temporal image sequences.
        text: Instruction strings.
        states: Optional state tensor.
        actions: Ground-truth action chunks, shape [B, T, action_dim].
        beta_a: Alpha parameter for Beta timestep sampling.
        beta_b: Beta parameter for Beta timestep sampling.
        loss_weighting: Optional weighting scheme for per-sample loss.
    """

    if actions.ndim != 3:
        raise ValueError("actions must have shape [batch, chunk, action_dim]")

    device = actions.device
    batch_size = actions.shape[0]

    timesteps = _sample_timesteps(batch_size, device, beta_a, beta_b)
    t = timesteps.view(batch_size, 1, 1)

    noise = torch.randn_like(actions)
    noisy_actions = t * actions + (1.0 - t) * noise

    velocity = model(images, text, states, noisy_actions, timesteps)
    target = noise - actions

    per_sample_loss = F.mse_loss(velocity, target, reduction="none").mean(dim=(1, 2))

    if loss_weighting == "t":
        per_sample_loss = per_sample_loss * timesteps
    elif loss_weighting == "sqrt_t":
        per_sample_loss = per_sample_loss * torch.sqrt(timesteps)
    elif loss_weighting == "inv_sqrt_t":
        per_sample_loss = per_sample_loss / torch.sqrt(timesteps + 1e-6)

    return per_sample_loss.mean()
