"""Learning-rate scheduling utilities."""

from __future__ import annotations

import math
from typing import Iterable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Create a cosine schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if num_training_steps <= 0:
            return 1.0
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def get_trainable_parameters(parameters: Iterable) -> list:
    """Filter out frozen parameters for optimizer setup."""
    return [param for param in parameters if param.requires_grad]
