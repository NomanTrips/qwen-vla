"""Temporal feature handling for QwenVLA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
from torch import nn

from .qwen3_vl import Qwen3VLFeatureExtractor


@dataclass
class TemporalFeatureConfig:
    num_frames: int = 3
    add_temporal_embeddings: bool = True
    max_frames: int = 8
    use_token_pooler: bool = False


class TokenPooler(nn.Module):
    """Optional visual token pooling to reduce sequence length."""

    def __init__(self, hidden_dim: int, mode: str = "mean") -> None:
        super().__init__()
        if mode not in {"mean", "learned"}:
            raise ValueError(f"Unsupported pooling mode: {mode}")
        self.mode = mode
        if mode == "learned":
            self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean":
            return features.mean(dim=1, keepdim=True)
        weights = torch.softmax(self.attn(features), dim=1)
        return (features * weights).sum(dim=1, keepdim=True)


class TemporalFeatureExtractor(nn.Module):
    """Encode temporal frames and preserve temporal ordering."""

    def __init__(
        self,
        qwen_vl: Qwen3VLFeatureExtractor,
        config: TemporalFeatureConfig | None = None,
        token_pooler: TokenPooler | None = None,
    ) -> None:
        super().__init__()
        self.qwen_vl = qwen_vl
        self.config = config or TemporalFeatureConfig()
        self.token_pooler = token_pooler
        if self.config.add_temporal_embeddings:
            self.temporal_embeddings = nn.Embedding(
                self.config.max_frames, self.qwen_vl.hidden_size
            )
        else:
            self.temporal_embeddings = None

    @staticmethod
    def _normalize_frames(
        frames: Sequence[Iterable],
    ) -> List[List[Iterable]]:
        if not frames:
            raise ValueError("frames cannot be empty")
        if isinstance(frames[0], (list, tuple)):
            return [list(frame_set) for frame_set in frames]
        return [list(frames)]

    def _validate_num_frames(self, frames_batch: List[List[Iterable]]) -> int:
        num_frames = len(frames_batch[0])
        if num_frames == 0:
            raise ValueError("Each sample must include at least one frame")
        for sample in frames_batch[1:]:
            if len(sample) != num_frames:
                raise ValueError("All samples must have the same num_frames")
        return num_frames

    def forward(
        self, frames: Sequence[Iterable], instruction: str | Sequence[str]
    ) -> torch.Tensor:
        """Return temporal features with shape [batch, num_frames * seq_len, hidden]."""

        frames_batch = self._normalize_frames(frames)
        num_frames = self._validate_num_frames(frames_batch)
        feature_chunks: List[torch.Tensor] = []
        for frame_index in range(num_frames):
            frame_batch = [sample[frame_index] for sample in frames_batch]
            with torch.no_grad():
                features = self.qwen_vl(frame_batch, instruction)
            if self.temporal_embeddings is not None:
                # Ensure temporal embeddings are on the same device as features
                if self.temporal_embeddings.weight.device != features.device:
                    self.temporal_embeddings = self.temporal_embeddings.to(features.device)
                temporal_embed = self.temporal_embeddings(
                    torch.tensor(frame_index, device=features.device)
                )
                features = features + temporal_embed
            if self.token_pooler is not None:
                # Ensure token pooler is on the same device as features
                self.token_pooler = self.token_pooler.to(features.device)
                features = self.token_pooler(features)
            feature_chunks.append(features)
        return torch.cat(feature_chunks, dim=1)

