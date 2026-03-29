from __future__ import annotations

import torch
import torch.nn as nn


class Mlp(nn.Module):
    """Simple 2-layer MLP (timm-like) used to avoid extra dependencies."""

    def __init__(
        self,
        *,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden = int(hidden_features if hidden_features is not None else in_features)
        out = int(out_features if out_features is not None else in_features)
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden, out)
        self.drop = nn.Dropout(drop) if drop and drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
