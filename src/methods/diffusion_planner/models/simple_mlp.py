"""A tiny baseline model for sanity training.

This is not the real diffusion model. It's a placeholder that:
  - takes a context vector x [B, x_dim]
  - predicts ego future y_hat [B, T, 3]

Used to validate:
  - dataset loading from sharded NPZ
  - training loop stability
  - checkpointing + output directory wiring
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleFutureMLP(nn.Module):
    def __init__(self, x_dim: int, T: int = 80, hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        self.x_dim = int(x_dim)
        self.T = int(T)
        self.out_dim = self.T * 3

        self.net = nn.Sequential(
            nn.Linear(self.x_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted future: [B, T, 3]."""
        assert x.ndim == 2 and x.shape[1] == self.x_dim, f"x shape {tuple(x.shape)}"
        y = self.net(x)
        y = y.view(x.shape[0], self.T, 3)
        return y
