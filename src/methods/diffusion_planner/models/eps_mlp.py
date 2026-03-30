"""A minimal eps-pred MLP with timestep conditioning.

This is *not* the final architecture (no DiT / no map tokens), but it matches
Diffusion training interfaces:
  eps_hat = f(x_cond, y_t, t)

Where:
  - x_cond: context vector from dataset [B, x_dim]
  - y_t: noisy trajectory [B, T, 3]
  - t: diffusion step [B]

Output:
  - eps_hat: predicted noise [B, T, 3]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.methods.diffusion_planner.diffusion.timestep_embedding import sinusoidal_timestep_embedding


class EpsMLP(nn.Module):
    def __init__(
        self,
        *,
        x_dim: int,
        T: int = 80,
        hidden: int = 512,
        t_embed_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.x_dim = int(x_dim)
        self.T = int(T)
        self.t_embed_dim = int(t_embed_dim)

        y_dim = self.T * 3
        in_dim = self.x_dim + y_dim + self.t_embed_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, y_dim),
        )

    def forward(self, x: torch.Tensor, y_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2 and x.shape[1] == self.x_dim, f"x shape {tuple(x.shape)}"
        assert y_t.ndim == 3 and y_t.shape[1:] == (self.T, 3), f"y_t shape {tuple(y_t.shape)}"
        assert t.ndim == 1 and t.shape[0] == x.shape[0], f"t shape {tuple(t.shape)}"

        t_emb = sinusoidal_timestep_embedding(t.to(device=x.device), self.t_embed_dim)
        h = torch.cat([x, y_t.reshape(x.shape[0], -1), t_emb], dim=1)
        eps = self.net(h)
        eps = eps.view(x.shape[0], self.T, 3)
        assert torch.isfinite(eps).all(), "eps_hat contains NaN/Inf"
        return eps
