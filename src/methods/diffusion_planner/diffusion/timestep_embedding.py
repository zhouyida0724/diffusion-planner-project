"""Timestep embedding.

We use the standard sinusoidal embedding used in diffusion / transformers.
"""

from __future__ import annotations

import math

import torch


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.

    Args:
        t: int tensor [B] (timesteps)
        dim: embedding dimension

    Returns:
        emb: float tensor [B, dim]
    """

    assert t.ndim == 1
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((t.shape[0], 1), device=t.device, dtype=emb.dtype)], dim=1)
    assert emb.shape == (t.shape[0], dim)
    return emb
