"""Loss stubs for diffusion training."""

from __future__ import annotations

import torch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute mean squared error with an optional broadcastable mask.

    mask should be 0/1 with shape broadcastable to pred.
    """

    assert pred.shape == target.shape
    if mask is None:
        return torch.mean((pred - target) ** 2)

    # broadcast mask to pred
    m = mask.to(dtype=pred.dtype, device=pred.device)
    assert m.shape[0] == pred.shape[0]
    err = (pred - target) ** 2
    err = err * m
    denom = torch.clamp(m.sum(), min=1.0)
    return err.sum() / denom
