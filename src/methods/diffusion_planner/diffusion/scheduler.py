"""Noise scheduler utilities (beta schedule + q_sample).

We start with a simple linear beta schedule and the standard forward noising:
  q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

This is enough to train an eps-pred model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def linear_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    assert num_steps > 1
    betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
    assert torch.all(betas > 0) and torch.all(betas < 1)
    return betas


@dataclass(frozen=True)
class NoiseSchedule:
    num_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def make(self, device: torch.device | None = None) -> "NoiseScheduleTensors":
        betas = linear_beta_schedule(self.num_steps, self.beta_start, self.beta_end)
        if device is not None:
            betas = betas.to(device)
        return NoiseScheduleTensors.from_betas(betas)


@dataclass(frozen=True)
class NoiseScheduleTensors:
    betas: torch.Tensor  # [K]
    alphas: torch.Tensor  # [K]
    alpha_bar: torch.Tensor  # [K]
    sqrt_alpha_bar: torch.Tensor  # [K]
    sqrt_one_minus_alpha_bar: torch.Tensor  # [K]

    @staticmethod
    def from_betas(betas: torch.Tensor) -> "NoiseScheduleTensors":
        assert betas.ndim == 1
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        out = NoiseScheduleTensors(
            betas=betas,
            alphas=alphas,
            alpha_bar=alpha_bar,
            sqrt_alpha_bar=torch.sqrt(alpha_bar),
            sqrt_one_minus_alpha_bar=torch.sqrt(1.0 - alpha_bar),
        )
        # basic sanity
        assert torch.isfinite(out.sqrt_alpha_bar).all()
        assert torch.isfinite(out.sqrt_one_minus_alpha_bar).all()
        return out

    @property
    def num_steps(self) -> int:
        return int(self.betas.shape[0])


def _extract_1d(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Gather 1D schedule tensor a[t] and reshape to broadcast over x."""
    assert a.ndim == 1
    assert t.ndim == 1
    out = a.gather(0, t)
    assert out.shape == (t.shape[0],)
    # reshape to [B, 1, 1, ...]
    return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))


def q_sample(
    *,
    schedule: NoiseScheduleTensors,
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Forward noising: sample x_t given x0 and eps.

    Args:
        schedule: precomputed schedule tensors.
        x0: clean data [B, ...]
        t: int64 timesteps [B]
        noise: eps ~ N(0,1) same shape as x0

    Returns:
        x_t: [B, ...]
    """

    assert x0.shape == noise.shape
    assert t.dtype in (torch.int32, torch.int64)
    assert t.ndim == 1 and t.shape[0] == x0.shape[0]
    assert int(t.min()) >= 0 and int(t.max()) < schedule.num_steps

    s1 = _extract_1d(schedule.sqrt_alpha_bar, t, x0.shape)
    s2 = _extract_1d(schedule.sqrt_one_minus_alpha_bar, t, x0.shape)

    xt = s1 * x0 + s2 * noise
    assert torch.isfinite(xt).all()
    return xt
