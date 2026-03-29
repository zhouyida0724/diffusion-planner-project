from __future__ import annotations

import abc

import torch

STD_MIN = 1e-6


class SDE(abc.ABC):
    """SDE abstract class."""

    @property
    @abc.abstractmethod
    def T(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def sde(self, x: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError

    @abc.abstractmethod
    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError

    @abc.abstractmethod
    def diffusion_coeff(self, t: torch.Tensor):
        raise NotImplementedError

    @abc.abstractmethod
    def marginal_prob_std(self, t: torch.Tensor):
        raise NotImplementedError


class VPSDE_linear(SDE):
    def __init__(self, beta_max: float = 20.0, beta_min: float = 0.1):
        super().__init__()
        self._beta_max = float(beta_max)
        self._beta_min = float(beta_min)

    @property
    def T(self) -> float:
        return 1.0

    def sde(self, x: torch.Tensor, t: torch.Tensor):
        shape = x.shape
        reshape = [-1] + [1] * (len(shape) - 1)
        t = t.reshape(reshape)

        beta_t = (self._beta_max - self._beta_min) * t + self._beta_min
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor):
        shape = x.shape
        reshape = [-1] + [1] * (len(shape) - 1)
        t = t.reshape(reshape)
        mean_log_coeff = -0.25 * t**2 * (self._beta_max - self._beta_min) - 0.5 * self._beta_min * t

        mean = torch.exp(mean_log_coeff) * x
        std = torch.sqrt(1 - torch.exp(2.0 * mean_log_coeff))
        return mean, std

    def diffusion_coeff(self, t: torch.Tensor):
        beta_t = (self._beta_max - self._beta_min) * t + self._beta_min
        return torch.sqrt(beta_t)

    def marginal_prob_std(self, t: torch.Tensor):
        discount = torch.exp(-0.5 * t**2 * (self._beta_max - self._beta_min) - self._beta_min * t)
        std = torch.sqrt(1 - discount)
        return std
