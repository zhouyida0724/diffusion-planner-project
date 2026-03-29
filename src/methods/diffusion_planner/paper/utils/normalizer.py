from __future__ import annotations

from copy import copy

import torch


class StateNormalizer:
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.mean.to(data.device)) / self.std.to(data.device)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.std.to(data.device) + self.mean.to(data.device)

    def to_dict(self) -> dict:
        return {"mean": self.mean.detach().cpu().numpy().tolist(), "std": self.std.detach().cpu().numpy().tolist()}


class ObservationNormalizer:
    def __init__(self, normalization_dict: dict):
        self._normalization_dict = normalization_dict

    def __call__(self, data: dict) -> dict:
        norm_data = copy(data)
        for k, v in self._normalization_dict.items():
            if k not in data:
                continue
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            norm_data[k] = (data[k] - v["mean"].to(data[k].device)) / v["std"].to(data[k].device)
            norm_data[k][mask] = 0
        return norm_data

    def inverse(self, data: dict) -> dict:
        norm_data = copy(data)
        for k, v in self._normalization_dict.items():
            if k not in data:
                continue
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            norm_data[k] = data[k] * v["std"].to(data[k].device) + v["mean"].to(data[k].device)
            norm_data[k][mask] = 0
        return norm_data

    def to_dict(self) -> dict:
        return {k: {kk: vv.detach().cpu().numpy().tolist() for kk, vv in v.items()} for k, v in self._normalization_dict.items()}
