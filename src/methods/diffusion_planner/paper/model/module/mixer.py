from __future__ import annotations

import torch
import torch.nn as nn

from .mlp import Mlp
from .droppath import DropPath


class MixerBlock(nn.Module):
    """MLP-Mixer block used by the vendor encoders.

    This is a lightweight re-implementation of the vendor module, but without
    timm dependency.

    Input x shape: [B, tokens, channels]
    """

    def __init__(self, tokens_mlp_dim: int, channels_mlp_dim: int, drop_path_rate: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels_mlp_dim)
        self.mlp_tokens = Mlp(in_features=tokens_mlp_dim, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.0)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate and drop_path_rate > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(channels_mlp_dim)
        self.mlp_channels = Mlp(in_features=channels_mlp_dim, hidden_features=channels_mlp_dim * 2, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # token-mixing: (B, T, C) -> (B, C, T) apply MLP over tokens
        y = self.norm1(x)
        y = y.transpose(1, 2)
        y = self.mlp_tokens(y)
        y = y.transpose(1, 2)
        x = x + self.drop_path(y)

        # channel-mixing
        y = self.norm2(x)
        y = self.mlp_channels(y)
        x = x + self.drop_path(y)
        return x
