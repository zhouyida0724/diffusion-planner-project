from __future__ import annotations

import torch
import torch.nn as nn

from .mlp import Mlp
from .droppath import DropPath
from .mixer import MixerBlock


class Encoder(nn.Module):
    """Condition encoder (agents + static + lanes) matching vendor shapes."""

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = int(config.hidden_dim)
        self.token_num = int(config.agent_num + config.static_objects_num + config.lane_num)

        self.neighbor_encoder = AgentFusionEncoder(
            config.time_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth,
        )
        self.static_encoder = StaticFusionEncoder(
            config.static_objects_state_dim,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
        )
        self.lane_encoder = LaneFusionEncoder(
            config.lane_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth,
        )

        self.fusion = FusionEncoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            drop_path_rate=config.encoder_drop_path_rate,
            depth=config.encoder_depth,
            device=getattr(config, "device", "cpu"),
        )

        # position embedding encodes x,y,cos,sin, type-onehot(3)
        self.pos_emb = nn.Linear(7, config.hidden_dim)

    def forward(self, inputs: dict) -> dict:
        neighbors = inputs["neighbor_agents_past"]
        static = inputs["static_objects"]
        lanes = inputs["lanes"]
        lanes_speed_limit = inputs["lanes_speed_limit"]
        lanes_has_speed_limit = inputs["lanes_has_speed_limit"]

        B = neighbors.shape[0]

        encoding_neighbors, neighbors_mask, neighbor_pos = self.neighbor_encoder(neighbors)
        encoding_static, static_mask, static_pos = self.static_encoder(static)
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(lanes, lanes_speed_limit, lanes_has_speed_limit)

        encoding_input = torch.cat([encoding_neighbors, encoding_static, encoding_lanes], dim=1)

        encoding_pos = torch.cat([neighbor_pos, static_pos, lane_pos], dim=1).view(B * self.token_num, -1)
        encoding_mask = torch.cat([neighbors_mask, static_mask, lanes_mask], dim=1).view(-1)

        encoding_pos_valid = self.pos_emb(encoding_pos[~encoding_mask])
        # Under autocast, pos_emb can output bf16/fp16; keep the indexed destination in the same dtype.
        encoding_pos_result = encoding_pos_valid.new_zeros((B * self.token_num, self.hidden_dim))
        encoding_pos_result[~encoding_mask] = encoding_pos_valid

        encoding_input = encoding_input + encoding_pos_result.view(B, self.token_num, -1)

        return {"encoding": self.fusion(encoding_input, encoding_mask.view(B, self.token_num))}


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int = 192, heads: int = 6, dropout: float = 0.1, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.drop_path = DropPath(dropout) if dropout and dropout > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=nn.GELU, drop=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), x, x, key_padding_mask=mask)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AgentFusionEncoder(nn.Module):
    def __init__(
        self,
        time_len: int,
        drop_path_rate: float = 0.3,
        hidden_dim: int = 192,
        depth: int = 3,
        tokens_mlp_dim: int = 64,
        channels_mlp_dim: int = 128,
    ):
        super().__init__()
        self._hidden_dim = int(hidden_dim)
        self._channel = int(channels_mlp_dim)

        self.type_emb = nn.Linear(3, channels_mlp_dim)

        self.channel_pre_project = Mlp(in_features=8 + 1, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.0)
        self.token_pre_project = Mlp(in_features=time_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.0)

        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for _ in range(depth)])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x: torch.Tensor):
        """x: [B, P, V, D] D=(x,y,cos,sin,vx,vy,w,l,type(3))."""

        neighbor_type = x[:, :, -1, 8:]
        x = x[..., :8]

        pos = x[:, :, -1, :7].clone()
        pos[..., -3:] = 0.0
        pos[..., -3] = 1.0

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0

        x = torch.cat([x, (~mask_v).float().unsqueeze(-1)], dim=-1)
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1)
        x = x[valid_indices]

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        x = torch.mean(x, dim=1)

        neighbor_type = neighbor_type.view(B * P, -1)[valid_indices]
        x = x + self.type_emb(neighbor_type)

        x = self.emb_project(self.norm(x))

        # Under autocast, `x` can be bf16/fp16; keep the indexed destination in the same dtype.
        x_result = x.new_zeros((B * P, x.shape[-1]))
        x_result[valid_indices] = x

        return x_result.view(B, P, -1), mask_p.reshape(B, -1), pos.view(B, P, -1)


class StaticFusionEncoder(nn.Module):
    def __init__(self, dim: int, drop_path_rate: float = 0.3, hidden_dim: int = 192):
        super().__init__()
        self._hidden_dim = int(hidden_dim)
        self.projection = Mlp(in_features=dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x: torch.Tensor):
        """x: [B,P,D] (x,y,cos,sin,w,l,type(4)...)"""

        B, P, _ = x.shape
        pos = x[:, :, :7].clone()
        pos[..., -3:] = 0.0
        pos[..., -2] = 1.0

        # Under autocast, `x` stays fp32 but the projection output becomes bf16/fp16.
        # Allocate the indexed destination in the current autocast dtype to avoid dtype-mismatch crashes.
        if torch.is_autocast_enabled() and x.is_cuda:
            # torch.get_autocast_dtype("cuda") is only available in newer PyTorch.
            if hasattr(torch, "get_autocast_dtype"):
                dst_dtype = torch.get_autocast_dtype("cuda")
            elif hasattr(torch, "get_autocast_gpu_dtype"):
                dst_dtype = torch.get_autocast_gpu_dtype()
            else:
                dst_dtype = x.dtype
        else:
            dst_dtype = x.dtype
        x_result = torch.zeros((B * P, self._hidden_dim), device=x.device, dtype=dst_dtype)
        mask_p = torch.sum(torch.ne(x[..., :10], 0), dim=-1).to(x.device) == 0
        valid_indices = ~mask_p.view(-1)

        if valid_indices.sum() > 0:
            xx = x.view(B * P, -1)[valid_indices]
            xx = self.projection(xx)
            x_result[valid_indices] = xx

        return x_result.view(B, P, -1), mask_p.view(B, P), pos.view(B, P, -1)


class LaneFusionEncoder(nn.Module):
    def __init__(
        self,
        lane_len: int,
        drop_path_rate: float = 0.3,
        hidden_dim: int = 192,
        depth: int = 3,
        tokens_mlp_dim: int = 64,
        channels_mlp_dim: int = 128,
    ):
        super().__init__()
        self._lane_len = int(lane_len)
        self._channel = int(channels_mlp_dim)

        self.speed_limit_emb = nn.Linear(1, channels_mlp_dim)
        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)
        self.traffic_emb = nn.Linear(4, channels_mlp_dim)

        self.channel_pre_project = Mlp(in_features=8, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.0)
        self.token_pre_project = Mlp(in_features=lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.0)
        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for _ in range(depth)])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x: torch.Tensor, speed_limit: torch.Tensor, has_speed_limit: torch.Tensor):
        traffic = x[:, :, 0, 8:]
        x = x[..., :8]

        pos = x[:, :, int(self._lane_len / 2), :7].clone()
        heading = torch.atan2(pos[..., 3], pos[..., 2])
        pos[..., 2] = torch.cos(heading)
        pos[..., 3] = torch.sin(heading)
        pos[..., -3:] = 0.0
        pos[..., -1] = 1.0

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0

        x = x.view(B * P, V, -1)
        valid_indices = ~mask_p.view(-1)
        x = x[valid_indices]

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        x = torch.mean(x, dim=1)

        speed_limit = speed_limit.view(B * P, 1)
        has_speed_limit = has_speed_limit.view(B * P, 1)
        traffic = traffic.view(B * P, -1)

        has_speed_limit_v = has_speed_limit[valid_indices].squeeze(-1) > 0.5
        speed_limit_v = speed_limit[valid_indices].squeeze(-1)
        # Allocate embeddings in the same dtype as `x` (autocast bf16/fp16 safe).
        speed_limit_embedding = x.new_zeros((speed_limit_v.shape[0], self._channel))

        if has_speed_limit_v.sum() > 0:
            speed_limit_embedding[has_speed_limit_v] = self.speed_limit_emb(
                speed_limit_v[has_speed_limit_v].unsqueeze(-1)
            ).to(dtype=speed_limit_embedding.dtype)
        if (~has_speed_limit_v).sum() > 0:
            speed_limit_embedding[~has_speed_limit_v] = self.unknown_speed_emb.weight.to(dtype=speed_limit_embedding.dtype).expand(
                (~has_speed_limit_v).sum().item(), -1
            )

        traffic_v = traffic[valid_indices]
        traffic_light_embedding = self.traffic_emb(traffic_v)

        x = x + speed_limit_embedding + traffic_light_embedding
        x = self.emb_project(self.norm(x))

        # Under autocast, `x` can be bf16/fp16; keep the indexed destination in the same dtype.
        x_result = x.new_zeros((B * P, x.shape[-1]))
        x_result[valid_indices] = x

        return x_result.view(B, P, -1), mask_p.reshape(B, -1), pos.view(B, P, -1)


class FusionEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 192, num_heads: int = 6, drop_path_rate: float = 0.3, depth: int = 3, device: str = "cpu"):
        super().__init__()
        dpr = float(drop_path_rate)
        self.blocks = nn.ModuleList([SelfAttentionBlock(hidden_dim, num_heads, dropout=dpr) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # always keep ego token
        mask[:, 0] = False
        for b in self.blocks:
            x = b(x, mask)
        return self.norm(x)
