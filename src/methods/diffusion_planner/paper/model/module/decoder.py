from __future__ import annotations

import os

import torch
import torch.nn as nn

# Gated debug for diffusion sampler dynamics (kept lightweight)
_DEBUG_SAMPLER_TICKS = 0

from ..diffusion_utils.sampling import dpm_sampler
from ..diffusion_utils.sde import SDE, VPSDE_linear
from ...utils.normalizer import ObservationNormalizer, StateNormalizer

from .mlp import Mlp
from .mixer import MixerBlock
from .dit import TimestepEmbedder, DiTBlock, FinalLayer


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        dpr = float(config.decoder_drop_path_rate)
        self._predicted_neighbor_num = int(config.predicted_neighbor_num)
        self._future_len = int(config.future_len)
        self._sde: SDE = VPSDE_linear()

        self.dit = DiT(
            sde=self._sde,
            route_encoder=RouteEncoder(
                config.route_num,
                config.lane_len,
                drop_path_rate=config.encoder_drop_path_rate,
                hidden_dim=config.hidden_dim,
            ),
            depth=config.decoder_depth,
            output_dim=(config.future_len + 1) * 4,
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=dpr,
            model_type=config.diffusion_model_type,
        )

        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer
        self._guidance_fn = getattr(config, "guidance_fn", None)

    @property
    def sde(self) -> SDE:
        return self._sde

    def forward(self, encoder_outputs: dict, inputs: dict) -> dict:
        ego_current = inputs["ego_current_state"][:, None, :4]

        neighbors_past = inputs["neighbor_agents_past"]
        # Some datasets include ego as the first entry in neighbor_agents_past.
        if neighbors_past.shape[1] == self._predicted_neighbor_num + 1:
            neighbors_current = neighbors_past[:, 1 : 1 + self._predicted_neighbor_num, -1, :4]
        else:
            neighbors_current = neighbors_past[:, : self._predicted_neighbor_num, -1, :4]

        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        inputs["neighbor_current_mask"] = neighbor_current_mask

        current_states = torch.cat([ego_current, neighbors_current], dim=1)

        B, P, _ = current_states.shape
        assert P == (1 + self._predicted_neighbor_num)

        ego_neighbor_encoding = encoder_outputs["encoding"]
        route_lanes = inputs["route_lanes"]

        if self.training:
            sampled_trajectories = inputs["sampled_trajectories"].reshape(B, P, -1)
            diffusion_time = inputs["diffusion_time"]
            out = self.dit(sampled_trajectories, diffusion_time, ego_neighbor_encoding, route_lanes, neighbor_current_mask)
            return {"score": out.reshape(B, P, -1, 4)}

        # inference sampling: start from noisy trajectory but keep t=0 state constrained.
        # IMPORTANT: DiT is trained in *normalized* state space (via StateNormalizer).
        # Therefore sampling must also happen in normalized space. We normalize current
        # states, sample x0 in normalized space, then inverse-normalize for output.

        # Normalize current states (expects shape [B,P,1,4] for correct broadcasting).
        current_states_norm = self._state_normalizer(current_states[:, :, None, :])[:, :, 0, :]

        # Random noise for future states in normalized space.
        noise = torch.randn(B, P, self._future_len, 4, device=current_states.device) * 0.5
        xT = torch.cat([current_states_norm[:, :, None], noise], dim=2).reshape(B, P, -1)

        def initial_state_constraint(xt: torch.Tensor, t: torch.Tensor, step: int):
            xt2 = xt.reshape(B, P, -1, 4)
            xt2[:, :, 0, :] = current_states_norm
            return xt2.reshape(B, P, -1)

        diffusion_steps = int(inputs.get("diffusion_steps", 10))

        # Optional sampler debug: prints a few statistics for the initial noise xT and solved x0.
        # Enable with: DIFFPLANNER_DEBUG_SAMPLER=1 (or DP_RUNTIME_DEBUG=1).
        global _DEBUG_SAMPLER_TICKS
        _dbg_sampler = os.getenv("DP_RUNTIME_DEBUG", "0") not in ("", "0", "false", "False") or (
            os.getenv("DIFFPLANNER_DEBUG_SAMPLER", "0") not in ("", "0", "false", "False")
        )
        _dbg_k = int(os.getenv("DP_RUNTIME_DEBUG_K", os.getenv("DIFFPLANNER_DEBUG_K", "5")))
        if _dbg_sampler and (_DEBUG_SAMPLER_TICKS < _dbg_k):
            try:
                xT_view = xT.reshape(B, P, -1, 4)
                xT_fut = xT_view[:, :, 1:]
                # show whether noise is varying
                noise_sample = noise.reshape(-1)[:6].detach().cpu().float().tolist()
                print(
                    "[DP_SAMPLER_DEBUG] tick={} steps={} xT_future_mean={:.4f} xT_future_std={:.4f} noise0={}".format(
                        _DEBUG_SAMPLER_TICKS,
                        diffusion_steps,
                        float(xT_fut.mean().detach().cpu()),
                        float(xT_fut.std().detach().cpu()),
                        [round(float(v), 4) for v in noise_sample],
                    )
                )
            except Exception:
                pass

        x0 = dpm_sampler(
            self.dit,
            xT,
            other_model_params={
                "cross_c": ego_neighbor_encoding,
                "route_lanes": route_lanes,
                "neighbor_current_mask": neighbor_current_mask,
            },
            diffusion_steps=diffusion_steps,
            dpm_solver_params={"correcting_xt_fn": initial_state_constraint},
            model_wrapper_params={
                "classifier_fn": self._guidance_fn,
                "classifier_kwargs": {
                    "model": self.dit,
                    "model_condition": {
                        "cross_c": ego_neighbor_encoding,
                        "route_lanes": route_lanes,
                        "neighbor_current_mask": neighbor_current_mask,
                    },
                    "inputs": inputs,
                    "observation_normalizer": self._observation_normalizer,
                    "state_normalizer": self._state_normalizer,
                },
                "guidance_scale": 0.5,
                "guidance_type": "classifier" if self._guidance_fn is not None else "uncond",
            },
        )

        if _dbg_sampler and (_DEBUG_SAMPLER_TICKS < _dbg_k):
            try:
                x0_view = x0.reshape(B, P, -1, 4)
                x0_fut = x0_view[:, :, 1:]
                x0_inv = self._state_normalizer.inverse(x0_view)[:, :, 1:]
                print(
                    "[DP_SAMPLER_DEBUG] tick={} x0_future_mean={:.4f} x0_future_std={:.4f} inv_future_mean={:.4f} inv_future_std={:.4f}".format(
                        _DEBUG_SAMPLER_TICKS,
                        float(x0_fut.mean().detach().cpu()),
                        float(x0_fut.std().detach().cpu()),
                        float(x0_inv.mean().detach().cpu()),
                        float(x0_inv.std().detach().cpu()),
                    )
                )
            except Exception:
                pass
            _DEBUG_SAMPLER_TICKS += 1

        # x0 currently is normalized in [B, P, (1+T), 4] (includes current state at index 0).
        x0_view = x0.reshape(B, P, -1, 4)
        # Ensure the constrained current state is exactly preserved.
        x0_view[:, :, 0, :] = current_states_norm
        x0_norm_fut = x0_view[:, :, 1:]
        x0_inv_fut = self._state_normalizer.inverse(x0_view)[:, :, 1:]

        out = {"prediction": x0_inv_fut}

        # Optional: expose normalized prediction for downstream debug.
        # Gated by DP_NORM_DEBUG=1 to avoid extra tensor plumbing by default.
        if os.getenv("DP_NORM_DEBUG", "0") not in ("", "0", "false", "False"):
            out["prediction_norm"] = x0_norm_fut
        return out


class RouteEncoder(nn.Module):
    def __init__(
        self,
        route_num: int,
        lane_len: int,
        drop_path_rate: float = 0.3,
        hidden_dim: int = 192,
        tokens_mlp_dim: int = 32,
        channels_mlp_dim: int = 64,
    ):
        super().__init__()
        self._channel = int(channels_mlp_dim)

        self.channel_pre_project = Mlp(in_features=4, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.0)
        self.token_pre_project = Mlp(in_features=route_num * lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.0)
        self.mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[..., :4]
        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        mask_b = torch.sum(~mask_p, dim=-1) == 0

        x = x.view(B, P * V, -1)

        valid_indices = ~mask_b.view(-1)
        x = x[valid_indices]

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.mixer(x)

        x = torch.mean(x, dim=1)
        x = self.emb_project(self.norm(x))

        # Under autocast, `x` can be bf16/fp16; keep destination dtype consistent for indexed assignment.
        x_result = x.new_zeros((B, x.shape[-1]))
        x_result[valid_indices] = x
        return x_result.view(B, -1)


class DiT(nn.Module):
    def __init__(
        self,
        *,
        sde: SDE,
        route_encoder: nn.Module,
        depth: int,
        output_dim: int,
        hidden_dim: int = 192,
        heads: int = 6,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        model_type: str = "x_start",
    ):
        super().__init__()
        assert model_type in ["score", "x_start"], f"Unknown model type: {model_type}"
        self._model_type = model_type
        self.route_encoder = route_encoder
        self.agent_embedding = nn.Embedding(2, hidden_dim)
        self.preproj = Mlp(in_features=output_dim, hidden_features=512, out_features=hidden_dim, act_layer=nn.GELU, drop=0.0)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_dim, output_dim)

        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std

    @property
    def model_type(self) -> str:
        return self._model_type

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cross_c: torch.Tensor,
        route_lanes: torch.Tensor,
        neighbor_current_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, P, _ = x.shape

        x = self.preproj(x)

        x_embedding = torch.cat(
            [self.agent_embedding.weight[0][None, :], self.agent_embedding.weight[1][None, :].expand(P - 1, -1)],
            dim=0,
        )
        x_embedding = x_embedding[None, :, :].expand(B, -1, -1).to(dtype=x.dtype)
        x = x + x_embedding

        route_encoding = self.route_encoder(route_lanes)
        y = route_encoding + self.t_embedder(t)

        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        attn_mask[:, 1:] = neighbor_current_mask

        for block in self.blocks:
            x = block(x, cross_c, y, attn_mask)

        x = self.final_layer(x, y)

        if self._model_type == "score":
            return x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        if self._model_type == "x_start":
            return x
        raise ValueError(f"Unknown model type: {self._model_type}")
