from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from .module.encoder import Encoder
from .module.decoder import Decoder
from ..utils.normalizer import ObservationNormalizer, StateNormalizer


@dataclass
class PaperModelConfig:
    """Config subset needed by the paper model.

    This is intentionally a dataclass so we can store it in checkpoints as a dict.
    """

    # arch
    hidden_dim: int = 192
    num_heads: int = 6
    encoder_depth: int = 3
    decoder_depth: int = 3
    encoder_drop_path_rate: float = 0.3
    decoder_drop_path_rate: float = 0.3

    # feature sizes
    agent_num: int = 33
    static_objects_num: int = 5
    lane_num: int = 70
    route_num: int = 25

    time_len: int = 21
    future_len: int = 80
    lane_len: int = 20

    predicted_neighbor_num: int = 32
    static_objects_state_dim: int = 10

    diffusion_model_type: str = "x_start"  # 'x_start' or 'score'

    # device hint (not strict)
    device: str = "cpu"

    # normalization (defaults loosely match vendor placeholders)
    state_mean: list[list[list[float]]] | None = None
    state_std: list[list[list[float]]] | None = None

    # observation normalization: mapping key -> {mean: [...], std: [...]}
    # Keys should match the inputs dict (e.g. ego_current_state, neighbor_agents_past, lanes, ...)
    observation_norm: dict[str, dict[str, list[float]]] | None = None

    def build_state_normalizer(self) -> StateNormalizer:
        # mean/std for ego + neighbors (P = 1 + predicted_neighbor_num)
        if self.state_mean is None or self.state_std is None:
            # defaults: x,y,cos,sin; roughly center around 0 in ego frame
            ego_mean = [0.0, 0.0, 0.0, 0.0]
            ego_std = [20.0, 20.0, 1.0, 1.0]
            neighbor_mean = [0.0, 0.0, 0.0, 0.0]
            neighbor_std = [20.0, 20.0, 1.0, 1.0]
            mean = [[ego_mean]] + [[neighbor_mean]] * self.predicted_neighbor_num
            std = [[ego_std]] + [[neighbor_std]] * self.predicted_neighbor_num
        else:
            mean = self.state_mean
            std = self.state_std
        return StateNormalizer(mean, std)

    def build_observation_normalizer(self) -> ObservationNormalizer:
        if not self.observation_norm:
            return ObservationNormalizer({})
        ndt: dict[str, dict[str, torch.Tensor]] = {}
        for k, v in self.observation_norm.items():
            if not isinstance(v, dict):
                continue
            if "mean" not in v or "std" not in v:
                continue
            ndt[str(k)] = {
                "mean": torch.tensor(v["mean"], dtype=torch.float32),
                "std": torch.tensor(v["std"], dtype=torch.float32),
            }
        return ObservationNormalizer(ndt)


class PaperDiffusionPlanner(nn.Module):
    """Paper-consistent Diffusion Planner (Encoder + Decoder).

    Forward signature matches vendor: (encoder_outputs, decoder_outputs) = model(inputs).

    This module also exposes a `sample_trajectory(...)` helper that returns ego trajectory [T,3]
    in ego frame.
    """

    CKPT_KIND = "paper_dit_dpm"
    CKPT_VERSION = 1

    def __init__(self, config: PaperModelConfig):
        super().__init__()
        self.config = config

        # attach normalizers expected by Decoder
        config.state_normalizer = config.build_state_normalizer()
        config.observation_normalizer = config.build_observation_normalizer()
        config.guidance_fn = None

        self.encoder = _PaperEncoder(config)
        self.decoder = _PaperDecoder(config)

    @property
    def sde(self):
        return self.decoder.decoder.sde

    def forward(self, inputs: dict) -> tuple[dict, dict]:
        # Normalize conditioning features (matches inference path).
        # IMPORTANT: state normalization is handled explicitly in training and inside decoder sampling.
        inputs_n = self.config.observation_normalizer(inputs) if hasattr(self.config, "observation_normalizer") else inputs
        enc = self.encoder(inputs_n)
        dec = self.decoder(enc, inputs_n)
        return enc, dec

    @torch.no_grad()
    def sample_trajectory(self, inputs: dict, *, diffusion_steps: int = 10) -> torch.Tensor:
        """Run DPM sampling and return ego trajectory [T,3] (x,y,heading)."""

        self.eval()
        inputs = dict(inputs)
        inputs["diffusion_steps"] = int(diffusion_steps)

        inputs = self.config.observation_normalizer(inputs) if hasattr(self.config, "observation_normalizer") else inputs

        enc = self.encoder(inputs)
        # run decoder in eval mode
        was_training = self.decoder.training
        self.decoder.eval()
        out = self.decoder(enc, inputs)
        if was_training:
            self.decoder.train(True)

        pred = out["prediction"]  # [B,P,T,4]
        # ego is index 0
        ego = pred[:, 0]  # [B,T,4]
        x = ego[..., 0]
        y = ego[..., 1]
        cos = ego[..., 2]
        sin = ego[..., 3]
        heading = torch.atan2(sin, cos)
        traj = torch.stack([x, y, heading], dim=-1)
        # return first batch
        return traj[0]

    def ckpt_payload(self) -> dict[str, Any]:
        """Metadata to store in checkpoints."""
        return {
            "kind": self.CKPT_KIND,
            "version": self.CKPT_VERSION,
            # asdict() only serializes dataclass fields (avoids torch Tensor/Module attrs).
            "paper_config": asdict(self.config),
        }


class _PaperEncoder(nn.Module):
    def __init__(self, config: PaperModelConfig):
        super().__init__()
        self.encoder = Encoder(config)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)
        nn.init.normal_(self.encoder.pos_emb.weight, std=0.02)

    def forward(self, inputs: dict) -> dict:
        return self.encoder(inputs)


class _PaperDecoder(nn.Module):
    def __init__(self, config: PaperModelConfig):
        super().__init__()
        self.decoder = Decoder(config)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)

        # timestep embedding MLP
        try:
            nn.init.normal_(self.decoder.dit.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.decoder.dit.t_embedder.mlp[2].weight, std=0.02)
        except Exception:
            pass

        # Zero-out adaLN modulation layers
        for block in self.decoder.dit.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].bias, 0)

    def forward(self, encoder_outputs: dict, inputs: dict) -> dict:
        return self.decoder(encoder_outputs, inputs)
