from __future__ import annotations

"""Diffusion-Planner-style StatePerturbation augmentation.

This is a training-time augmentation that:
  1) perturbs ego_current_state (x,y,yaw,vx,vy,ax,ay,steer,yaw_rate) within bounds
  2) regenerates the first ~2s of ego future with a quintic spline to stay feasible
  3) applies a consistent ego-centric transform to all relevant inputs so the scene
     remains consistent in the perturbed ego frame.

All ops are torch / batch-vectorized (no python loops over timesteps).
"""

from dataclasses import dataclass
from typing import Any, Sequence

import torch


def _vector_transform(xy: torch.Tensor, rot: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Apply SE(2) rotation (and optional translation bias) to a (...,2) tensor.

    Args:
        xy: [B, ..., 2]
        rot: [B, 2, 2]
        bias: [B, 2] or [B, ..., 2] broadcastable; if given, subtract before rotation.
    """

    if xy.ndim < 2 or xy.shape[-1] != 2:
        raise ValueError(f"xy must be [...,2], got {tuple(xy.shape)}")

    B = int(xy.shape[0])
    shape = xy.shape
    n_expand = xy.ndim - 2

    if bias is not None:
        if bias.ndim == 2 and bias.shape == (B, 2):
            bias_r = bias.reshape(B, *([1] * n_expand), 2)
        else:
            bias_r = bias
        xy = xy - bias_r

    # [B, ..., 2] -> [B, 2, N]
    flat = xy.reshape(B, -1, 2).transpose(1, 2)
    out = torch.bmm(rot, flat).transpose(1, 2).reshape(shape)
    return out


def _heading_transform(heading: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Transform a heading angle by the rotation matrix.

    heading: [B, ...] in radians.
    rot: [B,2,2]
    """

    B = int(heading.shape[0])
    shape = heading.shape
    h = heading.reshape(B, -1)
    r = rot.reshape(B, 1, 2, 2)
    # Transform unit heading vector and recover angle.
    return torch.atan2(
        torch.cos(h) * r[..., 1, 0] + torch.sin(h) * r[..., 1, 1],
        torch.cos(h) * r[..., 0, 0] + torch.sin(h) * r[..., 0, 1],
    ).reshape(shape)


def _normalize_angle(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + torch.pi, 2.0 * torch.pi) - torch.pi


@dataclass
class StatePerturbationConfig:
    enabled: bool = False
    prob: float = 0.0
    low: Sequence[float] = (0.0, -0.75, -0.35, -1.0, -0.5, -0.2, -0.1, 0.0, 0.0)
    high: Sequence[float] = (0.0, 0.75, 0.35, 1.0, 0.5, 0.2, 0.1, 0.0, 0.0)
    # regenerate first ~2s (20 steps @ 10Hz)
    refine_horizon_s: float = 2.0
    dt_s: float = 0.1
    num_refine: int = 20
    # (optional) only apply when abs(vx) >= this threshold (matches reference behavior)
    min_vx_mps: float = 2.0


class StatePerturbation:
    def __init__(self, cfg: StatePerturbationConfig, *, device: torch.device):
        self.cfg = cfg
        self.device = device

        low = torch.tensor(list(cfg.low), dtype=torch.float32, device=device)
        high = torch.tensor(list(cfg.high), dtype=torch.float32, device=device)
        if low.shape != (9,) or high.shape != (9,):
            raise ValueError("StatePerturbation bounds must be length-9: [x,y,yaw,vx,vy,ax,ay,steer,yaw_rate]")
        self._low = low
        self._high = high

        # Quintic spline matrices.
        T = float(cfg.refine_horizon_s + cfg.dt_s)
        A = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [1, T, T**2, T**3, T**4, T**5],
                [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
                [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],
            ],
            dtype=torch.float32,
            device=device,
        )
        self._coeff_inv = torch.linalg.inv(A)  # [6,6]

        t = torch.linspace(float(cfg.dt_s), float(cfg.refine_horizon_s), int(cfg.num_refine), device=device)
        # [P,6] where columns are t^0..t^5
        self._t_matrix = torch.pow(t[:, None], torch.arange(6, device=device)[None, :]).to(torch.float32)

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        if (not self.cfg.enabled) or float(self.cfg.prob) <= 0.0:
            return batch

        ego_cs = batch.get("ego_current_state")
        ego_fut = batch.get("ego_agent_future")
        nb_fut = batch.get("neighbor_agents_future")
        if not (torch.is_tensor(ego_cs) and torch.is_tensor(ego_fut) and torch.is_tensor(nb_fut)):
            return batch

        B = int(ego_cs.shape[0])
        if B <= 0:
            return batch

        # Decide per-sample application.
        apply = (torch.rand((B,), device=ego_cs.device) < float(self.cfg.prob))
        if float(self.cfg.min_vx_mps) > 0:
            apply = apply & ~(torch.abs(ego_cs[:, 4]) < float(self.cfg.min_vx_mps))
        if not bool(apply.any().item()):
            return batch

        # Clone the tensors we mutate.
        ego_cs = ego_cs.clone()
        ego_fut = ego_fut.clone()
        nb_fut = nb_fut.clone()

        # -----------------
        # 1) Perturb ego_current_state.
        # -----------------
        noise_u = torch.rand((B, 9), device=ego_cs.device, dtype=torch.float32)
        delta = self._low + (self._high - self._low) * noise_u  # [B,9] via broadcast
        # Keep deltas for non-applied samples as 0.
        delta = delta * apply[:, None].to(delta.dtype)

        # Compose into state vector [x,y,yaw,vx,vy,ax,ay,steer,yaw_rate].
        yaw0 = torch.atan2(ego_cs[:, 3], ego_cs[:, 2])
        state0 = torch.stack(
            [
                ego_cs[:, 0],
                ego_cs[:, 1],
                yaw0,
                ego_cs[:, 4],
                ego_cs[:, 5],
                ego_cs[:, 6],
                ego_cs[:, 7],
                ego_cs[:, 8],
                ego_cs[:, 9],
            ],
            dim=-1,
        )
        state1 = state0 + delta
        # Simple safety clamps (match reference intent): v>=0, yaw_rate in [-0.85,0.85]
        state1[:, 3] = torch.clamp(state1[:, 3], min=0.0)
        state1[:, 8] = torch.clamp(state1[:, 8], -0.85, 0.85)
        state1[:, 2] = _normalize_angle(state1[:, 2])

        ego_cs[:, 0:2] = state1[:, 0:2]
        ego_cs[:, 2] = torch.cos(state1[:, 2])
        ego_cs[:, 3] = torch.sin(state1[:, 2])
        ego_cs[:, 4:8] = state1[:, 3:7]
        ego_cs[:, 8] = state1[:, 7]
        ego_cs[:, 9] = state1[:, 8]

        # -----------------
        # 2) Quintic spline: regenerate first ~2s of ego future.
        # -----------------
        ego_fut = self._interpolate_ego_future(ego_cs, ego_fut)

        # -----------------
        # 3) Ego-centric transform of all relevant inputs.
        # -----------------
        batch = dict(batch)
        batch["ego_current_state"] = ego_cs
        batch["ego_agent_future"] = ego_fut
        batch["neighbor_agents_future"] = nb_fut
        batch = self._centric_transform_inplace(batch)

        return batch

    def _rot_from_cos_sin(self, ego_cs: torch.Tensor) -> torch.Tensor:
        # rot = R(-yaw) = [[cos, sin],[-sin, cos]]
        cos = ego_cs[:, 2]
        sin = ego_cs[:, 3]
        return torch.stack(
            [
                torch.stack([cos, sin], dim=-1),
                torch.stack([-sin, cos], dim=-1),
            ],
            dim=-2,
        )

    def _centric_transform_inplace(self, batch: dict[str, Any]) -> dict[str, Any]:
        ego_cs = batch["ego_current_state"]
        center_xy = ego_cs[:, 0:2]
        rot = self._rot_from_cos_sin(ego_cs)

        # ego_current_state: xy / cos-sin / v / a are 2D vectors.
        ego_cs[:, 0:2] = _vector_transform(ego_cs[:, 0:2], rot, center_xy)
        ego_cs[:, 2:4] = _vector_transform(ego_cs[:, 2:4], rot)
        ego_cs[:, 4:6] = _vector_transform(ego_cs[:, 4:6], rot)
        ego_cs[:, 6:8] = _vector_transform(ego_cs[:, 6:8], rot)
        batch["ego_current_state"] = ego_cs

        # ego future: [B,T,3]
        ego_fut = batch.get("ego_agent_future")
        if torch.is_tensor(ego_fut) and ego_fut.ndim == 3 and ego_fut.shape[-1] >= 3:
            ego_fut_xy = _vector_transform(ego_fut[..., :2], rot, center_xy)
            ego_fut_h = _heading_transform(ego_fut[..., 2], rot)
            ego_fut = torch.cat([ego_fut_xy, ego_fut_h[..., None]], dim=-1)
            batch["ego_agent_future"] = ego_fut

        # neighbor past: [B,*,Tp,D] where xy/cos-sin/v are first dims.
        nb_past = batch.get("neighbor_agents_past")
        if torch.is_tensor(nb_past) and nb_past.ndim == 4 and nb_past.shape[-1] >= 6:
            # mask of rows that are fully zero (keep consistent with reference)
            mask = (torch.sum(torch.ne(nb_past[..., :6], 0.0), dim=-1) == 0)
            nb_past[..., :2] = _vector_transform(nb_past[..., :2], rot, center_xy)
            nb_past[..., 2:4] = _vector_transform(nb_past[..., 2:4], rot)
            nb_past[..., 4:6] = _vector_transform(nb_past[..., 4:6], rot)
            nb_past[mask] = 0.0
            batch["neighbor_agents_past"] = nb_past

        # neighbor future: [B,*,Tf,3]
        nb_fut = batch.get("neighbor_agents_future")
        if torch.is_tensor(nb_fut) and nb_fut.ndim == 4 and nb_fut.shape[-1] >= 3:
            mask = (torch.sum(torch.ne(nb_fut[..., :2], 0.0), dim=-1) == 0)
            nb_fut[..., :2] = _vector_transform(nb_fut[..., :2], rot, center_xy)
            nb_fut[..., 2] = _heading_transform(nb_fut[..., 2], rot)
            nb_fut[mask] = 0.0
            batch["neighbor_agents_future"] = nb_fut

        # lanes / route_lanes: [B,L,N,12], we transform (xy, dir, left/right offsets) as vectors.
        for k in ("lanes", "route_lanes"):
            lane = batch.get(k)
            if not (torch.is_tensor(lane) and lane.ndim == 4 and lane.shape[-1] >= 8):
                continue
            mask = (torch.sum(torch.ne(lane[..., :8], 0.0), dim=-1) == 0)
            lane[..., :2] = _vector_transform(lane[..., :2], rot, center_xy)
            lane[..., 2:4] = _vector_transform(lane[..., 2:4], rot)
            lane[..., 4:6] = _vector_transform(lane[..., 4:6], rot)
            lane[..., 6:8] = _vector_transform(lane[..., 6:8], rot)
            lane[mask] = 0.0
            batch[k] = lane

        # static objects: [B,S,10] (we only transform xy, and (cos,sin) if present)
        so = batch.get("static_objects")
        if torch.is_tensor(so) and so.ndim == 3 and so.shape[-1] >= 2:
            mask = (torch.sum(torch.ne(so[..., :10] if so.shape[-1] >= 10 else so, 0.0), dim=-1) == 0)
            so[..., :2] = _vector_transform(so[..., :2], rot, center_xy)
            if so.shape[-1] >= 4:
                so[..., 2:4] = _vector_transform(so[..., 2:4], rot)
            so[mask] = 0.0
            batch["static_objects"] = so

        return batch

    def _interpolate_ego_future(self, ego_cs: torch.Tensor, ego_future: torch.Tensor) -> torch.Tensor:
        """Replace ego_future[:num_refine] with a quintic spline (vectorized)."""

        if ego_future.ndim != 3 or ego_future.shape[-1] < 3:
            return ego_future

        P = int(self.cfg.num_refine)
        dt = float(self.cfg.dt_s)
        if ego_future.shape[1] <= P + 2:
            return ego_future

        B = int(ego_cs.shape[0])
        # Setup matrices on correct device/dtype.
        Ainv = self._coeff_inv.to(device=ego_cs.device, dtype=torch.float32)[None, :, :].expand(B, -1, -1)  # [B,6,6]
        M = self._t_matrix.to(device=ego_cs.device, dtype=torch.float32)[None, :, :].expand(B, -1, -1)  # [B,P,6]

        x0 = ego_cs[:, 0]
        y0 = ego_cs[:, 1]
        # Use a lookahead point to infer heading (as reference does).
        mid = max(0, int(P // 2))
        theta0 = torch.atan2(ego_future[:, mid, 1] - y0, ego_future[:, mid, 0] - x0)
        v0 = torch.linalg.norm(ego_cs[:, 4:6], dim=-1)
        a0 = torch.linalg.norm(ego_cs[:, 6:8], dim=-1)
        omega0 = ego_cs[:, 9]

        xT = ego_future[:, P, 0]
        yT = ego_future[:, P, 1]
        thetaT = ego_future[:, P, 2]
        vT = torch.linalg.norm(ego_future[:, P, :2] - ego_future[:, P - 1, :2], dim=-1) / dt
        aT = torch.linalg.norm(
            ego_future[:, P, :2] - 2.0 * ego_future[:, P - 1, :2] + ego_future[:, P - 2, :2], dim=-1
        ) / (dt**2)
        omegaT = _normalize_angle(ego_future[:, P, 2] - ego_future[:, P - 1, 2]) / dt

        sx = torch.stack(
            [
                x0,
                v0 * torch.cos(theta0),
                a0 * torch.cos(theta0) - v0 * torch.sin(theta0) * omega0,
                xT,
                vT * torch.cos(thetaT),
                aT * torch.cos(thetaT) - vT * torch.sin(thetaT) * omegaT,
            ],
            dim=-1,
        )
        sy = torch.stack(
            [
                y0,
                v0 * torch.sin(theta0),
                a0 * torch.sin(theta0) + v0 * torch.cos(theta0) * omega0,
                yT,
                vT * torch.sin(thetaT),
                aT * torch.sin(thetaT) + vT * torch.cos(thetaT) * omegaT,
            ],
            dim=-1,
        )

        ax = torch.bmm(Ainv, sx[:, :, None])  # [B,6,1]
        ay = torch.bmm(Ainv, sy[:, :, None])  # [B,6,1]

        traj_x = torch.bmm(M, ax)  # [B,P,1]
        traj_y = torch.bmm(M, ay)  # [B,P,1]

        # Heading from derivatives: first heading from (x0,y0)->(x1,y1), then finite diff.
        hd0 = torch.atan2(traj_y[:, :1, 0] - y0[:, None], traj_x[:, :1, 0] - x0[:, None])
        hd_rest = torch.atan2(traj_y[:, 1:, 0] - traj_y[:, :-1, 0], traj_x[:, 1:, 0] - traj_x[:, :-1, 0])
        traj_h = torch.cat([hd0, hd_rest], dim=1)

        refined = torch.cat([traj_x, traj_y, traj_h[..., None]], dim=-1)  # [B,P,3]
        return torch.cat([refined, ego_future[:, P:, :]], dim=1)

