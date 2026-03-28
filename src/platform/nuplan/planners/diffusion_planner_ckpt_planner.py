"""Diffusion Planner (checkpoint-backed) nuPlan planner.

Design goals
- Do NOT modify vendor code under `nuplan-visualization/`.
- Wire our repo-local training checkpoints into nuPlan closed-loop simulation.
- Fail fast on load/inference errors (hard fallback).

This planner is meant for Milestone-B: end-to-end closure (train → ckpt → sim).
Model quality is not the focus yet.

Checkpoint contract (from PR #25 training skeleton)
- torch.load(ckpt) returns a dict with at least:
  - model_state: state_dict for `SimpleFutureMLP`

Runtime features
- We build the same x vector as `ShardedNpzDataset`:
  x = concat(ego_current_state (10,), ego_past (21,3) flattened)
- The MLP predicts relative poses [T,3] in ego frame, which matches
  `transform_predictions_to_states` expectation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np
import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from src.methods.diffusion_planner.models.simple_mlp import SimpleFutureMLP


@dataclass
class _EgoKinematics:
    x: float
    y: float
    heading: float
    vx: float
    vy: float
    ax: float
    ay: float


def _rot2d(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _wrap_pi(a: float) -> float:
    while a > np.pi:
        a -= 2 * np.pi
    while a < -np.pi:
        a += 2 * np.pi
    return float(a)


def _ego_kinematics(ego: EgoState) -> _EgoKinematics:
    # Position/heading: prefer rear axle pose.
    try:
        x = float(ego.rear_axle.x)
        y = float(ego.rear_axle.y)
        heading = float(ego.rear_axle.heading)
    except Exception:
        x = float(ego.center.x)
        y = float(ego.center.y)
        heading = float(ego.center.heading)

    vx = vy = ax = ay = 0.0
    try:
        v = ego.dynamic_car_state.rear_axle_velocity_2d
        vx = float(v.x)
        vy = float(v.y)
    except Exception:
        pass
    try:
        a = ego.dynamic_car_state.rear_axle_acceleration_2d
        ax = float(a.x)
        ay = float(a.y)
    except Exception:
        pass

    return _EgoKinematics(x=x, y=y, heading=heading, vx=vx, vy=vy, ax=ax, ay=ay)


def _build_x_from_history(ego_states: Deque[EgoState], *, history_len: int = 21) -> np.ndarray:
    """Build x matching `ShardedNpzDataset` (see PR #25)."""
    states = list(ego_states)
    if not states:
        raise RuntimeError("PlannerInput.history.ego_states is empty")

    cur = states[-1]
    cur_k = _ego_kinematics(cur)

    R = _rot2d(-cur_k.heading)
    v_local = R @ np.array([cur_k.vx, cur_k.vy], dtype=np.float64)

    ego_current_state = np.array(
        [
            0.0,
            0.0,
            float(np.cos(cur_k.heading)),
            float(np.sin(cur_k.heading)),
            float(v_local[0]),
            float(v_local[1]),
            float(cur_k.ax),
            float(cur_k.ay),
            1.0,
            1.0,
        ],
        dtype=np.float32,
    )

    past = np.zeros((history_len, 3), dtype=np.float32)

    if len(states) > 1:
        if len(states) >= history_len:
            idxs = np.linspace(len(states) - history_len, len(states) - 1, history_len).round().astype(int)
        else:
            idxs = np.linspace(0, len(states) - 1, len(states)).round().astype(int)

        offset = history_len - len(idxs)
        for j, idx in enumerate(idxs):
            s = states[int(idx)]
            k = _ego_kinematics(s)
            dx_dy = R @ np.array([k.x - cur_k.x, k.y - cur_k.y], dtype=np.float64)
            d_heading = _wrap_pi(k.heading - cur_k.heading)
            past[offset + j, 0] = float(dx_dy[0])
            past[offset + j, 1] = float(dx_dy[1])
            past[offset + j, 2] = float(d_heading)

        past[-1] = 0.0

    x = np.concatenate([ego_current_state.reshape(-1), past.reshape(-1)], axis=0)
    assert x.shape == (10 + history_len * 3,), f"x shape {x.shape}"
    return x.astype(np.float32)


class DiffusionPlannerCkpt(AbstractPlanner):
    """nuPlan planner that loads a repo-local checkpoint and predicts a trajectory."""

    def __init__(
        self,
        ckpt_path: str,
        past_trajectory_sampling: TrajectorySampling,
        future_trajectory_sampling: TrajectorySampling,
        config: object | None = None,  # accepted for hydra compatibility (unused)
        device: str = "cpu",
        enable_ema: bool = True,  # accepted for hydra compatibility (unused)
        **_: object,
    ):
        self._ckpt_path = str(ckpt_path)
        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling
        self._device = device

        if self._device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("device=cuda requested but torch.cuda.is_available() is False")

        self._model: Optional[SimpleFutureMLP] = None

        # matches export: 80 @ 10Hz = 8s
        self._T = int(self._future_trajectory_sampling.num_poses)
        self._future_horizon = float(self._future_trajectory_sampling.time_horizon)
        self._step_interval = self._future_horizon / self._T

        self._load_checkpoint_or_raise()

    def _load_checkpoint_or_raise(self) -> None:
        ckpt = torch.load(self._ckpt_path, map_location="cpu")
        if not isinstance(ckpt, dict) or "model_state" not in ckpt:
            raise RuntimeError(f"Unsupported ckpt format: expected dict with 'model_state', got keys={list(getattr(ckpt,'keys',lambda:[])())}")

        x_dim = 10 + 21 * 3
        model = SimpleFutureMLP(x_dim=x_dim, T=self._T)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()

        if self._device == "cuda":
            model = model.cuda()

        self._model = model

    def initialize(self, initialization: PlannerInitialization) -> None:
        # nothing else required
        return

    def name(self) -> str:
        return "diffusion_planner_ckpt"

    def observation_type(self) -> type[Observation]:
        # We only need ego history; detections are unused for now.
        return DetectionsTracks

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        if self._model is None:
            raise RuntimeError("Model is not loaded")

        ego_history = current_input.history.ego_states
        x = _build_x_from_history(ego_history)
        x_t = torch.from_numpy(x).unsqueeze(0)
        if self._device == "cuda":
            x_t = x_t.cuda(non_blocking=True)

        with torch.no_grad():
            y = self._model(x_t)

        y_np = y.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(y_np).all():
            raise RuntimeError("Non-finite trajectory prediction (NaN/Inf) from model")
        if y_np.shape != (self._T, 3):
            raise RuntimeError(f"Unexpected predicted shape: {y_np.shape}, expected ({self._T},3)")

        # y_np is relative poses [x,y,heading] in ego frame.
        states = transform_predictions_to_states(
            predicted_poses=y_np,
            ego_history=ego_history,
            future_horizon=self._future_horizon,
            step_interval=self._step_interval,
            include_ego_state=True,
        )
        return InterpolatedTrajectory(states)
