"""nuPlan DiffusionPlanner adapter.

This repository has two "diffusion planner" lineages:

1) Legacy upstream diffusion model under `nuplan.diffusion_planner.model.*`.
2) Our minimal training skeleton under `src.methods.diffusion_planner`, which
   writes checkpoints like:
     {
       'model_state': <state_dict>,
       'cfg': {...},
       ...
     }

To keep the nuPlan Hydra interface stable (`planner=diffusion_planner`), this
module auto-detects checkpoint format at runtime:
- if the checkpoint contains `model_state`, we load the lightweight
  `SimpleFutureMLP` and run a minimal feature extractor compatible with our
  exported NPZ format (`ego_current_state` + `ego_past`).
- otherwise we fall back to the legacy diffusion model.

If anything fails during MLP setup/inference, we fall back to a simple
constant-velocity trajectory (so simulation can still proceed and failures are
explicit in logs).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Type

import numpy as np
import torch

warnings.filterwarnings("ignore")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from .data_process.data_processor import DataProcessor
from .model.diffusion_planner import Diffusion_Planner
from .utils.config import Config


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
    """Extract ego pose/vel/acc in global frame with best-effort attribute access."""

    # Position/heading: prefer rear axle pose to match nuPlan defaults.
    try:
        x = float(ego.rear_axle.x)
        y = float(ego.rear_axle.y)
        heading = float(ego.rear_axle.heading)
    except Exception:
        x = float(ego.center.x)
        y = float(ego.center.y)
        heading = float(ego.center.heading)

    # Velocity / acceleration: may vary by nuPlan version.
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


def _build_skeleton_mlp_x(
    ego_states: Deque[EgoState], *, history_len: int = 21
) -> np.ndarray:
    """Build x matching `src.methods.diffusion_planner.data.npz_dataset`.

    x = concat(
      ego_current_state (10,),
      ego_past flattened (history_len, 3)
    )

    ego_current_state = [
      0, 0,
      cos(heading), sin(heading),
      v_local_x, v_local_y,
      ax, ay,
      1, 1
    ]

    ego_past[i] = [dx, dy, d_heading] in ego frame relative to current.
    The last element corresponds to current state -> [0,0,0].
    """

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

    # Choose `history_len` samples ending at current. If not enough history,
    # pad with zeros and keep the last as current.
    past = np.zeros((history_len, 3), dtype=np.float32)

    if len(states) == 1:
        # no past; keep zeros
        x = np.concatenate([ego_current_state.reshape(-1), past.reshape(-1)], axis=0)
        return x.astype(np.float32)

    # indices evenly spaced over available history
    if len(states) >= history_len:
        idxs = np.linspace(len(states) - history_len, len(states) - 1, history_len).round().astype(int)
    else:
        idxs = np.linspace(0, len(states) - 1, len(states)).round().astype(int)

    # fill from the end
    offset = history_len - len(idxs)
    for j, idx in enumerate(idxs):
        s = states[int(idx)]
        k = _ego_kinematics(s)
        dx_dy = R @ np.array([k.x - cur_k.x, k.y - cur_k.y], dtype=np.float64)
        d_heading = _wrap_pi(k.heading - cur_k.heading)
        past[offset + j, 0] = float(dx_dy[0])
        past[offset + j, 1] = float(dx_dy[1])
        past[offset + j, 2] = float(d_heading)

    # Force last element to be exactly current.
    past[-1] = 0.0

    x = np.concatenate([ego_current_state.reshape(-1), past.reshape(-1)], axis=0)
    return x.astype(np.float32)


class DiffusionPlanner(AbstractPlanner):
    def __init__(
        self,
        config: Config,
        ckpt_path: str,
        past_trajectory_sampling: TrajectorySampling,
        future_trajectory_sampling: TrajectorySampling,
        enable_ema: bool = True,
        device: str = "cpu",
    ):
        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"

        self._future_horizon = future_trajectory_sampling.time_horizon
        self._step_interval = (
            future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses
        )

        self._config = config
        self._ckpt_path = ckpt_path
        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self._ema_enabled = enable_ema
        self._device = device

        # Legacy components (initialized lazily if needed)
        self._legacy_planner: Optional[torch.nn.Module] = None
        self._legacy_data_processor: Optional[DataProcessor] = None
        self.observation_normalizer = getattr(config, "observation_normalizer", None)

        # Skeleton MLP policy (initialized if ckpt has 'model_state')
        self._mlp_policy: Optional[torch.nn.Module] = None

        self._mode: str = "uninitialized"  # 'skeleton_mlp' | 'legacy_diffusion' | 'fallback'

    def name(self) -> str:
        return "diffusion_planner"

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids
        self._initialization = initialization

        if self._ckpt_path is None:
            print("[DiffusionPlanner] ckpt_path is None; using fallback")
            self._mode = "fallback"
            return

        ckpt: Dict = torch.load(self._ckpt_path, map_location=self._device)

        # ---- Our training skeleton checkpoint: {'model_state': ...} ----
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            try:
                from src.methods.diffusion_planner.models.simple_mlp import SimpleFutureMLP

                x_dim = 10 + 21 * 3
                T = 80
                model = SimpleFutureMLP(x_dim=x_dim, T=T)
                model.load_state_dict(ckpt["model_state"], strict=True)
                model.eval()
                model.to(self._device)
                self._mlp_policy = model
                self._mode = "skeleton_mlp"
                print(f"[DiffusionPlanner] loaded skeleton MLP checkpoint: {self._ckpt_path}")
                return
            except Exception as e:
                print(
                    f"[DiffusionPlanner] Failed to init skeleton MLP from ckpt {self._ckpt_path}: {e}. "
                    "Falling back to constant-velocity trajectory."
                )
                self._mode = "fallback"
                return

        # ---- Legacy diffusion checkpoint ----
        try:
            self._legacy_planner = Diffusion_Planner(self._config)
            self._legacy_data_processor = DataProcessor(self._config)

            state_dict: Dict = ckpt
            if self._ema_enabled and "ema_state_dict" in state_dict:
                state_dict = state_dict["ema_state_dict"]
            else:
                if "model" in state_dict.keys():
                    state_dict = state_dict["model"]
                elif "model_state_dict" in state_dict.keys():
                    state_dict = state_dict["model_state_dict"]

            # DDP: strip 'module.'
            model_state_dict = {
                k[len("module.") :]: v for k, v in state_dict.items() if k.startswith("module.")
            }
            if not model_state_dict:
                model_state_dict = state_dict

            self._legacy_planner.load_state_dict(model_state_dict, strict=False)
            self._legacy_planner.eval()
            self._legacy_planner.to(self._device)
            self._mode = "legacy_diffusion"
            print(f"[DiffusionPlanner] loaded legacy diffusion checkpoint: {self._ckpt_path}")
        except Exception as e:
            print(
                f"[DiffusionPlanner] Failed to init legacy diffusion from ckpt {self._ckpt_path}: {e}. "
                "Falling back to constant-velocity trajectory."
            )
            self._mode = "fallback"

    def _fallback_predictions(self, ego_state_history: Deque[EgoState]) -> np.ndarray:
        """Constant-velocity in ego frame: dx = v_local_x * t, dy=0, dheading=0."""
        cur_k = _ego_kinematics(list(ego_state_history)[-1])
        R = _rot2d(-cur_k.heading)
        v_local = R @ np.array([cur_k.vx, cur_k.vy], dtype=np.float64)
        v = float(v_local[0])

        num = int(self._future_trajectory_sampling.num_poses)
        dt = float(self._step_interval)
        t = (np.arange(num, dtype=np.float64) + 1.0) * dt
        dx = v * t
        dy = np.zeros_like(dx)
        dhead = np.zeros_like(dx)
        return np.stack([dx, dy, dhead], axis=-1)

    def _predictions_to_states(
        self, predictions: np.ndarray, ego_state_history: Deque[EgoState]
    ) -> List[InterpolatableState]:
        """predictions: [T, 3] = [dx, dy, d_heading] in ego frame."""
        predictions = predictions.astype(np.float64)
        return transform_predictions_to_states(
            predictions, ego_state_history, self._future_horizon, self._step_interval
        )

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        ego_hist = current_input.history.ego_states

        # ---- Skeleton MLP ----
        if self._mode == "skeleton_mlp" and self._mlp_policy is not None:
            try:
                x = _build_skeleton_mlp_x(ego_hist, history_len=21)
                x_t = torch.from_numpy(x[None, :]).to(device=self._device, dtype=torch.float32)
                with torch.no_grad():
                    y_hat = self._mlp_policy(x_t)[0].detach().cpu().numpy()  # [80, 3]

                # If nuPlan config requests a different number of poses, slice or interpolate.
                num = int(self._future_trajectory_sampling.num_poses)
                if y_hat.shape[0] != num:
                    # Simple linear interpolation in time for dx/dy/dhead
                    t_src = np.linspace(0.0, 1.0, y_hat.shape[0], dtype=np.float64)
                    t_dst = np.linspace(0.0, 1.0, num, dtype=np.float64)
                    y_new = np.zeros((num, 3), dtype=np.float64)
                    for k in range(3):
                        y_new[:, k] = np.interp(t_dst, t_src, y_hat[:, k].astype(np.float64))
                    y_hat = y_new

                states = self._predictions_to_states(y_hat, ego_hist)
                return InterpolatedTrajectory(trajectory=states)
            except Exception as e:
                print(
                    f"[DiffusionPlanner] skeleton MLP inference failed: {e}. "
                    "Falling back to constant-velocity trajectory."
                )
                self._mode = "fallback"

        # ---- Legacy diffusion ----
        if self._mode == "legacy_diffusion" and self._legacy_planner is not None:
            assert self._legacy_data_processor is not None
            inputs = self._legacy_data_processor.observation_adapter(
                current_input.history,
                list(current_input.traffic_light_data),
                self._map_api,
                self._route_roadblock_ids,
                self._device,
            )
            if self.observation_normalizer is not None:
                inputs = self.observation_normalizer(inputs)
            with torch.no_grad():
                _, outputs = self._legacy_planner(inputs)

            # legacy model output: outputs['prediction'][0, 0] -> (T,4) with [x,y,cos,sin]
            pred = outputs["prediction"][0, 0].detach().cpu().numpy().astype(np.float64)
            heading = np.arctan2(pred[:, 3], pred[:, 2])[..., None]
            pred = np.concatenate([pred[..., :2], heading], axis=-1)
            states = transform_predictions_to_states(
                pred, ego_hist, self._future_horizon, self._step_interval
            )
            return InterpolatedTrajectory(trajectory=states)

        # ---- Fallback ----
        pred = self._fallback_predictions(ego_hist)
        states = self._predictions_to_states(pred, ego_hist)
        return InterpolatedTrajectory(trajectory=states)
