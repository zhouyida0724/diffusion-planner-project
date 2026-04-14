import warnings
import os
from dataclasses import fields
from types import SimpleNamespace
from typing import Deque, Dict, List, Type

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

# Paper model (trained in this repo)
from src.methods.diffusion_planner.paper.model.diffusion_planner import (  # type: ignore
    PaperDiffusionPlanner,
    PaperModelConfig,
)


class PaperDiTDpmPlanner(AbstractPlanner):
    """nuPlan planner wrapper for the repo's paper_dit_dpm model checkpoints."""

    def __init__(
        self,
        ckpt_path: str,
        diffusion_steps: int,
        past_trajectory_sampling: TrajectorySampling,
        future_trajectory_sampling: TrajectorySampling,
        device: str = "cuda",
    ):
        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"

        self._ckpt_path = ckpt_path
        self._diffusion_steps = int(diffusion_steps)
        self._device = device

        self._future_horizon = future_trajectory_sampling.time_horizon
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses
        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self._model: PaperDiffusionPlanner | None = None
        self._paper_cfg: PaperModelConfig | None = None
        self._data_processor: DataProcessor | None = None

    def name(self) -> str:
        return "paper_dit_dpm_planner"

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids

        device = torch.device(self._device)
        ckpt = torch.load(self._ckpt_path, map_location=device)
        if not isinstance(ckpt, dict) or "model_state" not in ckpt:
            raise RuntimeError(f"Unexpected checkpoint format at {self._ckpt_path}")

        # Build PaperModelConfig from checkpoint's paper_config (filter dataclass fields only)
        pc = ckpt.get("paper_config", {})
        if not isinstance(pc, dict):
            pc = {}
        valid = {f.name for f in fields(PaperModelConfig)}
        cfg_kwargs = {k: v for k, v in pc.items() if k in valid}
        # Route length is not part of PaperModelConfig, but DataProcessor needs it.
        paper_cfg = PaperModelConfig(**cfg_kwargs)

        model = PaperDiffusionPlanner(paper_cfg).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()

        # Build DataProcessor adapter config (needs route_len)
        adapter_cfg = SimpleNamespace(
            agent_num=int(paper_cfg.agent_num),
            static_objects_num=int(paper_cfg.static_objects_num),
            lane_num=int(paper_cfg.lane_num),
            lane_len=int(paper_cfg.lane_len),
            route_num=int(paper_cfg.route_num),
            route_len=int(paper_cfg.lane_len),
        )
        data_processor = DataProcessor(adapter_cfg)

        self._model = model
        self._paper_cfg = paper_cfg
        self._data_processor = data_processor
        self._initialization = initialization

    def _planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        assert self._data_processor is not None
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        inputs = self._data_processor.observation_adapter(
            history,
            traffic_light_data,
            self._map_api,
            self._route_roadblock_ids,
            device=self._device,
        )
        # Paper model expects diffusion_steps in the input dict
        inputs["diffusion_steps"] = int(self._diffusion_steps)
        return inputs

    @staticmethod
    def _outputs_to_trajectory(
        outputs: Dict[str, torch.Tensor],
        ego_state_history: Deque[EgoState],
        future_horizon: float,
        step_interval: float,
    ) -> List[InterpolatableState]:
        """Convert model outputs to nuPlan trajectory states.

        Note: The paper decoder predicts (future_len + 1) states, where index 0 is the *current* state.
        nuPlan's closed-loop expects exactly `future_horizon / step_interval` future states.
        """

        # outputs['prediction']: [B, P, T, 4] (x,y,cos,sin) in ego frame
        pred = outputs["prediction"][0, 0].detach().float().cpu().numpy().astype(np.float64)  # [T,4]

        expected_T = int(round(float(future_horizon) / float(step_interval)))
        if pred.shape[0] == expected_T + 1:
            # Drop the t=0 (current state) entry.
            pred = pred[1:]
        elif pred.shape[0] != expected_T:
            # Best-effort fallback: slice/pad to expected length.
            if pred.shape[0] > expected_T:
                pred = pred[:expected_T]
            else:
                pad = np.repeat(pred[-1:], expected_T - pred.shape[0], axis=0)
                pred = np.concatenate([pred, pad], axis=0)

        heading = np.arctan2(pred[:, 3], pred[:, 2])[..., None]
        traj = np.concatenate([pred[:, :2], heading], axis=-1)
        return transform_predictions_to_states(traj, ego_state_history, future_horizon, step_interval)

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        assert self._model is not None
        inputs = self._planner_input_to_model_inputs(current_input)
        _, outputs = self._model(inputs)

        # Lightweight debug to validate output space/time alignment.
        # Enable with DP_RUNTIME_DEBUG=1. Limit prints with DP_RUNTIME_DEBUG_K.
        if os.getenv("DP_RUNTIME_DEBUG", "0") not in ("", "0", "false", "False"):
            k = int(os.getenv("DP_RUNTIME_DEBUG_K", "3"))
            t = getattr(self, "_debug_ticks", 0)
            if t < k:
                try:
                    pred = outputs["prediction"][0, 0].detach().float().cpu().numpy()  # [T,4]
                    # show both t=0 and first future step for sanity
                    p0 = pred[0].tolist() if pred.shape[0] > 0 else []
                    p1 = pred[1].tolist() if pred.shape[0] > 1 else []
                    print(
                        f"[PAPER_PLANNER_DEBUG] tick={t} pred_shape={tuple(pred.shape)} "
                        f"p0_xy=({p0[0]:.3f},{p0[1]:.3f}) p1_xy=({p1[0]:.3f},{p1[1]:.3f})"
                    )
                except Exception:
                    pass
            setattr(self, "_debug_ticks", t + 1)

        trajectory = InterpolatedTrajectory(
            trajectory=self._outputs_to_trajectory(outputs, current_input.history.ego_states, self._future_horizon, self._step_interval)
        )
        return trajectory
