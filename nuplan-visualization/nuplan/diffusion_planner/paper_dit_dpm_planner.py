import warnings
import os
import bisect
import sqlite3
from dataclasses import fields
from types import SimpleNamespace
from typing import Deque, Dict, List, Type, Any

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

    # We need scenario access to do DB-backed feature extraction consistent with training/export.
    requires_scenario = True

    def __init__(
        self,
        ckpt_path: str,
        diffusion_steps: int,
        past_trajectory_sampling: TrajectorySampling,
        future_trajectory_sampling: TrajectorySampling,
        device: str = "cuda",
        scenario: object | None = None,  # injected by nuPlan when requires_scenario=True
        **_: Any,  # hydra compatibility
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
        # NOTE: We intentionally do NOT use nuplan-visualization's DataProcessor for inference,
        # because its feature semantics differ from our training/export extractor.
        self._data_processor: DataProcessor | None = None

        self._scenario = scenario
        self._feature_conn: sqlite3.Connection | None = None
        self._db_path: str | None = None
        self._scene_token_hex: str | None = None
        self._log_token: bytes | None = None
        self._ego_pose_timestamps: list[int] | None = None
        self._ts_to_frame_index: dict[int, int] | None = None

    def _get_feature_conn(self) -> sqlite3.Connection:
        if self._feature_conn is None:
            if not self._db_path:
                raise RuntimeError("DB path not initialized for feature extraction")
            self._feature_conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            self._feature_conn.row_factory = sqlite3.Row
        return self._feature_conn

    def _init_db_feature_extractor(self) -> None:
        """Initialize DB-backed context so we can call extract_single_frame.extract_features per tick."""
        if self._db_path is not None:
            return

        sc = self._scenario
        if sc is None:
            raise RuntimeError("PaperDiTDpmPlanner requires scenario but got None")

        # Locate DB path from scenario object.
        db_path = None
        for attr in ["log_file_load_path", "_log_file_load_path"]:
            if hasattr(sc, attr):
                db_path = getattr(sc, attr)
                break
        if not db_path:
            raise RuntimeError("Unable to locate scenario DB path (log_file_load_path)")
        self._db_path = str(db_path)

        # Resolve initial lidar token hex from scenario.
        initial_lidar_token_hex = None
        for attr in ["initial_lidar_token", "_initial_lidar_token"]:
            if hasattr(sc, attr):
                initial_lidar_token_hex = getattr(sc, attr)
                break
        if not initial_lidar_token_hex:
            raise RuntimeError("Unable to locate scenario initial_lidar_token")
        initial_lidar_token_hex = str(initial_lidar_token_hex)

        conn = self._get_feature_conn()
        cur = conn.cursor()

        # scene_token from lidar_pc.token
        row = cur.execute(
            "select scene_token from lidar_pc where token = ? limit 1", (bytes.fromhex(initial_lidar_token_hex),)
        ).fetchone()
        if not row or row[0] is None:
            raise RuntimeError(f"Unable to resolve scene_token from initial lidar_pc token={initial_lidar_token_hex}")
        scene_token_bytes = row[0]
        self._scene_token_hex = scene_token_bytes.hex() if isinstance(scene_token_bytes, (bytes, bytearray)) else str(scene_token_bytes)

        # log_token from scene
        row = cur.execute("select log_token from scene where token=? limit 1", (scene_token_bytes,)).fetchone()
        if not row or row[0] is None:
            raise RuntimeError(f"Unable to resolve log_token from scene token={self._scene_token_hex}")
        self._log_token = row[0]

        # Precompute timestamp->frame_index mapping for this log.
        rows = cur.execute(
            "select timestamp from ego_pose where log_token=? order by timestamp", (self._log_token,)
        ).fetchall()
        ts = [int(r[0]) for r in rows]
        if not ts:
            # fallback: global ego_pose ordering
            rows = cur.execute("select timestamp from ego_pose order by timestamp").fetchall()
            ts = [int(r[0]) for r in rows]
        self._ego_pose_timestamps = ts
        self._ts_to_frame_index = {t: i for i, t in enumerate(ts)}

    def _timestamp_to_frame_index(self, timestamp_us: int) -> int:
        assert self._ego_pose_timestamps is not None
        assert self._ts_to_frame_index is not None
        t = int(timestamp_us)
        if t in self._ts_to_frame_index:
            return int(self._ts_to_frame_index[t])
        # Nearest match (robust to tiny drift)
        ts = self._ego_pose_timestamps
        j = bisect.bisect_left(ts, t)
        if j <= 0:
            return 0
        if j >= len(ts):
            return len(ts) - 1
        before = ts[j - 1]
        after = ts[j]
        return j - 1 if abs(t - before) <= abs(after - t) else j

    def _extract_inputs_ours(self, *, timestamp_us: int) -> Dict[str, torch.Tensor]:
        """Extract conditioning features using the same extractor used for training/export."""
        self._init_db_feature_extractor()
        assert self._scene_token_hex is not None

        from src.platform.nuplan.features.extract_single_frame import extract_features

        frame_index = self._timestamp_to_frame_index(int(timestamp_us))
        feats = extract_features(
            self._get_feature_conn(),
            self._map_api,
            self._scene_token_hex,
            int(frame_index),
            debug_log=False,
            routing_mode="auto",
        )

        # Convert to batched torch tensors (B=1) on desired device.
        device = torch.device(self._device)
        out: Dict[str, torch.Tensor] = {}
        for k, v in feats.items():
            if isinstance(v, np.ndarray) and v.dtype == np.bool_:
                out[k] = torch.tensor(v, dtype=torch.bool, device=device).unsqueeze(0)
            else:
                out[k] = torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
        out["diffusion_steps"] = torch.tensor(int(self._diffusion_steps), dtype=torch.int64, device=device)
        return out

    def name(self) -> str:
        return "paper_dit_dpm_planner"

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids

        device = torch.device(self._device)
        # PyTorch newer versions default `weights_only=True`, which breaks checkpoints
        # that include non-tensor metadata (we rely on paper_config/cfg).
        try:
            ckpt = torch.load(self._ckpt_path, map_location=device, weights_only=False)
        except TypeError:
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

        # Prepare DB-backed extraction for this scenario.
        # This will throw early if scenario/db is not usable.
        self._init_db_feature_extractor()

    def _planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        # Use DB-backed extractor aligned with training/export semantics.
        # Current timestamp comes from the simulation history's current ego state.
        ego_state = planner_input.history.current_state[0]
        timestamp_us = int(ego_state.time_point.time_us)
        return self._extract_inputs_ours(timestamp_us=timestamp_us)

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
