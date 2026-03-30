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

import os
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
from src.methods.diffusion_planner.paper.model.diffusion_planner import PaperDiffusionPlanner, PaperModelConfig

# Runtime feature extraction utilities (reuse export logic)
from src.platform.nuplan.features.extract_single_frame import extract_lanes, extract_route_lanes


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


def _safe_obj_id(obj: object) -> str | None:
    """Best-effort stable id for a tracked object across history frames."""
    for attr in ("track_token", "token", "track_id", "uuid", "id"):
        v = getattr(obj, attr, None)
        if v is None:
            continue
        try:
            if isinstance(v, (bytes, bytearray)):
                return v.hex()
            return str(v)
        except Exception:
            continue
    return None


def _safe_xy_heading_v_wl_type(
    obj: object,
) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None, float | None, np.ndarray]:
    """Extract (x,y,heading,vx,vy,width,length,type_onehot3) from a nuPlan tracked object."""
    x = y = heading = vx = vy = width = length = None

    # pose
    try:
        c = getattr(obj, "center")
        x = float(getattr(c, "x"))
        y = float(getattr(c, "y"))
        heading = float(getattr(c, "heading"))
    except Exception:
        try:
            box = getattr(obj, "box")
            c = getattr(box, "center")
            x = float(getattr(c, "x"))
            y = float(getattr(c, "y"))
            heading = float(getattr(c, "heading"))
        except Exception:
            pass

    # velocity
    for attr in ("velocity", "velocity_2d", "predicted_velocity"):
        try:
            v = getattr(obj, attr)
            vx = float(getattr(v, "x"))
            vy = float(getattr(v, "y"))
            break
        except Exception:
            continue

    # size
    try:
        box = getattr(obj, "box")
        width = float(getattr(box, "width"))
        length = float(getattr(box, "length"))
    except Exception:
        try:
            width = float(getattr(obj, "width"))
            length = float(getattr(obj, "length"))
        except Exception:
            pass

    # type onehot(3): vehicle, pedestrian, bicycle
    t = getattr(obj, "tracked_object_type", None)
    if t is None:
        t = getattr(obj, "object_type", None)
    t_name = None
    try:
        t_name = str(getattr(t, "name", t))
    except Exception:
        t_name = None

    onehot = np.zeros((3,), dtype=np.float32)
    if t_name:
        n = str(t_name).upper()
        if "VEH" in n or "CAR" in n:
            onehot[0] = 1.0
        elif "PED" in n:
            onehot[1] = 1.0
        elif "BIC" in n or "CYC" in n:
            onehot[2] = 1.0

    return x, y, heading, vx, vy, width, length, onehot


def _build_neighbor_agents_past_from_history(
    *,
    history: object,
    cur_ego: _EgoKinematics,
    predicted_neighbor_num: int,
    time_len: int,
    runtime_debug: bool = False,
) -> tuple[np.ndarray, int]:
    """Build neighbor_agents_past: [P, time_len, 11] in current ego frame."""

    Pn = int(predicted_neighbor_num)
    V = int(time_len)
    out = np.zeros((Pn, V, 11), dtype=np.float32)

    obs_deque = getattr(history, "observations", None)
    if not obs_deque:
        return out, 0

    obs_list = list(obs_deque)
    if not obs_list:
        return out, 0

    obs_tail = obs_list[-V:]
    aligned: list[object | None] = [None] * V
    offset = V - len(obs_tail)
    for i, o in enumerate(obs_tail):
        aligned[offset + i] = o

    last_obs = aligned[-1]
    if not isinstance(last_obs, DetectionsTracks):
        if runtime_debug:
            try:
                print(f"[DP_RUNTIME_DEBUG] neighbors: last_obs not DetectionsTracks: {type(last_obs)}")
            except Exception:
                pass
        return out, 0

    tracked = getattr(last_obs, "tracked_objects", None)
    if tracked is None:
        return out, 0

    candidates: list[object] = []
    # prefer vehicles
    try:
        from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

        try:
            candidates = list(tracked.get_tracked_objects_of_type(TrackedObjectType.VEHICLE))
        except Exception:
            candidates = []
    except Exception:
        candidates = []

    if not candidates:
        for attr in ("tracked_objects", "get_tracked_objects"):
            try:
                v = getattr(tracked, attr)
                candidates = list(v() if callable(v) else v)
                break
            except Exception:
                continue

    if not candidates:
        return out, 0

    scored: list[tuple[float, str]] = []
    for obj in candidates:
        oid = _safe_obj_id(obj)
        if oid is None:
            continue
        x, y, heading, vx, vy, width, length, onehot = _safe_xy_heading_v_wl_type(obj)
        if x is None or y is None:
            continue
        d2 = float((x - cur_ego.x) ** 2 + (y - cur_ego.y) ** 2)
        scored.append((d2, oid))

    if not scored:
        return out, 0

    scored.sort(key=lambda t: t[0])
    chosen_ids = [oid for _, oid in scored[:Pn]]

    frame_maps: list[dict[str, object]] = []
    for o in aligned:
        m: dict[str, object] = {}
        if isinstance(o, DetectionsTracks):
            tr = getattr(o, "tracked_objects", None)
            if tr is not None:
                objs: list[object] = []
                for attr in ("tracked_objects", "get_tracked_objects"):
                    try:
                        v = getattr(tr, attr)
                        objs = list(v() if callable(v) else v)
                        break
                    except Exception:
                        continue
                for obj in objs:
                    oid = _safe_obj_id(obj)
                    if oid is not None:
                        m[oid] = obj
        frame_maps.append(m)

    R = _rot2d(-cur_ego.heading)

    for p, oid in enumerate(chosen_ids):
        for t in range(V):
            obj = frame_maps[t].get(oid)
            if obj is None:
                continue
            x, y, heading, vx, vy, width, length, onehot = _safe_xy_heading_v_wl_type(obj)
            if x is None or y is None or heading is None:
                continue

            dx_dy = R @ np.array([x - cur_ego.x, y - cur_ego.y], dtype=np.float64)
            d_heading = _wrap_pi(float(heading) - cur_ego.heading)
            out[p, t, 0] = float(dx_dy[0])
            out[p, t, 1] = float(dx_dy[1])
            out[p, t, 2] = float(np.cos(d_heading))
            out[p, t, 3] = float(np.sin(d_heading))

            if vx is not None and vy is not None:
                vv = R @ np.array([vx, vy], dtype=np.float64)
                out[p, t, 4] = float(vv[0])
                out[p, t, 5] = float(vv[1])

            out[p, t, 6] = float(width) if width is not None else 0.0
            out[p, t, 7] = float(length) if length is not None else 0.0
            out[p, t, 8:11] = onehot

        # constant-hold backfill
        valid = np.any(out[p, :, :8] != 0.0, axis=1)
        if not np.any(valid):
            continue
        first = int(np.argmax(valid))
        for t in range(0, first):
            out[p, t] = out[p, first]
        for t in range(1, V):
            if not valid[t]:
                out[p, t] = out[p, t - 1]

    nonzero = int(np.count_nonzero(out))
    return out, nonzero


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
        sampling_steps: int = 10,
        **_: object,
    ):
        self._ckpt_path = str(ckpt_path)
        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling
        self._device = device

        if self._device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("device=cuda requested but torch.cuda.is_available() is False")

        self._model: Optional[torch.nn.Module] = None
        self._ckpt_kind: str = "mlp"
        self._sampling_steps = int(sampling_steps)

        # for runtime feature extraction
        self._map_api = None
        self._route_roadblock_ids = None
        self._init_done = False

        # Runtime debug (gated by env var)
        # Back-compat: DP_RUNTIME_DEBUG=1 enables all debug.
        # New flags:
        #  - DIFFPLANNER_DEBUG_COND=1: conditioning tensor stats
        #  - DIFFPLANNER_DEBUG_SAMPLER=1: diffusion sampler xT/x0 stats
        self._runtime_debug = os.getenv("DP_RUNTIME_DEBUG", "0") not in ("", "0", "false", "False")
        self._runtime_debug_k = int(os.getenv("DP_RUNTIME_DEBUG_K", os.getenv("DIFFPLANNER_DEBUG_K", "5")))
        self._runtime_debug_ticks = 0
        self._debug_cond = self._runtime_debug or (os.getenv("DIFFPLANNER_DEBUG_COND", "0") not in ("", "0", "false", "False"))
        self._debug_sampler = self._runtime_debug or (os.getenv("DIFFPLANNER_DEBUG_SAMPLER", "0") not in ("", "0", "false", "False"))

        # matches export: 80 @ 10Hz = 8s
        self._T = int(self._future_trajectory_sampling.num_poses)
        self._future_horizon = float(self._future_trajectory_sampling.time_horizon)
        self._step_interval = self._future_horizon / self._T

        self._load_checkpoint_or_raise()

    def _load_checkpoint_or_raise(self) -> None:
        ckpt = torch.load(self._ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state" not in ckpt and "model_state_dict" in ckpt:
            ckpt = dict(ckpt)
            ckpt["model_state"] = ckpt["model_state_dict"]

        if not isinstance(ckpt, dict) or "model_state" not in ckpt:
            raise RuntimeError(
                f"Unsupported ckpt format: expected dict with 'model_state', got keys={list(getattr(ckpt,'keys',lambda:[])())}"
            )

        kind = str(ckpt.get("kind", "mlp"))
        # Heuristic: paper-model checkpoints may omit `kind`.
        try:
            sd_keys = list(getattr(ckpt.get("model_state"), "keys", lambda: [])())
        except Exception:
            sd_keys = []
        if kind == "mlp" and any(("encoder.encoder.neighbor_encoder" in k) or ("decoder.decoder.dit" in k) for k in sd_keys):
            kind = PaperDiffusionPlanner.CKPT_KIND

        # Strip common DDP prefix + patch known key naming drifts.
        try:
            sd = ckpt.get("model_state")
            if isinstance(sd, dict) and any(str(k).startswith("module.") for k in sd.keys()):
                sd = {str(k)[len("module.") :]: v for k, v in sd.items()}
            if kind == PaperDiffusionPlanner.CKPT_KIND and isinstance(sd, dict):
                remapped = {}
                for k, v in sd.items():
                    nk = str(k)
                    nk = nk.replace(".tokens_mlp.", ".mlp_tokens.")
                    nk = nk.replace(".channels_mlp.", ".mlp_channels.")
                    nk = nk.replace("route_encoder.Mixer.", "route_encoder.mixer.")
                    remapped[nk] = v
                sd = remapped
            ckpt = dict(ckpt)
            ckpt["model_state"] = sd
        except Exception:
            pass

        self._ckpt_kind = kind

        if kind == PaperDiffusionPlanner.CKPT_KIND:
            paper_cfg_dict = ckpt.get("paper_config") or ckpt.get("paper_cfg") or ckpt.get("paper_model_cfg")
            if not isinstance(paper_cfg_dict, dict):
                paper_cfg_dict = {}
            paper_cfg_dict = dict(paper_cfg_dict)
            paper_cfg_dict["device"] = self._device
            paper_cfg = PaperModelConfig(**{k: v for k, v in paper_cfg_dict.items() if k in PaperModelConfig.__annotations__})
            model = PaperDiffusionPlanner(paper_cfg)
            model.load_state_dict(ckpt["model_state"], strict=True)
            model.eval()
        else:
            # legacy baseline: SimpleFutureMLP
            x_dim = 10 + 21 * 3
            model = SimpleFutureMLP(x_dim=x_dim, T=self._T)
            model.load_state_dict(ckpt["model_state"], strict=True)
            model.eval()

        if self._device == "cuda":
            model = model.cuda()

        self._model = model

    def initialize(self, initialization: PlannerInitialization) -> None:
        # Save map + route for runtime feature extraction.
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids
        self._init_done = True
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

        neighbor_agents_past_nonzero_count: int = 0

        if self._ckpt_kind == PaperDiffusionPlanner.CKPT_KIND:
            assert isinstance(self._model, PaperDiffusionPlanner)
            if not self._init_done or self._map_api is None:
                raise RuntimeError("Planner.initialize() was not called or map_api missing")

            cfg = self._model.config
            B = 1
            Pn = int(cfg.predicted_neighbor_num)
            time_len = int(cfg.time_len)
            future_len = int(cfg.future_len)
            lane_len = int(cfg.lane_len)

            # Runtime route/lanes extraction (matches export logic, not a placeholder).
            cur = ego_history[-1]
            cur_k = _ego_kinematics(cur)
            from nuplan.common.actor_state.state_representation import Point2D

            point = Point2D(cur_k.x, cur_k.y)

            lanes_np, lanes_avails_np, lanes_speed_np, lanes_has_speed_np = extract_lanes(
                point=point,
                map_api=self._map_api,
                radius=150,
                max_lanes=int(cfg.lane_num),
                ego_heading=cur_k.heading,
                traffic_light_data=None,
            )
            route_lanes_np, route_lanes_avails_np, route_speed_np, route_has_speed_np = extract_route_lanes(
                point=point,
                map_api=self._map_api,
                radius=150,
                max_route_lanes=int(cfg.route_num),
                ego_heading=cur_k.heading,
                traffic_light_data=None,
                route_roadblock_ids=list(self._route_roadblock_ids) if self._route_roadblock_ids else None,
            )

            ego_current_state = torch.from_numpy(x[:10]).view(1, 10)

            dbg_should_print = self._runtime_debug and (self._runtime_debug_ticks < self._runtime_debug_k)

            # Neighbors: minimal runtime track history from DetectionsTracks.
            # Paper model expects `agent_num = 1 + predicted_neighbor_num` and may include ego as slot 0.
            agent_num = int(getattr(cfg, "agent_num", 1 + Pn))
            neighbor_all = np.zeros((agent_num, time_len, 11), dtype=np.float32)

            # Fill ego history into slot 0.
            ego_states = list(current_input.history.ego_states)
            ego_tail = ego_states[-time_len:]
            aligned_ego: list[EgoState | None] = [None] * time_len
            ego_off = time_len - len(ego_tail)
            for i, s in enumerate(ego_tail):
                aligned_ego[ego_off + i] = s

            R0 = _rot2d(-cur_k.heading)
            for t, s in enumerate(aligned_ego):
                if s is None:
                    continue
                k = _ego_kinematics(s)
                dx_dy = R0 @ np.array([k.x - cur_k.x, k.y - cur_k.y], dtype=np.float64)
                d_heading = _wrap_pi(k.heading - cur_k.heading)
                vv = R0 @ np.array([k.vx, k.vy], dtype=np.float64)
                neighbor_all[0, t, 0] = float(dx_dy[0])
                neighbor_all[0, t, 1] = float(dx_dy[1])
                neighbor_all[0, t, 2] = float(np.cos(d_heading))
                neighbor_all[0, t, 3] = float(np.sin(d_heading))
                neighbor_all[0, t, 4] = float(vv[0])
                neighbor_all[0, t, 5] = float(vv[1])
                neighbor_all[0, t, 6] = 1.8
                neighbor_all[0, t, 7] = 4.5
                neighbor_all[0, t, 8] = 1.0

            # Fill tracked neighbor vehicles into slots [1:1+Pn].
            neighbor_np, neighbor_nz = _build_neighbor_agents_past_from_history(
                history=current_input.history,
                cur_ego=cur_k,
                predicted_neighbor_num=Pn,
                time_len=time_len,
                runtime_debug=dbg_should_print,
            )
            hi = min(Pn, max(0, agent_num - 1))
            if hi > 0:
                neighbor_all[1 : 1 + hi] = neighbor_np[:hi]

            neighbor_agents_past = torch.from_numpy(neighbor_all).view(B, agent_num, time_len, 11)
            neighbor_agents_past_nonzero_count = int(np.count_nonzero(neighbor_all))

            neighbor_agents_future = torch.zeros((B, Pn, future_len, 3), dtype=torch.float32)
            static_objects = torch.zeros((B, int(cfg.static_objects_num), int(cfg.static_objects_state_dim)), dtype=torch.float32)

            lanes = torch.from_numpy(lanes_np).view(1, int(cfg.lane_num), lane_len, 12)
            lanes_speed_limit = torch.from_numpy(lanes_speed_np).view(1, int(cfg.lane_num), 1)
            lanes_has_speed_limit = torch.from_numpy(lanes_has_speed_np).view(1, int(cfg.lane_num), 1)

            route_lanes = torch.from_numpy(route_lanes_np).view(1, int(cfg.route_num), lane_len, 12)
            route_lanes_speed_limit = torch.from_numpy(route_speed_np).view(1, int(cfg.route_num), 1)
            route_lanes_has_speed_limit = torch.from_numpy(route_has_speed_np).view(1, int(cfg.route_num), 1)

            inputs = {
                "ego_current_state": ego_current_state,
                "neighbor_agents_past": neighbor_agents_past,
                "neighbor_agents_future": neighbor_agents_future,
                "ego_agent_future": torch.zeros((B, future_len, 3), dtype=torch.float32),
                "static_objects": static_objects,
                "lanes": lanes,
                "lanes_speed_limit": lanes_speed_limit,
                "lanes_has_speed_limit": lanes_has_speed_limit,
                "route_lanes": route_lanes,
                "route_lanes_speed_limit": route_lanes_speed_limit,
                "route_lanes_has_speed_limit": route_lanes_has_speed_limit,
            }
            if self._device == "cuda":
                inputs = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in inputs.items()}

            # Optional sampler debug: expose raw (x,y,cos,sin) before heading conversion.
            ego_x0 = ego_y0 = ego_cos0 = ego_sin0 = None

            with torch.no_grad():
                if self._debug_sampler:
                    # Run full forward pass to access decoder prediction [B,P,T,4].
                    self._model.eval()
                    _enc, dec = self._model(inputs)
                    pred = dec.get("prediction")
                    if pred is None:
                        raise RuntimeError("Paper model decoder did not return prediction")
                    ego = pred[:, 0]  # [B,T,4]
                    ego_x0 = ego[..., 0]
                    ego_y0 = ego[..., 1]
                    ego_cos0 = ego[..., 2]
                    ego_sin0 = ego[..., 3]
                    ego_heading = torch.atan2(ego_sin0, ego_cos0)
                    y = torch.stack([ego_x0[0], ego_y0[0], ego_heading[0]], dim=-1)  # [T,3]
                else:
                    y = self._model.sample_trajectory(inputs, diffusion_steps=self._sampling_steps)  # [T,3]

            y_np = y.detach().cpu().numpy().astype(np.float32)
        else:
            with torch.no_grad():
                y = self._model(x_t)
            y_np = y.squeeze(0).detach().cpu().numpy().astype(np.float32)

        # --- Runtime debug (first K ticks only, gated by env vars) ---
        if (self._runtime_debug or self._debug_cond or self._debug_sampler) and (self._runtime_debug_ticks < self._runtime_debug_k):
            try:
                xy = y_np[:, :2]
                dxy = xy[1:] - xy[:-1]
                seg_lens = np.sqrt((dxy**2).sum(axis=1))
                path_len = float(seg_lens.sum())
                start_end = float(np.sqrt(((xy[-1] - xy[0]) ** 2).sum()))

                parts = [
                    f"tick={self._runtime_debug_ticks}",
                    f"sampling_steps={int(self._sampling_steps)}",
                    f"neighbor_agents_past_nonzero_count={int(neighbor_agents_past_nonzero_count)}",
                    f"traj_path_len={path_len:.3f}",
                    f"traj_start_end_dist={start_end:.3f}",
                ]

                if self._debug_sampler and (self._ckpt_kind == PaperDiffusionPlanner.CKPT_KIND) and (ego_x0 is not None):
                    try:
                        # ego_x0 etc are [B,T]
                        x0 = ego_x0[0].detach().cpu().numpy()
                        y0 = ego_y0[0].detach().cpu().numpy()
                        cos0 = ego_cos0[0].detach().cpu().numpy()
                        sin0 = ego_sin0[0].detach().cpu().numpy()
                        norm = (cos0 ** 2 + sin0 ** 2) ** 0.5
                        parts += [
                            f"x0_x_minmax=[{float(x0.min()):.3f},{float(x0.max()):.3f}]",
                            f"x0_y_minmax=[{float(y0.min()):.3f},{float(y0.max()):.3f}]",
                            f"x0_cossin_norm_minmax=[{float(norm.min()):.3f},{float(norm.max()):.3f}]",
                        ]
                    except Exception:
                        pass

                if self._debug_cond and (self._ckpt_kind == PaperDiffusionPlanner.CKPT_KIND):
                    # numpy features (lanes/route_lanes)
                    try:
                        parts += [
                            f"lanes_shape={tuple(lanes_np.shape)}",
                            f"lanes_sum={float(np.sum(lanes_np)):.3f}",
                            f"lanes_nz={int(np.count_nonzero(lanes_np))}",
                            f"lanes_avails_sum={float(np.sum(lanes_avails_np)):.3f}",
                            f"route_lanes_shape={tuple(route_lanes_np.shape)}",
                            f"route_lanes_sum={float(np.sum(route_lanes_np)):.3f}",
                            f"route_lanes_nz={int(np.count_nonzero(route_lanes_np))}",
                            f"route_lanes_avails_sum={float(np.sum(route_lanes_avails_np)):.3f}",
                            f"route_roadblock_ids_len={0 if (self._route_roadblock_ids is None) else len(list(self._route_roadblock_ids))}",
                        ]
                    except Exception:
                        pass

                    # torch tensors (dynamic context)
                    try:
                        parts += [
                            f"ego_current_state_first4={ego_current_state.reshape(-1)[:4].detach().cpu().numpy().round(4).tolist()}",
                            f"neighbor_agents_future_nz={int(torch.count_nonzero(neighbor_agents_future).item())}",
                            f"ego_agent_future_nz={int(torch.count_nonzero(inputs['ego_agent_future']).item())}",
                            f"static_objects_nz={int(torch.count_nonzero(static_objects).item())}",
                        ]
                    except Exception:
                        pass

                print("[DP_DEBUG] " + " ".join(parts))
            except Exception:
                pass
            self._runtime_debug_ticks += 1

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
