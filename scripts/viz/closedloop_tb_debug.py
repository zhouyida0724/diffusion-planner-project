#!/usr/bin/env python3
"""TensorBoard debug viz for closed-loop per-tick dumps.

Reads per-tick NPZ dumps (created by our closed-loop debug hook) and logs
NPZ-style scene renderings with multiple trajectory overlays.

Primary use: compare diffusion planner plan vs IDM plan on the same scenario.

Expected dump keys (per tick):
- ego_current_state, ego_past, lanes, route_lanes, route_lanes_avails
- neighbor_agents_past, static_objects
- closed_loop_y: (80,3) in ego frame (x,y,dheading)

Usage (inside nuplan-simulation container recommended):
  PYTHONPATH=/workspace/nuplan-visualization:/workspace \
    python3 scripts/viz/closedloop_tb_debug.py \
      --dump-dir /workspace/outputs/debug_d32c40c_dump_v3 \
      --scenario d32c40cedb4d5025 \
      --idm-simlog /workspace/outputs/simulations/<idm_run>/.../d32c40cedb4d5025.msgpack.xz \
      --logdir /workspace/outputs/tb_closedloop_debug_d32c40c
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None  # type: ignore


def _load_idm_plan_ego_frame(simlog_path: Path, tick: int, horizon: int = 80) -> np.ndarray | None:
    """Return (horizon,2) planned relative xy in ego frame for IDM at a given tick."""

    import lzma
    import msgpack
    import pickle

    raw = lzma.open(simlog_path, "rb").read()
    blob = msgpack.unpackb(raw, raw=False)
    # SimulationLog is pickled Python object.
    simlog = pickle.loads(blob)
    hist = simlog.simulation_history

    if tick < 0 or tick >= len(hist.data):
        return None

    sample = hist.data[tick]
    cur = sample.ego_state
    cur_rear = cur.rear_axle
    x0 = float(cur_rear.x)
    y0 = float(cur_rear.y)
    h0 = float(cur_rear.heading)

    traj = sample.trajectory.get_sampled_trajectory()  # list[EgoState]
    # traj[0] is current, traj[1] is next.
    n = min(horizon, max(0, len(traj) - 1))
    if n <= 0:
        return None

    out = np.zeros((horizon, 2), dtype=np.float32)
    c = float(np.cos(-h0))
    s = float(np.sin(-h0))

    for i in range(n):
        st = traj[i + 1]
        r = st.rear_axle
        dxw = float(r.x) - x0
        dyw = float(r.y) - y0
        dx = c * dxw - s * dyw
        dy = s * dxw + c * dyw
        out[i, 0] = dx
        out[i, 1] = dy

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", type=str, required=True)
    ap.add_argument("--scenario", type=str, required=True)
    ap.add_argument("--idm-simlog", type=str, required=True)
    ap.add_argument("--logdir", type=str, required=True)
    ap.add_argument("--ticks", type=str, default="0-40")
    ap.add_argument("--image-size", type=int, default=800)
    ap.add_argument(
        "--traj-convention",
        type=str,
        default="current_plus_future",
        choices=["future_only", "current_plus_future"],
        help=(
            "How to plot trajectories. 'future_only' plots only predicted future points. "
            "'current_plus_future' prepends the current ego point (0,0) for nuBoard parity."
        ),
    )
    args = ap.parse_args()

    if SummaryWriter is None:
        raise SystemExit("torch.utils.tensorboard is unavailable")

    dump_dir = Path(args.dump_dir)
    idm_simlog = Path(args.idm_simlog)
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # Parse tick range like 0-40
    ticks: list[int] = []
    s = str(args.ticks).strip()
    if "-" in s:
        a, b = s.split("-", 1)
        ticks = list(range(int(a), int(b) + 1))
    else:
        ticks = [int(x) for x in s.split(",") if x.strip()]

    # Import renderer
    from src.methods.diffusion_planner.paper.train.tb_visualizer import render_npz_style_scene

    writer = SummaryWriter(log_dir=str(logdir))
    writer.add_text("meta/scenario", args.scenario, 0)
    writer.add_text("meta/dump_dir", str(dump_dir), 0)
    writer.add_text("meta/idm_simlog", str(idm_simlog), 0)

    for t in ticks:
        p = dump_dir / f"tick_{t:03d}_iter_{t:03d}.npz"
        if not p.is_file():
            continue

        z = np.load(p, allow_pickle=False)

        # Build batch-like dict (with leading batch dim = 1)
        batch = {
            "ego_current_state": z["ego_current_state"][None, ...],
            "ego_past": z["ego_past"][None, ...],
            "ego_agent_future": z.get("ego_agent_future", np.zeros((80, 3), dtype=np.float32))[None, ...],
            "lanes": z["lanes"][None, ...],
            "route_lanes": z["route_lanes"][None, ...],
            "route_lanes_avails": z["route_lanes_avails"][None, ...],
            "neighbor_agents_past": z["neighbor_agents_past"][None, ...],
            "neighbor_agents_future": z.get("neighbor_agents_future", np.zeros((32, 80, 3), dtype=np.float32))[None, ...],
            "static_objects": z.get("static_objects", np.zeros((5, 10), dtype=np.float32))[None, ...],
        }

        diff_xy = z["closed_loop_y"][..., 0:2].astype(np.float32)
        horizon = int(diff_xy.shape[0])
        idm_xy = _load_idm_plan_ego_frame(idm_simlog, tick=int(t), horizon=horizon)

        # nuBoard / simlog convention is: traj[0] is current ego, traj[1:] are future.
        # Our dump convention is: closed_loop_y[0] is the first future point (t=+0.1s), i.e. future-only.
        if args.traj_convention == "current_plus_future":
            diff_xy = np.concatenate([np.zeros((1, 2), dtype=np.float32), diff_xy], axis=0)
            if idm_xy is not None:
                idm_xy = np.concatenate([np.zeros((1, 2), dtype=np.float32), idm_xy], axis=0)

        overlays = [
            {"xy": diff_xy, "label": "diffusion_plan", "color": "orange", "lw": 3.0, "alpha": 0.95},
        ]
        if idm_xy is not None:
            overlays.append({"xy": idm_xy, "label": "idm_plan", "color": "cyan", "lw": 2.5, "alpha": 0.9})

        img = render_npz_style_scene(
            batch,
            sample_idx=0,
            ego_past_xy=z["ego_past"][..., 0:2],
            ego_future_overlays=overlays,
            title=f"{args.scenario} tick={t:02d}",
            image_size=int(args.image_size),
        )

        if img is not None:
            tag = f"closedloop/scene/{args.scenario}/tick_{t:03d}"
            writer.add_image(tag, img.transpose(2, 0, 1), global_step=int(t))

            # quick scalars for conditioning presence
            writer.add_scalar(
                f"closedloop/route_avails_sum/{args.scenario}",
                float(np.sum(z["route_lanes_avails"] > 0)),
                int(t),
            )
            writer.add_scalar(
                f"closedloop/static_objects_nnz/{args.scenario}",
                float(np.count_nonzero(z.get("static_objects", np.zeros((5, 10), dtype=np.float32)))),
                int(t),
            )

    writer.flush()
    writer.close()

    print(f"Wrote TB images to: {logdir}")
    return 0


if __name__ == "__main__":
    # Make PYTHONPATH hint explicit when run outside container.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
