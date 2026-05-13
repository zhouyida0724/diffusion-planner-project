"""Plot ego trajectories for scenarios that went off drivable area.

Usage:
  python3 scripts/analysis/plot_offroad_trajectories.py \
    --run data/nuplan/exp/exp/simulation/closed_loop_nonreactive_agents/2026.05.07.09.53.32 \
    --out outputs/eval/closedloop_regression_20260506/diverse60_offroad_trajectories.pdf
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from nuplan.planning.simulation.simulation_log import SimulationLog


def _rot(xy: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return (R @ xy.T).T


def load_xy_from_msgpack(msgpack_xz: Path) -> tuple[np.ndarray, float]:
    log = SimulationLog.load_data(msgpack_xz)
    samples = log.simulation_history.data
    xy = np.array([[s.ego_state.rear_axle.x, s.ego_state.rear_axle.y] for s in samples], dtype=float)
    yaw0 = float(samples[0].ego_state.rear_axle.heading)
    return xy, yaw0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True, help="nuPlan simulation run directory")
    ap.add_argument("--out", type=str, required=True, help="output PDF path")
    ap.add_argument("--per-page", type=int, default=36)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--use-local-frame", action="store_true", default=True)
    args = ap.parse_args()

    run = Path(args.run)
    metrics_dir = run / "metrics"
    simlog_root = run / "simulation_log" / "diffusion_planner_ckpt"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Determine offroad tokens by drivable_area_compliance stat value.
    dr = pd.read_parquet(metrics_dir / "drivable_area_compliance.parquet")
    stat_cols = [c for c in dr.columns if c.endswith("_stat_value")]
    if not stat_cols:
        raise RuntimeError(f"Cannot find *_stat_value in {metrics_dir/'drivable_area_compliance.parquet'}")
    stat_col = stat_cols[0]
    dr = dr[["scenario_name", "log_name", "scenario_type", stat_col]].rename(
        columns={"scenario_name": "token", stat_col: "drivable"}
    )
    dr["drivable"] = dr["drivable"].astype(float)
    off = dr[dr["drivable"] < 0.5].copy().sort_values(["scenario_type", "log_name", "token"])

    if len(off) == 0:
        print("No offroad scenarios found (drivable_area_compliance==1 for all)")
        return

    # Resolve msgpack.xz path for each token.
    def find_msgpack(row) -> Path | None:
        token = row["token"]
        log_name = row["log_name"]
        scenario_type = row["scenario_type"]
        p = simlog_root / str(scenario_type) / str(log_name) / str(token) / f"{token}.msgpack.xz"
        return p if p.exists() else None

    off["msgpack"] = off.apply(find_msgpack, axis=1)
    missing = off[off["msgpack"].isna()]
    if len(missing):
        print(f"WARNING: missing msgpack for {len(missing)}/{len(off)} offroad scenarios")
        print(missing[["token", "log_name", "scenario_type"]].head(20).to_string(index=False))

    off = off[off["msgpack"].notna()].copy()

    per_page = int(args.per_page)
    cols = int(args.cols)
    rows = int(math.ceil(per_page / cols))

    with PdfPages(out) as pdf:
        for start in range(0, len(off), per_page):
            chunk = off.iloc[start : start + per_page]
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
            axes = np.array(axes).reshape(-1)

            for ax in axes:
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, linewidth=0.3, alpha=0.4)

            for i, (_, r) in enumerate(chunk.iterrows()):
                ax = axes[i]
                msgpack_path = Path(r["msgpack"])
                xy, yaw0 = load_xy_from_msgpack(msgpack_path)
                if args.use_local_frame:
                    xy = xy - xy[0]
                    xy = _rot(xy, -yaw0)
                ax.plot(xy[:, 0], xy[:, 1], linewidth=1.0)
                ax.scatter([xy[0, 0]], [xy[0, 1]], s=8)
                ax.set_title(str(r["token"]), fontsize=8)
                ax.tick_params(labelsize=7)
            # blank remaining
            for j in range(len(chunk), len(axes)):
                axes[j].axis("off")

            fig.suptitle(
                f"Offroad trajectories (drivable_area_compliance=0): {len(off)} scenarios | run={run.name}",
                fontsize=11,
            )
            fig.tight_layout(rect=[0, 0.01, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {out} ({len(off)} offroad scenarios)")


if __name__ == "__main__":
    main()

