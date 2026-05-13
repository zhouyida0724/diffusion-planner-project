"""Plot ego trajectories for *all* scenarios in a nuPlan run.

Outputs a PDF with many small subplots (one per scenario). Optionally overlays
offroad frames using `corners_in_drivable_area` metric time-series.

Example:
  python3 scripts/analysis/plot_all_trajectories.py \
    --run data/nuplan/exp/exp/simulation/closed_loop_nonreactive_agents/2026.05.07.09.53.32 \
    --out outputs/eval/closedloop_regression_20260506/diverse60_all_trajectories.pdf
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


def load_xy(msgpack_xz: Path) -> tuple[np.ndarray, float]:
    log = SimulationLog.load_data(msgpack_xz)
    samples = log.simulation_history.data
    xy = np.array([[s.ego_state.rear_axle.x, s.ego_state.rear_axle.y] for s in samples], dtype=float)
    yaw0 = float(samples[0].ego_state.rear_axle.heading)
    return xy, yaw0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--per-page", type=int, default=36)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--local-frame", action="store_true", default=True)
    args = ap.parse_args()

    run = Path(args.run)
    simlog_root = run / "simulation_log" / "diffusion_planner_ckpt"
    metrics_dir = run / "metrics"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rr = pd.read_parquet(run / "runner_report.parquet")
    rr = rr[rr["succeeded"] == True].copy()  # noqa: E712
    # runner_report doesn't include scenario_type; fetch it from any metric parquet.
    scenario_types = pd.read_parquet(metrics_dir / "ego_is_making_progress.parquet")[
        ["scenario_name", "scenario_type"]
    ].rename(columns={"scenario_name": "token"})
    rr = rr.rename(columns={"scenario_name": "token"})
    rr = rr.merge(scenario_types, on="token", how="left")
    rr = rr.sort_values(["scenario_type", "log_name", "token"], na_position="last")

    # Join a few scalar metrics for titles.
    prog = pd.read_parquet(metrics_dir / "ego_progress_along_expert_route.parquet")[
        ["scenario_name", "ego_expert_progress_along_route_ratio_stat_value"]
    ].rename(columns={"scenario_name": "token", "ego_expert_progress_along_route_ratio_stat_value": "progress_ratio"})
    mp = pd.read_parquet(metrics_dir / "ego_is_making_progress.parquet")[
        ["scenario_name", "ego_is_making_progress_stat_value"]
    ].rename(columns={"scenario_name": "token", "ego_is_making_progress_stat_value": "making_progress"})
    dr = pd.read_parquet(metrics_dir / "drivable_area_compliance.parquet")
    dr_stat = [c for c in dr.columns if c.endswith("_stat_value")][0]
    dr = dr[["scenario_name", dr_stat]].rename(columns={"scenario_name": "token", dr_stat: "drivable"})
    dr["drivable"] = dr["drivable"].astype(float)

    corners = None
    corners_path = metrics_dir / "corners_in_drivable_area.parquet"
    if corners_path.exists():
        corners = pd.read_parquet(corners_path)
        # pick time-series values column if present
        # usually: time_series_values (list[bool])
        if "time_series_values" not in corners.columns:
            corners = None

    df = rr.merge(prog, on="token", how="left").merge(mp, on="token", how="left").merge(dr, on="token", how="left")

    def msgpack_path(row) -> Path:
        token = row["token"]
        return simlog_root / str(row["scenario_type"]) / str(row["log_name"]) / str(token) / f"{token}.msgpack.xz"

    df["msgpack"] = df.apply(lambda r: msgpack_path(r), axis=1)
    df = df[df["msgpack"].apply(lambda p: Path(p).exists())].copy()

    per_page = int(args.per_page)
    cols = int(args.cols)
    rows = int(math.ceil(per_page / cols))

    # Build a dict token -> corners time-series (True means inside drivable).
    corners_ts: dict[str, list] = {}
    if corners is not None:
        for _, r in corners.iterrows():
            corners_ts[str(r["scenario_name"])] = r["time_series_values"]

    with PdfPages(out) as pdf:
        for start in range(0, len(df), per_page):
            chunk = df.iloc[start : start + per_page]
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.4))
            axes = np.array(axes).reshape(-1)

            for ax in axes:
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, linewidth=0.3, alpha=0.35)

            for i, (_, r) in enumerate(chunk.iterrows()):
                ax = axes[i]
                token = str(r["token"])
                xy, yaw0 = load_xy(Path(r["msgpack"]))
                if args.local_frame:
                    xy = xy - xy[0]
                    xy = _rot(xy, -yaw0)

                ts = corners_ts.get(token)
                if ts is not None and len(ts) == len(xy):
                    inside = np.array(ts, dtype=bool)
                    ax.plot(xy[:, 0], xy[:, 1], linewidth=0.8, color="#4C78A8")
                    # mark offroad frames red
                    off = ~inside
                    if off.any():
                        ax.scatter(xy[off, 0], xy[off, 1], s=3, c="#E45756", alpha=0.8)
                else:
                    ax.plot(xy[:, 0], xy[:, 1], linewidth=0.9, color="#4C78A8")

                ax.scatter([xy[0, 0]], [xy[0, 1]], s=8, c="black")

                pr = r.get("progress_ratio")
                pr_s = "?" if pd.isna(pr) else f"{float(pr):.3f}"
                mpv = r.get("making_progress")
                mp_s = "?" if pd.isna(mpv) else ("T" if bool(mpv) else "F")
                drv = r.get("drivable")
                drv_s = "?" if pd.isna(drv) else ("1" if float(drv) > 0.5 else "0")
                ax.set_title(f"{token}\npr={pr_s} mp={mp_s} drv={drv_s}", fontsize=7)
                ax.tick_params(labelsize=7)

            for j in range(len(chunk), len(axes)):
                axes[j].axis("off")

            fig.suptitle(
                f"All ego trajectories (success only): {len(df)} scenarios | blue=path red=offroad frames | run={run.name}",
                fontsize=11,
            )
            fig.tight_layout(rect=[0, 0.01, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {out} ({len(df)} scenarios)")


if __name__ == "__main__":
    main()
