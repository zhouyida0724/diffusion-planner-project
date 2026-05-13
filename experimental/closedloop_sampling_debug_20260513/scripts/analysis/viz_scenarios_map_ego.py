"""Scenario-style visualization: map + ego trajectory (NuBoard-like, but static).

This produces a PDF where each page is one scenario:
  - background: proximal vector map polygons (lanes, intersections, crosswalks, etc.)
  - overlay: ego rear-axle trajectory (blue)
  - overlay: offroad frames (red dots) using `corners_in_drivable_area` time-series when available

Example:
  python3 scripts/analysis/viz_scenarios_map_ego.py \
    --run data/nuplan/exp/exp/simulation/closed_loop_nonreactive_agents/2026.05.07.09.53.32 \
    --out outputs/eval/closedloop_regression_20260506/diverse60_viz_map_ego.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon

from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.planning.simulation.simulation_log import SimulationLog


LAYER_STYLE = {
    SemanticMapLayer.LANE: ("#F0F0F0", 0.8),
    SemanticMapLayer.LANE_CONNECTOR: ("#F7F7F7", 0.7),
    SemanticMapLayer.INTERSECTION: ("#EFE7DA", 0.45),
    SemanticMapLayer.CROSSWALK: ("#FFF2CC", 0.55),
    SemanticMapLayer.WALKWAYS: ("#F5F5F5", 0.4),
    SemanticMapLayer.CARPARK_AREA: ("#F5F5F5", 0.35),
    SemanticMapLayer.STOP_LINE: ("#FFD1D1", 0.55),
    SemanticMapLayer.ROADBLOCK: ("#D9E8FF", 0.25),
}


def _poly_to_patch(coords: Iterable[tuple[float, float]]) -> MplPolygon:
    pts = np.asarray(list(coords), dtype=float)
    return MplPolygon(pts, closed=True)


def _add_polys(ax, polys, color: str, alpha: float) -> None:
    patches = []
    for poly in polys:
        try:
            coords = poly.polygon.exterior.coords
        except Exception:
            continue
        patches.append(_poly_to_patch(coords))
    if patches:
        pc = PatchCollection(patches, facecolor=color, edgecolor="none", alpha=alpha, zorder=0)
        ax.add_collection(pc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=float, default=140.0, help="map query radius around start pose")
    ap.add_argument(
        "--view-padding",
        type=float,
        default=25.0,
        help="axis padding (m) added around min/max of ego+expert trajectories",
    )
    ap.add_argument("--max", type=int, default=0, help="limit scenarios (0 means all)")
    ap.add_argument("--obstacles-frames", type=int, nargs="*", default=[0, -1], help="frames to draw obstacles")
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rr = pd.read_parquet(run / "runner_report.parquet")
    rr = rr[rr["succeeded"] == True].copy()  # noqa: E712
    rr = rr.rename(columns={"scenario_name": "token"})

    corners_ts: dict[str, list] = {}
    corners_path = run / "metrics" / "corners_in_drivable_area.parquet"
    if corners_path.exists():
        cdf = pd.read_parquet(corners_path)
        if "time_series_values" in cdf.columns:
            for _, r in cdf.iterrows():
                corners_ts[str(r["scenario_name"])] = r["time_series_values"]

    # sort by log then token for stable browsing
    rr = rr.sort_values(["log_name", "token"]).reset_index(drop=True)
    if args.max and int(args.max) > 0:
        rr = rr.iloc[: int(args.max)]

    simlog_root = run / "simulation_log" / "diffusion_planner_ckpt"

    with PdfPages(out) as pdf:
        for _, row in rr.iterrows():
            token = str(row["token"])
            log_name = str(row["log_name"])

            # locate msgpack path (scenario_type folder unknown here) by searching.
            # structure: simlog_root/<scenario_type>/<log_name>/<token>/<token>.msgpack.xz
            matches = list(simlog_root.glob(f"*/{log_name}/{token}/{token}.msgpack.xz"))
            if not matches:
                continue
            msgpack_path = matches[0]

            slog = SimulationLog.load_data(msgpack_path)
            scenario = slog.scenario
            map_api = scenario.map_api

            samples = slog.simulation_history.data
            xy = np.array([[s.ego_state.rear_axle.x, s.ego_state.rear_axle.y] for s in samples], dtype=float)

            # Expert (reference) trajectory from scenario ("routing" proxy).
            try:
                expert = list(scenario.get_expert_ego_trajectory())
                expert_xy = np.array([[s.rear_axle.x, s.rear_axle.y] for s in expert], dtype=float)
            except Exception:
                expert_xy = None

            # map neighborhood (query around start pose)
            center = Point2D(x=float(xy[0, 0]), y=float(xy[0, 1]))
            layer_names = [
                SemanticMapLayer.LANE_CONNECTOR,
                SemanticMapLayer.LANE,
                SemanticMapLayer.CROSSWALK,
                SemanticMapLayer.INTERSECTION,
                SemanticMapLayer.STOP_LINE,
                SemanticMapLayer.WALKWAYS,
                SemanticMapLayer.CARPARK_AREA,
            ]
            nearest = map_api.get_proximal_map_objects(center, float(args.radius), layer_names)
            # remove turn stop
            if SemanticMapLayer.STOP_LINE in nearest:
                nearest[SemanticMapLayer.STOP_LINE] = [
                    s for s in nearest[SemanticMapLayer.STOP_LINE] if s.stop_line_type != StopLineType.TURN_STOP
                ]

            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

            # render map polygons
            for layer in [
                SemanticMapLayer.LANE,
                SemanticMapLayer.LANE_CONNECTOR,
                SemanticMapLayer.INTERSECTION,
                SemanticMapLayer.CROSSWALK,
                SemanticMapLayer.WALKWAYS,
                SemanticMapLayer.CARPARK_AREA,
                SemanticMapLayer.STOP_LINE,
            ]:
                if layer in nearest and layer in LAYER_STYLE:
                    color, alpha = LAYER_STYLE[layer]
                    _add_polys(ax, nearest[layer], color=color, alpha=alpha)

            # ego trajectory
            ax.plot(xy[:, 0], xy[:, 1], color="#1f77b4", linewidth=2.0, zorder=5)
            ax.scatter([xy[0, 0]], [xy[0, 1]], s=25, c="black", zorder=6)
            ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=25, c="#1f77b4", zorder=6)

            # routing / reference overlay
            if expert_xy is not None and len(expert_xy) > 1:
                ax.plot(expert_xy[:, 0], expert_xy[:, 1], color="#2ca02c", linewidth=1.6, alpha=0.9, zorder=4)

            # obstacles overlay (a few frames to avoid clutter)
            def draw_box(cx: float, cy: float, yaw: float, length: float, width: float, *, color: str, alpha: float):
                # rectangle corners in local frame
                dx = length / 2.0
                dy = width / 2.0
                pts = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]], dtype=float)
                c, s = np.cos(yaw), np.sin(yaw)
                R = np.array([[c, -s], [s, c]], dtype=float)
                pts = (R @ pts.T).T + np.array([cx, cy])
                ax.plot(
                    [*pts[:, 0], pts[0, 0]],
                    [*pts[:, 1], pts[0, 1]],
                    color=color,
                    linewidth=1.0,
                    alpha=alpha,
                    zorder=3,
                )

            # select frames
            frames = []
            for f in args.obstacles_frames:
                if f < 0:
                    idx = max(0, len(samples) + int(f))
                else:
                    idx = min(len(samples) - 1, int(f))
                frames.append(idx)
            frames = sorted(set(frames))

            for k, idx in enumerate(frames):
                try:
                    obs = samples[idx].observation
                    # DetectionsTracks exposes tracked_objects
                    tobs = getattr(obs, "tracked_objects", None)
                    objs = getattr(tobs, "tracked_objects", None)
                    if objs is None:
                        continue
                    alpha = 0.25 if len(frames) > 1 else 0.35
                    if k == 0:
                        alpha = 0.35
                    for o in objs:
                        box = getattr(o, "box", None)
                        if box is None:
                            continue
                        center = box.center
                        draw_box(
                            float(center.x),
                            float(center.y),
                            float(box.center.heading),
                            float(box.length),
                            float(box.width),
                            color="#9467bd",
                            alpha=alpha,
                        )
                except Exception:
                    continue

            ts = corners_ts.get(token)
            if ts is not None and len(ts) == len(xy):
                inside = np.array(ts, dtype=bool)
                off = ~inside
                if off.any():
                    ax.scatter(xy[off, 0], xy[off, 1], s=8, c="#d62728", alpha=0.85, zorder=7)

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.grid(True, linewidth=0.4, alpha=0.35)
            ax.set_title(f"{token}\n{log_name}\nblue=ego  green=expert  purple=obstacles  red=offroad", fontsize=9)

            # auto view: fit to (ego + expert) so we don't crop long routes/trajectories.
            all_xy = [xy]
            if expert_xy is not None and len(expert_xy) > 1:
                all_xy.append(expert_xy)
            stack = np.concatenate(all_xy, axis=0)
            minx, miny = float(stack[:, 0].min()), float(stack[:, 1].min())
            maxx, maxy = float(stack[:, 0].max()), float(stack[:, 1].max())
            pad = float(args.view_padding)
            # avoid degenerate tiny ranges
            if (maxx - minx) < 10.0:
                minx -= 5.0
                maxx += 5.0
            if (maxy - miny) < 10.0:
                miny -= 5.0
                maxy += 5.0
            ax.set_xlim(minx - pad, maxx + pad)
            ax.set_ylim(miny - pad, maxy + pad)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
