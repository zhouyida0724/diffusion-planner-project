"""Overlay ego trajectories from two nuPlan closed-loop runs on the same scenario visualization.

For each token:
  - map polygons from the base run scenario
  - expert/reference trajectory (green)
  - obstacles (purple boxes) from base run (few frames)
  - base run ego trajectory (blue)
  - overlay run ego trajectory (orange)
  - offroad frames (red dots) from base run corners_in_drivable_area (if available)

Example:
  python3 scripts/analysis/viz_overlay_two_runs.py \
    --base-run data/.../2026.05.07.09.53.32 \
    --overlay-run data/.../2026.05.07.12.16.40 \
    --tokens outputs/eval/closedloop_regression_20260506/diverse60_success10_for_selector.csv \
    --out outputs/eval/closedloop_regression_20260506/overlay_base_vs_selector_success10.pdf \
    --max 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


def _poly_to_patch(coords) -> MplPolygon:
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


def _find_msgpack(run: Path, token: str, log_name: str | None = None) -> Path | None:
    root = run / "simulation_log" / "diffusion_planner_ckpt"
    if log_name:
        matches = list(root.glob(f"*/{log_name}/{token}/{token}.msgpack.xz"))
        return matches[0] if matches else None
    # fallback search
    matches = list(root.glob(f"*/**/{token}/{token}.msgpack.xz"))
    return matches[0] if matches else None


def _draw_box(ax, cx: float, cy: float, yaw: float, length: float, width: float, *, color: str, alpha: float):
    dx = length / 2.0
    dy = width / 2.0
    pts = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]], dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=float)
    pts = (R @ pts.T).T + np.array([cx, cy])
    ax.plot([*pts[:, 0], pts[0, 0]], [*pts[:, 1], pts[0, 1]], color=color, linewidth=1.0, alpha=alpha, zorder=3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-run", required=True)
    ap.add_argument("--overlay-run", required=True)
    ap.add_argument("--tokens", required=True, help="csv with column token")
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=float, default=140.0)
    ap.add_argument("--view-padding", type=float, default=25.0)
    ap.add_argument("--obstacles-frames", type=int, nargs="*", default=[0, -1])
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    base_run = Path(args.base_run)
    overlay_run = Path(args.overlay_run)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tok_df = pd.read_csv(args.tokens)
    tokens = [str(t) for t in tok_df["token"].tolist()]
    if args.max and int(args.max) > 0:
        tokens = tokens[: int(args.max)]

    rr_base = pd.read_parquet(base_run / "runner_report.parquet").rename(columns={"scenario_name": "token"})
    rr_base = rr_base[rr_base["succeeded"] == True]  # noqa: E712
    rr_overlay = pd.read_parquet(overlay_run / "runner_report.parquet").rename(columns={"scenario_name": "token"})
    rr_overlay = rr_overlay[rr_overlay["succeeded"] == True]  # noqa: E712

    base_meta = rr_base.set_index("token")["log_name"].to_dict()
    overlay_meta = rr_overlay.set_index("token")["log_name"].to_dict()

    # offroad frames from base run
    corners_ts: dict[str, list] = {}
    corners_path = base_run / "metrics" / "corners_in_drivable_area.parquet"
    if corners_path.exists():
        cdf = pd.read_parquet(corners_path)
        if "time_series_values" in cdf.columns:
            for _, r in cdf.iterrows():
                corners_ts[str(r["scenario_name"])] = r["time_series_values"]

    with PdfPages(out) as pdf:
        for token in tokens:
            log_name = base_meta.get(token) or overlay_meta.get(token)
            if log_name is None:
                continue
            p_base = _find_msgpack(base_run, token, log_name)
            p_ov = _find_msgpack(overlay_run, token, overlay_meta.get(token, log_name))
            if p_base is None or p_ov is None:
                continue

            slog_base = SimulationLog.load_data(p_base)
            slog_ov = SimulationLog.load_data(p_ov)

            scenario = slog_base.scenario
            map_api = scenario.map_api

            samples_base = slog_base.simulation_history.data
            samples_ov = slog_ov.simulation_history.data
            xy_base = np.array([[s.ego_state.rear_axle.x, s.ego_state.rear_axle.y] for s in samples_base], dtype=float)
            xy_ov = np.array([[s.ego_state.rear_axle.x, s.ego_state.rear_axle.y] for s in samples_ov], dtype=float)

            # expert trajectory
            try:
                expert = list(scenario.get_expert_ego_trajectory())
                expert_xy = np.array([[s.rear_axle.x, s.rear_axle.y] for s in expert], dtype=float)
            except Exception:
                expert_xy = None

            # map neighborhood
            center = Point2D(x=float(xy_base[0, 0]), y=float(xy_base[0, 1]))
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
            if SemanticMapLayer.STOP_LINE in nearest:
                nearest[SemanticMapLayer.STOP_LINE] = [
                    s for s in nearest[SemanticMapLayer.STOP_LINE] if s.stop_line_type != StopLineType.TURN_STOP
                ]

            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
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

            # expert / routing
            if expert_xy is not None and len(expert_xy) > 1:
                ax.plot(expert_xy[:, 0], expert_xy[:, 1], color="#2ca02c", linewidth=1.6, alpha=0.9, zorder=4)

            # obstacles from base run
            frames = []
            for f in args.obstacles_frames:
                if f < 0:
                    idx = max(0, len(samples_base) + int(f))
                else:
                    idx = min(len(samples_base) - 1, int(f))
                frames.append(idx)
            frames = sorted(set(frames))
            for k, idx in enumerate(frames):
                try:
                    obs = samples_base[idx].observation
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
                        c = box.center
                        _draw_box(
                            ax,
                            float(c.x),
                            float(c.y),
                            float(box.center.heading),
                            float(box.length),
                            float(box.width),
                            color="#9467bd",
                            alpha=alpha,
                        )
                except Exception:
                    continue

            # trajectories
            ax.plot(xy_base[:, 0], xy_base[:, 1], color="#1f77b4", linewidth=2.0, zorder=5, label="base")
            ax.plot(xy_ov[:, 0], xy_ov[:, 1], color="#ff7f0e", linewidth=2.0, zorder=5, label="overlay")
            ax.scatter([xy_base[0, 0]], [xy_base[0, 1]], s=25, c="black", zorder=6)

            # offroad frames from base
            ts = corners_ts.get(token)
            if ts is not None and len(ts) == len(xy_base):
                inside = np.array(ts, dtype=bool)
                off = ~inside
                if off.any():
                    ax.scatter(xy_base[off, 0], xy_base[off, 1], s=8, c="#d62728", alpha=0.85, zorder=7)

            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linewidth=0.4, alpha=0.35)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title(f"{token}\n{log_name}\nblue=base  orange=selector  green=expert  purple=obs  red=offroad(base)", fontsize=9)

            # view
            all_xy = [xy_base, xy_ov]
            if expert_xy is not None and len(expert_xy) > 1:
                all_xy.append(expert_xy)
            stack = np.concatenate(all_xy, axis=0)
            minx, miny = float(stack[:, 0].min()), float(stack[:, 1].min())
            maxx, maxy = float(stack[:, 0].max()), float(stack[:, 1].max())
            pad = float(args.view_padding)
            ax.set_xlim(minx - pad, maxx + pad)
            ax.set_ylim(miny - pad, maxy + pad)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

