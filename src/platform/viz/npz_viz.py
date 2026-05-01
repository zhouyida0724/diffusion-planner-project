"""NPZ -> PNG visualization helper.

This module is intentionally imported by `scripts/visualize_npz.py`.
Keep behavior stable: the CLI script remains the entrypoint.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np


def _ensure_pillow_resampling_compat() -> None:
    """Provide PIL.Image.Resampling fallback for older Pillow releases."""

    try:
        from PIL import Image
    except Exception:
        return

    if getattr(Image, "Resampling", None) is not None:
        return

    lanczos = getattr(Image, "LANCZOS", None)
    bicubic = getattr(Image, "BICUBIC", None)
    antialias = getattr(Image, "ANTIALIAS", None)
    bilinear = getattr(Image, "BILINEAR", None)
    nearest = getattr(Image, "NEAREST", None)

    class _CompatResampling:
        NEAREST = nearest if nearest is not None else 0
        BILINEAR = bilinear if bilinear is not None else (nearest if nearest is not None else 0)
        BICUBIC = bicubic if bicubic is not None else (bilinear if bilinear is not None else (nearest if nearest is not None else 0))
        LANCZOS = lanczos if lanczos is not None else (bicubic if bicubic is not None else (antialias if antialias is not None else (bilinear if bilinear is not None else (nearest if nearest is not None else 0))))

    Image.Resampling = _CompatResampling


def angle_difference_radians(angle1_rad: float, angle2_rad: float) -> float:
    """Return the smallest absolute difference between two angles (radians)."""
    diff = abs(angle1_rad - angle2_rad)
    normalized_diff = diff % (2 * math.pi)
    return (2 * math.pi) - normalized_diff if normalized_diff > math.pi else normalized_diff


def _env_flag(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _tl_color_from_onehot(onehot: np.ndarray) -> str:
    """Map traffic light onehot [g,y,r,unk] to a matplotlib color."""

    if onehot is None or len(onehot) < 4:
        return "#AAAAAA"
    i = int(np.argmax(onehot[:4]))
    if i == 0:
        return "#00AA00"  # green
    if i == 1:
        return "#CCAA00"  # yellow
    if i == 2:
        return "#CC0000"  # red
    return "#AAAAAA"  # unknown


def _draw_dir_arrows(ax, xs: np.ndarray, ys: np.ndarray, dxs: np.ndarray, dys: np.ndarray, *, color: str, every: int = 4):
    """Draw short direction arrows at sampled points."""

    if xs.size == 0:
        return
    every = max(1, int(every))
    for i in range(0, int(xs.shape[0]), every):
        x = float(xs[i])
        y = float(ys[i])
        dx = float(dxs[i])
        dy = float(dys[i])
        n = (dx * dx + dy * dy) ** 0.5
        if not np.isfinite(n) or n < 1e-6:
            continue
        dx /= n
        dy /= n
        # scale arrow length to be visible but not overwhelming
        L = 3.0
        ax.annotate(
            "",
            xy=(x + L * dx, y + L * dy),
            xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.85),
        )


def visualize_npz(npz_path: str | Path, output_path: Optional[str | Path] = None) -> Path:
    """Render a single-row NPZ export as a PNG.

    Args:
        npz_path: Path to the .npz file.
        output_path: Where to write the PNG. If None, defaults to "<token>_viz.png".

    Returns:
        Path to the saved PNG.
    """

    # Heavy imports live inside the function so this module can be imported in
    # headless contexts without eagerly configuring matplotlib backends.
    _ensure_pillow_resampling_compat()
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyArrowPatch
    from matplotlib import transforms

    npz_path = Path(npz_path)
    data = np.load(npz_path)

    def _squeeze1(x: np.ndarray) -> np.ndarray:
        """Handle both unbatched NPZ (legacy) and batched NPZ with leading dim=1."""

        x = np.asarray(x)
        if x.ndim >= 1 and x.shape[0] == 1:
            return x[0]
        return x

    token = str(data.get("token", "scene"))
    if output_path is None:
        output_path = f"{token}_viz.png"

    output_path = Path(output_path)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Debug overlay toggles (default OFF to keep legacy images stable)
    show_lane_dir = _env_flag("NPZ_VIZ_SHOW_LANE_DIR", "0")
    show_tl = _env_flag("NPZ_VIZ_SHOW_TRAFFIC_LIGHTS", "0")
    show_neighbor_heading = _env_flag("NPZ_VIZ_SHOW_NEIGHBOR_HEADING", "0")
    show_neighbor_vdir = _env_flag("NPZ_VIZ_SHOW_NEIGHBOR_VDIR", "0")
    show_acc = _env_flag("NPZ_VIZ_SHOW_ACC", "0")

    # Set range [-50, 50] for both axes
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"NPZ Visualization - {token}")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5)

    # ========== 1. Lanes (boundaries only, no centerline) ==========
    lanes = _squeeze1(data["lanes"])  # (70, 20, 12)
    route_lanes = _squeeze1(data["route_lanes"]) if ("route_lanes" in data.files) else None

    # Only draw normal lane boundaries
    for lane_idx in range(lanes.shape[0]):
        lane_x = lanes[lane_idx, :, 0]
        lane_y = lanes[lane_idx, :, 1]

        # Skip if all zeros
        if np.all(lane_x == 0) and np.all(lane_y == 0):
            continue

        # Skip if too far
        dist = np.sqrt(lane_x**2 + lane_y**2)
        if np.min(dist[dist > 0]) > 50:
            continue

        # Left boundary (blue)
        left_x = lanes[lane_idx, :, 0] + lanes[lane_idx, :, 4]
        left_y = lanes[lane_idx, :, 1] + lanes[lane_idx, :, 5]
        valid_left = (left_x != 0) | (left_y != 0)
        ax.plot(left_x[valid_left], left_y[valid_left], "b--", linewidth=1, alpha=0.5)

        # Right boundary (red)
        right_x = lanes[lane_idx, :, 0] + lanes[lane_idx, :, 6]
        right_y = lanes[lane_idx, :, 1] + lanes[lane_idx, :, 7]
        valid_right = (right_x != 0) | (right_y != 0)
        ax.plot(right_x[valid_right], right_y[valid_right], "r--", linewidth=1, alpha=0.5)

    # ============================================================
    # Route lanes visualization (light yellow solid lines with gold arrows)
    # ============================================================
    if route_lanes is not None:
        route_lanes_avails = _squeeze1(data.get("route_lanes_avails")) if ("route_lanes_avails" in data.files) else None
        for rlane_idx in range(route_lanes.shape[0]):
            lane_x = route_lanes[rlane_idx, :, 0]
            lane_y = route_lanes[rlane_idx, :, 1]

            # Skip if all zeros
            if np.all(lane_x == 0) and np.all(lane_y == 0):
                continue

            # Skip if too far
            dist = np.sqrt(lane_x**2 + lane_y**2)
            if np.min(dist[dist > 0]) > 50:
                continue

            # Get availability mask if available
            if route_lanes_avails is not None:
                avail = route_lanes_avails[rlane_idx]
                valid_mask = avail > 0
                if not np.any(valid_mask):
                    continue
                x_coords = lane_x[valid_mask]
                y_coords = lane_y[valid_mask]
            else:
                valid_mask = (lane_x != 0) | (lane_y != 0)
                x_coords = lane_x[valid_mask]
                y_coords = lane_y[valid_mask]

            if len(x_coords) == 0:
                continue

            # Optional: color route lanes by traffic light state stored in lane_feature[:,8:12]
            lane_color = "#FFFF99"
            if show_tl:
                try:
                    # Use the first valid point as representative.
                    j0 = int(np.argmax(valid_mask))
                    onehot = route_lanes[rlane_idx, j0, 8:12]
                    lane_color = _tl_color_from_onehot(onehot)
                except Exception:
                    lane_color = "#AAAAAA"

            ax.plot(x_coords, y_coords, "-", color=lane_color, alpha=0.9, linewidth=2)

            # Plot gold direction arrows every few points
            arrow_interval = max(1, len(x_coords) // 5)  # ~5 arrows per lane
            for i in range(0, len(x_coords) - 1, arrow_interval):
                if i + 1 < len(x_coords):
                    dx = x_coords[i + 1] - x_coords[i]
                    dy = y_coords[i + 1] - y_coords[i]
                    d = np.sqrt(dx**2 + dy**2)
                    if d > 0.01:
                        ax.annotate(
                            "",
                            xy=(x_coords[i + 1], y_coords[i + 1]),
                            xytext=(x_coords[i], y_coords[i]),
                            arrowprops=dict(arrowstyle="->", color="#FFD700", lw=1.5),
                        )

            # Optional: visualize the *exported* lane direction vectors (route_lanes[:,:,2:4]).
            # This is useful for debugging coordinate-frame contract mismatches.
            if show_lane_dir:
                try:
                    dxs = route_lanes[rlane_idx, :, 2][valid_mask]
                    dys = route_lanes[rlane_idx, :, 3][valid_mask]
                    _draw_dir_arrows(ax, x_coords, y_coords, dxs, dys, color="#AA00FF", every=max(1, len(x_coords) // 6))
                except Exception:
                    pass

    # ========== 2. Ego (arrow at origin) ==========
    ego_state = _squeeze1(data["ego_current_state"])
    cos_h, sin_h = ego_state[2], ego_state[3]
    ego_heading = np.arctan2(sin_h, cos_h)

    arrow_length = 4
    arrow = FancyArrowPatch(
        (0, 0),
        (arrow_length, 0),
        arrowstyle="-|>",
        mutation_scale=15,
        facecolor="red",
        edgecolor="darkred",
        linewidth=2,
    )
    ax.add_patch(arrow)

    ego_circle = patches.Circle((0, 0), radius=1.5, facecolor="red", edgecolor="darkred", linewidth=2)
    ax.add_patch(ego_circle)

    ego_future = _squeeze1(data["ego_agent_future"])
    if ego_future is not None:
        ax.plot(ego_future[:, 0], ego_future[:, 1], "b-", linewidth=3, alpha=0.8)

    # Load neighbor agents data
    neighbor_past = _squeeze1(data["neighbor_agents_past"])
    neighbor_future = _squeeze1(data["neighbor_agents_future"]) if ("neighbor_agents_future" in data.files) else None

    # ========== 2.1 Ego past trajectory ==========
    # Use neighbor_agents_past[0] (ego slot0) as the single source of truth.
    try:
        nb0 = neighbor_past[0, :, 0:2]
        valid_mask = (np.abs(nb0[:, 0]) > 1e-6) | (np.abs(nb0[:, 1]) > 1e-6)
        if np.any(valid_mask):
            # Make ego-past visually unmistakable: use a unique color + higher zorder.
            ax.plot(
                nb0[valid_mask, 0],
                nb0[valid_mask, 1],
                color="#00FF00",
                linestyle="--",
                linewidth=4.0,
                alpha=1.0,
                zorder=50,
                label="Ego Past (slot0)",
            )
            ax.scatter(
                nb0[valid_mask, 0],
                nb0[valid_mask, 1],
                s=18,
                color="#00FF00",
                alpha=0.9,
                zorder=51,
            )
    except Exception:
        pass

    if show_acc:
        try:
            ax.text(
                0.02,
                0.98,
                f"ego_ax={float(ego_state[6]):.3f}, ego_ay={float(ego_state[7]):.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC", alpha=0.8),
            )
        except Exception:
            pass

    agent_id = 0
    agent_ids: dict[int, int] = {}

    # ========== 3. Neighbor agents past trajectories ==========
    for agent_idx in range(neighbor_past.shape[0]):
        curr_x = neighbor_past[agent_idx, -1, 0]
        curr_y = neighbor_past[agent_idx, -1, 1]
        past = neighbor_past[agent_idx, :, 0:2]

        # NOTE: do not skip ego slot0 when heading debug is enabled; otherwise its arrow is never shown.
        if agent_idx != 0 and abs(curr_x) < 0.1 and abs(curr_y) < 0.1:
            continue
        if abs(curr_x) > 50 or abs(curr_y) > 50:
            continue
        if not np.any(past != 0):
            continue

        agent_ids[agent_idx] = agent_id
        agent_id += 1

        valid_mask = (past[:, 0] != 0) | (past[:, 1] != 0)
        if not np.any(valid_mask):
            continue
        start_idx = int(np.argmax(valid_mask))
        start_x, start_y = past[start_idx, 0], past[start_idx, 1]

        ax.annotate(
            str(agent_ids[agent_idx]),
            (curr_x, curr_y),
            fontsize=10,
            fontweight="bold",
            color="red",
            ha="left",
            va="bottom",
            xytext=(5, 5),
            textcoords="offset points",
        )

        ax.plot(past[start_idx:, 0], past[start_idx:, 1], "b-", linewidth=3, alpha=0.8)
        ax.plot(start_x, start_y, "s", markersize=8, color="blue", markeredgecolor="darkblue", markeredgewidth=2)
        ax.plot(curr_x, curr_y, "o", markersize=8, color="blue", markeredgecolor="darkblue", markeredgewidth=2)

        if show_neighbor_heading and neighbor_past.shape[-1] >= 4:
            try:
                cos_h = float(neighbor_past[agent_idx, -1, 2])
                sin_h = float(neighbor_past[agent_idx, -1, 3])
                h = float(np.arctan2(sin_h, cos_h))
                L = 4.0
                ax.annotate(
                    "",
                    xy=(float(curr_x + L * np.cos(h)), float(curr_y + L * np.sin(h))),
                    xytext=(float(curr_x), float(curr_y)),
                    arrowprops=dict(arrowstyle="->", color="#00AAFF", lw=2.0, alpha=0.9),
                )
            except Exception:
                pass

        # Optional: visualize velocity direction (ego frame) using v_local (dims 4:6).
        if show_neighbor_vdir and neighbor_past.shape[-1] >= 6:
            try:
                vx = float(neighbor_past[agent_idx, -1, 4])
                vy = float(neighbor_past[agent_idx, -1, 5])
                n = float((vx * vx + vy * vy) ** 0.5)
                if n > 1e-3:
                    vx /= n
                    vy /= n
                    L = 4.0
                    ax.annotate(
                        "",
                        xy=(float(curr_x + L * vx), float(curr_y + L * vy)),
                        xytext=(float(curr_x), float(curr_y)),
                        arrowprops=dict(arrowstyle="->", color="#FF00AA", lw=2.0, alpha=0.85),
                    )
            except Exception:
                pass

    # ========== 4. Neighbor agents future trajectories ==========
    if neighbor_future is not None:
        for agent_idx in range(neighbor_future.shape[0]):
            curr_x = neighbor_past[agent_idx, -1, 0]
            curr_y = neighbor_past[agent_idx, -1, 1]
            future = neighbor_future[agent_idx, :, 0:2]

            if abs(curr_x) < 0.1 and abs(curr_y) < 0.1:
                continue
            if abs(curr_x) > 50 or abs(curr_y) > 50:
                continue
            if not np.any(future != 0):
                continue

            valid_mask = (future[:, 0] != 0) | (future[:, 1] != 0)
            if not np.any(valid_mask):
                continue
            end_idx = int(np.argmax(valid_mask))
            end_x, end_y = future[end_idx, 0], future[end_idx, 1]

            ax.plot(future[:, 0], future[:, 1], "g-", linewidth=3, alpha=0.8)
            ax.plot(curr_x, curr_y, "o", markersize=8, color="green", markeredgecolor="darkgreen", markeredgewidth=2)
            ax.plot(end_x, end_y, "*", markersize=12, color="green", markeredgecolor="darkgreen", markeredgewidth=1)

    # ========== 5. Neighbor agents with box visualization ==========
    for agent_idx in range(neighbor_past.shape[0]):
        type_v = neighbor_past[agent_idx, -1, 8]
        type_p = neighbor_past[agent_idx, -1, 9]
        type_b = neighbor_past[agent_idx, -1, 10]

        if type_v > 0.5:
            agent_type = "Vehicle"
            color = "blue"
        elif type_p > 0.5:
            agent_type = "Pedestrian"
            color = "orange"
        elif type_b > 0.5:
            agent_type = "Bicycle"
            color = "purple"
        else:
            continue

        curr_x = neighbor_past[agent_idx, -1, 0]
        curr_y = neighbor_past[agent_idx, -1, 1]

        if abs(curr_x) < 0.1 and abs(curr_y) < 0.1:
            continue
        if abs(curr_x) > 50 or abs(curr_y) > 50:
            continue

        width = neighbor_past[agent_idx, -1, 6]
        length = neighbor_past[agent_idx, -1, 7]

        cos_h = neighbor_past[agent_idx, -1, 2]
        sin_h = neighbor_past[agent_idx, -1, 3]
        # neighbor_agents_past stores ego-relative heading as cos/sin.
        neighbor_heading = float(np.arctan2(sin_h, cos_h))

        if width == 0:
            width = 2.0
        if length == 0:
            length = 4.0

        if agent_type == "Vehicle":
            # Matplotlib Rectangle rotates about its lower-left corner; rotate around box center.
            rect = patches.Rectangle(
                (curr_x - length / 2, curr_y - width / 2),
                length,
                width,
                facecolor=color,
                edgecolor="black",
                linewidth=1,
                alpha=0.7,
            )
            rect.set_transform(transforms.Affine2D().rotate_around(curr_x, curr_y, neighbor_heading) + ax.transData)
            ax.add_patch(rect)
        elif agent_type == "Pedestrian":
            radius = max(width, length) / 2
            circle = patches.Circle((curr_x, curr_y), radius, facecolor=color, edgecolor="black", linewidth=1, alpha=0.7)
            ax.add_patch(circle)
        elif agent_type == "Bicycle":
            diamond = patches.RegularPolygon(
                (curr_x, curr_y),
                numVertices=4,
                radius=max(width, length) / 2,
                orientation=neighbor_heading,
                facecolor=color,
                edgecolor="black",
                linewidth=1,
                alpha=0.7,
            )
            ax.add_patch(diamond)

    # ========== 6. Static objects ==========
    static_objs = _squeeze1(data.get("static_objects")) if ("static_objects" in data.files) else None
    if static_objs is not None and len(static_objs) > 0:
        for obj_idx in range(static_objs.shape[0]):
            x, y = static_objs[obj_idx, 0], static_objs[obj_idx, 1]
            if float(x) == 0.0 and float(y) == 0.0:
                continue
            if abs(x) > 50 or abs(y) > 50:
                continue
            ax.plot(x, y, color="red", marker="^", markersize=20, alpha=0.7)

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")
    return output_path
