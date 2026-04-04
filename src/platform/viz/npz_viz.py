"""NPZ -> PNG visualization helper.

This module is intentionally imported by `scripts/visualize_npz.py`.
Keep behavior stable: the CLI script remains the entrypoint.
"""

from __future__ import annotations

import math
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

    npz_path = Path(npz_path)
    data = np.load(npz_path)

    token = str(data.get("token", "scene"))
    if output_path is None:
        output_path = f"{token}_viz.png"

    output_path = Path(output_path)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

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
    lanes = data["lanes"]  # (70, 20, 12)
    route_lanes = data.get("route_lanes", None)

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
        route_lanes_avails = data.get("route_lanes_avails")
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

            ax.plot(x_coords, y_coords, "-", color="#FFFF99", alpha=0.9, linewidth=2)

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

    # ========== 2. Ego (arrow at origin) ==========
    ego_state = data["ego_current_state"]
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

    ego_future = data["ego_agent_future"]
    if ego_future is not None:
        ax.plot(ego_future[:, 0], ego_future[:, 1], "b-", linewidth=3, alpha=0.8)

    # ========== 2.1 Ego past trajectory ==========
    if "ego_past" in data.files:
        ego_past = data["ego_past"]
        valid_mask = (ego_past[:, 0] != 0) | (ego_past[:, 1] != 0)
        if np.any(valid_mask):
            ax.plot(
                ego_past[valid_mask, 0],
                ego_past[valid_mask, 1],
                "g--",
                linewidth=3,
                alpha=0.8,
                label="Ego Past",
            )

    # Load neighbor agents data
    neighbor_past = data["neighbor_agents_past"]
    neighbor_future = data.get("neighbor_agents_future")

    agent_id = 0
    agent_ids: dict[int, int] = {}

    # ========== 3. Neighbor agents past trajectories ==========
    for agent_idx in range(neighbor_past.shape[0]):
        curr_x = neighbor_past[agent_idx, -1, 0]
        curr_y = neighbor_past[agent_idx, -1, 1]
        past = neighbor_past[agent_idx, :, 0:2]

        if abs(curr_x) < 0.1 and abs(curr_y) < 0.1:
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
        neighbor_heading = angle_difference_radians(ego_heading, np.arctan2(sin_h, cos_h))

        if width == 0:
            width = 2.0
        if length == 0:
            length = 4.0

        if agent_type == "Vehicle":
            rect = patches.Rectangle(
                (curr_x - length / 2, curr_y - width / 2),
                length,
                width,
                angle=math.degrees(neighbor_heading),
                facecolor=color,
                edgecolor="black",
                linewidth=1,
                alpha=0.7,
            )
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
    static_objs = data.get("static_objects")
    if static_objs is not None and len(static_objs) > 0:
        for obj_idx in range(static_objs.shape[0]):
            x, y = static_objs[obj_idx, 0], static_objs[obj_idx, 1]
            if x == 0 and y == 0:
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
