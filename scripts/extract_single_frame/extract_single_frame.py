#!/usr/bin/env python3
"""Complete feature extraction for nuPlan scenario using map_api with valid marking.

This script is a backwards-compatible entrypoint.

Core extraction logic lives in:
  src/platform/nuplan/features/extract_single_frame.py

Do NOT change the CLI/usage of this script.
"""

from __future__ import annotations

import os
import sys
import numpy as np

# Add nuplan-visualization to path (legacy behavior)
sys.path.insert(0, "/workspace/nuplan-visualization")
_local_nuplan_vis = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "nuplan-visualization"))
if os.path.isdir(_local_nuplan_vis):
    sys.path.insert(0, _local_nuplan_vis)

# Ensure repo root is importable so we can import src/ modules when running this file directly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nuplan.common.maps.nuplan_map.map_factory import get_maps_api  # noqa: E402

from src.platform.nuplan.features import extract_single_frame as core  # noqa: E402

# --------------------------------------------------------------------------------------
# Public constants (kept for backwards compatibility with other scripts/tests)
# --------------------------------------------------------------------------------------

DB_PATH = core.DB_PATH
MAP_ROOT = core.MAP_ROOT
MAP_VERSION = core.MAP_VERSION
MAP_NAME = core.MAP_NAME

EGO_FUTURE_LEN = core.EGO_FUTURE_LEN
EGO_HISTORY_LEN = core.EGO_HISTORY_LEN
NEIGHBOR_HISTORY_LEN = core.NEIGHBOR_HISTORY_LEN
NEIGHBOR_FUTURE_LEN = core.NEIGHBOR_FUTURE_LEN
MAX_NEIGHBORS = core.MAX_NEIGHBORS
MAX_STATIC_OBJECTS = core.MAX_STATIC_OBJECTS
MAX_LANES = core.MAX_LANES
MAX_ROUTE_LANES = core.MAX_ROUTE_LANES
POLYLINE_LEN = core.POLYLINE_LEN
LANE_DIM = core.LANE_DIM

# Target scenario (legacy defaults)
SCENARIO_TOKEN = "037db12ac9125b9a"
CENTER_FRAME_INDEX = 17486

OUTPUT_PATH = "/workspace/data_process/npz_scenes/test_ego_past.npz"
CSV_OUTPUT_PATH = "/workspace/diffusion-planner-project/data_process/npz_scenes/las_vegas_hs_17486.csv"

# Public logger/stats API (used by batch exporters)
EXTRACT_LOG_STYLE = core.EXTRACT_LOG_STYLE
DEBUG_LOGGER = core.DEBUG_LOGGER
reset_log_stats = core.reset_log_stats
get_log_stats = core.get_log_stats


def _sync_core_globals() -> None:
    """Keep core module globals in sync with this script's user-configurable globals."""
    core.DB_PATH = DB_PATH
    core.MAP_ROOT = MAP_ROOT
    core.MAP_VERSION = MAP_VERSION
    core.MAP_NAME = MAP_NAME

    # Constants are identical; keep them in sync for safety.
    core.EGO_FUTURE_LEN = EGO_FUTURE_LEN
    core.EGO_HISTORY_LEN = EGO_HISTORY_LEN
    core.NEIGHBOR_HISTORY_LEN = NEIGHBOR_HISTORY_LEN
    core.NEIGHBOR_FUTURE_LEN = NEIGHBOR_FUTURE_LEN
    core.MAX_NEIGHBORS = MAX_NEIGHBORS
    core.MAX_STATIC_OBJECTS = MAX_STATIC_OBJECTS
    core.MAX_LANES = MAX_LANES
    core.MAX_ROUTE_LANES = MAX_ROUTE_LANES
    core.POLYLINE_LEN = POLYLINE_LEN
    core.LANE_DIM = LANE_DIM

    global DEBUG_LOGGER
    core.DEBUG_LOGGER = DEBUG_LOGGER


# --------------------------------------------------------------------------------------
# Re-export core helper functions (imported by regression tests and batch exporters)
# --------------------------------------------------------------------------------------

load_db = core.load_db
get_location_from_log = core.get_location_from_log
map_name_from_location = core.map_name_from_location
extract_features = core.extract_features


def generate_csv_summary(features, csv_path):
    """Generate CSV file with field summaries."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w") as f:
        f.write("field_name,shape,dtype,min,max,mean,nonzero_count,nonzero_percent\n")

        for name, arr in features.items():
            arr_min = float(arr.min())
            arr_max = float(arr.max())
            arr_mean = float(arr.mean())
            nonzero = np.count_nonzero(arr)
            total = arr.size
            nonzero_pct = nonzero / total * 100
            f.write(
                f"{name},{arr.shape},{arr.dtype},{arr_min:.4f},{arr_max:.4f},{arr_mean:.4f},{nonzero},{nonzero_pct:.2f}%\n"
            )

    print(f"CSV summary saved to: {csv_path}")


def run_extraction(
    db_path: str,
    scenario_token: str,
    center_frame_index: int,
    output_path: str,
    csv_output_path: str | None = None,
):
    """Programmatic entrypoint for batch runs.

    Kept for backwards compatibility.
    """

    global DB_PATH, SCENARIO_TOKEN, CENTER_FRAME_INDEX, OUTPUT_PATH, CSV_OUTPUT_PATH, DEBUG_LOGGER

    DB_PATH = db_path
    SCENARIO_TOKEN = scenario_token
    CENTER_FRAME_INDEX = int(center_frame_index)
    OUTPUT_PATH = output_path
    if csv_output_path is not None:
        CSV_OUTPUT_PATH = csv_output_path

    _sync_core_globals()

    # Re-init debug logger so each output gets its own log file name (legacy behavior).
    try:
        if EXTRACT_LOG_STYLE == "legacy":
            DEBUG_LOGGER = core._init_debug_logger(output_path=OUTPUT_PATH)
        else:
            DEBUG_LOGGER = core.DEBUG_LOGGER
        core.DEBUG_LOGGER = DEBUG_LOGGER
    except Exception:
        pass

    return main()


def main():
    print("=" * 60)
    print("Starting complete feature extraction WITH VALID MARKING...")
    print(f"Scenario: {SCENARIO_TOKEN}, Frame: {CENTER_FRAME_INDEX}")
    print("=" * 60)

    _sync_core_globals()

    conn = load_db()

    # Determine the correct map from DB metadata.
    global MAP_NAME
    location = get_location_from_log(conn)
    MAP_NAME = map_name_from_location(location)
    core.MAP_NAME = MAP_NAME
    print(f"DB location: {location} -> MAP_NAME: {MAP_NAME}")

    print("\n[5/7] Loading map API...")
    map_api = get_maps_api(MAP_ROOT, MAP_VERSION, MAP_NAME)
    print(f"  Map: {map_api.map_name}")

    features = extract_features(conn, map_api, SCENARIO_TOKEN, CENTER_FRAME_INDEX)

    conn.close()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez(OUTPUT_PATH, **features)

    print(f"\n{'=' * 60}")
    print(f"Saved NPZ to: {OUTPUT_PATH}")
    print(f"{'=' * 60}")

    generate_csv_summary(features, CSV_OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("FEATURE SUMMARY REPORT")
    print("=" * 60)

    for name, arr in features.items():
        arr_min = arr.min()
        arr_max = arr.max()
        nonzero = np.count_nonzero(arr)
        total = arr.size
        nonzero_pct = nonzero / total * 100
        print(f"\n{name}:")
        print(f"  Shape: {arr.shape}")
        print(f"  Value range: [{arr_min:.4f}, {arr_max:.4f}]")
        print(f"  Non-zero: {nonzero}/{total} ({nonzero_pct:.1f}%)")

    print("\n" + "=" * 60)
    print("LANE FEATURE DETAIL (dim 0-11)")
    print("=" * 60)
    lane_example = features["lanes"][0]
    for dim in range(LANE_DIM):
        dim_data = lane_example[:, dim]
        nonzero = np.count_nonzero(dim_data)
        print(f"  dim {dim}: nonzero={nonzero}/{POLYLINE_LEN}, min={dim_data.min():.4f}, max={dim_data.max():.4f}")

    print("\n" + "=" * 60)
    print("LANE VALID MARKING DETAIL")
    print("=" * 60)
    lanes_avails = features["lanes_avails"]
    valid_lanes = np.sum(np.any(lanes_avails, axis=1))
    total_valid_points = np.sum(lanes_avails)
    total_points = lanes_avails.size
    print(f"  Valid lanes: {valid_lanes}/{MAX_LANES}")
    print(
        f"  Valid points: {total_valid_points}/{total_points} ({total_valid_points/total_points*100:.1f}%)"
    )

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
