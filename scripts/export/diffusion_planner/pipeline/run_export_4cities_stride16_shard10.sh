#!/usr/bin/env bash
set -euo pipefail

# Sequential exporter for 4 cities.
# - Uses scene-local lidar_pc frame_index (stride16 plans).
# - Runs 10 shards per city, cap 200k kept per shard.
# - If a city fails, record it and continue to next.
#
# Usage:
#   bash scripts/export/diffusion_planner/pipeline/run_export_4cities_stride16_shard10.sh [OUT_ROOT]
#
# Optional env:
#   LIMIT_PER_SHARD (default 200000)
#   NUM_SHARDS (default 10)
#   EXPORT_SCHEDULE (default db_grouped)
#   EXTRACT_SINGLE_FRAME_IMPL must be 'py' (guarded in extractor)
#   EXPORT_FRAME_INDEX_MODE (default scene_lidar)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
OUT_ROOT="${1:-/media/zhouyida/新加卷1/exports_stride16_v0.1}"

LIMIT_PER_SHARD="${LIMIT_PER_SHARD:-200000}"
NUM_SHARDS="${NUM_SHARDS:-10}"
EXPORT_SCHEDULE="${EXPORT_SCHEDULE:-db_grouped}"

export EXTRACT_SINGLE_FRAME_IMPL="py"
export EXPORT_FRAME_INDEX_MODE="${EXPORT_FRAME_INDEX_MODE:-scene_lidar}"
export EXPORT_SCHEDULE

mkdir -p "$OUT_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
STATUS_DIR="$OUT_ROOT/_status"
mkdir -p "$STATUS_DIR"
STATUS_JSON="$STATUS_DIR/run_${STAMP}.jsonl"

declare -a CITIES=(boston vegas_1 pittsburgh singapore)

declare -A PLAN_DIRS
PLAN_DIRS[boston]="$REPO_ROOT/plans/plan_v0.1_scene_lidar_stride16_boston"
PLAN_DIRS[vegas_1]="$REPO_ROOT/plans/plan_v0.1_scene_lidar_stride16_vegas_1"
PLAN_DIRS[pittsburgh]="$REPO_ROOT/plans/plan_v0.1_scene_lidar_stride16_pittsburgh"
PLAN_DIRS[singapore]="$REPO_ROOT/plans/plan_v0.1_scene_lidar_stride16_singapore"

run_city() {
  local city="$1"
  local plan_dir="${PLAN_DIRS[$city]}"
  local out_city="$OUT_ROOT/$city"

  if [[ ! -f "$plan_dir/index.jsonl" ]]; then
    echo "[skip] missing plan: $plan_dir/index.jsonl" | tee -a "$STATUS_DIR/run_${STAMP}.log"
    printf '{"t":"%s","city":"%s","status":"missing_plan","plan_dir":"%s"}\n' "$(date -Is)" "$city" "$plan_dir" >> "$STATUS_JSON"
    return 0
  fi

  mkdir -p "$out_city"

  echo "[city] $city plan=$plan_dir out=$out_city" | tee -a "$STATUS_DIR/run_${STAMP}.log"
  printf '{"t":"%s","city":"%s","status":"start","plan_dir":"%s","out":"%s","limit_per_shard":%s,"num_shards":%s,"schedule":"%s"}\n' \
    "$(date -Is)" "$city" "$plan_dir" "$out_city" "$LIMIT_PER_SHARD" "$NUM_SHARDS" "$EXPORT_SCHEDULE" >> "$STATUS_JSON"

  set +e
  bash "$REPO_ROOT/scripts/export/diffusion_planner/pipeline/run_export_sharded_mod.sh" \
    "$plan_dir" \
    "$out_city" \
    "$LIMIT_PER_SHARD" \
    "$NUM_SHARDS"
  local rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "[fail] $city rc=$rc (continuing)" | tee -a "$STATUS_DIR/run_${STAMP}.log"
    printf '{"t":"%s","city":"%s","status":"failed","rc":%s}\n' "$(date -Is)" "$city" "$rc" >> "$STATUS_JSON"
    return 0
  fi

  echo "[ok] $city" | tee -a "$STATUS_DIR/run_${STAMP}.log"
  printf '{"t":"%s","city":"%s","status":"done"}\n' "$(date -Is)" "$city" >> "$STATUS_JSON"
  return 0
}

for c in "${CITIES[@]}"; do
  run_city "$c"
  # small cooldown to reduce IO spikes between cities
  sleep 5
  
  # If user wants to stop after a city, allow early exit.
  if [[ "${STOP_AFTER_CITY:-}" == "$c" ]]; then
    echo "[stop] STOP_AFTER_CITY=$c" | tee -a "$STATUS_DIR/run_${STAMP}.log"
    break
  fi
done

echo "[all done] status=$STATUS_JSON" | tee -a "$STATUS_DIR/run_${STAMP}.log"
