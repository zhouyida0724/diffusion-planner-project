#!/usr/bin/env bash
set -euo pipefail

# Exporter runner for stride16 v0.1 plans, split into 4 parts per city.
# Total parts: 4 cities * 4 quarters = 16 jobs, executed sequentially.
#
# Output layout:
#   <OUT_ROOT>/<city>_q0/shard_000...
#   <OUT_ROOT>/<city>_q1/...
#   <OUT_ROOT>/<city>_q2/...
#   <OUT_ROOT>/<city>_q3/...
#
# Usage:
#   bash scripts/export/diffusion_planner/pipeline/run_export_4cities_stride16_shard10_split4.sh [OUT_ROOT]
#
# Optional env:
#   LIMIT_PER_SHARD (default 200000)
#   NUM_SHARDS (default 10)
#   EXPORT_SCHEDULE (default db_grouped)
#   MAX_PARALLEL (default NUM_SHARDS)  # concurrency throttle inside run_export_sharded_mod.sh
#   START_CITY (e.g. vegas_1)
#   START_Q (0..3)

ulimit -n 65535 2>/dev/null || true
echo "ULIMIT_NOFILE=$(ulimit -n 2>/dev/null || echo unknown)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if command -v git >/dev/null 2>&1; then
  if git_root=$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel 2>/dev/null); then
    REPO_ROOT="$git_root"
  else
    REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
  fi
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
fi

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
PID_FILE="$STATUS_DIR/active.pid"
DONE_FILE="$STATUS_DIR/DONE"

echo $$ > "$PID_FILE"
rm -f "$DONE_FILE"

# Boston is already done; only export the remaining 3 cities.
declare -a CITIES=(vegas_1 pittsburgh singapore)

declare -A BASE_PLAN_DIRS
BASE_PLAN_DIRS[boston]="$REPO_ROOT/plans/plan_v0.1_scene_lidar_stride16_boston"
BASE_PLAN_DIRS[vegas_1]="$REPO_ROOT/plans/plan_v0.1_scene_lidar_stride16_vegas_1"
BASE_PLAN_DIRS[pittsburgh]="$REPO_ROOT/plans/plan_v0.1_scene_lidar_stride16_pittsburgh"
BASE_PLAN_DIRS[singapore]="$REPO_ROOT/plans/plan_v0.1_scene_lidar_stride16_singapore"

split_if_needed() {
  local base="$1"   # base plan dir
  local prefix="$2" # output prefix, will create ${prefix}0..${prefix}3

  local ok=1
  for i in 0 1 2 3; do
    if [[ ! -f "${prefix}${i}/index.jsonl" ]]; then
      ok=0
      break
    fi
  done
  if [[ $ok -eq 1 ]]; then
    return 0
  fi

  python3 "$REPO_ROOT/scripts/export/diffusion_planner/pipeline/split_plan_index_jsonl.py" \
    --in "$base" \
    --out-prefix "$prefix" \
    --parts 4
}

run_quarter() {
  local city="$1"
  local q="$2"   # 0..3
  local base_plan="${BASE_PLAN_DIRS[$city]}"
  local plan_dir="${base_plan}_q${q}"
  local out_city="$OUT_ROOT/${city}_q${q}"

  if [[ ! -f "$base_plan/index.jsonl" ]]; then
    echo "[skip] missing base plan: $base_plan/index.jsonl" | tee -a "$STATUS_DIR/run_${STAMP}.log"
    printf '{"t":"%s","city":"%s","part":"q%s","status":"missing_plan","plan_dir":"%s"}\n' "$(date -Is)" "$city" "$q" "$base_plan" >> "$STATUS_JSON"
    return 0
  fi

  split_if_needed "$base_plan" "${base_plan}_q"

  # Skip already-complete outputs to avoid overwrites.
  if [[ -d "$out_city" ]]; then
    local complete=1
    for ((SID=0; SID<NUM_SHARDS; SID++)); do
      local sd="$out_city/shard_$(printf '%03d' $SID)"
      if [[ ! -f "$sd/data.npz" || ! -f "$sd/metrics.json" || ! -f "$sd/manifest.jsonl" ]]; then
        complete=0
        break
      fi
    done
    if [[ $complete -eq 1 ]]; then
      echo "[skip] $city q$q already complete: $out_city" | tee -a "$STATUS_DIR/run_${STAMP}.log"
      printf '{"t":"%s","city":"%s","part":"q%s","status":"skip_complete","out":"%s"}\n' "$(date -Is)" "$city" "$q" "$out_city" >> "$STATUS_JSON"
      return 0
    fi
  fi

  mkdir -p "$out_city"

  echo "[part] $city q$q plan=$plan_dir out=$out_city" | tee -a "$STATUS_DIR/run_${STAMP}.log"
  printf '{"t":"%s","city":"%s","part":"q%s","status":"start","plan_dir":"%s","out":"%s","limit_per_shard":%s,"num_shards":%s,"schedule":"%s"}\n' \
    "$(date -Is)" "$city" "$q" "$plan_dir" "$out_city" "$LIMIT_PER_SHARD" "$NUM_SHARDS" "$EXPORT_SCHEDULE" >> "$STATUS_JSON"

  set +e
  bash "$REPO_ROOT/scripts/export/diffusion_planner/pipeline/run_export_sharded_mod.sh" \
    "$plan_dir" \
    "$out_city" \
    "$LIMIT_PER_SHARD" \
    "$NUM_SHARDS"
  local rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "[fail] $city q$q rc=$rc (continuing)" | tee -a "$STATUS_DIR/run_${STAMP}.log"
    printf '{"t":"%s","city":"%s","part":"q%s","status":"failed","rc":%s}\n' "$(date -Is)" "$city" "$q" "$rc" >> "$STATUS_JSON"
    return 0
  fi

  echo "[ok] $city q$q" | tee -a "$STATUS_DIR/run_${STAMP}.log"
  printf '{"t":"%s","city":"%s","part":"q%s","status":"done"}\n' "$(date -Is)" "$city" "$q" >> "$STATUS_JSON"
  return 0
}

for c in "${CITIES[@]}"; do
  for q in 0 1 2 3; do
    run_quarter "$c" "$q"
    sleep 4
  done
done

echo "[all done] status=$STATUS_JSON" | tee -a "$STATUS_DIR/run_${STAMP}.log"
date -Is > "$DONE_FILE"
rm -f "$PID_FILE"
