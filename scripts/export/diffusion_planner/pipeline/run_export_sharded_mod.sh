#!/usr/bin/env bash
set -euo pipefail

# Raise FD soft limit to avoid "Too many open files" (Errno 24) under multi-shard export.
# Keep best-effort: if hard limit is lower, ulimit will fail and we continue.
ulimit -n 65535 2>/dev/null || true
echo "ULIMIT_NOFILE=$(ulimit -n 2>/dev/null || echo unknown)"

PLAN_DIR=${1:?Usage: run_export_sharded_mod.sh <plan_dir> <out_root> [limit] [num_shards]}
OUT_ROOT=${2:?Usage: run_export_sharded_mod.sh <plan_dir> <out_root> [limit] [num_shards]}
LIMIT=${3:-0}
NUM_SHARDS=${4:-4}
MAX_PARALLEL=${MAX_PARALLEL:-$NUM_SHARDS}

mkdir -p "$OUT_ROOT"

echo "PLAN_DIR=$PLAN_DIR"
echo "OUT_ROOT=$OUT_ROOT"
echo "LIMIT=$LIMIT"
echo "NUM_SHARDS=$NUM_SHARDS"
echo "MAX_PARALLEL=$MAX_PARALLEL"

# Optional: locality-aware scheduling within each shard.
# Set EXPORT_SCHEDULE=db_grouped to group samples by db_path (deterministic) to reduce cross-DB random access.
EXPORT_SCHEDULE=${EXPORT_SCHEDULE:-}
if [[ -n "$EXPORT_SCHEDULE" ]]; then
  echo "EXPORT_SCHEDULE=$EXPORT_SCHEDULE"
fi

for ((SID=0; SID<NUM_SHARDS; SID++)); do
  OUT_DIR="$OUT_ROOT/shard_$(printf '%03d' $SID)"
  mkdir -p "$OUT_DIR"
  echo "[launch] shard=$SID -> $OUT_DIR"
  (
    # Run from repo root (works on host; container can still bind-mount repo to /workspace).
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
    cd "$REPO_ROOT"

    EXTRACT_LOG_STYLE=quiet EXTRACT_WARN_PRINT_N=0 \
      python3 scripts/export/diffusion_planner/pipeline/export_v0_1_single_npz.py \
        --plan "$PLAN_DIR" \
        --out "$OUT_DIR" \
        --limit "$LIMIT" \
        --num-shards "$NUM_SHARDS" \
        --shard-id "$SID" \
        ${EXPORT_SCHEDULE:+--schedule "$EXPORT_SCHEDULE"} \
        2> "$OUT_DIR/run.stderr.log"
  ) &

  # Concurrency throttle to avoid OOM when multiple shards hit stack/npz_save simultaneously.
  while true; do
    running=$(jobs -pr | wc -l | tr -d ' ')
    if [[ "${running:-0}" -lt "$MAX_PARALLEL" ]]; then
      break
    fi
    sleep 0.2
  done
done

wait

echo "DONE all shards"
