#!/usr/bin/env bash
set -euo pipefail

PLAN_DIR=${1:?Usage: run_export_sharded_mod.sh <plan_dir> <out_root> [limit] [num_shards]}
OUT_ROOT=${2:?Usage: run_export_sharded_mod.sh <plan_dir> <out_root> [limit] [num_shards]}
LIMIT=${3:-0}
NUM_SHARDS=${4:-4}

mkdir -p "$OUT_ROOT"

echo "PLAN_DIR=$PLAN_DIR"
echo "OUT_ROOT=$OUT_ROOT"
echo "LIMIT=$LIMIT"
echo "NUM_SHARDS=$NUM_SHARDS"

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
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
    if command -v git >/dev/null 2>&1; then
      if git_root=$(cd "$REPO_ROOT" && git rev-parse --show-toplevel 2>/dev/null); then
        REPO_ROOT="$git_root"
      fi
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
done

wait

echo "DONE all shards"
