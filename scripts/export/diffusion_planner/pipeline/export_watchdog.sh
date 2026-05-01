#!/usr/bin/env bash
set -euo pipefail

# Cron-friendly watchdog:
# - Writes watch_latest.json (progress snapshot)
# - If no active exporter PID and not DONE, starts the sequential export runner.
# - If PID exists but is dead, restarts.
# - If export appears stalled (no manifest updates for a while) while no exporter PID, restarts.
# - Never kills a running exporter (safe by default).

OUT_ROOT="${1:-/media/zhouyida/新加卷1/exports_stride16_v0.1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
if command -v git >/dev/null 2>&1; then
  if git_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null); then
    REPO_ROOT="$git_root"
  fi
fi

STATUS_DIR="$OUT_ROOT/_status"
PID_FILE="$STATUS_DIR/active.pid"
DONE_FILE="$STATUS_DIR/DONE"
LOG_FILE="$STATUS_DIR/watchdog.log"

mkdir -p "$STATUS_DIR"

echo "[$(date -Is)] tick" >> "$LOG_FILE"

# Always write a snapshot.
python3 "$REPO_ROOT/scripts/export/diffusion_planner/pipeline/watch_export_4cities.py" "$OUT_ROOT" >> "$LOG_FILE" 2>&1 || true

start_runner() {
  echo "[$(date -Is)] starting runner" >> "$LOG_FILE"
  # Run detached so cron won't block; runner writes active.pid + DONE.
  nohup bash "$REPO_ROOT/scripts/export/diffusion_planner/pipeline/run_export_4cities_stride16_shard10.sh" "$OUT_ROOT" >> "$STATUS_DIR/runner.out.log" 2>&1 &
  echo "[$(date -Is)] runner launched pid=$!" >> "$LOG_FILE"
}

# Extra safety: if any exporter processes are already running, don't start another.
if pgrep -f "export_v0_1_single_npz.py" >/dev/null 2>&1 || pgrep -f "run_export_4cities_stride16_shard10.sh" >/dev/null 2>&1; then
  echo "[$(date -Is)] exporter processes already running; skip start" >> "$LOG_FILE"
  exit 0
fi

# If already done, don't restart.
if [[ -f "$DONE_FILE" ]]; then
  echo "[$(date -Is)] DONE present, skip" >> "$LOG_FILE"
  exit 0
fi

# If pid file exists, check liveness.
if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE" || true)
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "[$(date -Is)] active pid=$pid" >> "$LOG_FILE"
    exit 0
  fi
  echo "[$(date -Is)] stale pid_file (pid=$pid), restarting" >> "$LOG_FILE"
  rm -f "$PID_FILE"
  start_runner
  exit 0
fi

# No active pid and not done => start.
start_runner
