#!/usr/bin/env bash
set -euo pipefail

# run_nuboard_outputs.sh
# One-click launcher: open nuBoard pointing at the latest simulation output under:
#   outputs/sim/exp/simulation/**/<timestamp>/
#
# Usage:
#   ./run_nuboard_outputs.sh                # uses latest output, port 5007
#   ./run_nuboard_outputs.sh 5010           # uses latest output, port 5010
#   ./run_nuboard_outputs.sh <sim_dir>      # uses given output dir, port 5007
#   ./run_nuboard_outputs.sh 5010 <sim_dir> # uses given output dir, port 5010

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# In Docker we usually have:
#   - repo mounted at: /workspace/<repo_name> (this script lives inside it)
#   - nuplan-visualization mounted at: /workspace/nuplan-visualization
# We should therefore use:
#   - REPO_ROOT for outputs
#   - /workspace/nuplan-visualization for nuBoard code when available
if [[ -d /workspace/nuplan-visualization ]]; then
  NUPLAN_VIZ_ROOT="/workspace/nuplan-visualization"
else
  NUPLAN_VIZ_ROOT="${REPO_ROOT}/nuplan-visualization"
fi

# Default outputs root is relative to this repo.
OUTPUTS_ROOT_DEFAULT="${REPO_ROOT}/outputs/sim/exp/simulation"

usage() {
  cat <<'EOF'
Usage:
  ./run_nuboard_outputs.sh                # latest output, port 5007
  ./run_nuboard_outputs.sh 5010           # latest output, port 5010
  ./run_nuboard_outputs.sh <sim_dir>      # given output dir, port 5007
  ./run_nuboard_outputs.sh 5010 <sim_dir> # given output dir, port 5010

Notes:
  - <sim_dir> should be a timestamped simulation output directory like:
      outputs/sim/exp/simulation/<experiment>/<timestamp>/
  - Override outputs root via SIM_OUTPUTS_ROOT env var.
EOF
}

is_number() {
  [[ "${1:-}" =~ ^[0-9]+$ ]]
}

PORT=5007
SIM_DIR=""

if [[ $# -eq 0 ]]; then
  :
elif [[ $# -eq 1 ]]; then
  if is_number "$1"; then
    PORT="$1"
  else
    SIM_DIR="$1"
  fi
elif [[ $# -eq 2 ]]; then
  if is_number "$1"; then
    PORT="$1"
    SIM_DIR="$2"
  else
    echo "ERROR: If providing two args, first must be port number." >&2
    usage >&2
    exit 2
  fi
else
  usage >&2
  exit 2
fi

export PYTHONPATH="${NUPLAN_VIZ_ROOT}"

OUTPUTS_ROOT="${SIM_OUTPUTS_ROOT:-${OUTPUTS_ROOT_DEFAULT}}"

find_latest_sim_dir() {
  local root="$1"

  if [[ ! -d "${root}" ]]; then
    echo ""  # no outputs
    return 0
  fi

  # Expect directories like: <root>/<experiment>/<timestamp>
  # Timestamp format used by nuPlan: YYYY.MM.DD.HH.MM.SS (lexicographically sortable)
  local latest
  latest=$(find "${root}" -mindepth 2 -maxdepth 2 -type d \
    -regextype posix-extended \
    -regex '.*/[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]{2}\.[0-9]{2}\.[0-9]{2}$' \
    2>/dev/null | sort | tail -n 1 || true)

  echo "${latest}"
}

if [[ -z "${SIM_DIR}" ]]; then
  SIM_DIR="$(find_latest_sim_dir "${OUTPUTS_ROOT}")"
  if [[ -z "${SIM_DIR}" ]]; then
    cat >&2 <<EOF
ERROR: No simulation outputs found.

Expected at least one directory under:
  ${OUTPUTS_ROOT}/<experiment>/<timestamp>/

Example:
  ${OUTPUTS_ROOT}/closed_loop_nonreactive_agents/2026.03.28.17.17.40/

Tips:
  - Run a simulation first (see scripts/run_simulation.py).
  - If your outputs live elsewhere, set SIM_OUTPUTS_ROOT=/path/to/outputs/sim/exp/simulation
EOF
    exit 1
  fi
fi

# Normalize relative paths to absolute (relative to repo root).
if [[ "${SIM_DIR}" != /* ]]; then
  SIM_DIR="${REPO_ROOT}/${SIM_DIR}"
fi

if [[ ! -d "${SIM_DIR}" ]]; then
  echo "ERROR: simulation output dir does not exist: ${SIM_DIR}" >&2
  exit 1
fi

echo "[nuBoard] Using simulation_path: ${SIM_DIR}"
echo "[nuBoard] Starting on port: ${PORT}"

# nuBoard needs a scenario_builder config to load scenario metadata.
# Default to the nuPlan mini cache path used by our containers.
NUPLAN_DATA_ROOT_DEFAULT="/workspace/data/nuplan/data/cache/mini"
NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-${NUPLAN_DATA_ROOT_DEFAULT}}"

if [[ ! -d "${NUPLAN_DATA_ROOT}" ]]; then
  cat >&2 <<EOF
ERROR: NUPLAN_DATA_ROOT does not exist: ${NUPLAN_DATA_ROOT}

Set it explicitly, e.g.:
  NUPLAN_DATA_ROOT=/workspace/data/nuplan/data/cache/mini \
    ./scripts/run_nuboard.py
EOF
  exit 1
fi

echo "[nuBoard] scenario_builder.data_root=${NUPLAN_DATA_ROOT}"

cd "${NUPLAN_VIZ_ROOT}"

python3 -m nuplan.planning.script.run_nuboard \
  scenario_builder=nuplan \
  "scenario_builder.data_root=${NUPLAN_DATA_ROOT}" \
  "simulation_path=[${SIM_DIR}]" \
  "port_number=${PORT}"
