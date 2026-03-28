#!/usr/bin/env bash
set -euo pipefail

# run_nuboard_legacy.sh - Legacy nuBoard launcher (no simulation_path preloaded)
# Usage: ./run_nuboard_legacy.sh [port_number]
# Default port: 5007 (to avoid conflict with training container's TensorBoard on 5006)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Prefer container path layout when available.
if [[ -d /workspace/nuplan-visualization ]]; then
  WORKSPACE=/workspace
else
  WORKSPACE="${REPO_ROOT}"
fi

export PYTHONPATH="${WORKSPACE}/nuplan-visualization"

PORT=${1:-5007}

cd "${WORKSPACE}/nuplan-visualization"
python3 -m nuplan.planning.script.run_nuboard \
  scenario_builder=nuplan \
  'scenario_builder.data_root=/workspace/data/nuplan/data/cache/mini' \
  "port_number=${PORT}"
