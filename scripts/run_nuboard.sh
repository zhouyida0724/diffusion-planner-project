#!/bin/bash
# run_nuboard.sh - Start nuBoard visualization server
# Usage: ./run_nuboard.sh

set -e

# Configuration
NUPLAN_DATA_ROOT="/workspace/data/nuplan"
NUPLAN_MAPS_ROOT="${NUPLAN_DATA_ROOT}/maps"
NUPLAN_EXP_ROOT="${NUPLAN_DATA_ROOT}/exp"

# Kill existing nuBoard process
echo "Stopping any existing nuBoard processes..."
pkill -f run_nuboard || true
sleep 2

# Set environment variables
export NUPLAN_DEVKIT_ROOT=/workspace/nuplan-devkit
export NUPLAN_DATA_ROOT=$NUPLAN_DATA_ROOT
export NUPLAN_MAPS_ROOT=$NUPLAN_MAPS_ROOT
export NUPLAN_EXP_ROOT=$NUPLAN_EXP_ROOT
export PYTHONPATH=/workspace/nuplan-devkit:$PYTHONPATH

cd /workspace/nuplan-devkit

echo "Starting nuBoard on http://localhost:5006 ..."
python3 -m nuplan.planning.script.run_nuboard \
    scenario_builder=nuplan \
    'scenario_builder.data_root=/workspace/data/nuplan/data/cache/mini' \
    > /tmp/nuboard.log 2>&1 &

# Wait for server to start
sleep 5

# Check if server is running
if curl -s http://localhost:5006 > /dev/null 2>&1; then
    echo "✅ nuBoard is running at http://localhost:5006"
else
    echo "❌ Failed to start nuBoard. Check /tmp/nuboard.log for details."
    exit 1
fi
