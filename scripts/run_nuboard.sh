#!/bin/bash
# run_nuboard.sh - Launch nuBoard visualization
# Usage: ./run_nuboard.sh [port_number]
# Default port: 5007 (to avoid conflict with training container's TensorBoard on 5006)

export PYTHONPATH=/workspace/nuplan-visualization

PORT=${1:-5007}

cd /workspace/nuplan-visualization
python3 -m nuplan.planning.script.run_nuboard \
    scenario_builder=nuplan \
    'scenario_builder.data_root=/workspace/data/nuplan/data/cache/mini' \
    port_number=$PORT
