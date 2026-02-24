#!/bin/bash
# run_nuboard.sh - Launch nuBoard visualization

export PYTHONPATH=/workspace/nuplan-visualization

cd /workspace/nuplan-visualization
python3 -m nuplan.planning.script.run_nuboard \
    scenario_builder=nuplan \
    'scenario_builder.data_root=/workspace/data/nuplan/data/cache/mini'
