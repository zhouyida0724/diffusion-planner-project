#!/bin/bash
# run_simulation.sh - Run nuPlan closed-loop simulation
# Usage: 
#   ./run_simulation.sh                           # Run all scenarios in default test split
#   ./run_simulation.sh --scenario=folder_path   # Run scenarios in specific folder
#   ./run_simulation.sh --num=5                  # Run 5 scenarios

set -e

# Configuration
NUPLAN_DATA_ROOT="/workspace/data/nuplan"
NUPLAN_MAPS_ROOT="/workspace/data/nuplan/maps"
NUPLAN_EXP_ROOT="/workspace/data/nuplan/exp"
DATA_PATH="${NUPLAN_DATA_ROOT}/data/cache/mini"

# Parse arguments
SCENARIO_FILTER="simulation_test_split"
NUM_SCENARIOS=""
CUSTOM_SCENARIO=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --scenario=*)
            CUSTOM_SCENARIO="${1#*=}"
            SCENARIO_FILTER="one_hand_picked_scenario"
            ;;
        --scenario)
            CUSTOM_SCENARIO="$2"
            SCENARIO_FILTER="one_hand_picked_scenario"
            shift
            ;;
        --num=*)
            NUM_SCENARIOS="+num_scenarios=${1#*=}"
            ;;
        --num)
            NUM_SCENARIOS="+num_scenarios=$2"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --scenario=FOLDER   Run specific scenario folder"
            echo "  --num=N             Run N scenarios"
            echo "  --help, -h          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# Build command
CMD="python3 -m nuplan.planning.script.run_simulation \
    '+simulation=closed_loop_nonreactive_agents' \
    planner=idm_planner \
    scenario_builder=nuplan \
    scenario_filter=${SCENARIO_FILTER} \
    'scenario_builder.data_root=${DATA_PATH}' \
    worker=sequential \
    verbose=true"

# Add custom scenario if specified
if [ -n "$CUSTOM_SCENARIO" ]; then
    CMD="$CMD 'scenario_filter.log_name=${CUSTOM_SCENARIO}'"
fi

# Add num scenarios if specified
if [ -n "$NUM_SCENARIOS" ]; then
    CMD="$CMD $NUM_SCENARIOS"
fi

echo "Running simulation..."
echo "Command: $CMD"
eval $CMD
