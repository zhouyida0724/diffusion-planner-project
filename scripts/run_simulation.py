#!/usr/bin/env python3
"""
run_simulation.py - Run nuPlan closed-loop simulation with IDM Planner

Usage:
    python run_simulation.py                     # Run 1 scenario
    python run_simulation.py --num=5             # Run 5 scenarios
    python run_simulation.py --scenario=TOKEN    # Run specific scenario
"""

import argparse
import subprocess
import os

CONFIG_FILE = '/workspace/nuplan-visualization/nuplan/planning/script/config/common/scenario_filter/one_hand_picked_scenario.yaml'

# Default config template
DEFAULT_CONFIG = """_target_: nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter
scenario_types: null
scenario_tokens: null
log_names: null
map_names: null
num_scenarios_per_type: 1
limit_total_scenarios: null
timestamp_threshold_s: null
ego_displacement_minimum_m: null
expand_scenarios: false
remove_invalid_goals: true
shuffle: false
"""

def main():
    parser = argparse.ArgumentParser(description='Run IDM simulation')
    parser.add_argument('--num', type=int, default=1, help='Number of scenarios')
    parser.add_argument('--scenario', type=str, default=None, help='Specific scenario token')
    args = parser.parse_args()
    
    os.environ['PYTHONPATH'] = '/workspace/nuplan-visualization'
    
    if args.scenario:
        # Modify config to use specific scenario
        with open(CONFIG_FILE, 'w') as f:
            f.write(f"""_target_: nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter
scenario_types: null
scenario_tokens:
  - '{args.scenario}'
log_names: null
map_names: null
num_scenarios_per_type: 1
limit_total_scenarios: null
timestamp_threshold_s: null
ego_displacement_minimum_m: null
expand_scenarios: false
remove_invalid_goals: true
shuffle: false
""")
        
        cmd = [
            'python3', '-m', 'nuplan.planning.script.run_simulation',
            '+simulation=closed_loop_nonreactive_agents',
            'planner=idm_planner',
            'scenario_builder=nuplan',
            'scenario_filter=one_hand_picked_scenario',
            'scenario_builder.data_root=/workspace/data/nuplan/data/cache/mini',
            'worker=sequential',
            'verbose=true'
        ]
    else:
        cmd = [
            'python3', '-m', 'nuplan.planning.script.run_simulation',
            '+simulation=closed_loop_nonreactive_agents',
            'planner=idm_planner',
            'scenario_builder=nuplan',
            'scenario_filter=simulation_test_split',
            'scenario_builder.data_root=/workspace/data/nuplan/data/cache/mini',
            f'+num_scenarios_per_type={args.num}',
            'worker=sequential',
            'verbose=true'
        ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=os.environ)

if __name__ == '__main__':
    main()
