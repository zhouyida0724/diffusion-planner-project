#!/usr/bin/env python3
"""
run_simulation.py - Run nuPlan closed-loop simulation with IDM Planner
"""

import argparse
import subprocess
import os
import json
import yaml

CONFIG_FILE = '/workspace/nuplan-visualization/nuplan/planning/script/config/common/scenario_filter/one_hand_picked_scenario.yaml'

def main():
    parser = argparse.ArgumentParser(description='Run IDM simulation')
    parser.add_argument('--scenarios_file', type=str, default=None, 
                        help='CSV file with log names')
    parser.add_argument('--yaml_file', type=str, default=None,
                        help='YAML config file with scenario filter')
    parser.add_argument('--scenario', type=str, default=None, help='Specific scenario token')
    parser.add_argument('--num', type=int, default=None, help='Number of random scenarios')
    args = parser.parse_args()
    
    # Require explicit specification
    if not args.scenarios_file and not args.scenario and not args.num:
        parser.error("必须指定 --scenarios_file, --scenario 或 --num")
    
    os.environ['PYTHONPATH'] = '/workspace/nuplan-visualization'
    
    # If YAML file provided, copy it to config location
    if args.yaml_file:
        with open(args.yaml_file, 'r') as f:
            config_content = f.read()
        with open(CONFIG_FILE, 'w') as f:
            f.write(config_content)
        
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
    elif args.scenarios_file:
        # Read log names from CSV (first column is log_name)
        log_names = []
        with open(args.scenarios_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                if parts:
                    log_names.append(parts[0])
        
        print(f"Loaded {len(log_names)} log names from {args.scenarios_file}")
        
        # Write config with log_names
        with open(CONFIG_FILE, 'w') as f:
            f.write("""_target_: nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter

scenario_tokens: null
log_names:
""")
            for log in log_names:
                f.write(f"  - '{log}'\n")
            f.write("""scenario_type: null
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
    elif args.num:
        cmd = [
            'python3', '-m', 'nuplan.planning.script.run_simulation',
            '+simulation=closed_loop_nonreactive_agents',
            'planner=idm_planner',
            'scenario_builder=nuplan',
            'scenario_filter=simulation_test_split',
            'scenario_builder.data_root=/workspace/data/nuplan/data/cache/mini',
            f'+num_scenarios={args.num}',
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
            '+num_scenarios=1',
            'worker=sequential',
            'verbose=true'
        ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=os.environ)

if __name__ == '__main__':
    main()
