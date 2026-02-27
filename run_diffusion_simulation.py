#!/usr/bin/env python3
"""
run_diffusion_simulation.py - Run nuPlan closed-loop simulation with Diffusion Planner

Usage:
    python run_diffusion_simulation.py --scenarios_file=data/scenarios_200_valid.csv
    python run_diffusion_simulation.py --scenario=TOKEN
    python run_diffusion_simulation.py --num=200
"""

import argparse
import subprocess
import os

CONFIG_FILE = '/workspace/nuplan-visualization/nuplan/planning/script/config/common/scenario_filter/one_hand_picked_scenario.yaml'

def main():
    parser = argparse.ArgumentParser(description='Run Diffusion Planner simulation')
    parser.add_argument('--scenarios_file', type=str, default=None,
                        help='CSV file with scenario tokens')
    parser.add_argument('--scenario', type=str, default=None, help='Specific scenario token')
    parser.add_argument('--num', type=int, default=None, help='Number of random scenarios')
    args = parser.parse_args()
    
    # Require explicit specification
    if not args.scenarios_file and not args.scenario and not args.num:
        parser.error("必须指定 --scenarios_file, --scenario 或 --num")
    
    os.environ['PYTHONPATH'] = '/workspace/nuplan-visualization:/workspace/diffusion_planner'
    
    if args.scenarios_file:
        # Read tokens from file
        tokens = []
        with open(args.scenarios_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                if parts:
                    tokens.append(parts[0])
        
        print(f"Loaded {len(tokens)} scenarios from {args.scenarios_file}")
        
        # Write config with tokens
        tokens_yaml = '\n'.join([f"  - '{t}'" for t in tokens])
        with open(CONFIG_FILE, 'w') as f:
            f.write(f"""_target_: nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter
scenario_types: null
scenario_tokens:
{tokens_yaml}
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
            'planner=diffusion_planner',
            'scenario_builder=nuplan',
            'scenario_filter=one_hand_picked_scenario',
            'scenario_builder.data_root=/workspace/data/nuplan/data/cache/mini',
            'worker=sequential',
            'verbose=true'
        ]
    elif args.scenario:
        # Single scenario
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
            'planner=diffusion_planner',
            'scenario_builder=nuplan',
            'scenario_filter=one_hand_picked_scenario',
            'scenario_builder.data_root=/workspace/data/nuplan/data/cache/mini',
            'worker=sequential',
            'verbose=true'
        ]
    elif args.num:
        # Random N scenarios
        cmd = [
            'python3', '-m', 'nuplan.planning.script.run_simulation',
            '+simulation=closed_loop_nonreactive_agents',
            'planner=diffusion_planner',
            'scenario_builder=nuplan',
            'scenario_filter=simulation_test_split',
            'scenario_builder.data_root=/workspace/data/nuplan/data/cache/mini',
            f'+num_scenarios={args.num}',
            'worker=sequential',
            'verbose=true'
        ]
    else:
        # Default: run 1 random scenario
        cmd = [
            'python3', '-m', 'nuplan.planning.script.run_simulation',
            '+simulation=closed_loop_nonreactive_agents',
            'planner=diffusion_planner',
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
