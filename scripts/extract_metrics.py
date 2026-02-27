#!/usr/bin/env python3
"""
extract_metrics.py - Extract metrics from nuPlan simulation results

Usage:
    python extract_metrics.py --exp_dir <path> --planner <planner_name> --output <output.csv>
    python extract_metrics.py --exp_dir 2026.02.25.10.00.00 --planner IDMPlanner --output idm_metrics.csv
    python extract_metrics.py --exp_dir 2026.02.25.10.00.00 --planner DiffusionPlanner --output diffusion_metrics.csv
"""

import argparse
import pickle
import os
import glob
import sys

DEFAULT_METRICS_DIR = '/workspace/data/nuplan/exp/exp/simulation/closed_loop_nonreactive_agents'
DEFAULT_SCENARIOS_FILE = '/workspace/data/scenarios_200_stratified.txt'


def extract_metrics(exp_dir, planner, scenarios_file, output_file):
    """Extract metrics from simulation results."""
    
    # Resolve experiment directory
    if not os.path.isabs(exp_dir):
        metrics_dir = os.path.join(DEFAULT_METRICS_DIR, exp_dir, 'metrics')
    else:
        metrics_dir = os.path.join(exp_dir, 'metrics')
    
    if not os.path.exists(metrics_dir):
        print(f"Error: Metrics directory not found: {metrics_dir}", file=sys.stderr)
        return False
    
    # Read scenario tokens
    if not os.path.exists(scenarios_file):
        print(f"Error: Scenarios file not found: {scenarios_file}", file=sys.stderr)
        return False
    
    with open(scenarios_file, 'r') as f:
        tokens = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Extracting metrics for {len(tokens)} scenarios from {exp_dir}")
    print(f"Planner: {planner}")
    print(f"Metrics dir: {metrics_dir}")
    
    results = []
    found_count = 0
    
    for token in tokens:
        # Find files with this token and planner
        # File pattern: *_{token}_{planner}.pickle.temp
        pattern = os.path.join(metrics_dir, f'*_{token}_{planner}.pickle.temp')
        files = glob.glob(pattern)
        
        if not files:
            results.append(f"{token}|NO_DATA")
            continue
        
        found_count += 1
        
        # Read all metric files and collect scores
        scores = {}
        for filepath in files:
            try:
                with open(filepath, 'rb') as pf:
                    data = pickle.load(pf)
                    for item in data:
                        metric_name = item.get('metric_computator', 'unknown')
                        score = item.get('metric_score')
                        if score is not None:
                            scores[metric_name] = score
            except Exception as e:
                print(f"Warning: Error reading {filepath}: {e}", file=sys.stderr)
        
        # Format as string
        score_str = ';'.join([f"{k}={v}" for k, v in scores.items()])
        results.append(f"{token}|{score_str}")
    
    # Save results
    with open(output_file, 'w') as f:
        for r in results:
            f.write(r + '\n')
    
    print(f"\nDone! Saved {len(results)} entries to {output_file}")
    print(f"Found data for {found_count}/{len(tokens)} scenarios")
    
    return True


def list_experiments():
    """List available experiment directories."""
    exp_root = DEFAULT_METRICS_DIR
    if not os.path.exists(exp_root):
        print(f"Error: Experiment root not found: {exp_root}")
        return
    
    experiments = sorted([d for d in os.listdir(exp_root) if os.path.isdir(os.path.join(exp_root, d))])
    
    print("Available experiments:")
    for exp in experiments:
        metrics_path = os.path.join(exp_root, exp, 'metrics')
        if os.path.exists(metrics_path):
            num_files = len(os.listdir(metrics_path))
            print(f"  {exp} ({num_files} metric files)")


def main():
    parser = argparse.ArgumentParser(description='Extract metrics from nuPlan simulation')
    parser.add_argument('--exp_dir', type=str, required=True, help='Experiment directory name or path')
    parser.add_argument('--planner', type=str, required=True, help='Planner name (e.g., IDMPlanner, DiffusionPlanner)')
    parser.add_argument('--scenarios', type=str, default=DEFAULT_SCENARIOS_FILE, help='Scenarios token file')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    parser.add_argument('--list', action='store_true', help='List available experiments')
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    success = extract_metrics(args.exp_dir, args.planner, args.scenarios, args.output)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
