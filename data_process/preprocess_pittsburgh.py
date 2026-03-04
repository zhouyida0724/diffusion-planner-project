#!/usr/bin/env python3
"""
Data Preprocessing Module for Diffusion Planner
==============================================
Processes nuPlan data with advanced sampling:
- Configurable scenario distribution (e.g., type A: 30%, type B: 20%)
- Frame sampling: sample every N frames from scenarios
- Progress tracking and validation

Usage:
    # Basic preprocessing
    python preprocess_pittsburgh.py --data_path /data/pittsburgh \
                                   --map_path /data/maps \
                                   --save_path /data/output
    
    # With custom distribution
    python preprocess_pittsburgh.py --data_path /data/pittsburgh \
                                   --map_path /data/maps \
                                   --save_path /data/output \
                                   --distribution_file distribution.json
    
    # With frame sampling
    python preprocess_pittsburgh.py --data_path /data/pittsburgh \
                                   --map_path /data/maps \
                                   --save_path /data/output \
                                   --frame_sample_rate 2
    
    # Distribution JSON format:
    # {
    #   "accelerating_at_crosswalk": 0.1,
    #   "changing_lane": 0.15,
    #   "following_lane_with_lead": 0.2,
    #   ...
    # }
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import random
import math

# Add paths
sys.path.insert(0, '/workspace/nuplan-visualization')


class SamplingConfig:
    """Configuration for data sampling"""
    
    def __init__(
        self,
        total_scenarios: Optional[int] = None,
        scenarios_per_type: Optional[int] = None,
        distribution: Optional[Dict[str, float]] = None,
        frame_sample_rate: int = 1,
        shuffle: bool = True
    ):
        """
        Args:
            total_scenarios: Total number of scenarios to sample
            scenarios_per_type: Sample N scenarios per type (used if distribution not provided)
            distribution: Custom distribution as {scenario_type: ratio}
            frame_sample_rate: Sample every N frames (1 = all frames)
            shuffle: Whether to shuffle scenarios
        """
        self.total_scenarios = total_scenarios
        self.scenarios_per_type = scenarios_per_type
        self.distribution = distribution
        self.frame_sample_rate = frame_sample_rate
        self.shuffle = shuffle
    
    def validate(self):
        """Validate configuration"""
        if self.distribution:
            total_ratio = sum(self.distribution.values())
            if abs(total_ratio - 1.0) > 0.01:
                print(f"Warning: Distribution ratios sum to {total_ratio}, normalizing...")
                self.distribution = {
                    k: v / total_ratio for k, v in self.distribution.items()
                }
    
    def get_target_counts(self, available_types: List[str], total_available: int) -> Dict[str, int]:
        """
        Calculate target counts for each scenario type
        
        Args:
            available_types: List of available scenario types
            total_available: Total scenarios available
            
        Returns:
            Dict of {scenario_type: target_count}
        """
        if self.distribution:
            # Use custom distribution
            targets = {}
            for stype in available_types:
                ratio = self.distribution.get(stype, 0)
                if self.total_scenarios:
                    targets[stype] = int(ratio * self.total_scenarios)
                else:
                    # Use ratio of available
                    targets[stype] = max(1, int(ratio * total_available * 0.1))  # Default to 10% of available
            return targets
        elif self.scenarios_per_type:
            # Uniform sampling per type
            return {stype: self.scenarios_per_type for stype in available_types}
        elif self.total_scenarios:
            # Evenly distribute
            per_type = self.total_scenarios // len(available_types)
            return {stype: per_type for stype in available_types}
        else:
            # Return all available
            return {stype: 999999 for stype in available_types}


class DataPreprocessor:
    """Advanced data preprocessor with distribution control"""
    
    def __init__(
        self,
        data_path: str,
        map_path: str,
        save_path: str,
        sampling_config: SamplingConfig,
        agent_num: int = 32,
        static_objects_num: int = 5,
        lane_num: int = 70,
        lane_len: int = 20,
        route_num: int = 25,
        route_len: int = 20,
        time_len: int = 21,
        future_len: int = 80
    ):
        self.data_path = data_path
        self.map_path = map_path
        self.save_path = save_path
        self.sampling_config = sampling_config
        self.map_version = "nuplan-maps-v1.0"
        
        # Model params
        self.agent_num = agent_num
        self.static_objects_num = static_objects_num
        self.lane_num = lane_num
        self.lane_len = lane_len
        self.route_num = route_num
        self.route_len = route_len
        self.time_len = time_len
        self.future_len = future_len
        
        os.makedirs(save_path, exist_ok=True)
    
    def extract_all_scenarios(self) -> Dict[str, List]:
        """Extract all scenarios and group by type"""
        print("[1/5] Extracting all scenarios...")
        
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
        from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
        from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
        
        sensor_root = None
        db_files = None
        
        builder = NuPlanScenarioBuilder(
            self.data_path, self.map_path, sensor_root, db_files, self.map_version
        )
        
        # Get all scenarios
        scenario_filter = ScenarioFilter(
            scenario_types=None,
            scenario_tokens=None,
            log_names=None,
            map_names=None,
            num_scenarios_per_type=10000,
            limit_total_scenarios=None,
            timestamp_threshold_s=None,
            ego_displacement_minimum_m=None,
            expand_scenarios=True,
            remove_invalid_goals=False,
            shuffle=False,
            ego_start_speed_threshold=None,
            ego_stop_speed_threshold=None,
            speed_noise_tolerance=None
        )
        
        worker = SingleMachineParallelExecutor(use_process_pool=True)
        all_scenarios = builder.get_scenarios(scenario_filter, worker)
        
        # Group by type
        by_type = defaultdict(list)
        for scenario in all_scenarios:
            by_type[scenario.scenario_type].append(scenario)
        
        print(f"  Total scenarios: {len(all_scenarios)}")
        print(f"  Scenario types: {len(by_type)}")
        
        return dict(by_type)
    
    def sample_by_distribution(self, scenarios_by_type: Dict[str, List]) -> List:
        """
        Sample scenarios according to distribution config
        
        Args:
            scenarios_by_type: Dict of {scenario_type: [scenarios]}
            
        Returns:
            List of sampled scenarios
        """
        print("[2/5] Sampling scenarios by distribution...")
        
        config = self.sampling_config
        config.validate()
        
        available_types = list(scenarios_by_type.keys())
        total_available = sum(len(v) for v in scenarios_by_type.values())
        
        # Get target counts
        target_counts = config.get_target_counts(available_types, total_available)
        
        sampled = []
        
        for stype, target_count in target_counts.items():
            available = scenarios_by_type.get(stype, [])
            actual_count = min(target_count, len(available))
            
            if actual_count == 0:
                continue
            
            # Sample
            if config.shuffle:
                selected = random.sample(available, actual_count)
            else:
                selected = available[:actual_count]
            
            sampled.extend(selected)
            print(f"  {stype}: {len(selected)} / {len(available)}")
        
        # Shuffle final result
        if config.shuffle:
            random.shuffle(sampled)
        
        # Limit total
        if config.total_scenarios and len(sampled) > config.total_scenarios:
            sampled = sampled[:config.total_scenarios]
        
        print(f"  Total sampled: {len(sampled)}")
        
        return sampled
    
    def apply_frame_sampling(self, scenarios: List) -> List:
        """
        Apply frame sampling to scenarios
        
       抽帧: From each scenario, sample every N frames
        
        Args:
            scenarios: List of scenarios
            
        Returns:
            Scenarios with frame sampling applied
        """
        rate = self.sampling_config.frame_sample_rate
        
        if rate <= 1:
            print(f"[3/5] Frame sampling disabled (rate={rate})")
            return scenarios
        
        print(f"[3/5] Applying frame sampling (rate={rate})...")
        
        # Frame sampling is done during processing
        # Store the rate for processor to use
        self.frame_sample_rate = rate
        
        print(f"  Will sample every {rate} frames from each scenario")
        
        return scenarios
    
    def process_scenarios(self, scenarios: List):
        """Process scenarios into NPZ files"""
        print(f"[4/5] Processing {len(scenarios)} scenarios...")
        
        from diffusion_planner.data_process.data_processor import DataProcessor
        
        # Create processor with frame sampling
        processor = DataProcessor(self)
        processor.frame_sample_rate = self.sampling_config.frame_sample_rate
        
        processor.work(scenarios)
        
        print(f"  Saved to: {self.save_path}")
    
    def save_metadata(self, scenarios: List):
        """Save training metadata"""
        print("[5/5] Saving metadata...")
        
        # Save NPZ file list
        npz_files = [
            f for f in os.listdir(self.save_path)
            if f.endswith('.npz')
        ]
        
        list_path = os.path.join(self.save_path, 'training_data_list.json')
        with open(list_path, 'w') as f:
            json.dump(npz_files, f, indent=2)
        
        # Save sampling config used
        config_path = os.path.join(self.save_path, 'sampling_config.json')
        sampling_info = {
            'total_scenarios': self.sampling_config.total_scenarios,
            'distribution': self.sampling_config.distribution,
            'frame_sample_rate': self.sampling_config.frame_sample_rate,
            'shuffle': self.sampling_config.shuffle,
            'actual_scenarios': len(scenarios),
            'npz_files': len(npz_files)
        }
        with open(config_path, 'w') as f:
            json.dump(sampling_info, f, indent=2)
        
        print(f"  Data list: {list_path}")
        print(f"  Config: {config_path}")
        print(f"  NPZ files: {len(npz_files)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess nuPlan data with advanced sampling'
    )
    
    # Data paths
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--map_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    
    # Sampling options
    parser.add_argument('--total_scenarios', type=int, default=None,
                        help='Total scenarios to sample')
    parser.add_argument('--scenarios_per_type', type=int, default=None,
                        help='Scenarios per type (uniform)')
    parser.add_argument('--distribution_file', type=str, default=None,
                        help='JSON file with scenario distribution')
    parser.add_argument('--frame_sample_rate', type=int, default=1,
                        help='Sample every N frames (1=all)')
    parser.add_argument('--shuffle', type=bool, default=True)
    
    # Model params
    parser.add_argument('--agent_num', type=int, default=32)
    parser.add_argument('--static_objects_num', type=int, default=5)
    parser.add_argument('--lane_num', type=int, default=70)
    parser.add_argument('--lane_len', type=int, default=20)
    parser.add_argument('--route_num', type=int, default=25)
    parser.add_argument('--route_len', type=int, default=20)
    parser.add_argument('--time_len', type=int, default=21)
    parser.add_argument('--future_len', type=int, default=80)
    
    return parser.parse_args()


def create_sample_distribution_file():
    """Create a sample distribution file"""
    sample = {
        # Example distribution (must sum to 1.0)
        "accelerating_at_crosswalk": 0.05,
        "accelerating_at_stop_sign": 0.05,
        "changing_lane": 0.10,
        "changing_lane_to_left": 0.05,
        "changing_lane_to_right": 0.05,
        "following_lane_with_lead": 0.15,
        "following_lane_with_slow_lead": 0.10,
        "following_lane_without_lead": 0.10,
        "high_magnitude_speed": 0.05,
        "low_magnitude_speed": 0.05,
        "near_long_vehicle": 0.10,
        "near_pedestrian_on_crosswalk": 0.10,
        "on_intersection": 0.05,
        "starting_left_turn": 0.05,
        "starting_right_turn": 0.05,
        "stationary": 0.05
    }
    return sample


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Advanced Data Preprocessing for Diffusion Planner")
    print("=" * 60)
    print(f"Data: {args.data_path}")
    print(f"Save: {args.save_path}")
    
    # Load distribution
    distribution = None
    if args.distribution_file:
        with open(args.distribution_file, 'r') as f:
            distribution = json.load(f)
        print(f"Distribution: {args.distribution_file}")
    else:
        print("Distribution: uniform (scenarios_per_type)")
    
    # Create sampling config
    sampling_config = SamplingConfig(
        total_scenarios=args.total_scenarios,
        scenarios_per_type=args.scenarios_per_type,
        distribution=distribution,
        frame_sample_rate=args.frame_sample_rate,
        shuffle=args.shuffle
    )
    
    # Create preprocessor
    preprocessor = DataPreprocessor(
        data_path=args.data_path,
        map_path=args.map_path,
        save_path=args.save_path,
        sampling_config=sampling_config,
        agent_num=args.agent_num,
        static_objects_num=args.static_objects_num,
        lane_num=args.lane_num,
        lane_len=args.lane_len,
        route_num=args.route_num,
        route_len=args.route_len,
        time_len=args.time_len,
        future_len=args.future_len
    )
    
    # Pipeline
    scenarios_by_type = preprocessor.extract_all_scenarios()
    scenarios = preprocessor.sample_by_distribution(scenarios_by_type)
    scenarios = preprocessor.apply_frame_sampling(scenarios)
    preprocessor.process_scenarios(scenarios)
    preprocessor.save_metadata(scenarios)
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
