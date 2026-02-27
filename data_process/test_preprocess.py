#!/usr/bin/env python3
"""
Unit tests for nuPlan data preprocessing pipeline.

Usage:
    python -m pytest test_preprocess.py -v
    python test_preprocess.py
"""

import os
import sys
import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Add paths
NUPLAN_VIS_PATH = '/workspace/nuplan-visualization'
DIFFUSION_PATH = '/workspace'
sys.path.insert(0, NUPLAN_VIS_PATH)
sys.path.insert(0, DIFFUSION_PATH)


class TestNuPlanDataProcessor(unittest.TestCase):
    """Test cases for NuPlanDataProcessor"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.data_path = '/workspace/data/nuplan/data/cache/mini'
        cls.map_path = '/workspace/data/nuplan/maps'
        
        # Create temporary directory for output
        cls.temp_dir = tempfile.mkdtemp()
        cls.output_path = os.path.join(cls.temp_dir, 'test_output')
        os.makedirs(cls.output_path, exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
    
    def test_01_scenario_builder_import(self):
        """Test that scenario builder can be imported"""
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
        from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
        from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
        self.assertIsNotNone(NuPlanScenarioBuilder)
        self.assertIsNotNone(ScenarioFilter)
        self.assertIsNotNone(SingleMachineParallelExecutor)
        print("✓ Scenario builder imports successful")
    
    def test_02_load_single_scenario(self):
        """Test loading a single scenario"""
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
        from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
        from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
        
        # Get a specific log name to load scenarios from
        import sqlite3
        db_path = self.data_path
        db_files = [f for f in os.listdir(db_path) if f.endswith('.db')][:1]
        
        log_name = None
        if db_files:
            log_name = db_files[0].replace('.db', '')
        
        self.assertIsNotNone(log_name, "No log name found")
        
        # Build filter using log_names to get a scenario
        scenario_filter = ScenarioFilter(
            scenario_types=None,
            scenario_tokens=None,
            log_names=[log_name],
            map_names=None,
            num_scenarios_per_type=None,
            limit_total_scenarios=1,
            timestamp_threshold_s=None,
            ego_displacement_minimum_m=None,
            expand_scenarios=True,
            remove_invalid_goals=False,
            shuffle=False
        )
        
        # Create builder
        builder = NuPlanScenarioBuilder(
            self.data_path,
            self.map_path,
            None,
            None,
            "nuplan-maps-v1.0"
        )
        
        worker = SingleMachineParallelExecutor(use_process_pool=False)
        scenarios = builder.get_scenarios(scenario_filter, worker)
        
        self.assertEqual(len(scenarios), 1)
        print(f"✓ Loaded scenario: {scenarios[0].token}")
        
        return scenarios[0]
    
    def test_03_scenario_has_required_attributes(self):
        """Test that scenario has required attributes for processing"""
        scenario = self.test_02_load_single_scenario()
        
        # Check required attributes
        self.assertTrue(hasattr(scenario, 'token'))
        self.assertTrue(hasattr(scenario, '_map_name'))
        self.assertTrue(hasattr(scenario, 'map_api'))
        self.assertTrue(hasattr(scenario, 'initial_ego_state'))
        self.assertTrue(hasattr(scenario, 'initial_tracked_objects'))
        self.assertTrue(hasattr(scenario, 'get_route_roadblock_ids'))
        self.assertTrue(hasattr(scenario, 'get_past_tracked_objects'))
        self.assertTrue(hasattr(scenario, 'get_future_tracked_objects'))
        self.assertTrue(hasattr(scenario, 'get_traffic_light_status_at_iteration'))
        
        print(f"✓ Scenario has all required attributes")
    
    def test_04_process_single_scenario_to_npz(self):
        """Test processing a single scenario to NPZ format"""
        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.diffusion_planner.data_process.agent_process import (
            sampled_tracked_objects_to_array_list,
            sampled_static_objects_to_array_list,
            agent_past_process,
            agent_future_process
        )
        from nuplan.diffusion_planner.data_process.ego_process import (
            get_ego_past_array_from_scenario,
            get_ego_future_array_from_scenario,
            calculate_additional_ego_states
        )
        
        # Load scenario
        scenario = self.test_02_load_single_scenario()
        
        # Parameters
        past_time_horizon = 2.0
        num_past_poses = int(10 * past_time_horizon)
        future_time_horizon = 8.0
        num_future_poses = int(10 * future_time_horizon)
        num_agents = 32
        num_static = 5
        
        # Process ego past
        ego_state = scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.array([
            ego_state.rear_axle.x, 
            ego_state.rear_axle.y, 
            ego_state.rear_axle.heading
        ], dtype=np.float64)
        
        ego_agent_past, time_stamps_past = get_ego_past_array_from_scenario(
            scenario, num_past_poses, past_time_horizon
        )
        
        # Process agents
        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_past_tracked_objects(
                iteration=0, time_horizon=past_time_horizon, num_samples=num_past_poses
            )
        ]
        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        neighbor_agents_past, neighbor_agents_types = sampled_tracked_objects_to_array_list(
            sampled_past_observations
        )
        
        static_objects, static_objects_types = sampled_static_objects_to_array_list(
            present_tracked_objects
        )
        
        ego_agent_past, neighbor_agents_past, neighbor_indices, static_objects = \
            agent_past_process(
                ego_agent_past, neighbor_agents_past, neighbor_agents_types,
                num_agents, static_objects, static_objects_types,
                num_static, 10, anchor_ego_state
            )
        
        # Process future
        ego_agent_future = get_ego_future_array_from_scenario(
            scenario, ego_state, num_future_poses, future_time_horizon
        )
        
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=0, time_horizon=future_time_horizon, num_samples=num_future_poses
            )
        ]
        
        sampled_future_observations = [present_tracked_objects] + future_tracked_objects
        future_tracked_objects_array_list, _ = sampled_tracked_objects_to_array_list(
            sampled_future_observations
        )
        neighbor_agents_future = agent_future_process(
            anchor_ego_state, future_tracked_objects_array_list,
            num_agents, neighbor_indices
        )
        
        # Ego current state
        ego_current_state = calculate_additional_ego_states(ego_agent_past, time_stamps_past)
        
        # Verify shapes (allowing for variable length scenarios)
        self.assertGreaterEqual(ego_agent_past.shape[1], 1, "Ego past should have at least 1 pose")
        self.assertGreaterEqual(ego_agent_future.shape[0], 1, "Ego future should have at least 1 pose")
        self.assertEqual(neighbor_agents_past.shape[0], num_agents, "Neighbor agents shape incorrect")
        self.assertEqual(static_objects.shape[0], num_static, "Static objects shape incorrect")
        
        # Save to NPZ
        output_file = os.path.join(self.output_path, f"{scenario._map_name}_{scenario.token}.npz")
        np.savez(
            output_file,
            map_name=scenario._map_name,
            token=scenario.token,
            ego_current_state=ego_current_state,
            ego_agent_future=ego_agent_future,
            neighbor_agents_past=neighbor_agents_past,
            neighbor_agents_future=neighbor_agents_future,
            static_objects=static_objects
        )
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_file), "NPZ file not created")
        
        # Verify NPZ content
        loaded = np.load(output_file)
        self.assertIn('ego_agent_future', loaded.files)
        self.assertIn('neighbor_agents_past', loaded.files)
        
        print(f"✓ Processed scenario to NPZ: {output_file}")
        print(f"  - ego_agent_future shape: {loaded['ego_agent_future'].shape}")
        print(f"  - neighbor_agents_past shape: {loaded['neighbor_agents_past'].shape}")
    
    def test_05_npz_file_list_generation(self):
        """Test that we can generate a list of NPZ files"""
        npz_files = list(Path(self.output_path).glob('*.npz'))
        file_list = [f.name for f in npz_files]
        
        import json
        list_path = os.path.join(self.output_path, 'test_file_list.json')
        with open(list_path, 'w') as f:
            json.dump(file_list, f, indent=2)
        
        self.assertGreater(len(file_list), 0, "No NPZ files found")
        print(f"✓ Generated file list with {len(file_list)} files")


def run_tests():
    """Run all test cases"""
    print("=" * 60)
    print("Running nuPlan Data Preprocessing Tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestNuPlanDataProcessor)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
