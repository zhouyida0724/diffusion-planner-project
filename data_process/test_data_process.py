#!/usr/bin/env python3
"""
Unit Tests for analyze_dataset.py
=================================
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add paths
sys.path.insert(0, '/workspace/nuplan-visualization')
sys.path.insert(0, '/workspace')


class TestDatasetAnalyzer(unittest.TestCase):
    """Test cases for DatasetAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock the analyze_dataset module
        with patch('sys.argv', ['analyze_dataset.py', '--data_path', '/tmp/test']):
            pass
    
    @patch('analyze_dataset.sqlite3')
    def test_count_logs(self, mock_sqlite):
        """Test log counting"""
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test db files
            for i in range(5):
                Path(f"{tmpdir}/log_{i}.db").touch()
            
            # Import and test
            from analyze_dataset import DatasetAnalyzer
            analyzer = DatasetAnalyzer(tmpdir)
            count = analyzer.count_logs()
            
            self.assertEqual(count, 5)
    
    @patch('analyze_dataset.sqlite3')
    def test_aggregate_log_stats(self, mock_sqlite):
        """Test log statistics aggregation"""
        from analyze_dataset import DatasetAnalyzer
        
        analyzer = DatasetAnalyzer('/tmp/test')
        analyzer.log_stats = {
            'log1': {'frames': 1000, 'duration_s': 3600},
            'log2': {'frames': 2000, 'duration_s': 7200},
        }
        
        result = analyzer._aggregate_log_stats()
        
        self.assertEqual(result['log_count'], 2)
        self.assertEqual(result['total_frames'], 3000)
        self.assertEqual(result['total_hours'], 3.0)
    
    def test_get_notion_format(self):
        """Test Notion format generation"""
        from analyze_dataset import DatasetAnalyzer
        
        analyzer = DatasetAnalyzer('/tmp/test')
        analyzer.log_stats = {'log1': {}}
        analyzer.scenario_stats = {
            'type_a': {
                'count': 100,
                'total_duration_hours': 1.5,
                'total_frames': 5000,
                'unique_logs': 10,
                'avg_duration_s': 10.0
            },
            'type_b': {
                'count': 50,
                'total_duration_hours': 0.8,
                'total_frames': 2500,
                'unique_logs': 5,
                'avg_duration_s': 9.5
            }
        }
        
        result = analyzer.get_notion_format()
        
        self.assertIn('markdown_table', result)
        self.assertIn('type_a', result['markdown_table'])
        self.assertIn('type_b', result['markdown_table'])


class TestVisualizations(unittest.TestCase):
    """Test visualization generation"""
    
    def test_generate_visualizations(self):
        """Test that visualizations can be generated with mock data"""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        from analyze_dataset import DatasetAnalyzer
        
        analyzer = DatasetAnalyzer('/tmp/test')
        analyzer.scenario_stats = {
            f'type_{i}': {
                'count': 100 * i,
                'total_duration_hours': 10.0 * i,
                'total_frames': 5000 * i,
                'unique_logs': 10 * i,
                'avg_duration_s': 10.0
            }
            for i in range(1, 11)
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer.generate_visualizations(tmpdir)
            
            # Check files exist
            self.assertTrue(os.path.exists(f"{tmpdir}/scenario_distribution_bar.png"))
            self.assertTrue(os.path.exists(f"{tmpdir}/scenario_distribution_pie.png"))
            self.assertTrue(os.path.exists(f"{tmpdir}/duration_distribution.png"))
            self.assertTrue(os.path.exists(f"{tmpdir}/frame_distribution.png"))
            self.assertTrue(os.path.exists(f"{tmpdir}/duration_boxplot.png"))


class TestSamplingConfig(unittest.TestCase):
    """Test sampling configuration"""
    
    def test_distribution_validation(self):
        """Test distribution ratio validation"""
        from preprocess_pittsburgh import SamplingConfig
        
        # Test invalid distribution (doesn't sum to 1)
        config = SamplingConfig(
            distribution={'a': 0.3, 'b': 0.5},  # Sum = 0.8
            total_scenarios=100
        )
        
        with patch('preprocess_pittsburgh.print') as mock_print:
            config.validate()
        
        # Should normalize
        self.assertAlmostEqual(config.distribution['a'], 0.375, places=2)
        self.assertAlmostEqual(config.distribution['b'], 0.625, places=2)
    
    def test_get_target_counts_uniform(self):
        """Test uniform target count calculation"""
        from preprocess_pittsburgh import SamplingConfig
        
        config = SamplingConfig(scenarios_per_type=50)
        available_types = ['type_a', 'type_b', 'type_c']
        
        targets = config.get_target_counts(available_types, 1000)
        
        self.assertEqual(targets['type_a'], 50)
        self.assertEqual(targets['type_b'], 50)
        self.assertEqual(targets['type_c'], 50)
    
    def test_get_target_counts_distribution(self):
        """Test distribution-based target count calculation"""
        from preprocess_pittsburgh import SamplingConfig
        
        config = SamplingConfig(
            distribution={'a': 0.5, 'b': 0.3, 'c': 0.2},
            total_scenarios=100
        )
        
        available_types = ['a', 'b', 'c']
        
        targets = config.get_target_counts(available_types, 1000)
        
        self.assertEqual(targets['a'], 50)
        self.assertEqual(targets['b'], 30)
        self.assertEqual(targets['c'], 20)


class TestDataPreprocessor(unittest.TestCase):
    """Test DataPreprocessor class"""
    
    def test_init(self):
        """Test preprocessor initialization"""
        from preprocess_pittsburgh import DataPreprocessor, SamplingConfig
        
        config = SamplingConfig(total_scenarios=100)
        preprocessor = DataPreprocessor(
            data_path='/data/train',
            map_path='/data/maps',
            save_path='/data/output',
            sampling_config=config
        )
        
        self.assertEqual(preprocessor.data_path, '/data/train')
        self.assertEqual(preprocessor.save_path, '/data/output')
    
    def test_get_scenario_distribution(self):
        """Test scenario distribution calculation"""
        from preprocess_pittsburgh import DataPreprocessor, SamplingConfig
        
        # Mock scenarios
        class MockScenario:
            def __init__(self, scenario_type, log_name):
                self.scenario_type = scenario_type
                self.log_name = log_name
        
        scenarios = [
            MockScenario('type_a', 'log1'),
            MockScenario('type_a', 'log1'),
            MockScenario('type_b', 'log2'),
            MockScenario('type_c', 'log3'),
        ]
        
        config = SamplingConfig()
        preprocessor = DataPreprocessor(
            data_path='/data/train',
            map_path='/data/maps',
            save_path='/data/output',
            sampling_config=config
        )
        
        dist = preprocessor.get_scenario_distribution(scenarios, by='type')
        
        self.assertEqual(dist['type_a'], 2)
        self.assertEqual(dist['type_b'], 1)
        self.assertEqual(dist['type_c'], 1)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_cli_interface(self):
        """Test CLI argument parsing"""
        import argparse
        from analyze_dataset import parse_args
        
        # Test with mock args
        with patch('sys.argv', [
            'analyze_dataset.py',
            '--data_path', '/test/data',
            '--visualize',
            '--output', 'result.json'
        ]):
            args = parse_args()
            
            self.assertEqual(args.data_path, '/test/data')
            self.assertTrue(args.visualize)
            self.assertEqual(args.output, 'result.json')


if __name__ == '__main__':
    unittest.main()
