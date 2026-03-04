#!/usr/bin/env python3
"""
Data Analysis Script for nuPlan Dataset
========================================
Analyzes nuPlan dataset and provides detailed statistics:
- Number of logs, scenarios, hours, frames
- Per scenario type: count, total duration, total frames
- Visualizations: bar charts, pie charts
- Export to Notion-ready format
- Auto-upload images to Notion

Usage:
    python analyze_dataset.py --data_path /path/to/pittsburgh/data
    python analyze_dataset.py --data_path /path/to/data --output analysis.json
    python analyze_dataset.py --data_path /path/to/data --visualize --notion
    python analyze_dataset.py --data_path /path/to/data --visualize --notion --notion_page_id <page_id>
"""

import os
import sys
import argparse
import json
import requests
from pathlib import Path
from typing import Dict, List, Tuple
import sqlite3
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add nuplan to path
sys.path.insert(0, '/workspace/nuplan-visualization')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze nuPlan dataset statistics')
    parser.add_argument('--data_path', type=str, required=True, help='Path to nuPlan data')
    parser.add_argument('--map_path', type=str, default=None, help='Path to nuPlan maps')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--notion', action='store_true', help='Generate Notion-compatible output')
    parser.add_argument('--notion_page_id', type=str, default=None, help='Notion page ID to upload to')
    parser.add_argument('--notion_api_key', type=str, default=None, help='Notion API key (or use ~/.config/notion/api_key)')
    return parser.parse_args()


class NotionUploader:
    """Upload images to Notion via external hosting + API"""
    
    def __init__(self, api_key: str = None, page_id: str = None):
        self.api_key = api_key or self._load_api_key()
        self.page_id = page_id
        self.notion_version = "2025-09-03"
        
    def _load_api_key(self) -> str:
        """Load API key from config file"""
        config_path = os.path.expanduser("~/.config/notion/api_key")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return f.read().strip()
        raise ValueError("Notion API key not found. Set --notion_api_key or create ~/.config/notion/api_key")
    
    def upload_to_imgbb(self, image_path: str) -> str:
        """Upload image to external hosting and return URL"""
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            return None
            
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Try catbox.moe (free, no API key needed)
        try:
            files = {
                'reqtype': (None, 'fileupload'),
                'time': (None, '72h'),
                'fileToUpload': (os.path.basename(image_path), image_data, 'image/png')
            }
            response = requests.post("https://catbox.moe/user/api.php", files=files, timeout=60)
            if response.status_code == 200 and response.text.startswith('https://'):
                return response.text.strip()
        except Exception as e:
            print(f"catbox.moe upload failed: {e}")
        
        return None
    
    def add_image_to_notion(self, page_id: str, image_url: str, caption: str = None):
        """Add image block to Notion page"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": self.notion_version,
            "Content-Type": "application/json"
        }
        
        block = {
            "object": "block",
            "type": "image",
            "image": {
                "type": "external",
                "external": {"url": image_url}
            }
        }
        
        if caption:
            block["image"]["caption"] = [{"type": "text", "text": {"content": caption}}]
        
        response = requests.patch(
            f"https://api.notion.com/v1/blocks/{page_id}/children",
            headers=headers,
            json={"children": [block]}
        )
        
        return response.status_code == 200
    
    def add_heading(self, page_id: str, text: str, level: int = 2):
        """Add heading block to Notion"""
        heading_type = f"heading_{level}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": self.notion_version,
            "Content-Type": "application/json"
        }
        
        response = requests.patch(
            f"https://api.notion.com/v1/blocks/{page_id}/children",
            headers=headers,
            json={
                "children": [{
                    "object": "block",
                    "type": heading_type,
                    heading_type: {"rich_text": [{"type": "text", "text": {"content": text}}]}
                }]
            }
        )
        return response.status_code == 200
    
    def upload_images_to_notion(self, image_dir: str, page_id: str = None):
        """Upload all images from directory to Notion"""
        target_page = page_id or self.page_id
        if not target_page:
            print("Error: No page_id specified")
            return False
        
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        
        print(f"\n[NOTION] Uploading {len(image_files)} images to Notion...")
        
        for img_file in sorted(image_files):
            img_path = os.path.join(image_dir, img_file)
            
            # Upload to external host
            print(f"  Uploading {img_file}...", end=" ")
            img_url = self.upload_to_imgbb(img_path)
            
            if img_url:
                # Add to Notion
                self.add_image_to_notion(target_page, img_url, caption=img_file)
                print(f"✅ {img_url}")
            else:
                print(f"❌ Failed")
        
        print("[NOTION] Done!")
        return True


class DatasetAnalyzer:
    """Analyze nuPlan dataset comprehensively"""
    
    def __init__(self, data_path: str, map_path: str = None):
        self.data_path = data_path
        self.map_path = map_path
        self.log_stats = {}
        self.scenario_stats = {}
        
    def count_logs(self) -> int:
        """Count number of log files"""
        count = 0
        for root, dirs, files in os.walk(self.data_path):
            count += sum(1 for f in files if f.endswith('.db'))
        return count
    
    def analyze_logs(self) -> Dict:
        """Analyze each log file for metadata"""
        print("[1/4] Analyzing log files...")
        
        log_info = {}
        
        for root, dirs, files in os.walk(self.data_path):
            for f in files:
                if not f.endswith('.db'):
                    continue
                    
                db_path = os.path.join(root, f)
                log_name = f.replace('.db', '')
                
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    # Get frame count
                    cursor.execute("SELECT COUNT(*) FROM lidar_sweep")
                    num_frames = cursor.fetchone()[0]
                    
                    # Get duration
                    cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM lidar_sweep")
                    result = cursor.fetchone()
                    duration_s = 0
                    if result[0] and result[1]:
                        duration_s = (result[1] - result[0]) / 1e9
                    
                    # Get scenario count from metadata
                    cursor.execute("SELECT COUNT(*) FROM meta")
                    num_meta = cursor.fetchone()[0]
                    
                    log_info[log_name] = {
                        'frames': num_frames,
                        'duration_s': duration_s,
                        'duration_min': round(duration_s / 60, 2)
                    }
                    
                    conn.close()
                    
                except Exception as e:
                    print(f"Warning: Error reading {db_path}: {e}")
                    continue
        
        self.log_stats = log_info
        return self._aggregate_log_stats()
    
    def _aggregate_log_stats(self) -> Dict:
        """Aggregate log statistics"""
        total_frames = sum(v['frames'] for v in self.log_stats.values())
        total_duration = sum(v['duration_s'] for v in self.log_stats.values())
        
        return {
            'log_count': len(self.log_stats),
            'total_frames': total_frames,
            'total_hours': round(total_duration / 3600, 2),
            'avg_frames_per_log': round(total_frames / len(self.log_stats), 0) if self.log_stats else 0,
            'avg_duration_per_log_min': round(total_duration / 3600 / len(self.log_stats) * 60, 2) if self.log_stats else 0
        }
    
    def analyze_scenarios(self, verbose: bool = False) -> Dict:
        """Analyze scenarios using nuPlan scenario builder"""
        print("[2/4] Building scenario builder...")
        
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
        from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
        from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
        
        sensor_root = None
        db_files = None
        map_version = "nuplan-maps-v1.0"
        
        builder = NuPlanScenarioBuilder(
            self.data_path, self.map_path, sensor_root, db_files, map_version
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
        
        print("[3/4] Extracting scenarios...")
        worker = SingleMachineParallelExecutor(use_process_pool=True)
        scenarios = builder.get_scenarios(scenario_filter, worker)
        
        # Group by scenario type
        type_stats = defaultdict(lambda: {
            'count': 0,
            'total_duration_s': 0,
            'total_frames': 0,
            'logs': set()
        })
        
        print("[4/4] Computing scenario statistics...")
        for scenario in scenarios:
            stype = scenario.scenario_type
            
            # Estimate duration and frames from scenario
            # (scenarios are typically 8-20 seconds)
            duration_s = 10  # approximate
            frames = int(duration_s * 2)  # ~2Hz
            
            type_stats[stype]['count'] += 1
            type_stats[stype]['total_duration_s'] += duration_s
            type_stats[stype]['total_frames'] += frames
            type_stats[stype]['logs'].add(scenario.log_name)
        
        # Convert to regular dict
        scenario_type_stats = {}
        for stype, stats in type_stats.items():
            scenario_type_stats[stype] = {
                'count': stats['count'],
                'total_duration_hours': round(stats['total_duration_s'] / 3600, 2),
                'total_frames': stats['total_frames'],
                'unique_logs': len(stats['logs']),
                'avg_duration_s': round(stats['total_duration_s'] / stats['count'], 2) if stats['count'] > 0 else 0
            }
        
        self.scenario_stats = scenario_type_stats
        
        return {
            'total_scenarios': len(scenarios),
            'scenario_types': len(scenario_type_stats),
            'by_type': scenario_type_stats
        }
    
    def generate_visualizations(self, output_dir: str = '.'):
        """Generate bar chart, pie chart, and distribution visualizations"""
        print("[VIS] Generating visualizations...")
        
        if not self.scenario_stats:
            print("Warning: No scenario stats to visualize")
            return
        
        # Sort by count
        sorted_types = sorted(
            self.scenario_stats.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        types = [s[0] for s in sorted_types[:20]]  # Top 20
        counts = [s[1]['count'] for s in sorted_types[:20]]
        
        # =============================================
        # 1. Bar chart - Scenario Type Distribution
        # =============================================
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.barh(types[::-1], counts[::-1], color='steelblue')
        ax.set_xlabel('Number of Scenarios')
        ax.set_title('Scenario Type Distribution (Top 20)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scenario_distribution_bar.png'), dpi=150)
        plt.close()
        
        # =============================================
        # 2. Pie chart - Scenario Type Proportion
        # =============================================
        top_10_types = [s[0] for s in sorted_types[:10]]
        top_10_counts = [s[1]['count'] for s in sorted_types[:10]]
        other_count = sum(s[1]['count'] for s in sorted_types[10:])
        
        pie_types = top_10_types + ['Others']
        pie_counts = top_10_counts + [other_count]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = plt.cm.tab20(range(len(pie_types)))
        ax.pie(pie_counts, labels=pie_types, autopct='%1.1f%%', colors=colors)
        ax.set_title('Scenario Type Proportion')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scenario_distribution_pie.png'), dpi=150)
        plt.close()
        
        # =============================================
        # 3. Duration Distribution Histogram
        # =============================================
        durations = [s[1]['total_duration_hours'] for s in sorted_types]
        type_names = [s[0] for s in sorted_types]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(range(len(type_names)), durations, color='coral')
        ax.set_xticks(range(len(type_names)))
        ax.set_xticklabels(type_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Total Duration (hours)')
        ax.set_title('Total Duration by Scenario Type')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'duration_distribution.png'), dpi=150)
        plt.close()
        
        # =============================================
        # 4. Frame Distribution Histogram
        # =============================================
        frames = [s[1]['total_frames'] for s in sorted_types]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(range(len(type_names)), frames, color='seagreen')
        ax.set_xticks(range(len(type_names)))
        ax.set_xticklabels(type_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Total Frames')
        ax.set_title('Total Frames by Scenario Type')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'frame_distribution.png'), dpi=150)
        plt.close()
        
        # =============================================
        # 5. Duration per Scenario Box Plot
        # =============================================
        # Group durations by scenario type
        duration_data = []
        labels = []
        for stype, stats in sorted_types[:15]:
            duration_data.append(stats['avg_duration_s'])
            labels.append(stype[:15])  # Truncate for readability
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot(duration_data, labels=labels, patch_artist=True)
        ax.set_ylabel('Average Duration per Scenario (seconds)')
        ax.set_title('Average Scenario Duration by Type (Top 15)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'duration_boxplot.png'), dpi=150)
        plt.close()
        
        print(f"[VIS] Saved to {output_dir}")
        print(f"[VIS] Generated files:")
        print(f"  - scenario_distribution_bar.png")
        print(f"  - scenario_distribution_pie.png")
        print(f"  - duration_distribution.png")
        print(f"  - frame_distribution.png")
        print(f"  - duration_boxplot.png")
    
    def get_notion_format(self) -> Dict:
        """Get Notion-compatible format for documentation"""
        if not self.scenario_stats:
            return {'error': 'No stats available'}
        
        sorted_types = sorted(
            self.scenario_stats.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        # Summary table with all metrics
        summary_table = "| Scenario Type | Count | Hours | Frames | Avg Duration(s) | Unique Logs |\n"
        summary_table += "|--------------|-------|-------|--------|----------------|-------------|\n"
        
        for stype, stats in sorted_types:
            summary_table += f"| {stype} | {stats['count']} | {stats['total_duration_hours']} | {stats['total_frames']:,} | {stats['avg_duration_s']} | {stats['unique_logs']} |\n"
        
        # Top scenarios for quick view
        top_scenarios = [
            {
                'type': s[0], 
                'count': s[1]['count'],
                'hours': s[1]['total_duration_hours'],
                'frames': s[1]['total_frames']
            } 
            for s in sorted_types[:10]
        ]
        
        return {
            'total_logs': len(self.log_stats),
            'total_scenarios': sum(s['count'] for s in self.scenario_stats.values()),
            'total_hours': sum(s['total_duration_hours'] for s in self.scenario_stats.values()),
            'total_frames': sum(s['total_frames'] for s in self.scenario_stats.values()),
            'scenario_types': len(self.scenario_stats),
            'top_10': top_scenarios,
            'markdown_table': summary_table,
            'visualizations': [
                'scenario_distribution_bar.png',
                'scenario_distribution_pie.png', 
                'duration_distribution.png',
                'frame_distribution.png',
                'duration_boxplot.png'
            ]
        }


def main():
    args = parse_args()
    
    print("=" * 60)
    print("nuPlan Dataset Analysis")
    print("=" * 60)
    
    analyzer = DatasetAnalyzer(args.data_path, args.map_path)
    
    # Run analysis
    log_summary = analyzer.analyze_logs()
    scenario_summary = analyzer.analyze_scenarios(verbose=args.visualize)
    
    # Combine results
    results = {
        'data_path': args.data_path,
        'logs': log_summary,
        'scenarios': scenario_summary
    }
    
    # Generate visualizations
    vis_output_dir = '.'
    if args.visualize:
        vis_output_dir = os.path.dirname(args.output) if args.output else '.'
        if vis_output_dir and not os.path.exists(vis_output_dir):
            os.makedirs(vis_output_dir, exist_ok=True)
        analyzer.generate_visualizations(vis_output_dir)
    
    # Notion format and upload
    notion_data = None
    if args.notion:
        notion_data = analyzer.get_notion_format()
        results['notion'] = notion_data
        
        # Upload images to Notion if page_id provided
        if args.notion_page_id:
            try:
                notion_uploader = NotionUploader(
                    api_key=args.notion_api_key,
                    page_id=args.notion_page_id
                )
                notion_uploader.upload_images_to_notion(vis_output_dir, args.notion_page_id)
            except Exception as e:
                print(f"[NOTION] Error: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total logs: {log_summary['log_count']}")
    print(f"Total hours: {log_summary['total_hours']}")
    print(f"Total frames: {log_summary['total_frames']:,}")
    print(f"Total scenarios: {scenario_summary['total_scenarios']:,}")
    print(f"Scenario types: {scenario_summary['scenario_types']}")
    
    # Save output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved to: {args.output}")
    
    if notion_data:
        print("\n--- Notion Format ---")
        print(notion_data.get('markdown_table', '')[:500] + "...")


if __name__ == '__main__':
    main()
