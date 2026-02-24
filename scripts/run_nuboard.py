#!/usr/bin/env python3
"""
run_nuboard.py - Launch nuBoard visualization
"""

import subprocess
import os

os.environ['PYTHONPATH'] = '/workspace/nuplan-visualization'

subprocess.run([
    'python3', '-m', 'nuplan.planning.script.run_nuboard',
    'scenario_builder=nuplan',
    'scenario_builder.data_root=/workspace/data/nuplan/data/cache/mini',
])
