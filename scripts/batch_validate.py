#!/usr/bin/env python3
import sqlite3
import os
import random
import subprocess
import sys

DATA_BASE = '/workspace/data/nuplan/data/cache/mini/'
MAP_ROOT = '/workspace/data/nuplan/maps'
MAP_VERSION = 'v1.0'

CITY_MAP = {
    'us-ma-boston': 'us-ma-boston',
    'us-pa-pittsburgh-hazelwood': 'us-pa-pittsburgh-hazelwood', 
    'sg-one-north': 'sg-one-north',
    'las_vegas': 'us-nv-las-vegas-strip',
}

OUTPUT_DIR = '/workspace/validation_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

from collections import defaultdict
dbs_by_loc = defaultdict(list)
for f in os.listdir(DATA_BASE):
    if f.endswith('.db'):
        db_path = os.path.join(DATA_BASE, f)
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT location FROM log LIMIT 1')
            row = cursor.fetchone()
            if row:
                dbs_by_loc[row[0]].append(db_path)
            conn.close()
        except:
            pass

print(f'各城市: {dict((k, len(v)) for k, v in dbs_by_loc.items())}')

random.seed(42)
samples = []
for loc, dbs in dbs_by_loc.items():
    n = min(2, len(dbs))
    selected = random.sample(dbs, n)
    for db_path in selected:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM ego_pose')
        total = cursor.fetchone()[0]
        frame_idx = random.randint(100, total - 100)
        cursor.execute('SELECT token FROM scene LIMIT 1')
        scene_token = cursor.fetchone()[0].hex()
        cursor.execute('SELECT location FROM log LIMIT 1')
        location = cursor.fetchone()[0]
        conn.close()
        
        samples.append({
            'db_path': db_path,
            'frame_idx': frame_idx,
            'scene_token': scene_token,
            'location': location,
            'map_name': CITY_MAP.get(location, location),
        })

print(f'\n处理 {len(samples)} 个样本...')

for i, s in enumerate(samples):
    db_name = os.path.basename(s['db_path']).replace('.db', '')
    output_prefix = f"{s['location']}_{db_name}_{s['frame_idx']}"
    # extract_single_frame.py saves to: /workspace/diffusion-planner-project/data_process/npz_scenes/{SCENARIO_TOKEN}_{CENTER_FRAME_INDEX}.npz
    npz_path = f"/workspace/diffusion-planner-project/data_process/npz_scenes/{s['scene_token']}_{s['frame_idx']}.npz"
    png_path = os.path.join(OUTPUT_DIR, f"{output_prefix}.png")
    
    print(f'\n[{i+1}/{len(samples)}] {output_prefix}')
    
    # 写临时脚本
    temp_script = f'/tmp/run_extract_{i}.py'
    extract_dir = '/workspace/diffusion-planner-project/scripts/extract_single_frame'
    with open(temp_script, 'w') as f:
        f.write(f'''
import sys
sys.path.insert(0, '{extract_dir}')

DB_PATH = '{s['db_path']}'
MAP_NAME = '{s['map_name']}'
MAP_ROOT = '{MAP_ROOT}'
MAP_VERSION = '{MAP_VERSION}'
SCENARIO_TOKEN = '{s['scene_token']}'
CENTER_FRAME_INDEX = {s['frame_idx']}

from extract_single_frame import main
main()
''')
    
    # 运行提取
    result = subprocess.run(['python3', temp_script], capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        print(f'  提取错误: {result.stderr[:300]}')
        continue
    
    if not os.path.exists(npz_path):
        print(f'  NPZ 未生成')
        continue
    
    # 可视化
    viz_script = '/workspace/diffusion-planner-project/scripts/visualize_npz.py'
    result = subprocess.run(['python3', viz_script, npz_path, png_path], capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        print(f'  完成!')
    else:
        print(f'  可视化错误: {result.stderr[:100]}')

print('\n全部完成!')
print(f'输出目录: {OUTPUT_DIR}')
