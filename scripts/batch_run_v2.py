#!/usr/bin/env python3
"""
批量特征提取与可视化 - 直接运行版
"""
import sys
import os
import sqlite3
import random
import subprocess
from collections import defaultdict

# 路径配置
DATA_BASE = '/workspace/data/nuplan/data/cache/mini/'
MAP_ROOT = '/workspace/data/nuplan/maps'
MAP_VERSION = 'v1.0'
EXTRACT_SCRIPT = '/workspace/diffusion-planner-project/scripts/extract_single_frame/extract_single_frame.py'
VIZ_SCRIPT = '/workspace/diffusion-planner-project/scripts/visualize_npz.py'
OUTPUT_BASE = '/workspace/diffusion-planner-project/data_process/npz_scenes/'
VALIDATION_OUTPUT = '/workspace/diffusion-planner-project/validation_output/'

CITY_MAP = {
    'us-ma-boston': 'us-ma-boston',
    'us-pa-pittsburgh-hazelwood': 'us-pa-pittsburgh-hazelwood', 
    'sg-one-north': 'sg-one-north',
    'las_vegas': 'us-nv-las-vegas-strip',
}

def get_samples():
    """获取各城市的随机样本"""
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
    
    return samples

def run_extraction(sample):
    """运行单次特征提取"""
    npz_path = f"{OUTPUT_BASE}{sample['scene_token']}_{sample['frame_idx']}.npz"
    
    # 读取原始脚本
    with open(EXTRACT_SCRIPT, 'r') as f:
        code = f.read()
    
    # 替换变量
    code = code.replace(
        "DB_PATH = '/workspace/data/nuplan/data/cache/mini/2021.10.01.19.16.42_veh-28_03307_03808.db'",
        f"DB_PATH = '{sample['db_path']}'"
    )
    code = code.replace("MAP_NAME = 'us-ma-boston'", f"MAP_NAME = '{sample['map_name']}'")
    code = code.replace("SCENARIO_TOKEN = 'e0b441bd54dc59cd'", f"SCENARIO_TOKEN = '{sample['scene_token']}'")
    code = code.replace('CENTER_FRAME_INDEX = 33250', f"CENTER_FRAME_INDEX = {sample['frame_idx']}")
    code = code.replace(
        "OUTPUT_PATH = '/workspace/diffusion-planner-project/data_process/npz_scenes/'+ SCENARIO_TOKEN + '_'+ str(CENTER_FRAME_INDEX) + '.npz'",
        f"OUTPUT_PATH = '{npz_path}'"
    )
    
    # 写入临时文件
    temp_script = '/tmp/temp_extract.py'
    with open(temp_script, 'w') as f:
        f.write(code)
    
    # 运行
    result = subprocess.run(['python3', temp_script], capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        return None, result.stderr
    
    if not os.path.exists(npz_path):
        return None, "NPZ file not created"
    
    return npz_path, None

def run_visualization(npz_path, output_png):
    """运行可视化"""
    result = subprocess.run(
        ['python3', VIZ_SCRIPT, npz_path, output_png],
        capture_output=True, text=True, timeout=30
    )
    return result.returncode == 0, result.stderr

def main():
    os.makedirs(VALIDATION_OUTPUT, exist_ok=True)
    
    samples = get_samples()
    print(f'\n处理 {len(samples)} 个样本...\n')
    
    results = []
    for i, s in enumerate(samples):
        db_name = os.path.basename(s['db_path']).replace('.db', '')
        output_prefix = f"{s['location']}_{db_name}_{s['frame_idx']}"
        png_path = os.path.join(VALIDATION_OUTPUT, f"{output_prefix}.png")
        
        print(f'[{i+1}/{len(samples)}] {output_prefix}')
        print(f'  Scene: {s["scene_token"]}, Frame: {s["frame_idx"]}, Map: {s["map_name"]}')
        
        npz_path, err = run_extraction(s)
        if err:
            print(f'  提取错误: {err[:200]}')
            results.append((output_prefix, False, err))
            continue
        
        print(f'  NPZ: {npz_path}')
        
        ok, err = run_visualization(npz_path, png_path)
        if ok:
            print(f'  PNG: {png_path}')
            results.append((output_prefix, True, None))
        else:
            print(f'  可视化错误: {err[:100]}')
            results.append((output_prefix, False, err))
        
        print()
    
    # 总结
    success = sum(1 for _, ok, _ in results if ok)
    print(f'完成! 成功: {success}/{len(results)}')
    print(f'输出目录: {VALIDATION_OUTPUT}')
    
    # 列出生成的文件
    print('\n生成的文件:')
    for f in sorted(os.listdir(VALIDATION_OUTPUT)):
        print(f'  {f}')

if __name__ == '__main__':
    main()
