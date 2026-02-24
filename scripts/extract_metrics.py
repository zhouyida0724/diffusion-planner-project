import pickle
import os
import glob

metrics_dir = '/workspace/data/nuplan/exp/exp/simulation/closed_loop_nonreactive_agents/2026.02.23.18.21.42/metrics'

# Read tokens
with open('/workspace/data/scenarios_200.txt', 'r') as f:
    tokens = [line.strip() for line in f.readlines()]

results = []
for token in tokens:
    # Find files with this token
    pattern = os.path.join(metrics_dir, f'*_{token}_IDMPlanner.pickle.temp')
    files = glob.glob(pattern)
    
    if not files:
        results.append(f"{token}|NO_DATA")
        continue
    
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
            print(f"Error: {e}")
    
    # Format as string
    score_str = ';'.join([f"{k}={v}" for k, v in scores.items()])
    results.append(f"{token}|{score_str}")

# Save
with open('/workspace/data/idm_metrics_200.csv', 'w') as f:
    for r in results:
        f.write(r + '\n')

print(f"Done! Saved {len(results)} entries")
