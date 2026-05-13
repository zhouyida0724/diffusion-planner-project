#!/usr/bin/env bash
set -euo pipefail

# Poll fast_eval.jsonl every 5 minutes and write newest macro record into a queue file.
# This script does NOT send messages by itself; it updates files that the main agent can read.

EXP_DIR="/home/zhouyida/.openclaw/workspace/diffusion-planner-project/outputs/training/stride16_v01_4city_e10_bs96_bf16_linear_lr1e4_mask1p0_keep0_sp0p5_20260507"
JSONL="$EXP_DIR/fast_eval.jsonl"
STATE="/home/zhouyida/.openclaw/workspace/memory/fast_eval_watch.json"
QUEUE="/home/zhouyida/.openclaw/workspace/memory/fast_eval_pending.jsonl"

mkdir -p "$(dirname "$STATE")"
mkdir -p "$(dirname "$QUEUE")"

touch "$QUEUE"

while true; do
  if [[ -f "$JSONL" ]]; then
    python3 - "$JSONL" "$STATE" "$QUEUE" <<'PY'
import json, sys
from pathlib import Path

jsonl=Path(sys.argv[1])
state=Path(sys.argv[2])
queue=Path(sys.argv[3])

try:
    st=json.loads(state.read_text())
    last=int(st.get('last_reported_step', -1))
except Exception:
    last=-1

new_records=[]
try:
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj=json.loads(line)
        except Exception:
            continue
        if obj.get('city')!='macro':
            continue
        step=int(obj.get('step',-1))
        if step>last:
            new_records.append(obj)
except Exception:
    new_records=[]

if new_records:
    new_records.sort(key=lambda o:int(o.get('step',-1)))
    # append all unseen
    with queue.open('a', encoding='utf-8') as f:
        for obj in new_records:
            f.write(json.dumps(obj, ensure_ascii=False)+"\n")
    last=int(new_records[-1].get('step', last))
    # update state
    st={'exp_dir': st.get('exp_dir',''), 'fast_eval_jsonl': st.get('fast_eval_jsonl',''), 'last_reported_step': last}
    state.write_text(json.dumps(st, ensure_ascii=False, indent=2)+"\n")
PY
  fi

  sleep 300
done
