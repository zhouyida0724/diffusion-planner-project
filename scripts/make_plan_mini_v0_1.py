#!/usr/bin/env python3
"""Create a portable v0.1 export plan for nuPlan *mini* DBs.

Runs inside nuplan-simulation container.

Output directory contains:
- index.jsonl: one record per sample
- plan_config.json: how the plan was generated

Each record minimally contains:
  {"db_path": ..., "db_name": ..., "scene_token_hex": ..., "log_token_hex": ..., "frame_index": ...}

`frame_index` is defined as the index within ego_pose rows filtered by scene.log_token (ORDER BY timestamp),
matching extract_single_frame.get_target_frame behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
from pathlib import Path


def ro_connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="/workspace/data/nuplan/data/cache/mini")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-ego-poses", type=int, default=200, help="Skip scenes with too few ego_pose rows")
    args = ap.parse_args()

    random.seed(args.seed)

    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_paths = sorted([p for p in data_root.iterdir() if p.suffix == ".db"])
    if not db_paths:
        raise RuntimeError(f"No .db files under {data_root}")

    # Shuffle to avoid single-log dominance
    random.shuffle(db_paths)

    records = []
    for db_path in db_paths:
        if len(records) >= args.limit:
            break

        con = ro_connect(str(db_path))
        try:
            cur = con.cursor()
            cur.execute("SELECT token, log_token FROM scene")
            scenes = cur.fetchall() or []
            # Shuffle scenes too
            scenes = list(scenes)
            random.shuffle(scenes)

            for s in scenes:
                if len(records) >= args.limit:
                    break

                scene_token = s[0]
                log_token = s[1]
                if scene_token is None or log_token is None:
                    continue

                # Count ego_pose rows for that log
                cur.execute("SELECT COUNT(*) AS n FROM ego_pose WHERE log_token = ?", (log_token,))
                n = int(cur.fetchone()[0])
                if n < args.min_ego_poses:
                    continue

                # Choose a safe-ish center index: leave some margin for future samples.
                # extract_ego_data samples future at + (i+1)*10, i up to 79 => +800.
                # So keep index <= n-801. Similarly past needs 20*10=200 history.
                lo = 200
                hi = max(lo, n - 801)
                if hi <= lo:
                    frame_index = n // 2
                else:
                    frame_index = random.randint(lo, hi)

                records.append(
                    {
                        "db_path": str(db_path),
                        "db_name": db_path.name,
                        "scene_token_hex": scene_token.hex(),
                        "log_token_hex": log_token.hex(),
                        "frame_index": int(frame_index),
                    }
                )
        finally:
            con.close()

    if len(records) < args.limit:
        raise RuntimeError(f"Only collected {len(records)} samples (<{args.limit}). Try lowering --min-ego-poses")

    index_path = out_dir / "index.jsonl"
    with open(index_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(records[: args.limit]):
            r = dict(r)
            r["sample_id"] = f"{r['db_name']}:{r['scene_token_hex']}:{r['frame_index']}"
            r["index"] = i
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_dir / "plan_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_root": str(data_root),
                "limit": int(args.limit),
                "seed": int(args.seed),
                "min_ego_poses": int(args.min_ego_poses),
                "db_count": len(db_paths),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"WROTE {len(records[:args.limit])} records -> {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
