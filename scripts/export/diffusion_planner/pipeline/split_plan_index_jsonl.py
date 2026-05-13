#!/usr/bin/env python3
"""Split a plan directory (index.jsonl only) into N roughly-equal parts.

This repo's stride16 export plans currently contain only index.jsonl.

Usage:
  python3 split_plan_index_jsonl.py --in <plan_dir> --out-prefix <out_dir_prefix> --parts 2

Example:
  python3 split_plan_index_jsonl.py \
    --in plans/plan_v0.1_scene_lidar_stride16_boston \
    --out-prefix plans/plan_v0.1_scene_lidar_stride16_boston_h \
    --parts 2

Creates:
  <out-prefix>0/index.jsonl
  <out-prefix>1/index.jsonl
  ...
"""

from __future__ import annotations

import argparse
from pathlib import Path


def count_lines(p: Path) -> int:
    n = 0
    with p.open("rb") as f:
        for _ in f:
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="Input plan dir containing index.jsonl")
    ap.add_argument("--out-prefix", required=True, help="Output dir prefix (suffix 0..parts-1 will be appended)")
    ap.add_argument("--parts", type=int, default=2)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    idx = in_dir / "index.jsonl"
    if not idx.exists():
        raise FileNotFoundError(idx)
    parts = int(args.parts)
    if parts < 2:
        raise ValueError("--parts must be >= 2")

    total = count_lines(idx)
    # Split by contiguous chunks (deterministic, stable order).
    # Parts differ by at most 1 line.
    base = total // parts
    rem = total % parts
    part_sizes = [base + (1 if i < rem else 0) for i in range(parts)]

    out_paths = []
    for i in range(parts):
        out_dir = Path(f"{args.out_prefix}{i}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_paths.append(out_dir / "index.jsonl")

    writers = [p.open("w", encoding="utf-8") for p in out_paths]
    try:
        cur_part = 0
        cur_left = part_sizes[0]
        with idx.open("r", encoding="utf-8") as f:
            for line in f:
                if cur_left <= 0:
                    cur_part += 1
                    if cur_part >= parts:
                        raise RuntimeError("Internal split error: ran out of parts")
                    cur_left = part_sizes[cur_part]
                writers[cur_part].write(line)
                cur_left -= 1
    finally:
        for w in writers:
            try:
                w.close()
            except Exception:
                pass

    print(f"in={in_dir} total_lines={total} parts={parts} sizes={part_sizes}")
    for p in out_paths:
        print(f"out={p}")


if __name__ == "__main__":
    main()

