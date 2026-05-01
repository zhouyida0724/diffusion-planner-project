#!/usr/bin/env python3
"""Watch export progress for 4-city sequential export.

Usage:
  python3 watch_export_4cities.py [OUT_ROOT]

Writes:
  <OUT_ROOT>/_status/watch_latest.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def read_jsonl_counts(path: Path) -> tuple[int, int]:
    kept = 0
    hard = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if o.get("qc_hard_skip", False):
                    hard += 1
                else:
                    kept += 1
    except FileNotFoundError:
        pass
    return kept, hard


def summarize_city(out_root: Path, city: str) -> dict:
    root = out_root / city
    if not root.is_dir():
        return {"city": city, "exists": False}

    shard_dirs = sorted([p for p in root.glob("shard_*") if p.is_dir()])
    if not shard_dirs:
        return {"city": city, "exists": True, "shards": 0}

    newest = 0
    kept_total = 0
    hard_total = 0

    for sd in shard_dirs:
        mf = sd / "manifest.jsonl"
        if mf.exists():
            kept, hard = read_jsonl_counts(mf)
            kept_total += kept
            hard_total += hard
            newest = max(newest, int(mf.stat().st_mtime))

    age_s = int(time.time()) - newest if newest else None

    return {
        "city": city,
        "exists": True,
        "shards": len(shard_dirs),
        "kept": kept_total,
        "hard_skip": hard_total,
        "manifest_age_s": age_s,
    }


def main() -> int:
    out_root = Path(
        os.environ.get("OUT_ROOT", "")
        or (os.sys.argv[1] if len(os.sys.argv) > 1 else "/media/zhouyida/新加卷1/exports_stride16_v0.1")
    )
    status_dir = out_root / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)

    cities = ["boston", "vegas_1", "pittsburgh", "singapore"]
    payload = {
        "t": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "out_root": str(out_root),
        "cities": [summarize_city(out_root, c) for c in cities],
    }
    (status_dir / "watch_latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {status_dir/'watch_latest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
