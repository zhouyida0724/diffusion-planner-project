#!/usr/bin/env python3
"""QC utility: compare two export outputs (baseline vs db_grouped).

Goal:
- Verify no-overlap/no-missing by plan_row_idx.
- Verify kept/hard-skip decisions match (allowing different manifest order / different t).

Usage:
  python3 scripts/qc_compare_export_manifests.py \
    --a /path/to/baseline/shard_000/manifest.jsonl \
    --b /path/to/grouped/shard_000/manifest.jsonl

Exit code:
- 0 if equivalent for all compared fields
- 2 if mismatch detected
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Rec:
    plan_row_idx: int
    qc_hard_skip: bool
    qc_error: str
    qc_flags: tuple[str, ...]
    route_lanes_avails_sum: int | None
    lanes_avails_sum: int | None


def _load_manifest(path: Path) -> dict[int, Rec]:
    out: dict[int, Rec] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r: dict[str, Any] = json.loads(line)
            pri = int(r["plan_row_idx"])
            if pri in out:
                raise RuntimeError(f"duplicate plan_row_idx={pri} in {path}")
            out[pri] = Rec(
                plan_row_idx=pri,
                qc_hard_skip=bool(r.get("qc_hard_skip", False)),
                qc_error=str(r.get("qc_error", "")) if r.get("qc_hard_skip", False) else "",
                qc_flags=tuple(r.get("qc_flags", []) or []),
                route_lanes_avails_sum=(None if r.get("qc_hard_skip", False) else int(r.get("route_lanes_avails_sum"))),
                lanes_avails_sum=(None if r.get("qc_hard_skip", False) else int(r.get("lanes_avails_sum"))),
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, type=str)
    ap.add_argument("--b", required=True, type=str)
    ap.add_argument(
        "--show",
        type=int,
        default=20,
        help="Show up to N mismatches (default 20)",
    )
    args = ap.parse_args()

    a_path = Path(args.a)
    b_path = Path(args.b)

    a = _load_manifest(a_path)
    b = _load_manifest(b_path)

    a_keys = set(a.keys())
    b_keys = set(b.keys())

    only_a = sorted(a_keys - b_keys)
    only_b = sorted(b_keys - a_keys)

    mismatches: list[str] = []

    if only_a:
        mismatches.append(f"plan_row_idx only in A ({len(only_a)}): {only_a[:args.show]}")
    if only_b:
        mismatches.append(f"plan_row_idx only in B ({len(only_b)}): {only_b[:args.show]}")

    common = sorted(a_keys & b_keys)
    for k in common:
        ra = a[k]
        rb = b[k]
        if ra != rb:
            mismatches.append(f"plan_row_idx={k}:\n  A={ra}\n  B={rb}")
            if len(mismatches) >= args.show:
                break

    if mismatches:
        print("MISMATCHES:")
        for m in mismatches:
            print(m)
        print(
            json.dumps(
                {
                    "a": str(a_path),
                    "b": str(b_path),
                    "n_a": len(a),
                    "n_b": len(b),
                    "n_common": len(common),
                },
                indent=2,
            )
        )
        return 2

    print(
        json.dumps(
            {
                "status": "ok",
                "a": str(a_path),
                "b": str(b_path),
                "n": len(common),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
