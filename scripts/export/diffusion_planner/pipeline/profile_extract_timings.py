#!/usr/bin/env python3
"""Pretty-print extractor per-submethod timing stats from metrics.json.

Usage:
  python3 scripts/profile_extract_timings.py /path/to/metrics.json

The exporter writes metrics.json with:
  metrics['extract_profile']['by_step_s'][step] = {count, mean, p50, p90, p99, max}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _fmt(x: object) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x)*1e3:.2f}ms"
    except Exception:
        return str(x)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: profile_extract_timings.py /path/to/metrics.json", file=sys.stderr)
        return 2

    p = Path(sys.argv[1])
    m = json.loads(p.read_text(encoding="utf-8"))
    prof = (m.get("extract_profile") or {}).get("by_step_s") or {}

    if not prof:
        print("No extract_profile.by_step_s found. Did you run with EXTRACT_PROFILE=1?", file=sys.stderr)
        return 1

    rows = []
    for k, s in prof.items():
        rows.append(
            (
                k,
                s.get("count", 0),
                s.get("mean"),
                s.get("p50"),
                s.get("p90"),
                s.get("p99"),
                s.get("max"),
            )
        )

    def _key_mean(r):
        return (r[2] is None, r[2] or 0.0)

    def _key_p99(r):
        return (r[5] is None, r[5] or 0.0)

    print(f"metrics: {p}")
    print("\nTop by mean:")
    for r in sorted(rows, key=_key_mean, reverse=True)[:15]:
        print(
            f"  {r[0]:35s}  n={r[1]:4d}  mean={_fmt(r[2])}  p50={_fmt(r[3])}  p90={_fmt(r[4])}  p99={_fmt(r[5])}  max={_fmt(r[6])}"
        )

    print("\nTop by p99:")
    for r in sorted(rows, key=_key_p99, reverse=True)[:15]:
        print(
            f"  {r[0]:35s}  n={r[1]:4d}  mean={_fmt(r[2])}  p50={_fmt(r[3])}  p90={_fmt(r[4])}  p99={_fmt(r[5])}  max={_fmt(r[6])}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
