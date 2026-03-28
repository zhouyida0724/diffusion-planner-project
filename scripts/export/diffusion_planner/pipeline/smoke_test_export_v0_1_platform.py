#!/usr/bin/env python3
"""Smoke test for v0.1 export entrypoint.

Runs a small export (limit=100) using an existing slice plan and writes into
outputs/export/ so it is safe on dev machines.

This intentionally calls the existing CLI entrypoint (scripts/export_v0_1_single_npz.py)
so we do not accidentally change behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def count_lines(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--plan",
        type=str,
        default="plans/plan_boston50w_slice01_100k_20260326_093606",
        help="Plan directory containing index.jsonl (default: Boston 50w slice plan)",
    )
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--schedule", type=str, default="db_grouped")
    ap.add_argument("--bfs-max-time-s", type=float, default=0.5)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    plan_dir = (repo_root / args.plan).resolve()
    if not (plan_dir / "index.jsonl").exists():
        raise FileNotFoundError(f"plan index.jsonl not found under: {plan_dir}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (repo_root / "outputs" / "export" / f"smoke_export_platform_{ts}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exporter = (repo_root / "scripts" / "export_v0_1_single_npz.py").resolve()

    env = dict(os.environ)
    env["BFS_MAX_TIME_S"] = str(args.bfs_max_time_s)

    cmd = [
        "python3",
        str(exporter),
        "--plan",
        str(plan_dir),
        "--out",
        str(out_dir),
        "--limit",
        str(args.limit),
        "--schedule",
        str(args.schedule),
    ]

    print("Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, env=env)

    manifest_path = out_dir / "manifest.jsonl"
    metrics_path = out_dir / "metrics.json"
    npz_path = out_dir / "data.npz"

    assert manifest_path.exists(), f"missing {manifest_path}"
    assert metrics_path.exists(), f"missing {metrics_path}"
    assert npz_path.exists(), f"missing {npz_path}"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    planned = int(metrics["planned"])
    kept = int(metrics["kept"])
    hard = int(metrics["hard_skipped"])

    manifest_lines = count_lines(manifest_path)

    assert manifest_lines == planned, f"manifest lines {manifest_lines} != planned {planned}"
    assert kept + hard == planned, f"kept+hard {kept}+{hard} != planned {planned}"

    print(f"OK: out_dir={out_dir}")
    print(f"planned={planned} kept={kept} hard_skipped={hard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
