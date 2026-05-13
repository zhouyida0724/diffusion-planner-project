#!/usr/bin/env python3
"""Convert a token list to scene.token tokens (export/extractor contract).

This is an explicit preprocessing step (not a silent fallback):
- If a token exists in `scene.token`, keep it.
- Else if it exists in `lidar_pc.token`, map to `lidar_pc.scene_token`.
- Else: error.

Usage:
  python3 scripts/diagnostics/convert_tokens_to_scene_tokens.py \
    --tokens_csv outputs/eval/test_maneuver20/test_maneuver20_tokens_final.csv \
    --db_root /home/zhouyida/nuplan_cache_test/test \
    --out_csv /tmp/test_maneuver20_scene_tokens.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sqlite3
from pathlib import Path


def _hex(b: bytes) -> str:
    return b.hex()


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    return conn


def _exists_scene(conn: sqlite3.Connection, tok: bytes) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM scene WHERE token=? LIMIT 1", (tok,))
    return cur.fetchone() is not None


def _map_lidar_to_scene(conn: sqlite3.Connection, tok: bytes) -> bytes | None:
    cur = conn.cursor()
    cur.execute("SELECT scene_token FROM lidar_pc WHERE token=? LIMIT 1", (tok,))
    row = cur.fetchone()
    return row[0] if row else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens_csv", required=True)
    ap.add_argument("--db_root", required=True, help="dir containing *.db")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--dedup", action="store_true", help="Deduplicate scene tokens (preserve order)")
    args = ap.parse_args()

    in_csv = Path(args.tokens_csv)
    out_csv = Path(args.out_csv)

    # Load tokens
    with in_csv.open() as f:
        r = csv.DictReader(f)
        if "token" not in r.fieldnames:
            raise SystemExit(f"CSV must have column 'token', got {r.fieldnames}")
        tokens = [row["token"].strip() for row in r if row.get("token")]

    # Find dbs
    dbs = sorted(glob.glob(os.path.join(args.db_root, "*.db")))
    if not dbs:
        raise SystemExit(f"No .db found under {args.db_root}")

    resolved: list[str] = []
    for t in tokens:
        tb = bytes.fromhex(t)
        found_scene = None
        # brute-force over dbs (60 tokens -> fine)
        for db in dbs:
            conn = _connect(db)
            try:
                if _exists_scene(conn, tb):
                    found_scene = t
                    break
                st = _map_lidar_to_scene(conn, tb)
                if st is not None:
                    found_scene = _hex(st)
                    break
            finally:
                conn.close()
        if found_scene is None:
            raise SystemExit(f"Token not found as scene.token or lidar_pc.token in any db: {t}")
        resolved.append(found_scene)

    # Strict sanity: if conversion collapses many tokens to the same scene, the input list was
    # likely lidar tokens (frames) rather than scenario(scene) tokens.
    if len(set(resolved)) != len(resolved) and not args.dedup:
        # show a tiny summary
        from collections import Counter

        c = Counter(resolved)
        top = c.most_common(5)
        raise SystemExit(
            "Converted scene tokens contain duplicates. "
            "This usually means the input tokens are lidar_pc tokens (multiple frames from the same scene) "
            "rather than scene tokens. Re-generate the list as scene tokens, or re-run with --dedup. "
            f"Top duplicates: {top}"
        )

    if args.dedup:
        out = []
        seen = set()
        for t in resolved:
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
        resolved = out

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["token"])
        for t in resolved:
            w.writerow([t])

    print(f"Wrote {len(resolved)} scene tokens to: {out_csv}")


if __name__ == "__main__":
    main()
