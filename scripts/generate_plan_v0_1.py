#!/usr/bin/env python3
"""Generate nuPlan v0.1 plan (scene token level) with tag enrichment.

Requirement (v0.1):
- Choose smallest train_* location by total ego_pose rows (exact count via sqlite).
- Enumerate TRUE scenes via `scene` table tokens across all DBs in that location.
- Randomly select N_SCENES scenes.
- For each selected scene, export frame indices: 0, STRIDE, 2*STRIDE, ... (stride=STRIDE)
  with cap_per_scene = CAP_PER_SCENE.
- Enrich each sample with tags:
  - center_ts = ego_pose.timestamp at the chosen frame index (ego_pose filtered by scene.log_token)
  - align lidar_pc_token by last lidar_pc.timestamp <= center_ts, preferring:
      1) lidar_pc.scene_token == scene.token (when populated)
      2) fallback join lidar_pc.ego_pose_token -> ego_pose.log_token == scene.log_token
  - tags = scenario_tag.type list for that lidar_pc_token

Outputs:
- plans/plan_v0.1_stride8_scene200_cap100_<location>_<YYYYMMDD_HHMM>/{index.jsonl,scene_list.json,plan_config.yaml,stats.json,tag_stats.json}

Designed to run on host (uses sqlite3). DBs must be accessible at CACHE_BASE.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import glob
import json
import os
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Defaults (override via CLI)
CACHE_BASE = Path("/media/zhouyida/新加卷1/nuplan_datasets/data/cache")
OUT_BASE = Path("plans")

SEED = 20260322
N_SCENES = 200
STRIDE = 8
CAP_PER_SCENE = 100

# We only need first (cap_per_scene-1)*stride + 1 poses to generate capped indices.
# This is computed from CLI args inside main.


def ro_connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def count_table(con: sqlite3.Connection, table: str) -> int:
    cur = con.cursor()
    cur.execute(f"SELECT count(*) FROM {table}")
    (n,) = cur.fetchone()
    return int(n)


def inventory_locations(cache_base: Path) -> Tuple[str, Dict[str, Dict]]:
    locs = sorted([p.name for p in cache_base.iterdir() if p.is_dir() and p.name.startswith("train_")])
    inv: Dict[str, Dict] = {}
    smallest_loc: Optional[str] = None
    smallest_frames: Optional[int] = None
    for loc in locs:
        dbs = sorted(glob.glob(str(cache_base / loc / "*.db")))
        total = 0
        for db in dbs:
            con = ro_connect(Path(db))
            try:
                total += count_table(con, "ego_pose")
            finally:
                con.close()
        inv[loc] = {"db_count": len(dbs), "ego_pose_frames": total}
        if smallest_frames is None or total < smallest_frames:
            smallest_frames = total
            smallest_loc = loc
    assert smallest_loc is not None
    return smallest_loc, inv


@dataclass(frozen=True)
class SceneRef:
    db_path: Path
    scene_token: bytes
    log_token: bytes


def iter_scenes_in_db(con: sqlite3.Connection) -> Iterable[Tuple[bytes, bytes]]:
    """Yield (scene_token, log_token)."""
    cur = con.cursor()
    cur.execute("SELECT token, log_token FROM scene")
    for tok, log_tok in cur.fetchall():
        yield tok, log_tok


def get_location_from_log(con: sqlite3.Connection) -> str:
    cur = con.cursor()
    cur.execute("SELECT location FROM log LIMIT 1")
    row = cur.fetchone()
    return str(row[0]) if row else "unknown"


def fetch_ego_pose_prefix(con: sqlite3.Connection, log_token: bytes, limit_n: int) -> List[Tuple[int, bytes]]:
    """Fetch first limit_n ego_pose rows (timestamp, token) for given log_token ordered by timestamp."""
    cur = con.cursor()
    cur.execute(
        "SELECT timestamp, token FROM ego_pose WHERE log_token=? ORDER BY timestamp LIMIT ?",
        (log_token, int(limit_n)),
    )
    rows = [(int(ts), tok) for ts, tok in cur.fetchall()]
    return rows


def _table_has_column(con: sqlite3.Connection, table: str, column: str) -> bool:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]  # (cid, name, type, notnull, dflt, pk)
    return column in cols


def prefetch_lidar_pc_for_scene(con: sqlite3.Connection, scene_token: bytes, max_ts: int) -> Tuple[List[int], List[bytes]]:
    """Fetch lidar_pc (timestamp, token) for a given scene up to max_ts, ordered by timestamp."""
    cur = con.cursor()
    cur.execute(
        "SELECT timestamp, token FROM lidar_pc WHERE scene_token=? AND timestamp<=? ORDER BY timestamp",
        (scene_token, int(max_ts)),
    )
    ts: List[int] = []
    tok: List[bytes] = []
    for t, b in cur.fetchall():
        ts.append(int(t))
        tok.append(b)
    return ts, tok


def prefetch_lidar_pc_for_log(con: sqlite3.Connection, log_token: bytes, max_ts: int) -> Tuple[List[int], List[bytes]]:
    """Fetch lidar_pc (timestamp, token) aligned to a given log via lidar_pc.ego_pose_token -> ego_pose.log_token.

    Note: nuPlan DB `lidar_pc` typically does NOT have `log_token`.
    """
    cur = con.cursor()
    cur.execute(
        """
        SELECT lp.timestamp, lp.token
        FROM lidar_pc lp
        JOIN ego_pose ep ON lp.ego_pose_token = ep.token
        WHERE ep.log_token=? AND lp.timestamp<=?
        ORDER BY lp.timestamp
        """,
        (log_token, int(max_ts)),
    )
    ts: List[int] = []
    tok: List[bytes] = []
    for t, b in cur.fetchall():
        ts.append(int(t))
        tok.append(b)
    return ts, tok


def align_lidar_token(lidar_ts: List[int], lidar_tok: List[bytes], center_ts: int) -> Optional[bytes]:
    j = bisect.bisect_right(lidar_ts, center_ts) - 1
    if j < 0:
        return None
    return lidar_tok[j]


def fetch_tags_for_lidar_tokens(con: sqlite3.Connection, lidar_tokens: List[bytes]) -> Dict[bytes, List[str]]:
    """Return map lidar_pc_token -> [tag types]."""
    if not lidar_tokens:
        return {}
    out: Dict[bytes, List[str]] = {}
    cur = con.cursor()
    # sqlite has a max variable limit (commonly 999). batch.
    B = 900
    for i in range(0, len(lidar_tokens), B):
        batch = lidar_tokens[i : i + B]
        q = ",".join(["?"] * len(batch))
        cur.execute(f"SELECT lidar_pc_token, type FROM scenario_tag WHERE lidar_pc_token IN ({q})", batch)
        for tok, typ in cur.fetchall():
            out.setdefault(tok, []).append(str(typ))
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate nuPlan v0.1 export plan (scene-level selection + stride/cap + tag enrichment).")
    ap.add_argument("--cache-base", type=str, default=str(CACHE_BASE), help="nuPlan cache root containing train_* dirs")
    ap.add_argument("--out-base", type=str, default=str(OUT_BASE), help="Output base directory for plans")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-scenes", type=int, default=N_SCENES)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--cap-per-scene", type=int, default=CAP_PER_SCENE)
    ap.add_argument("--location", type=str, default="", help="Optional: force a specific location dir (e.g. train_boston). If empty, pick smallest train_* by ego_pose rows")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    now = dt.datetime.now()

    cache_base = Path(args.cache_base)
    out_base = Path(args.out_base)

    if args.location:
        location = str(args.location)
        inv = {}
    else:
        location, inv = inventory_locations(cache_base)

    location_dir = cache_base / location
    db_paths = sorted(location_dir.glob("*.db"))

    # Enumerate all scene tokens across all DBs.
    all_scenes: List[SceneRef] = []
    for db in db_paths:
        con = ro_connect(db)
        try:
            for scene_tok, log_tok in iter_scenes_in_db(con):
                all_scenes.append(SceneRef(db_path=db, scene_token=scene_tok, log_token=log_tok))
        finally:
            con.close()

    n_scenes = int(args.n_scenes)
    stride = int(args.stride)
    cap_per_scene = int(args.cap_per_scene)
    poses_prefix_limit = (cap_per_scene - 1) * stride + 1

    if len(all_scenes) < n_scenes:
        raise RuntimeError(f"Not enough scenes in {location}: have {len(all_scenes)}, need {n_scenes}")

    chosen = random.sample(all_scenes, n_scenes)

    out_dir = out_base / f"plan_v0.1_stride{int(args.stride)}_scene{int(args.n_scenes)}_cap{int(args.cap_per_scene)}_{location}_{now.strftime('%Y%m%d_%H%M')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "index.jsonl"
    scene_list_path = out_dir / "scene_list.json"
    stats_path = out_dir / "stats.json"
    tag_stats_path = out_dir / "tag_stats.json"
    cfg_path = out_dir / "plan_config.yaml"

    # We'll write index and compute tag stats on the fly.
    tag_freq: Dict[str, int] = {}
    no_tag_count = 0

    # For reproducibility and debugging.
    scene_list_dump: List[Dict] = []

    produced = 0

    with open(index_path, "w", encoding="utf-8") as fidx:
        for sref in chosen:
            con = ro_connect(sref.db_path)
            try:
                # DB location (should match directory, but use metadata truth).
                loc = get_location_from_log(con)

                ego_prefix = fetch_ego_pose_prefix(con, sref.log_token, poses_prefix_limit)
                if not ego_prefix:
                    continue
                max_ts = ego_prefix[-1][0]

                # Align lidar_pc to this scene/log.
                # Preferred: use lidar_pc.scene_token (if populated).
                lidar_ts: List[int] = []
                lidar_tok: List[bytes] = []
                if _table_has_column(con, "lidar_pc", "scene_token"):
                    try:
                        lidar_ts, lidar_tok = prefetch_lidar_pc_for_scene(con, sref.scene_token, max_ts)
                    except Exception:
                        lidar_ts, lidar_tok = [], []
                # Fallback: join via lidar_pc.ego_pose_token -> ego_pose.log_token
                if not lidar_ts:
                    lidar_ts, lidar_tok = prefetch_lidar_pc_for_log(con, sref.log_token, max_ts)
                # Pre-fetch tags for all lidar tokens in this prefix.
                tags_map = fetch_tags_for_lidar_tokens(con, lidar_tok)

                # Record scene list metadata (approx).
                scene_list_dump.append(
                    {
                        "db_path": str(sref.db_path),
                        "db_name": sref.db_path.name,
                        "scene_token_hex": sref.scene_token.hex(),
                        "log_token_hex": sref.log_token.hex(),
                        "ego_pose_prefix_len": len(ego_prefix),
                    }
                )

                # Generate indices 0, stride, ... capped by cap_per_scene and prefix length.
                for k in range(cap_per_scene):
                    idx = k * stride
                    if idx >= len(ego_prefix):
                        break
                    ts_us, _pose_tok = ego_prefix[idx]

                    lidar_token = align_lidar_token(lidar_ts, lidar_tok, ts_us) if lidar_ts else None
                    lidar_hex = lidar_token.hex() if lidar_token is not None else None

                    tags = tags_map.get(lidar_token, []) if lidar_token is not None else []
                    if not tags:
                        no_tag_count += 1
                    else:
                        for t in tags:
                            tag_freq[t] = tag_freq.get(t, 0) + 1

                    rec = {
                        "sample_id": f"{sref.db_path.name}:{sref.scene_token.hex()}:{idx}",
                        "location": loc,
                        "db_path": str(sref.db_path),
                        "db_name": sref.db_path.name,
                        "scene_token_hex": sref.scene_token.hex(),
                        "log_token_hex": sref.log_token.hex(),
                        "frame_index": idx,
                        "timestamp_us": ts_us,
                        "lidar_pc_token_hex": lidar_hex,
                        "tags": tags,
                    }
                    fidx.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    produced += 1

            finally:
                con.close()

    # Write cfg
    cfg = {
        "seed": SEED,
        "location": location,
        "n_scenes": n_scenes,
        "stride": stride,
        "cap_per_scene": cap_per_scene,
        "poses_prefix_limit": POSES_PREFIX_LIMIT,
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")

    stats = {
        "location": location,
        "location_inventory": inv,
        "scene_pool_size": len(all_scenes),
        "scenes_selected": n_scenes,
        "stride": stride,
        "cap_per_scene": cap_per_scene,
        "poses_prefix_limit": POSES_PREFIX_LIMIT,
        "produced_samples": produced,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    with open(scene_list_path, "w", encoding="utf-8") as f:
        json.dump(scene_list_dump, f, ensure_ascii=False, indent=2)

    # Sort tag frequency for readability.
    tag_freq_sorted = dict(sorted(tag_freq.items(), key=lambda kv: kv[1], reverse=True))
    with open(tag_stats_path, "w", encoding="utf-8") as f:
        json.dump({"no_tag_count": no_tag_count, "tag_frequency": tag_freq_sorted}, f, ensure_ascii=False, indent=2)

    print(f"Wrote plan to: {out_dir}")
    print(f"Produced samples: {produced}")


if __name__ == "__main__":
    main()
