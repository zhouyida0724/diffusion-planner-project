#!/usr/bin/env python3
"""v0.1 batch export: read a plan index.jsonl and export into ONE NPZ + manifest + metrics.

Run INSIDE nuplan-simulation container.

Hard-skip rules:
- NaN/Inf in any required array
- shape mismatch
- route_lanes_avails_sum == 0 (routing missing)

Soft flag:
- route_min_dist_m > 30

Notes:
- Reuses map_api (one per run, since v0.1 uses single location).
- Disables per-frame debug logging inside extract_features to avoid massive I/O.

Scheduling:
- Default (--schedule=plan) processes samples in plan file order within this shard.
- Locality-aware (--schedule=db_grouped) groups samples by db_path (deterministic: db_path sorted) to reduce
  cross-DB random access / connection switching, while preserving modulo sharding (plan_row_idx % num_shards).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

# NOTE: This repo has a top-level package named `platform` which would shadow
# Python's stdlib `platform` module if we put `src/` on sys.path.
# To avoid breaking dependencies (e.g., pandas), we load our reusable IO helpers
# by file path.
_REPO_ROOT = Path(__file__).resolve().parents[4]

import numpy as np
import sqlite3

import importlib.util as _importlib_util

def _load_helper(module_name: str, path: Path):
    spec = _importlib_util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module {module_name} from {path}")
    mod = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_io_dir = _REPO_ROOT / "src" / "platform" / "io"
_manifest = _load_helper("_export_io_manifest", _io_dir / "manifest.py")
_metrics = _load_helper("_export_io_metrics", _io_dir / "metrics.py")
_npz = _load_helper("_export_io_npz_writer", _io_dir / "npz_writer.py")
_qc = _load_helper("_export_io_qc", _io_dir / "qc.py")

write_jsonl_line = _manifest.write_jsonl_line
bucketize = _metrics.bucketize
summarize_durations = _metrics.summarize_durations
save_npz_compressed = _npz.save_npz_compressed
is_finite = _qc.is_finite
route_min_dist_m = _qc.route_min_dist_m

# Import the refactored extractor by path (container may not treat /workspace/scripts as a package).
import importlib.util

_esf_path_candidates = [
    # New location after scripts/ reorg
    "/workspace/scripts/export/diffusion_planner/extract/extract_single_frame/extract_single_frame.py",
    # Legacy location (keep for backwards compatibility in older containers)
    "/workspace/scripts/extract_single_frame/extract_single_frame.py",
]

_esf_path = next((p for p in _esf_path_candidates if os.path.exists(p)), "")
if not _esf_path:
    # Host/dev fallback: resolve relative to this file.
    _esf_path = str(
        (
            Path(__file__).resolve().parents[1]
            / "extract"
            / "extract_single_frame"
            / "extract_single_frame.py"
        ).resolve()
    )

_spec = importlib.util.spec_from_file_location("extract_single_frame", _esf_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Failed to load extractor from {_esf_path}")
_esf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_esf)
esf = _esf


REQUIRED_KEYS = [
    "ego_current_state",
    "ego_agent_future",
    "neighbor_agents_past",
    "neighbor_agents_future",
    "static_objects",
    "lanes",
    "lanes_avails",
    "route_lanes",
    "route_lanes_avails",
    "lanes_speed_limit",
    "lanes_has_speed_limit",
    "route_lanes_speed_limit",
    "route_lanes_has_speed_limit",
]


def ro_connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=str, required=True, help="Path to plan directory containing index.jsonl")
    ap.add_argument("--out", type=str, required=True, help="Output directory")
    ap.add_argument("--map-root", type=str, default=esf.MAP_ROOT)
    ap.add_argument("--map-version", type=str, default=esf.MAP_VERSION)
    ap.add_argument("--limit", type=int, default=0, help="Optional: only process first N lines")
    ap.add_argument("--num-shards", type=int, default=1, help="Total shards for modulo sharding")
    ap.add_argument("--shard-id", type=int, default=0, help="This shard id in [0, num_shards)")
    ap.add_argument(
        "--schedule",
        type=str,
        default="plan",
        choices=["plan", "db_grouped"],
        help=(
            "Processing order within this shard. "
            "'plan' preserves plan file order; 'db_grouped' groups by db_path (deterministic, sorted by db_path) "
            "to reduce cross-DB random access."
        ),
    )
    args = ap.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard-id must satisfy 0 <= shard_id < num_shards")

    # Avoid extremely noisy map RuntimeWarnings that slow down batch export via stderr I/O.
    warnings.filterwarnings("ignore", message=".*invalid value encountered in cast.*", category=RuntimeWarning)

    # Reset aggregated log stats (warning counters) for this run.
    try:
        esf.reset_log_stats()
    except Exception:
        pass

    t_prog0 = time.time()

    plan_dir = Path(args.plan)
    index_path = plan_dir / "index.jsonl"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    metrics_path = out_dir / "metrics.json"
    run_info_path = out_dir / "RUN_INFO.json"
    npz_path = out_dir / "data.npz"

    # ---- Step: Load plan records ----
    # We keep plan_row_idx (= line index in index.jsonl) for traceability and sharding.
    t_plan0 = time.time()
    records = []
    with open(index_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            if args.num_shards > 1 and (i % args.num_shards) != args.shard_id:
                continue
            r = json.loads(line)
            r["plan_row_idx"] = i
            r["shard_id"] = int(args.shard_id)
            r["num_shards"] = int(args.num_shards)
            records.append(r)
    t_plan_s = time.time() - t_plan0

    if not records:
        raise RuntimeError("Empty plan")

    # ---- Step: Determine location/map from first record's DB metadata (truth) ----
    t_loc0 = time.time()
    first_db = records[0]["db_path"]
    con0 = ro_connect(first_db)
    try:
        location = esf.get_location_from_log(con0)
        map_name = esf.map_name_from_location(location)
    finally:
        con0.close()
    t_loc_s = time.time() - t_loc0

    # ---- Step: Init map api ----
    t_map0 = time.time()
    map_api = esf.get_maps_api(args.map_root, args.map_version, map_name)
    t_map_s = time.time() - t_map0

    # Prepare output buffers (append then stack at end)
    buffers: dict[str, list[np.ndarray]] = {k: [] for k in REQUIRED_KEYS}
    kept_meta = []

    hard_skip = 0
    timeout_count = 0
    bfs_timeout_count = 0
    hard_skip_reasons = Counter()

    soft_flag_far = 0
    soft_flag_counts = Counter()

    # Aggregated timing buckets
    t_extract_total = 0.0
    t_qc_total = 0.0
    t_manifest_total = 0.0
    t_append_total = 0.0

    # Optional per-submethod timings from extractor (EXTRACT_PROFILE=1)
    # key -> list of durations (seconds)
    extract_profile: dict[str, list[float]] = {}
    extract_profile_total: list[float] = []

    # Optional profiling flags from extractor (EXTRACT_PROFILE=1)
    flags_need_bridge = Counter()
    flags_bfs_called = Counter()
    flags_bridge_found = Counter()
    flags_bridge_reason = Counter()
    flags_intersection_pruned: list[int] = []
    flags_rmin_old_m: list[float] = []

    # Correlate BFS timing with bfs_called.
    bfs_time_called: list[float] = []
    bfs_time_not_called: list[float] = []

    t0 = time.time()
    t_prep_s = time.time() - t_prog0

    # ---- Step: Choose processing order within shard ----
    # NOTE: Regardless of schedule, shard membership is defined solely by plan_row_idx % num_shards.
    # We keep plan_row_idx in manifest for auditability and for no-overlap/no-missing checks.
    by_db: dict[str, list[dict]] = {}
    for r in records:
        by_db.setdefault(r["db_path"], []).append(r)

    if args.schedule == "plan":
        iter_records = records
    elif args.schedule == "db_grouped":
        # Deterministic grouping: process dbs in sorted(db_path) order; within each db keep plan order.
        iter_records = []
        for db_path in sorted(by_db.keys()):
            iter_records.extend(by_db[db_path])
    else:
        raise ValueError(f"Unknown --schedule={args.schedule}")

    # Reuse sqlite connections per db_path
    con_cache: dict[str, sqlite3.Connection] = {}

    def get_con(db_path: str) -> sqlite3.Connection:
        if db_path not in con_cache:
            con_cache[db_path] = ro_connect(db_path)
        return con_cache[db_path]

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for idx, r in enumerate(iter_records):
            db_path = r["db_path"]
            scene = r["scene_token_hex"]
            frame = int(r["frame_index"])
            sample_id = r.get("sample_id", f"{Path(db_path).name}:{scene}:{frame}")

            qc_flags: list[str] = []
            pf_for_manifest: dict | None = None

            try:
                con = get_con(db_path)

                # Global timeout guard per sample (some map/route queries can hang).
                import signal

                class _Timeout(Exception):
                    pass

                def _alarm_handler(signum, frame_):
                    raise _Timeout("timeout")

                old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(5)
                # ---- Step: extract_features ----
                t_ex0 = time.time()
                try:
                    # Silence per-frame prints/warnings from the legacy extractor during batch runs.
                    import contextlib
                    with open(os.devnull, 'w') as _dn, contextlib.redirect_stdout(_dn):
                        feats = esf.extract_features(con, map_api, scene, frame, debug_log=False)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                t_ex = time.time() - t_ex0
                t_extract_total += t_ex

                # Optional: aggregate per-submethod timings produced by extractor.
                # Must not affect REQUIRED_KEYS / npz buffers.
                sample_timing = None
                if isinstance(feats, dict) and '_timing' in feats:
                    try:
                        sample_timing = feats.pop('_timing')
                        if isinstance(sample_timing, dict):
                            total_s = 0.0
                            for k, v in sample_timing.items():
                                try:
                                    fv = float(v)
                                    extract_profile.setdefault(str(k), []).append(fv)
                                    total_s += fv
                                except Exception:
                                    continue
                            extract_profile_total.append(float(total_s))
                    except Exception:
                        # Never fail the export due to profiling metadata.
                        sample_timing = None

                # Optional: aggregate per-sample BFS trigger flags produced by extractor.
                if isinstance(feats, dict) and '_profile_flags' in feats:
                    try:
                        pf = feats.pop('_profile_flags')
                        if isinstance(pf, dict):
                            # Keep a compact subset in manifest for sampling/visualization/debug.
                            pf_for_manifest = {
                                k: pf.get(k)
                                for k in [
                                    'need_bridge',
                                    'bfs_called',
                                    'bfs_triggered_by',
                                    'bridge_found',
                                    'bridge_len',
                                    'bridge_reason',
                                    'intersection_pruned',
                                    'ego_rb',
                                    'ego_rb_in_route',
                                    'off_route',
                                    'bad_route_geom',
                                    'rmin_old_m',
                                    'realign_from_overlap',
                                    'overlap_idx',
                                    'route_len_old',
                                    'route_len_new',
                                    'avails_sum_old',
                                    'avails_sum_new',
                                    # Optional TL debug (may be the only key when profiling is disabled).
                                    'route_lane_tl_debug',
                                ]
                                if k in pf
                            }

                            nb = bool(pf.get('need_bridge', False))
                            bc = bool(pf.get('bfs_called', False))
                            bf = bool(pf.get('bridge_found', False))
                            br = str(pf.get('bridge_reason', ''))
                            ip = pf.get('intersection_pruned', None)
                            rm = pf.get('rmin_old_m', None)

                            flags_need_bridge[nb] += 1
                            flags_bfs_called[bc] += 1
                            flags_bridge_found[bf] += 1
                            if br:
                                flags_bridge_reason[br] += 1

                            try:
                                if ip is not None:
                                    flags_intersection_pruned.append(int(ip))
                            except Exception:
                                pass
                            try:
                                if rm is not None:
                                    flags_rmin_old_m.append(float(rm))
                            except Exception:
                                pass

                            # Correlate BFS timing (only meaningful when bfs_called=True)
                            # Treat BFS time as 0.0 when bfs is not called (key absent).
                            bfs_dt = 0.0
                            if isinstance(sample_timing, dict):
                                try:
                                    bfs_dt = float(sample_timing.get('bfs_bridge_route_if_needed', 0.0) or 0.0)
                                except Exception:
                                    bfs_dt = 0.0
                            if bc:
                                bfs_time_called.append(float(bfs_dt))
                            else:
                                bfs_time_not_called.append(float(bfs_dt))
                    except Exception:
                        pass

                # Hard-skip: BFS wall-clock timeout (controlled by BFS_MAX_TIME_S).
                # Exporter policy: treat this as a hard failure (do not keep sample) so downstream training doesn't
                # silently include broken routing.
                if pf_for_manifest is not None:
                    br = str(pf_for_manifest.get('bridge_reason', '') or '')
                    if br.startswith('bfs_exception: timeout'):
                        raise RuntimeError('bfs_timeout')

                # ---- Step: sanity checks + QC ----
                t_qc0 = time.time()
                for k in REQUIRED_KEYS:
                    if k not in feats:
                        raise RuntimeError(f"missing key {k}")
                    if not is_finite(feats[k]):
                        raise RuntimeError(f"non-finite {k}")

                # hard: routing missing
                route_av = feats["route_lanes_avails"]
                route_av_sum = int(np.sum(route_av))
                if route_av_sum == 0:
                    raise RuntimeError("route_lanes_avails_sum==0")

                # hard: lanes_avails all zero
                lanes_av_sum = int(np.sum(feats["lanes_avails"]))
                if lanes_av_sum == 0:
                    raise RuntimeError("lanes_avails_sum==0")

                # soft: route far
                rmin = route_min_dist_m(feats["route_lanes"], route_av)
                if rmin is not None and rmin > 30.0:
                    qc_flags.append("route_min_dist_gt_30m")
                    soft_flag_far += 1
                    soft_flag_counts["route_min_dist_gt_30m"] += 1
                t_qc_total += (time.time() - t_qc0)

                # ---- Step: append buffers ----
                t_ap0 = time.time()
                for k in REQUIRED_KEYS:
                    buffers[k].append(feats[k])
                # Record shard-local index t
                shard_t = len(kept_meta)

                kept_meta.append(r)
                t_append_total += (time.time() - t_ap0)

                # ---- Step: write manifest ----
                t_mf0 = time.time()
                write_jsonl_line(
                    mf,
                    {
                        **r,
                        "sample_id": sample_id,
                        "t": int(shard_t),
                        "qc_hard_skip": False,
                        "qc_flags": qc_flags,
                        "route_lanes_avails_sum": route_av_sum,
                        "lanes_avails_sum": lanes_av_sum,
                        "route_min_dist_m": rmin,
                        "_profile_flags": pf_for_manifest,
                    },
                )
                t_manifest_total += (time.time() - t_mf0)

            except _Timeout:
                hard_skip += 1
                timeout_count += 1
                hard_skip_reasons['timeout'] += 1
                write_jsonl_line(
                    mf,
                    {
                        **r,
                        "sample_id": sample_id,
                        "t": None,
                        "qc_hard_skip": True,
                        "qc_error": "timeout",
                    },
                )

            except Exception as e:
                hard_skip += 1
                err = str(e)
                # Classify common hard-skip reasons for metrics.
                if err == 'bfs_timeout':
                    bfs_timeout_count += 1
                    hard_skip_reasons['bfs_timeout'] += 1
                elif err == 'route_lanes_avails_sum==0':
                    hard_skip_reasons['route_lanes_avails_sum==0'] += 1
                elif err == 'lanes_avails_sum==0':
                    hard_skip_reasons['lanes_avails_sum==0'] += 1
                else:
                    hard_skip_reasons[err] += 1

                write_jsonl_line(
                    mf,
                    {
                        **r,
                        "sample_id": sample_id,
                        "t": None,
                        "qc_hard_skip": True,
                        "qc_error": err,
                    },
                )

            if (idx + 1) % 500 == 0:
                dt_s = time.time() - t0
                fps = (idx + 1) / max(dt_s, 1e-9)
                print(f"[{idx+1}/{len(iter_records)}] processed; kept={len(kept_meta)} skip={hard_skip} fps={fps:.2f}", file=sys.stderr, flush=True)

    # Close DB connections
    for con in con_cache.values():
        try:
            con.close()
        except Exception:
            pass

    # Stack and write NPZ
    kept_n = len(kept_meta)
    if kept_n == 0:
        raise RuntimeError("No samples kept")

    t_stack0 = time.time()
    stacked = {k: np.stack(buffers[k], axis=0) for k in REQUIRED_KEYS}
    t_stack_s = time.time() - t_stack0

    t_npz0 = time.time()
    save_npz_compressed(npz_path, stacked)
    t_npz_save_s = time.time() - t_npz0

    elapsed = time.time() - t0
    # Collect aggregated warnings from extractor (quiet mode counts without spamming).
    log_stats = {}
    try:
        log_stats = esf.get_log_stats()
    except Exception:
        log_stats = {}

    t_write_metrics0 = time.time()
    metrics = {
        "plan_dir": str(plan_dir),
        "map_name": map_name,
        "location": location,
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
        "planned": len(records),
        "schedule": str(args.schedule),
        "kept": kept_n,
        "hard_skipped": hard_skip,
        "timeout_count": timeout_count,
        "bfs_timeout_count": bfs_timeout_count,
        "hard_skip_reasons": dict(hard_skip_reasons),
        "soft_flag_counts": dict(soft_flag_counts),
        "elapsed_s": elapsed,
        "fps_kept": kept_n / max(elapsed, 1e-9),
        "warning_count": int(log_stats.get('warning_total', 0) or 0),
        "warning_by_key": log_stats.get('warning_by_key', {}),
        "npz_path": str(npz_path),
        "npz_size_bytes": int(npz_path.stat().st_size),
        "manifest_path": str(manifest_path),

        # Alias for convenience (same as timing_s.extract_total_s)
        "total_extract_s": float(t_extract_total),

        # Optional per-submethod timing profile (present only when EXTRACT_PROFILE=1)
        "extract_profile": {
            "by_step_s": {k: summarize_durations(v) for k, v in sorted(extract_profile.items())},
            "total_per_sample_s": summarize_durations(extract_profile_total),
            "flags": {
                "need_bridge": {str(k): int(v) for k, v in flags_need_bridge.items()},
                "bfs_called": {str(k): int(v) for k, v in flags_bfs_called.items()},
                "bridge_found": {str(k): int(v) for k, v in flags_bridge_found.items()},
                "bridge_reason": dict(flags_bridge_reason),
                "intersection_pruned": {
                    "summary": summarize_durations([float(x) for x in flags_intersection_pruned]),
                    "buckets": bucketize([float(x) for x in flags_intersection_pruned], bins=[0, 1, 2, 5, 10, 20]),
                },
                "rmin_old_m": {
                    "summary": summarize_durations([float(x) for x in flags_rmin_old_m]),
                    "buckets": bucketize([float(x) for x in flags_rmin_old_m], bins=[0, 1, 2, 5, 10, 20, 30, 50]),
                },
                "bfs_time_s": {
                    "called": summarize_durations(bfs_time_called),
                    "not_called": summarize_durations(bfs_time_not_called),
                },
            },
        },

        "timing_s": {
            "prep_total_s": float(t_prep_s),
            "plan_load_s": float(t_plan_s),
            "location_lookup_s": float(t_loc_s),
            "map_api_init_s": float(t_map_s),
            "loop_elapsed_s": float(t0 and (time.time() - t0) or 0.0),
            "extract_total_s": float(t_extract_total),
            "qc_total_s": float(t_qc_total),
            "manifest_write_total_s": float(t_manifest_total),
            "buffers_append_total_s": float(t_append_total),
            "stack_s": float(t_stack_s),
            "npz_save_s": float(t_npz_save_s),
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    t_write_metrics_s = time.time() - t_write_metrics0

    # Write run info with command + git metadata
    def _sh(cmd: str) -> str:
        import subprocess
        try:
            out = subprocess.check_output(cmd, shell=True, cwd=str(Path(__file__).resolve().parent.parent), stderr=subprocess.DEVNULL)
            return out.decode('utf-8', errors='ignore').strip()
        except Exception:
            return ""

    run_info = {
        "start_time_unix": t0,
        "end_time_unix": time.time(),
        "elapsed_s": elapsed,
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "python": sys.executable,
        "git_commit": _sh('git rev-parse HEAD'),
        "git_status_short": _sh('git status --porcelain'),
        "env": {
            "EXTRACT_LOG_STYLE": os.environ.get('EXTRACT_LOG_STYLE', ''),
            "EXTRACT_WARN_PRINT_N": os.environ.get('EXTRACT_WARN_PRINT_N', ''),
            "EXPORT_SCHEDULE": os.environ.get('EXPORT_SCHEDULE', ''),
        },
        "plan_dir": str(plan_dir),
        "out_dir": str(out_dir),
        "timing_write_metrics_s": float(t_write_metrics_s),
    }
    with open(run_info_path, 'w', encoding='utf-8') as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    print("DONE", file=sys.stderr, flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
