#!/usr/bin/env python3
"""Materialize mmap-friendly training cache for Diffusion Planner.

Converts each shard's:
  - data.npz
  - manifest.jsonl
into an on-disk cache of uncompressed .npy arrays suitable for np.load(..., mmap_mode='r').

Default output layout:
  outputs/cache/training_arrays/
    boston50w_prod/
      <slice>/
        <shard>/
          arrays/<key>.npy
          manifest_kept.jsonl
          CACHE_INFO.json

Incremental behavior:
  If CACHE_INFO.json matches the source files (mtime/size/hash), the shard is skipped.

Example:
  python3 scripts/export/diffusion_planner/pipeline/materialize_training_cache.py \
    --prod-root exports_local/boston50w_prod \
    --slices slice02_N12_20260326_105143 \
    --shards shard_000 \
    --cache-root outputs/cache/training_arrays
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


CACHE_VERSION = 1


@dataclass(frozen=True)
class SourceFiles:
    shard_dir: Path
    npz_path: Path
    manifest_path: Path


def _sha1_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _stat_dict(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path),
        "size": int(st.st_size),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
    }


def _maybe_hash(path: Path, *, do_hash: bool) -> Optional[str]:
    if not do_hash:
        return None
    return _sha1_file(path)


def _discover_slices(prod_root: Path, slices: Optional[List[str]]) -> List[Path]:
    prod_root = prod_root.expanduser().resolve()
    if slices:
        out = []
        for s in slices:
            p = (prod_root / s).resolve()
            if not p.is_dir():
                raise FileNotFoundError(f"Slice not found: {p}")
            out.append(p)
        return out

    # auto-discover slices (dirs containing shards/)
    out = []
    for p in sorted(prod_root.iterdir()):
        if p.is_dir() and (p / "shards").is_dir():
            out.append(p)
    if not out:
        raise FileNotFoundError(f"No slices found under {prod_root} (expected subdirs with shards/)")
    return out


def _discover_shards(slice_dir: Path, shards: Optional[List[str]]) -> List[SourceFiles]:
    shards_root = slice_dir / "shards"
    if not shards_root.is_dir():
        raise FileNotFoundError(f"Missing shards/ under slice: {slice_dir}")

    if shards:
        shard_dirs = [shards_root / s for s in shards]
    else:
        shard_dirs = sorted([p for p in shards_root.glob("shard_*") if p.is_dir()])

    out: List[SourceFiles] = []
    for d in shard_dirs:
        npz = d / "data.npz"
        manifest = d / "manifest.jsonl"
        if not (npz.is_file() and manifest.is_file()):
            raise FileNotFoundError(f"Expected {npz} and {manifest}")
        out.append(SourceFiles(shard_dir=d, npz_path=npz, manifest_path=manifest))
    return out


def _read_manifest_kept(manifest_path: Path) -> Tuple[List[dict], Dict[str, int]]:
    kept: List[dict] = []
    stats = {"manifest_lines": 0, "hard_skip": 0, "kept": 0}

    with manifest_path.open("r") as f:
        for i, line in enumerate(f):
            stats["manifest_lines"] += 1
            obj = json.loads(line)
            if obj.get("qc_hard_skip", False):
                stats["hard_skip"] += 1
                continue
            # Keep a stable mapping back to original manifest line idx (helpful for debugging).
            obj = dict(obj)
            obj["_orig_manifest_idx"] = int(i)
            kept.append(obj)
            stats["kept"] += 1

    return kept, stats


def _load_cache_info(cache_info_path: Path) -> Optional[dict]:
    if not cache_info_path.is_file():
        return None
    try:
        return json.loads(cache_info_path.read_text())
    except Exception:
        return None


def _sources_match(cache_info: dict, npz_path: Path, manifest_path: Path, *, do_hash: bool) -> bool:
    try:
        s_npz = cache_info["sources"]["npz"]
        s_man = cache_info["sources"]["manifest"]
    except Exception:
        return False

    cur_npz = _stat_dict(npz_path)
    cur_man = _stat_dict(manifest_path)

    # Always require size+mtime match.
    for key in ["size", "mtime_ns"]:
        if int(s_npz.get(key, -1)) != int(cur_npz.get(key, -2)):
            return False
        if int(s_man.get(key, -1)) != int(cur_man.get(key, -2)):
            return False

    if do_hash:
        if s_npz.get("sha1") != _sha1_file(npz_path):
            return False
        if s_man.get("sha1") != _sha1_file(manifest_path):
            return False

    return True


def materialize_one(
    *,
    dataset_name: str,
    slice_name: str,
    shard_name: str,
    src: SourceFiles,
    cache_root: Path,
    do_hash: bool,
    force: bool,
) -> Dict[str, Any]:
    shard_cache_dir = cache_root / dataset_name / slice_name / shard_name
    arrays_dir = shard_cache_dir / "arrays"
    cache_info_path = shard_cache_dir / "CACHE_INFO.json"
    manifest_kept_path = shard_cache_dir / "manifest_kept.jsonl"

    cache_info = _load_cache_info(cache_info_path)
    if (not force) and cache_info is not None:
        if _sources_match(cache_info, src.npz_path, src.manifest_path, do_hash=do_hash):
            return {"status": "skipped", "shard_cache_dir": str(shard_cache_dir)}

    arrays_dir.mkdir(parents=True, exist_ok=True)

    kept, mstats = _read_manifest_kept(src.manifest_path)

    # Load arrays once, write each key to .npy.
    with np.load(src.npz_path, allow_pickle=False) as z:
        keys = list(z.files)
        arrays_meta: Dict[str, Any] = {}
        for k in keys:
            arr = z[k]
            out_path = arrays_dir / f"{k}.npy"
            np.save(out_path, arr, allow_pickle=False)
            arrays_meta[k] = {
                "path": str(out_path),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }

        # Basic sanity: kept rows should match first dim for 2D+ arrays.
        # Many feature tensors have leading dimension N.
        n = int(len(kept))
        for k in keys:
            arr = z[k]
            if arr.ndim >= 1 and arr.shape[0] != n:
                raise RuntimeError(
                    f"Shard {src.shard_dir}: key={k} has shape[0]={arr.shape[0]} but kept={n}. "
                    "(Exporter invariant violated?)"
                )

    # Write manifest_kept.jsonl
    with manifest_kept_path.open("w") as f:
        for obj in kept:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    cache_info_out = {
        "cache_version": CACHE_VERSION,
        "generated_unix": int(time.time()),
        "dataset": dataset_name,
        "slice": slice_name,
        "shard": shard_name,
        "sources": {
            "npz": {
                **_stat_dict(src.npz_path),
                "sha1": _maybe_hash(src.npz_path, do_hash=do_hash),
            },
            "manifest": {
                **_stat_dict(src.manifest_path),
                "sha1": _maybe_hash(src.manifest_path, do_hash=do_hash),
            },
        },
        "kept": {
            **mstats,
        },
        "outputs": {
            "shard_cache_dir": str(shard_cache_dir),
            "arrays_dir": str(arrays_dir),
            "manifest_kept": str(manifest_kept_path),
        },
        "arrays": arrays_meta,
    }

    cache_info_path.write_text(json.dumps(cache_info_out, indent=2, sort_keys=True))

    return {
        "status": "materialized",
        "shard_cache_dir": str(shard_cache_dir),
        "kept": int(mstats["kept"]),
        "hard_skip": int(mstats["hard_skip"]),
        "manifest_lines": int(mstats["manifest_lines"]),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod-root", type=str, required=True, help="e.g. exports_local/boston50w_prod")
    ap.add_argument("--dataset-name", type=str, default=None, help="Default: prod-root basename")
    ap.add_argument("--slices", type=str, nargs="+", default=None, help="Slice directory names")
    ap.add_argument("--shards", type=str, nargs="+", default=None, help="Shard directory names")
    ap.add_argument(
        "--cache-root",
        type=str,
        default="outputs/cache/training_arrays",
        help="Cache root (default: outputs/cache/training_arrays)",
    )
    ap.add_argument("--no-hash", action="store_true", help="Skip SHA1 hashing (faster, weaker incrementality)")
    ap.add_argument("--force", action="store_true", help="Force re-materialization")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    prod_root = Path(args.prod_root).expanduser().resolve()
    dataset_name = args.dataset_name or prod_root.name
    cache_root = Path(args.cache_root).expanduser().resolve()

    slice_dirs = _discover_slices(prod_root, args.slices)

    total = {"materialized": 0, "skipped": 0}

    for slice_dir in slice_dirs:
        slice_name = slice_dir.name
        sources = _discover_shards(slice_dir, args.shards)
        for src in sources:
            shard_name = src.shard_dir.name
            res = materialize_one(
                dataset_name=dataset_name,
                slice_name=slice_name,
                shard_name=shard_name,
                src=src,
                cache_root=cache_root,
                do_hash=(not args.no_hash),
                force=args.force,
            )
            print(json.dumps({"slice": slice_name, "shard": shard_name, **res}, ensure_ascii=False))
            total[res["status"]] += 1

    print(f"Done. materialized={total['materialized']} skipped={total['skipped']} cache_root={cache_root}")


if __name__ == "__main__":
    main()
