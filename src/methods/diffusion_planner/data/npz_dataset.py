"""NPZ + manifest.jsonl dataset for Diffusion Planner.

Dataset format (per shard directory):
  shard_XXX/
    - data.npz
    - manifest.jsonl

`manifest.jsonl` contains one JSON object per *candidate* sample. Some rows are
filtered out via `qc_hard_skip: true`.

Important: the `data.npz` arrays are stored for *kept-only* samples, in the same
order as the manifest lines **after filtering qc_hard_skip**.

So mapping from manifest line -> npz row index is:
  row_idx_in_npz = count_non_skipped_lines_seen_so_far

This file provides a lightweight PyTorch Dataset that indexes across many
shards without loading all shards into RAM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ShardSpec:
    shard_dir: Path
    npz_path: Path
    manifest_path: Path


def _discover_shards_one(data_root: Path) -> List[ShardSpec]:
    """Discover shard directories under a single root.

    Accepts either:
      - a slice directory containing `shards/`
      - a `shards/` directory containing `shard_*/`
      - a direct path to a single `shard_*/` directory
    """

    data_root = data_root.expanduser().resolve()

    if (data_root / "shards").is_dir():
        shards_root = data_root / "shards"
    else:
        shards_root = data_root

    # direct shard directory
    if (shards_root / "data.npz").is_file() and (shards_root / "manifest.jsonl").is_file():
        shard_dirs = [shards_root]
    else:
        shard_dirs = sorted([p for p in shards_root.glob("shard_*") if p.is_dir()])

    out: List[ShardSpec] = []
    for d in shard_dirs:
        npz_path = d / "data.npz"
        manifest_path = d / "manifest.jsonl"
        if npz_path.is_file() and manifest_path.is_file():
            out.append(ShardSpec(shard_dir=d, npz_path=npz_path, manifest_path=manifest_path))

    if not out:
        raise FileNotFoundError(
            f"No shards found under {data_root}. Expected shard_*/data.npz + manifest.jsonl"
        )
    return out


def discover_shards(data_roots: List[str | Path] | str | Path) -> List[ShardSpec]:
    """Discover shards from one or many roots.

    This is a convenience wrapper used by the training entrypoint.

    Args:
        data_roots: either a single path or a list of paths. Each path can be:
          - slice dir
          - shards/ dir
          - shard_*/ dir

    Returns:
        List[ShardSpec] sorted by shard_dir.
    """

    if isinstance(data_roots, (str, Path)):
        roots = [data_roots]
    else:
        roots = list(data_roots)

    specs: List[ShardSpec] = []
    for r in roots:
        specs.extend(_discover_shards_one(Path(r)))

    # Deduplicate by absolute shard_dir
    uniq: Dict[Path, ShardSpec] = {}
    for s in specs:
        uniq[s.shard_dir.resolve()] = s

    return [uniq[k] for k in sorted(uniq.keys())]


class _NpzCache:
    """Small cache for opened npz files.

    np.load returns an NpzFile wrapper that lazily loads arrays on access.
    We keep a few open handles to avoid re-opening on every __getitem__.
    """

    def __init__(self, max_open: int = 4):
        self.max_open = max_open
        self._cache: Dict[Path, Any] = {}
        self._lru: List[Path] = []

    def get(self, npz_path: Path):
        if npz_path in self._cache:
            # refresh LRU
            self._lru.remove(npz_path)
            self._lru.append(npz_path)
            return self._cache[npz_path]

        z = np.load(npz_path, allow_pickle=False)
        self._cache[npz_path] = z
        self._lru.append(npz_path)

        while len(self._lru) > self.max_open:
            old = self._lru.pop(0)
            try:
                self._cache[old].close()
            finally:
                self._cache.pop(old, None)

        return z


class ShardedNpzDataset(Dataset):
    """Indexes kept-only samples across multiple shards.

    Returned sample contains:
      - x: float tensor [x_dim]
      - y: float tensor [T, 3] (ego future)
      - meta: dict with sample_id, shard_dir, etc.

    For now `x` is a simple concatenation of:
      ego_current_state (10,) + ego_past flattened (21*3,)

    This is intentionally minimal for a smoke/sanity training run.
    """

    def __init__(
        self,
        data_root: str | Path,
        *,
        max_samples: Optional[int] = None,
        cache_max_open: int = 4,
    ):
        self.data_root = data_root
        self.shards = discover_shards(data_root)
        self._cache = _NpzCache(max_open=cache_max_open)

        # Build global index: list[(shard_idx, row_idx_in_npz, manifest_obj)]
        self._index: List[Tuple[int, int, Dict[str, Any]]] = []

        # Stats while scanning manifests
        per_shard: Dict[str, Dict[str, int]] = {}
        total_manifest = 0
        total_hard_skip = 0

        for s_idx, spec in enumerate(self.shards):
            shard_key = str(spec.shard_dir)
            per_shard[shard_key] = {
                "manifest_lines": 0,
                "hard_skip": 0,
                "kept": 0,
                "kept_used": 0,
            }

            row = 0
            with spec.manifest_path.open("r") as f:
                for line in f:
                    per_shard[shard_key]["manifest_lines"] += 1
                    total_manifest += 1
                    obj = json.loads(line)
                    if obj.get("qc_hard_skip", False):
                        per_shard[shard_key]["hard_skip"] += 1
                        total_hard_skip += 1
                        continue

                    per_shard[shard_key]["kept"] += 1
                    self._index.append((s_idx, row, obj))
                    row += 1
                    per_shard[shard_key]["kept_used"] += 1

                    if max_samples is not None and len(self._index) >= max_samples:
                        break

            if max_samples is not None and len(self._index) >= max_samples:
                break

        if not self._index:
            raise RuntimeError(f"No kept samples found under {data_root}")

        self.data_stats = {
            "max_samples": max_samples,
            "num_shards": len(self.shards),
            "manifest_lines": int(total_manifest),
            "hard_skip": int(total_hard_skip),
            "kept_used": int(len(self._index)),
            "hard_skip_ratio": float(total_hard_skip / max(total_manifest, 1)),
            "per_shard": per_shard,
        }

        # Validate shapes from first sample.
        # Important: do NOT keep an NPZ file handle open here.
        # DataLoader may fork worker processes; inheriting an open ZipFile can
        # lead to zlib decompression errors under multiprocessing.
        s_idx0, row0, _ = self._index[0]
        with np.load(self.shards[s_idx0].npz_path, allow_pickle=False) as z:
            assert z["ego_current_state"].ndim == 2 and z["ego_current_state"].shape[1] == 10
            assert z["ego_past"].ndim == 3 and z["ego_past"].shape[1:] == (21, 3)
            assert z["ego_agent_future"].ndim == 3 and z["ego_agent_future"].shape[1:] == (80, 3)
            _ = z["ego_current_state"][row0]  # ensure row in range

        self.x_dim = 10 + 21 * 3
        self.y_T = 80

    def __len__(self) -> int:
        return len(self._index)

    def get_data_stats(self) -> Dict[str, Any]:
        """Return JSON-serializable dataset stats (manifest counts, skips, etc.)."""

        return dict(self.data_stats)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_idx, row_idx, meta = self._index[idx]
        spec = self.shards[shard_idx]
        z = self._cache.get(spec.npz_path)

        ego_current = z["ego_current_state"][row_idx].astype(np.float32)  # (10,)
        ego_past = z["ego_past"][row_idx].astype(np.float32)  # (21,3)
        y = z["ego_agent_future"][row_idx].astype(np.float32)  # (80,3)

        x = np.concatenate([ego_current.reshape(-1), ego_past.reshape(-1)], axis=0)

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)

        # Hard assertions to catch silent corruption early
        assert x_t.shape == (self.x_dim,), f"x shape {x_t.shape}"
        assert y_t.shape == (self.y_T, 3), f"y shape {y_t.shape}"
        assert torch.isfinite(x_t).all(), "x contains NaN/Inf"
        assert torch.isfinite(y_t).all(), "y contains NaN/Inf"

        return {
            "x": x_t,
            "y": y_t,
            "meta": {
                "sample_id": meta.get("sample_id"),
                "shard_dir": str(spec.shard_dir),
                "row_idx": int(row_idx),
                "t": meta.get("t"),
            },
        }
