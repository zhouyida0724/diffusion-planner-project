from __future__ import annotations

"""Feature-complete NPZ dataset (returns the full feature dict expected by paper model).

This reads the same sharded NPZ format as `ShardedNpzDataset` but returns a richer
set of tensors.

Expected NPZ keys (from our exporter):
  - ego_current_state: [N, 10]
  - neighbor_agents_past: [N, P, V, D]
  - neighbor_agents_future: [N, P, Tf, 3]
  - ego_agent_future: [N, Tf, 3]
  - static_objects: [N, S, Ds]
  - lanes: [N, L, lane_len, Dl]
  - lanes_speed_limit: [N, L, 1]
  - lanes_has_speed_limit: [N, L, 1]
  - route_lanes: [N, R, lane_len, Dl]
  - route_lanes_speed_limit: [N, R, 1]
  - route_lanes_has_speed_limit: [N, R, 1]

Shapes vary slightly by exporter version; we keep loading generic arrays and let
model-side assertions catch mismatches.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .npz_dataset import ShardSpec


def _discover_feature_shards_one(data_root: Path) -> List[ShardSpec]:
    """Discover shard directories for feature dataset.

    Supports both:
      1) exporter outputs: shard_*/data.npz + manifest.jsonl
      2) cache-only shards: shard_*/arrays/*.npy + manifest_kept.jsonl

    This dataset only uses the manifest for indexing and reads features from
    mmap-friendly cache arrays, so we accept cache shards even without data.npz.
    """

    data_root = data_root.expanduser().resolve()

    if (data_root / "shards").is_dir():
        shards_root = data_root / "shards"
    else:
        shards_root = data_root

    def _mk_spec(d: Path) -> ShardSpec | None:
        # Prefer kept manifest if present (already qc_hard_skip filtered).
        manifest_kept = d / "manifest_kept.jsonl"
        manifest = d / "manifest.jsonl"
        arrays_dir = d / "arrays"
        npz_path = d / "data.npz"

        if arrays_dir.is_dir() and manifest_kept.is_file():
            return ShardSpec(shard_dir=d, npz_path=npz_path, manifest_path=manifest_kept)
        if npz_path.is_file() and manifest.is_file():
            return ShardSpec(shard_dir=d, npz_path=npz_path, manifest_path=manifest)
        return None

    # direct shard directory
    spec = _mk_spec(shards_root)
    if spec is not None:
        return [spec]

    shard_dirs = sorted([p for p in shards_root.glob("shard_*") if p.is_dir()])
    out: List[ShardSpec] = []
    for d in shard_dirs:
        s = _mk_spec(d)
        if s is not None:
            out.append(s)

    if not out:
        raise FileNotFoundError(
            f"No shards found under {data_root}. Expected either shard_*/data.npz+manifest.jsonl or shard_*/arrays+manifest_kept.jsonl"
        )
    return out


def discover_feature_shards(data_roots: List[str | Path] | str | Path) -> List[ShardSpec]:
    """Discover shards from one or many roots (feature dataset).

    Unlike the generic discover_shards(), this also supports cache-only shards.
    """

    if isinstance(data_roots, (str, Path)):
        roots = [data_roots]
    else:
        roots = list(data_roots)

    specs: List[ShardSpec] = []
    for r in roots:
        specs.extend(_discover_feature_shards_one(Path(r)))

    uniq: Dict[Path, ShardSpec] = {}
    for s in specs:
        uniq[s.shard_dir.resolve()] = s

    return [uniq[k] for k in sorted(uniq.keys())]


class ShardedNpzFeatureDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        *,
        max_samples: Optional[int] = None,
        exclude_keys: Optional[set[tuple[str, int]]] = None,
        cache_root: str | Path = "outputs/cache/training_arrays",
        cache_max_open: int = 4,
    ):
        self.data_root = data_root
        self.cache_root = Path(cache_root)
        self.shards: list[ShardSpec] = discover_feature_shards(data_root)
        self.exclude_keys: set[tuple[str, int]] = set(exclude_keys or set())

        # NOTE: paper training MUST NOT fall back to compressed npz.
        # We only read from mmap-friendly arrays prepared by materialize_training_cache.py.

        self._index: List[Tuple[int, int, Dict[str, Any]]] = []

        total_manifest = 0
        total_hard_skip = 0

        for s_idx, spec in enumerate(self.shards):
            row = 0
            with spec.manifest_path.open("r") as f:
                for line in f:
                    total_manifest += 1
                    obj = json.loads(line)
                    if obj.get("qc_hard_skip", False):
                        total_hard_skip += 1
                        continue

                    # NOTE: arrays are stored for kept-only samples (after qc_hard_skip filtering),
                    # in the same order as manifest lines after filtering. Therefore `row` is the
                    # row index into cached arrays for this shard. If we exclude additional samples
                    # (for leakage-free fast-eval), we MUST still increment `row` to keep alignment.
                    key = (str(spec.shard_dir.resolve()), int(row))
                    if key not in self.exclude_keys:
                        self._index.append((s_idx, row, obj))
                    row += 1
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
        }

        # Cache key check on first sample.
        s_idx0, row0, _ = self._index[0]
        spec0 = self.shards[s_idx0]
        cache_dir = self._cache_dir_for_shard(spec0)
        required = [
            "ego_current_state",
            "neighbor_agents_past",
            "ego_agent_future",
            "neighbor_agents_future",
            "static_objects",
            "lanes",
            "lanes_speed_limit",
            "lanes_has_speed_limit",
            "route_lanes",
        ]
        for k in required:
            if not (cache_dir / "arrays" / f"{k}.npy").is_file():
                raise FileNotFoundError(
                    f"Missing cache file: {cache_dir / 'arrays' / (k + '.npy')}\n"
                    f"Please materialize cache first, e.g.:\n"
                    f"  python3 scripts/export/diffusion_planner/pipeline/materialize_training_cache.py "
                    f"--prod-root exports_local/boston50w_prod --cache-root {self.cache_root}" 
                )
        # simple read test
        _ = np.load(cache_dir / "arrays" / "ego_current_state.npy", mmap_mode="r")[row0]

    def __len__(self) -> int:
        return len(self._index)

    def get_data_stats(self) -> Dict[str, Any]:
        return dict(self.data_stats)

    def _cache_dir_for_shard(self, spec: ShardSpec) -> Path:
        """Map a shard spec to its cache directory.

        We keep cache layout stable and collision-free across datasets.

        Supported sources:
          - exports_local/boston50w_prod/<slice_dir>/shards/shard_XXX
            -> <cache_root>/boston50w_prod/<slice_dir>/shard_XXX

          - exports_local/boston200k_new/p0..p4/shard_XXX
            -> <cache_root>/train_data_boston150w/p0..p4/shard_XXX
            (we materialize 200k caches under train_data_boston150w to avoid
             name collisions like shard_000 across partitions).
        """

        # Cache-only shards: shard_dir already *is* the cache dir.
        if (Path(spec.shard_dir) / "arrays").is_dir():
            return Path(spec.shard_dir)

        parts = list(Path(spec.shard_dir).parts)

        # Boston 50w prod slices
        if "boston50w_prod" in parts:
            i = parts.index("boston50w_prod")
            slice_dir = parts[i + 1]
            shard_name = parts[-1]
            return self.cache_root / "boston50w_prod" / slice_dir / shard_name

        # Boston 200k partitions (p0..p4)
        if "boston200k_new" in parts:
            i = parts.index("boston200k_new")
            # expected: .../boston200k_new/p0/shard_000
            part_dir = parts[i + 1] if i + 1 < len(parts) else "unknown_part"
            shard_name = parts[-1]
            return self.cache_root / "train_data_boston150w" / part_dir / shard_name

        # Pittsburgh 200k partitions (p0..p4)
        if "pittsburgh_200k" in parts:
            i = parts.index("pittsburgh_200k")
            part_dir = parts[i + 1] if i + 1 < len(parts) else "unknown_part"
            shard_name = parts[-1]
            return self.cache_root / "pittsburgh_200k" / part_dir / shard_name

        # Vegas 200k partitions (p0..p4)
        if "vegas_200k" in parts:
            i = parts.index("vegas_200k")
            part_dir = parts[i + 1] if i + 1 < len(parts) else "unknown_part"
            shard_name = parts[-1]
            return self.cache_root / "vegas_200k" / part_dir / shard_name

        # generic fallback: use shard dir name only
        return self.cache_root / "unknown" / parts[-1]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_idx, row_idx, meta = self._index[idx]
        spec = self.shards[shard_idx]
        cache_dir = self._cache_dir_for_shard(spec)

        def _t(name: str) -> torch.Tensor:
            p = cache_dir / "arrays" / f"{name}.npy"
            arr = np.load(p, mmap_mode="r")[row_idx]
            arr = np.asarray(arr, dtype=np.float32)
            ten = torch.from_numpy(arr.copy())  # copy out of read-only mmap
            if not torch.isfinite(ten).all():
                raise RuntimeError(f"{name} contains NaN/Inf at idx={idx} shard={spec.shard_dir}")
            return ten

        sample = {
            "ego_current_state": _t("ego_current_state"),
            "neighbor_agents_past": _t("neighbor_agents_past"),
            "ego_agent_future": _t("ego_agent_future"),
            "neighbor_agents_future": _t("neighbor_agents_future"),
            "static_objects": _t("static_objects"),
            "lanes": _t("lanes"),
            "lanes_speed_limit": _t("lanes_speed_limit"),
            "lanes_has_speed_limit": _t("lanes_has_speed_limit"),
            "route_lanes": _t("route_lanes"),
            "route_lanes_speed_limit": _t("route_lanes_speed_limit") if (cache_dir / "arrays" / "route_lanes_speed_limit.npy").is_file() else torch.zeros((1,), dtype=torch.float32),
            "route_lanes_has_speed_limit": _t("route_lanes_has_speed_limit") if (cache_dir / "arrays" / "route_lanes_has_speed_limit.npy").is_file() else torch.zeros((1,), dtype=torch.float32),
            "meta": {
                "sample_id": meta.get("sample_id"),
                "shard_dir": str(spec.shard_dir),
                "row_idx": int(row_idx),
                "t": meta.get("t"),
                # Useful lightweight labels for analysis / fast-eval breakdowns.
                "location": meta.get("location"),
                "tags": meta.get("tags") or [],
            },
        }
        return sample
