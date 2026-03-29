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

from .npz_dataset import ShardSpec, discover_shards, _NpzCache


class ShardedNpzFeatureDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        *,
        max_samples: Optional[int] = None,
        cache_max_open: int = 4,
    ):
        self.data_root = data_root
        self.shards: list[ShardSpec] = discover_shards(data_root)
        self._cache = _NpzCache(max_open=cache_max_open)

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

        # Basic key check on first sample.
        s_idx0, row0, _ = self._index[0]
        with np.load(self.shards[s_idx0].npz_path, allow_pickle=False) as z:
            for k in [
                "ego_current_state",
                "neighbor_agents_past",
                "ego_agent_future",
                "neighbor_agents_future",
                "static_objects",
                "lanes",
                "lanes_speed_limit",
                "lanes_has_speed_limit",
                "route_lanes",
            ]:
                if k not in z:
                    raise KeyError(f"NPZ missing key '{k}'. Available keys: {list(z.keys())}")
            _ = z["ego_current_state"][row0]

    def __len__(self) -> int:
        return len(self._index)

    def get_data_stats(self) -> Dict[str, Any]:
        return dict(self.data_stats)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_idx, row_idx, meta = self._index[idx]
        spec = self.shards[shard_idx]
        z = self._cache.get(spec.npz_path)

        def _t(name: str) -> torch.Tensor:
            arr = z[name][row_idx]
            arr = np.asarray(arr, dtype=np.float32)
            ten = torch.from_numpy(arr)
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
            "route_lanes_speed_limit": _t("route_lanes_speed_limit") if "route_lanes_speed_limit" in z else torch.zeros((1,), dtype=torch.float32),
            "route_lanes_has_speed_limit": _t("route_lanes_has_speed_limit") if "route_lanes_has_speed_limit" in z else torch.zeros((1,), dtype=torch.float32),
            "meta": {
                "sample_id": meta.get("sample_id"),
                "shard_dir": str(spec.shard_dir),
                "row_idx": int(row_idx),
                "t": meta.get("t"),
            },
        }
        return sample
