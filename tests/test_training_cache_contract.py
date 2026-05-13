from __future__ import annotations

from pathlib import Path

from scripts.train.diffusion_planner.train import _discover_slice_dirs
from src.methods.diffusion_planner.data.feature_npz_dataset import ShardedNpzFeatureDataset
from src.methods.diffusion_planner.data.npz_dataset import ShardSpec


def test_discover_slice_dirs_accepts_materialized_stride16_layout(tmp_path: Path) -> None:
    shard = tmp_path / "exports_stride16_v0.1" / "vegas_q0" / "shard_000"
    (shard / "arrays").mkdir(parents=True)
    (shard / "manifest_kept.jsonl").write_text("{}\n", encoding="utf-8")

    assert _discover_slice_dirs(tmp_path / "exports_stride16_v0.1") == [tmp_path / "exports_stride16_v0.1" / "vegas_q0"]


def test_cache_dir_for_shard_preserves_stride16_part_name(tmp_path: Path) -> None:
    dataset = object.__new__(ShardedNpzFeatureDataset)
    dataset.cache_root = tmp_path / "cache"
    spec = ShardSpec(
        shard_dir=tmp_path / "exports_stride16_v0.1" / "singapore_q3" / "shard_042",
        npz_path=tmp_path / "exports_stride16_v0.1" / "singapore_q3" / "shard_042" / "data.npz",
        manifest_path=tmp_path / "exports_stride16_v0.1" / "singapore_q3" / "shard_042" / "manifest.jsonl",
    )

    assert dataset._cache_dir_for_shard(spec) == tmp_path / "cache" / "exports_stride16_v0.1" / "singapore_q3" / "shard_042"
