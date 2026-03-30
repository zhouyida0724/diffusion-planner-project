#!/usr/bin/env python3
"""Print distribution / stats for exported dataset shards using `npz_dataset.py` logic.

This is for quick QC over multiple slices/shards (e.g., Boston 50w slice01-05).
It only reads `manifest.jsonl` (fast) and does NOT load `data.npz`.

Examples
--------
# Scan a production root (auto-detect slice* directories)
python3 scripts/train/diffusion_planner/print_dataset_distribution.py \
  --prod-root exports_local/boston50w_prod

# Scan explicit slice/shard roots
python3 scripts/train/diffusion_planner/print_dataset_distribution.py \
  --roots exports_local/boston50w_prod/slice05_N12_20260328_221211/shards

Outputs
-------
- prints a human-readable summary
- optionally writes JSON to --out-json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _as_rel(p: Path, base: Path) -> str:
    try:
        return str(p.relative_to(base))
    except Exception:
        return str(p)


@dataclass
class ShardStats:
    manifest_lines: int = 0
    hard_skipped: int = 0
    kept: int = 0
    soft_flag_counts: Dict[str, int] = field(default_factory=dict)
    tag_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def hard_skip_ratio(self) -> float:
        return float(self.hard_skipped) / float(self.manifest_lines) if self.manifest_lines else 0.0


@dataclass
class DatasetScanStats:
    totals: ShardStats
    by_shard: Dict[str, ShardStats]


def _iter_shard_dirs(root: Path) -> Iterable[Path]:
    """Accept slice dir, shards dir, or shard dir and return shard_* dirs."""
    root = root.resolve()

    # shard dir
    if root.is_dir() and (root / "manifest.jsonl").exists() and (root / "data.npz").exists():
        yield root
        return

    # slice dir containing shards/
    if (root / "shards").is_dir():
        for d in sorted((root / "shards").glob("shard_*")):
            if d.is_dir():
                yield d
        return

    # shards dir
    for d in sorted(root.glob("shard_*")):
        if d.is_dir():
            yield d


def _scan_manifest(manifest_path: Path) -> ShardStats:
    st = ShardStats()
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            st.manifest_lines += 1
            obj = json.loads(line)

            hard = bool(obj.get("qc_hard_skip", False))
            if hard:
                st.hard_skipped += 1
            else:
                st.kept += 1

            flags = obj.get("qc_flags", None)
            if isinstance(flags, list):
                for flg in flags:
                    if not flg:
                        continue
                    st.soft_flag_counts[str(flg)] = st.soft_flag_counts.get(str(flg), 0) + 1

            tags = obj.get("tags", None)
            if isinstance(tags, list):
                for tg in tags:
                    if not tg:
                        continue
                    st.tag_counts[str(tg)] = st.tag_counts.get(str(tg), 0) + 1

    return st


def _expand_prod_root_if_needed(root: Path) -> List[Path]:
    """If `root` looks like a prod root containing slice* dirs, expand to those slices."""
    if not root.is_dir():
        return [root]
    slice_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("slice")])
    # Only treat as prod root if it has slice dirs and doesn't itself look like a shard/shards dir.
    if slice_dirs and not (root / "manifest.jsonl").exists() and not (root / "data.npz").exists():
        return slice_dirs
    return [root]


def scan_roots(roots: List[Path], *, repo_root: Path) -> DatasetScanStats:
    by_shard: Dict[str, ShardStats] = {}
    totals = ShardStats()

    expanded: List[Path] = []
    for r in roots:
        expanded.extend(_expand_prod_root_if_needed(r))

    for r in expanded:
        for shard_dir in _iter_shard_dirs(r):
            manifest = shard_dir / "manifest.jsonl"
            if not manifest.exists():
                continue
            st = _scan_manifest(manifest)
            key = _as_rel(shard_dir, repo_root)
            by_shard[key] = st

            totals.manifest_lines += st.manifest_lines
            totals.hard_skipped += st.hard_skipped
            totals.kept += st.kept
            for k, v in st.soft_flag_counts.items():
                totals.soft_flag_counts[k] = totals.soft_flag_counts.get(k, 0) + int(v)
            for k, v in st.tag_counts.items():
                totals.tag_counts[k] = totals.tag_counts.get(k, 0) + int(v)

    return DatasetScanStats(totals=totals, by_shard=by_shard)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod-root", type=str, default=None, help="e.g. exports_local/boston50w_prod")
    ap.add_argument("--roots", type=str, nargs="*", default=None, help="slice/shards root(s) or shard root(s)")
    ap.add_argument("--out-json", type=str, default=None, help="optional json output path")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[3]

    roots: List[Path] = []
    if args.prod_root:
        prod = (repo_root / args.prod_root).resolve() if not Path(args.prod_root).is_absolute() else Path(args.prod_root)
        if not prod.exists():
            raise FileNotFoundError(f"prod-root not found: {prod}")
        # Only pick slice* directories
        roots = sorted([p for p in prod.iterdir() if p.is_dir() and p.name.startswith("slice")])
    if args.roots:
        for r in args.roots:
            p = (repo_root / r).resolve() if not Path(r).is_absolute() else Path(r)
            roots.append(p)

    if not roots:
        raise SystemExit("No roots provided. Use --prod-root or --roots")

    stats: DatasetScanStats = scan_roots(roots, repo_root=repo_root)

    # Pretty print
    total = stats.totals
    print("=== Dataset distribution (manifest-only) ===")
    print(f"roots: {len(roots)}")
    print(f"shards_found: {len(stats.by_shard)}")
    print("--- totals ---")
    print(f"manifest_lines: {total.manifest_lines}")
    print(f"hard_skipped:   {total.hard_skipped}  (ratio={total.hard_skip_ratio:.4f})")
    print(f"kept:          {total.kept}")

    if total.soft_flag_counts:
        print("--- soft flags (total) ---")
        for k, v in sorted(total.soft_flag_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"{k}: {v}")

    if total.tag_counts:
        print("--- scenario tags (total, top 30) ---")
        items = sorted(total.tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:30]
        for k, v in items:
            print(f"{k}: {v}")

    print("--- per-shard ---")
    for shard_key, s in sorted(stats.by_shard.items(), key=lambda kv: kv[0]):
        print(
            f"{shard_key}: manifest={s.manifest_lines} kept={s.kept} hard={s.hard_skipped} "
            f"ratio={s.hard_skip_ratio:.4f}"
        )

    if args.out_json:
        out = (repo_root / args.out_json).resolve() if not Path(args.out_json).is_absolute() else Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "roots": [_as_rel(p, repo_root) for p in roots],
            "stats": {
                "totals": asdict(stats.totals),
                "by_shard": {k: asdict(v) for k, v in stats.by_shard.items()},
            },
        }
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
