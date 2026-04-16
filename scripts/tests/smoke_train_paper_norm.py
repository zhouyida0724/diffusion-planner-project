#!/usr/bin/env python3
"""Smoke test: paper_dit_dpm tiny train step with normalizers.

This is intentionally lightweight and meant to catch:
  - dataset loading from cache_root
  - normalization json loading
  - observation + state normalizers wired into forward/backward
  - one optimizer step without NaNs

Example:
  python scripts/tests/smoke_train_paper_norm.py \
    --arrays-dir outputs/cache/training_arrays/vegas_200k/p2/shard_003/arrays \
    --normalization-file outputs/cache/normalization_ours.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from src.methods.diffusion_planner.paper.model.diffusion_planner import PaperDiffusionPlanner, PaperModelConfig
from src.methods.diffusion_planner.paper.train import train_loop_paper_dit_xstart
from src.methods.diffusion_planner.train.trainer import TrainConfig


class TrainingArraysDataset(torch.utils.data.Dataset):
    """Minimal Dataset that reads a single arrays/ dir produced by training_arrays cache."""

    def __init__(self, arrays_dir: Path):
        super().__init__()
        import numpy as np

        self.arrays_dir = Path(arrays_dir)
        if not self.arrays_dir.is_dir():
            raise FileNotFoundError(f"arrays_dir not found: {self.arrays_dir}")

        def _load(name: str):
            fp = self.arrays_dir / f"{name}.npy"
            if not fp.exists():
                raise FileNotFoundError(f"Missing {fp}")
            return np.load(fp, mmap_mode="r")

        self.ego_current_state = _load("ego_current_state")
        self.ego_agent_future = _load("ego_agent_future")
        self.neighbor_agents_past = _load("neighbor_agents_past")
        self.neighbor_agents_future = _load("neighbor_agents_future")
        self.lanes = _load("lanes")
        self.lanes_speed_limit = _load("lanes_speed_limit")
        self.lanes_has_speed_limit = _load("lanes_has_speed_limit")
        self.route_lanes = _load("route_lanes")
        self.route_lanes_speed_limit = _load("route_lanes_speed_limit")
        self.route_lanes_has_speed_limit = _load("route_lanes_has_speed_limit")
        self.static_objects = _load("static_objects")

        self._n = int(self.ego_current_state.shape[0])

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        # torch tensors, float32
        return {
            "ego_current_state": torch.from_numpy(self.ego_current_state[idx].copy()).float(),
            "ego_agent_future": torch.from_numpy(self.ego_agent_future[idx].copy()).float(),
            "neighbor_agents_past": torch.from_numpy(self.neighbor_agents_past[idx].copy()).float(),
            "neighbor_agents_future": torch.from_numpy(self.neighbor_agents_future[idx].copy()).float(),
            "lanes": torch.from_numpy(self.lanes[idx].copy()).float(),
            "lanes_speed_limit": torch.from_numpy(self.lanes_speed_limit[idx].copy()).float(),
            "lanes_has_speed_limit": torch.from_numpy(self.lanes_has_speed_limit[idx].copy()).float(),
            "route_lanes": torch.from_numpy(self.route_lanes[idx].copy()).float(),
            "route_lanes_speed_limit": torch.from_numpy(self.route_lanes_speed_limit[idx].copy()).float(),
            "route_lanes_has_speed_limit": torch.from_numpy(self.route_lanes_has_speed_limit[idx].copy()).float(),
            "static_objects": torch.from_numpy(self.static_objects[idx].copy()).float(),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays-dir", type=str, required=True, help="Path to a single .../arrays directory")
    ap.add_argument("--normalization-file", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    ds = TrainingArraysDataset(Path(args.arrays_dir))
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True)

    s0 = ds[0]
    nb = s0["neighbor_agents_past"]
    st = s0["static_objects"]
    ln = s0["lanes"]
    rt = s0["route_lanes"]
    ego_f = s0["ego_agent_future"]

    future_len = int(ego_f.shape[0])
    if future_len == 81:
        future_len = 80

    # Use the same logic as train.py to load normalization.
    import json

    norm = json.loads(Path(args.normalization_file).read_text())
    ego_mean = norm["ego"]["mean"]
    ego_std = norm["ego"]["std"]
    nb_mean = norm["neighbor"]["mean"]
    nb_std = norm["neighbor"]["std"]

    paper_cfg = PaperModelConfig(
        device=args.device,
        agent_num=int(nb.shape[0]),
        predicted_neighbor_num=int(nb.shape[0] - 1) if int(nb.shape[0]) > 32 else int(nb.shape[0]),
        time_len=int(nb.shape[1]),
        static_objects_num=int(st.shape[0]),
        lane_num=int(ln.shape[0]),
        route_num=int(rt.shape[0]),
        lane_len=int(ln.shape[1]),
        future_len=future_len,
        state_mean=[[ego_mean]] + [[nb_mean]] * (int(nb.shape[0]) if int(nb.shape[0]) <= 32 else int(nb.shape[0]) - 1),
        state_std=[[ego_std]] + [[nb_std]] * (int(nb.shape[0]) if int(nb.shape[0]) <= 32 else int(nb.shape[0]) - 1),
        observation_norm={k: v for k, v in norm.items() if k not in ("ego", "neighbor")},
    )

    model = PaperDiffusionPlanner(paper_cfg)

    cfg = TrainConfig(
        exp_name="smoke_paper_norm",
        steps=int(args.steps),
        batch_size=2,
        lr=1e-4,
        num_workers=0,
        log_every=1,
        ckpt_every=999999,
        seed=0,
        device=args.device,
        amp="off",
        # keep outputs contained
        out_root=str(_REPO_ROOT / "outputs" / "training"),
        tensorboard=False,
        tb_enable=False,
    )

    train_loop_paper_dit_xstart(cfg=cfg, model=model, train_loader=dl)
    print("[smoke] ok")


if __name__ == "__main__":
    main()
