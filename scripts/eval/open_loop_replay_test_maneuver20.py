#!/usr/bin/env python3
"""Open-loop *replay* evaluation on test maneuver scenarios (straight/left/right).

This is the *aligned-with-closed-loop* open-loop definition:

- Use the same 60 lidar_pc tokens as closed-loop.
- For each scenario, iterate over frames (like closed-loop ticks), but keep ego state on
  the logged (GT) trajectory instead of simulated dynamics.
- At each frame, predict a future trajectory and compute ADE/FDE@1/3/5/8s.

Outputs:
- frame-level JSONL: one row per (scenario, frame)
- scenario-level JSONL: one row per scenario (mean over frames)
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# Ensure repo root importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# nuplan vendor code lives under nuplan-visualization/
_NUPLAN_VENDOR = _REPO_ROOT / "nuplan-visualization"
if str(_NUPLAN_VENDOR) not in sys.path:
    sys.path.insert(0, str(_NUPLAN_VENDOR))

from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

from src.platform.nuplan.features.extract_single_frame import extract_features, map_name_from_location
from src.methods.diffusion_planner.paper.model.diffusion_planner import PaperDiffusionPlanner, PaperModelConfig


def _torch_load_compat(path: Path, *, map_location: torch.device) -> Any:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def _load_model_from_ckpt(ckpt_path: Path, *, device: torch.device) -> PaperDiffusionPlanner:
    ckpt = _torch_load_compat(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise RuntimeError(
            f"unexpected ckpt payload keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )

    cfg_d = ckpt.get("paper_config")
    if isinstance(cfg_d, dict):
        try:
            cfg = PaperModelConfig(**cfg_d)
        except Exception:
            cfg = PaperModelConfig()
            for k, v in cfg_d.items():
                if hasattr(cfg, k):
                    try:
                        setattr(cfg, k, v)
                    except Exception:
                        pass
    else:
        cfg = PaperModelConfig()

    cfg.device = str(device)
    model = PaperDiffusionPlanner(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict strict=False missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device)
    model.eval()
    return model


def _ade_fde(pred_xy: torch.Tensor, gt_xy: torch.Tensor, horizon_idxs: list[int]) -> dict[str, float]:
    assert pred_xy.shape == gt_xy.shape
    d = torch.linalg.norm(pred_xy - gt_xy, dim=-1)  # [T]
    out: dict[str, float] = {}
    for hh in horizon_idxs:
        if hh < 0 or hh >= int(d.shape[0]):
            continue
        ade = d[: hh + 1].mean()
        fde = d[hh]
        sec = (hh + 1) / 10.0
        tag = f"{sec:g}s".replace(".", "p")
        out[f"ade_{tag}"] = float(ade.detach().cpu().item())
        out[f"fde_{tag}"] = float(fde.detach().cpu().item())
    return out


@dataclass
class ScenarioRow:
    group: str
    scenario_type: str
    lidar_pc_token: str
    scene_token: str
    frame_index: int
    db_path: str
    location: str
    map_version: str


def _read_csv(path: Path) -> list[ScenarioRow]:
    out: list[ScenarioRow] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(
                ScenarioRow(
                    group=row["group"],
                    scenario_type=row.get("scenario_type") or "",
                    lidar_pc_token=row["lidar_pc_token"],
                    scene_token=row["scene_token"],
                    frame_index=int(row["frame_index"]),
                    db_path=row["db_path"],
                    location=row.get("location") or "",
                    map_version=row.get("map_version") or "",
                )
            )
    return out


def _default_map_version(map_name: str) -> str:
    if map_name == "us-ma-boston":
        return "9.12.1817"
    if map_name == "us-pa-pittsburgh-hazelwood":
        return "9.17.1937"
    return "9.15.1915"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scenarios-csv",
        type=str,
        default="outputs/eval/test_maneuver20/test_maneuver20_scenarios_final_local.csv",
    )
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--diffusion-steps", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument(
        "--max-frames",
        type=int,
        default=80,
        help=(
            "Max number of frames to evaluate per scenario, starting from the scenario's frame_index. "
            "We stop early if feature extraction fails or GT future is too short."
        ),
    )
    ap.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help=(
            "Stride in ego_pose frame_index per evaluation tick. "
            "ego_pose in nuPlan DB is typically 20Hz; our features/future are 10Hz, "
            "so stride=2 corresponds to 0.1s ticks (aligned with closed-loop)."
        ),
    )
    ap.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    ap.add_argument(
        "--output-frames",
        type=str,
        default=None,
        help="Frame-level JSONL output path.",
    )
    ap.add_argument(
        "--output-scenarios",
        type=str,
        default=None,
        help="Scenario-level JSONL output path (mean over frames).",
    )
    args = ap.parse_args()

    scenarios_csv = Path(args.scenarios_csv).expanduser().resolve()
    rows = _read_csv(scenarios_csv)
    if args.max_scenarios is not None:
        rows = rows[: int(args.max_scenarios)]
    if not rows:
        raise RuntimeError(f"no rows in {scenarios_csv}")

    out_dir = scenarios_csv.parent
    out_frames = Path(args.output_frames) if args.output_frames else out_dir / "open_loop_replay_frames.jsonl"
    out_scenarios = (
        Path(args.output_scenarios) if args.output_scenarios else out_dir / "open_loop_replay_scenario_mean.jsonl"
    )
    out_frames.parent.mkdir(parents=True, exist_ok=True)
    out_scenarios.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = _load_model_from_ckpt(Path(args.ckpt).expanduser().resolve(), device=device)

    horizon_idxs = [9, 29, 49, 79]  # 10Hz: 1/3/5/8 seconds

    # Overwrite outputs (avoid accidental append mixes).
    out_frames.write_text("")
    out_scenarios.write_text("")

    for i, s in enumerate(rows):
        db_path = Path(s.db_path)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        map_name = map_name_from_location(s.location)
        map_version = str(s.map_version or "")
        if (not map_version) or (not all(part.isdigit() for part in map_version.split("."))):
            map_version = _default_map_version(map_name)

        map_root = str(_REPO_ROOT / "data" / "nuplan" / "maps")
        map_api = get_maps_api(map_root=map_root, map_version=map_version, map_name=map_name)

        def t(x: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(device=device)

        per_frame = []
        for j in range(int(args.max_frames)):
            fi = int(s.frame_index) + int(j) * int(args.frame_stride)
            try:
                feats = extract_features(
                    conn,
                    map_api,
                    s.scene_token,
                    fi,
                    debug_log=False,
                    routing_mode="auto",
                    map_root=map_root,
                    map_version=map_version,
                )
            except Exception as e:
                # Stop this scenario early.
                rec = {
                    "idx": int(i),
                    "frame_offset": int(j),
                    "frame_index": int(fi),
                    "group": s.group,
                    "scenario_type": s.scenario_type,
                    "lidar_pc_token": s.lidar_pc_token,
                    "scene_token": s.scene_token,
                    "db_path": str(db_path),
                    "location": s.location,
                    "map_name": map_name,
                    "map_version": map_version,
                    "error": repr(e),
                }
                with out_frames.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                break

            inputs = {
                "ego_current_state": t(feats["ego_current_state"]),
                "neighbor_agents_past": t(feats["neighbor_agents_past"]),
                "static_objects": t(feats["static_objects"]),
                "lanes": t(feats["lanes"]),
                "lanes_speed_limit": t(feats["lanes_speed_limit"]),
                "lanes_has_speed_limit": t(feats["lanes_has_speed_limit"]),
                "route_lanes": t(feats["route_lanes"]),
                "route_lanes_speed_limit": t(feats.get("route_lanes_speed_limit", np.zeros((1,), np.float32))),
                "route_lanes_has_speed_limit": t(feats.get("route_lanes_has_speed_limit", np.zeros((1,), np.float32))),
                "diffusion_steps": int(args.diffusion_steps),
            }

            with torch.no_grad():
                _enc, dec = model(inputs)
            pred = dec["prediction"][0, 0, :, :2]  # [T,2]

            gt = torch.from_numpy(np.asarray(feats["ego_agent_future"], dtype=np.float32)).to(device=device)
            # Some caches store (T+1) with current at idx0.
            if gt.shape[0] == pred.shape[0] + 1:
                gt = gt[1:, :]
            if gt.shape[0] != pred.shape[0]:
                # Can't compute the fixed horizons; stop.
                break

            gt_xy = gt[:, :2]
            m = _ade_fde(pred, gt_xy, horizon_idxs)

            rec = {
                "idx": int(i),
                "frame_offset": int(j),
                "frame_index": int(fi),
                "group": s.group,
                "scenario_type": s.scenario_type,
                "lidar_pc_token": s.lidar_pc_token,
                "scene_token": s.scene_token,
                "db_path": str(db_path),
                "location": s.location,
                "map_name": map_name,
                "map_version": map_version,
                **m,
            }
            with out_frames.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            per_frame.append(rec)

        conn.close()

        if per_frame:
            df = pd.DataFrame(per_frame)
            metric_cols = [
                "ade_1s",
                "ade_3s",
                "ade_5s",
                "ade_8s",
                "fde_1s",
                "fde_3s",
                "fde_5s",
                "fde_8s",
            ]
            metric_cols = [c for c in metric_cols if c in df.columns]
            mean = {c: float(df[c].mean()) for c in metric_cols}
            rec2 = {
                "idx": int(i),
                "group": s.group,
                "scenario_type": s.scenario_type,
                "lidar_pc_token": s.lidar_pc_token,
                "scene_token": s.scene_token,
                "db_path": str(db_path),
                "location": s.location,
                "map_name": map_name,
                "map_version": map_version,
                "n_frames": int(len(per_frame)),
                **mean,
            }
            with out_scenarios.open("a") as f:
                f.write(json.dumps(rec2) + "\n")

        if (i + 1) % 1 == 0:
            print(f"[ok] replay open-loop scenario {i+1}/{len(rows)} -> frames={out_frames.name}")

    print(f"[done] wrote frame-level: {out_frames}")
    print(f"[done] wrote scenario-level: {out_scenarios}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
