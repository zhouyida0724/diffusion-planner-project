#!/usr/bin/env python3
from __future__ import annotations
import json, math, sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.diffusion_planner.train import _build_balanced_subset_dataset, _expand_roots
from src.methods.diffusion_planner.data.feature_npz_dataset import ShardedNpzFeatureDataset
from src.methods.diffusion_planner.paper.model.diffusion_planner import PaperDiffusionPlanner, PaperModelConfig
from src.methods.diffusion_planner.paper.train.paper_trainer import _ade_fde_at_horizons

EXP = Path("outputs/training/stride16_v01_4city_e10_bs96_bf16_linear_lr2e5_20260505")
CKPT = EXP / "checkpoint_step_145000.pt"
CACHE_ROOT = "/media/zhouyida/新加卷1/training_arrays"
DATA_ROOT = "/media/zhouyida/新加卷1/training_arrays/exports_stride16_v0.1"
NORM = "/media/zhouyida/新加卷1/training_arrays/exports_stride16_v0.1/normalization_ours_20260503.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FAST_N = 1024
BATCH_SIZE = 32
NUM_WORKERS = 0
TB_NUM_SAMPLES = 3

CITY_ROOTS = {
    "boston": ([DATA_ROOT], ["boston_h0_redo_20260503_1703", "boston_h1"]),
    "pittsburgh": ([f"{DATA_ROOT}/pittsburgh_q{i}" for i in range(4)], None),
    "vegas": ([f"{DATA_ROOT}/vegas_1_q{i}" for i in range(4)], None),
    "singapore": ([f"{DATA_ROOT}/singapore_q{i}" for i in range(4)], None),
}

def tb_label(sample: dict[str, Any]) -> str:
    fut = sample.get("ego_agent_future")
    arr = fut.detach().float().cpu() if torch.is_tensor(fut) else torch.as_tensor(fut, dtype=torch.float32)
    if arr.ndim == 3:
        arr = arr[0]
    valid = torch.sum(torch.ne(arr[:, :2], 0.0), dim=-1) != 0
    idx = int(torch.nonzero(valid, as_tuple=False)[-1].item()) if torch.any(valid) else int(arr.shape[0]-1)
    yaw = float(arr[idx,2].item()) if arr.shape[-1] >= 3 else 0.0
    if yaw > 0.35: return "left"
    if yaw < -0.35: return "right"
    return "straight"

def sample_one(batch: dict[str, Any], i: int, device: torch.device) -> dict[str, Any]:
    out = {}
    for k,v in batch.items():
        if torch.is_tensor(v):
            out[k] = v[i:i+1].to(device=device, dtype=torch.float32)
        else:
            out[k] = v
    return out

def build_tb_samples(device: torch.device):
    out = {}
    for city,(roots,slices) in CITY_ROOTS.items():
        expanded = _expand_roots(list(roots), slices)
        ds, _ = _build_balanced_subset_dataset(roots=expanded, n_total=FAST_N, seed=0, cache_root=CACHE_ROOT)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False, drop_last=False)
        buckets = {"straight": [], "left": [], "right": []}
        for batch in dl:
            B = next(v.shape[0] for v in batch.values() if torch.is_tensor(v))
            for i in range(B):
                s = sample_one(batch, i, device)
                lab = tb_label(s)
                if len(buckets[lab]) < TB_NUM_SAMPLES:
                    buckets[lab].append(s)
            if all(len(buckets[m]) >= TB_NUM_SAMPLES for m in buckets):
                break
        out[city] = buckets
    return out

def build_model(device: torch.device):
    # Use one actual slice first sample only for shapes.
    shape_roots = _expand_roots([DATA_ROOT], ["boston_h0_redo_20260503_1703"])
    ds = ShardedNpzFeatureDataset(shape_roots, max_samples=None, cache_root=CACHE_ROOT)
    s0 = ds[0]
    nb, st, ln, rt, ego_f = s0["neighbor_agents_past"], s0["static_objects"], s0["lanes"], s0["route_lanes"], s0["ego_agent_future"]
    future_len = int(ego_f.shape[0])
    if future_len == 81: future_len = 80
    try:
        slot0_is_ego = bool(torch.allclose(nb[0, -1, :4], s0["ego_current_state"][:4], atol=1e-4, rtol=0.0))
    except Exception:
        slot0_is_ego = False
    predicted_neighbor_num = int(nb.shape[0] - 1) if slot0_is_ego else int(nb.shape[0])
    cfg = PaperModelConfig(
        device=str(device),
        agent_num=int(1 + predicted_neighbor_num),
        predicted_neighbor_num=predicted_neighbor_num,
        time_len=int(nb.shape[1]),
        static_objects_num=int(st.shape[0]),
        lane_num=int(ln.shape[0]),
        route_num=int(rt.shape[0]),
        lane_len=int(ln.shape[1]),
        future_len=future_len,
    )
    data = json.loads(Path(NORM).read_text())
    if "ego" in data and "neighbor" in data:
        cfg.state_mean = [[data["ego"]["mean"]]] + [[data["neighbor"]["mean"]]] * int(cfg.predicted_neighbor_num)
        cfg.state_std = [[data["ego"]["std"]]] + [[data["neighbor"]["std"]]] * int(cfg.predicted_neighbor_num)
    obs_norm = {k:v for k,v in data.items() if k not in ("ego","neighbor")}
    cfg.observation_norm = obs_norm if obs_norm else None
    model = PaperDiffusionPlanner(cfg).to(device)
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model

def gt_xy(batch):
    gt = batch["ego_agent_future"]
    if gt.shape[-2] == 81:
        gt = gt[:,1:,:]
    return gt[...,:2]

def path_len(xy: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(xy[:,1:,:] - xy[:,:-1,:], dim=-1).sum(dim=-1)

def eval_sample(model, batch, steps_s: int):
    inputs = {
        "ego_current_state": batch["ego_current_state"],
        "neighbor_agents_past": batch["neighbor_agents_past"],
        "static_objects": batch["static_objects"],
        "lanes": batch["lanes"],
        "lanes_speed_limit": batch["lanes_speed_limit"],
        "lanes_has_speed_limit": batch["lanes_has_speed_limit"],
        "route_lanes": batch["route_lanes"],
        "route_lanes_speed_limit": batch.get("route_lanes_speed_limit"),
        "route_lanes_has_speed_limit": batch.get("route_lanes_has_speed_limit"),
        "diffusion_steps": int(steps_s),
    }
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE=="cuda")):
        _, dec = model(inputs)
    pred = dec["prediction"][:,0,:,:2].float()
    gt = gt_xy(batch).float()
    m = _ade_fde_at_horizons(pred, gt, [9,29,49,79])
    d10 = torch.linalg.norm(pred[:,9,:]-gt[:,9,:],dim=-1).item()
    rec = {k: float(v) for k,v in m.items()}
    rec.update({
        "d_1s": float(d10),
        "pred_end_norm_m": float(torch.linalg.norm(pred[:,79,:],dim=-1).item()),
        "gt_end_norm_m": float(torch.linalg.norm(gt[:,79,:],dim=-1).item()),
        "pred_path_len_m": float(path_len(pred).item()),
        "gt_path_len_m": float(path_len(gt).item()),
        "pred_xy_1s": [float(x) for x in pred[0,9,:].detach().cpu().tolist()],
        "gt_xy_1s": [float(x) for x in gt[0,9,:].detach().cpu().tolist()],
        "pred_xy_8s": [float(x) for x in pred[0,79,:].detach().cpu().tolist()],
        "gt_xy_8s": [float(x) for x in gt[0,79,:].detach().cpu().tolist()],
    })
    return rec

def main():
    device=torch.device(DEVICE)
    torch.manual_seed(12345)
    samples = build_tb_samples(device)
    model = build_model(device)
    rows=[]
    for city,buckets in samples.items():
        for maneuver, ss in buckets.items():
            for slot,batch in enumerate(ss):
                rec={"city":city,"maneuver":maneuver,"slot":slot,"step":145000}
                # GT stats
                gt=gt_xy(batch).float()
                rec["gt_path_len_m"] = float(path_len(gt).item())
                rec["gt_end_norm_m"] = float(torch.linalg.norm(gt[:,79,:],dim=-1).item())
                for steps in (3,10):
                    r=eval_sample(model,batch,steps)
                    for k,v in r.items(): rec[f"s{steps}_{k}"]=v
                rows.append(rec)
                print(json.dumps(rec, ensure_ascii=False))
    out=Path('outputs/analysis/tb_viz_20260506/tb_viz_quant_step145000.jsonl')
    out.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
    # summary
    import collections
    for steps in (3,10):
        print('\nSUMMARY steps',steps)
        groups=collections.defaultdict(list)
        for r in rows:
            groups['ALL'].append(r)
            groups[r['maneuver']].append(r)
            groups[r['city']].append(r)
        for g,rs in groups.items():
            vals={k: sum(float(r[f's{steps}_{k}']) for r in rs)/len(rs) for k in ['ade_1s','ade_3s','ade_5s','ade_8s','fde_1s','fde_3s','fde_5s','fde_8s','pred_path_len_m','gt_path_len_m']}
            print(g, 'n',len(rs), ' '.join(f'{k}={v:.3f}' for k,v in vals.items()))

if __name__ == '__main__':
    main()
