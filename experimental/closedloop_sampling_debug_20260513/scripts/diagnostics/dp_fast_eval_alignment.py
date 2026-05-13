#!/usr/bin/env python3
"""Validate paper-model closed-loop runtime dump against fast-eval inference path.

Given a DP_RUNTIME_FEATURE_DUMP tick*.npz, run the same checkpoint with:
  A) closed-loop normal path: model.sample_trajectory(inputs, diffusion_steps=K)
  B) fast-eval path: inputs["diffusion_steps"]=K; _, dec = model(inputs); dec["prediction"][:,0]

With identical RNG seed these should be numerically identical. This proves the
sampler/model call path is aligned; feature alignment is covered separately by
dp_feature_dump_compare.py.
"""
from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

from src.methods.diffusion_planner.paper.model.diffusion_planner import PaperDiffusionPlanner, PaperModelConfig

REQ_KEYS = [
    'ego_current_state','neighbor_agents_past','neighbor_agents_future','ego_agent_future',
    'static_objects','lanes','lanes_speed_limit','lanes_has_speed_limit',
    'route_lanes','route_lanes_speed_limit','route_lanes_has_speed_limit',
    'lanes_avails','route_lanes_avails',
]


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['model_state']
    if any(str(k).startswith('module.') for k in sd.keys()):
        sd = {str(k)[len('module.'):]: v for k, v in sd.items()}
    pc = dict(ckpt.get('paper_config') or ckpt.get('paper_cfg') or ckpt.get('paper_model_cfg') or {})
    pc['device'] = device
    cfg = PaperModelConfig(**{k:v for k,v in pc.items() if k in PaperModelConfig.__annotations__})
    model = PaperDiffusionPlanner(cfg)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model.to(device)


def make_inputs(npz_path: str, cfg, device: str, include_avails: bool=True):
    a = np.load(npz_path, allow_pickle=False)
    missing = [k for k in REQ_KEYS if k not in a.files]
    if missing:
        raise RuntimeError(f'missing keys: {missing}')
    def t(k, shape=None, bool_=False):
        x = torch.from_numpy(np.asarray(a[k]))
        if x.dtype == torch.float64: x = x.float()
        if shape is not None: x = x.view(*shape)
        x = x.unsqueeze(0).to(device)
        return x.bool() if bool_ else x
    inputs = {
        'ego_current_state': t('ego_current_state', (10,)),
        'neighbor_agents_past': t('neighbor_agents_past', (int(cfg.agent_num), int(cfg.time_len), 11)),
        'neighbor_agents_future': t('neighbor_agents_future', (int(cfg.agent_num), int(cfg.future_len), 3)),
        'ego_agent_future': t('ego_agent_future', (int(cfg.future_len), 3)),
        'static_objects': t('static_objects', (int(cfg.static_objects_num), int(cfg.static_objects_state_dim))),
        'lanes': t('lanes', (int(cfg.lane_num), int(cfg.lane_len), 12)),
        'lanes_speed_limit': t('lanes_speed_limit', (int(cfg.lane_num),)),
        'lanes_has_speed_limit': t('lanes_has_speed_limit', (int(cfg.lane_num),)),
        'route_lanes': t('route_lanes', (int(cfg.route_num), int(cfg.lane_len), 12)),
        'route_lanes_speed_limit': t('route_lanes_speed_limit', (int(cfg.route_num),)),
        'route_lanes_has_speed_limit': t('route_lanes_has_speed_limit', (int(cfg.route_num),)),
    }
    if include_avails:
        inputs['lanes_avails'] = t('lanes_avails', (int(cfg.lane_num), int(cfg.lane_len)), True)
        inputs['route_lanes_avails'] = t('route_lanes_avails', (int(cfg.route_num), int(cfg.lane_len)), True)
    return inputs, a


def traj_stats(y: np.ndarray):
    xy = y[:, :2]
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return {
        'path_len': float(seg.sum()),
        'end_x': float(xy[-1,0]),
        'end_y': float(xy[-1,1]),
        'end_dist': float(np.linalg.norm(xy[-1] - xy[0])),
        'x_min': float(xy[:,0].min()), 'x_max': float(xy[:,0].max()),
        'y_min': float(xy[:,1].min()), 'y_max': float(xy[:,1].max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--npz', required=True)
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--no-avails', action='store_true')
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    model = load_model(args.ckpt, args.device)
    inputs, arr = make_inputs(args.npz, model.config, args.device, include_avails=not args.no_avails)

    # Basic actual-input sanity only, excluding legacy ego_past.
    print('INPUT_SANITY')
    for k in sorted(inputs.keys()):
        v = inputs[k]
        vf = v.float() if v.dtype == torch.bool else v
        nonzero = float((vf.abs() > 1e-9).float().mean().cpu())
        finite = float(torch.isfinite(vf).float().mean().cpu())
        print(f'  {k:30s} shape={tuple(v.shape)!s:24s} dtype={str(v.dtype):12s} finite={finite:.6f} nonzero_frac={nonzero:.6f} min={float(vf.min().cpu()):.6g} max={float(vf.max().cpu()):.6g}')

    torch.manual_seed(args.seed)
    if args.device == 'cuda': torch.cuda.manual_seed_all(args.seed)
    y_sample = model.sample_trajectory(dict(inputs), diffusion_steps=args.steps).detach().cpu()

    torch.manual_seed(args.seed)
    if args.device == 'cuda': torch.cuda.manual_seed_all(args.seed)
    inputs_fe = dict(inputs)
    inputs_fe['diffusion_steps'] = int(args.steps)
    _, dec = model(inputs_fe)
    pred = dec['prediction'][0,0].detach().cpu()
    y_fast = torch.stack([pred[:,0], pred[:,1], torch.atan2(pred[:,3], pred[:,2])], dim=-1)

    diff = (y_sample - y_fast).abs().numpy()
    ys = y_sample.numpy(); yf = y_fast.numpy()
    print('ALIGNMENT')
    print(f'  ckpt={args.ckpt}')
    print(f'  npz={args.npz}')
    print(f'  seed={args.seed} steps={args.steps} noise_scale={os.getenv("DP_SAMPLER_NOISE_SCALE", "0.5")} include_avails={not args.no_avails}')
    print(f'  max_abs_diff={float(diff.max()):.9g}')
    print(f'  mean_abs_diff={float(diff.mean()):.9g}')
    print(f'  sample_stats={json.dumps(traj_stats(ys), sort_keys=True)}')
    print(f'  fasteval_stats={json.dumps(traj_stats(yf), sort_keys=True)}')
    print('  sample_first5=', np.round(ys[:5], 6).tolist())
    print('  fasteval_first5=', np.round(yf[:5], 6).tolist())
    if float(diff.max()) > 1e-6:
        raise SystemExit(2)
    print('OK: closed-loop sample_trajectory path and fast-eval model(inputs) path are numerically aligned for this dump')

if __name__ == '__main__':
    main()
