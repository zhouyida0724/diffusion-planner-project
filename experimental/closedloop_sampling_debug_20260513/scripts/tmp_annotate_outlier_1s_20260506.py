#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location('q', str(ROOT / 'scripts/tmp_tb_viz_quant_20260506.py'))
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)  # type: ignore

from src.methods.diffusion_planner.paper.train.tb_visualizer import _safe_import_matplotlib, _to_numpy

OUT = ROOT / 'outputs/analysis/tb_viz_20260506/annotated_1s'
OUT.mkdir(parents=True, exist_ok=True)
CASES = [
    ('pittsburgh','straight',2),
    ('boston','right',0),
    ('boston','straight',0),
    ('vegas','right',0),
    ('vegas','straight',1),
    ('pittsburgh','left',2),
]

def get_pred(model, batch, steps):
    inp={
        'ego_current_state':batch['ego_current_state'], 'neighbor_agents_past':batch['neighbor_agents_past'],
        'static_objects':batch['static_objects'], 'lanes':batch['lanes'], 'lanes_speed_limit':batch['lanes_speed_limit'],
        'lanes_has_speed_limit':batch['lanes_has_speed_limit'], 'route_lanes':batch['route_lanes'],
        'route_lanes_speed_limit':batch.get('route_lanes_speed_limit'), 'route_lanes_has_speed_limit':batch.get('route_lanes_has_speed_limit'),
        'diffusion_steps':int(steps)}
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(q.DEVICE=='cuda')):
        _, dec = model(inp)
    return dec['prediction'][:,0,:,:2].float()

def render(batch, pred, gt, city, man, slot, steps):
    plt = _safe_import_matplotlib(); assert plt is not None
    import matplotlib.patches as patches
    ego_state = _to_numpy(batch['ego_current_state'])[0]
    lanes = _to_numpy(batch['lanes'])[0]
    route_lanes = _to_numpy(batch['route_lanes'])[0]
    route_av = _to_numpy(batch.get('route_lanes_avails'))[0] if batch.get('route_lanes_avails') is not None else None
    neighbor_past = _to_numpy(batch['neighbor_agents_past'])[0]
    gt_np = gt[0].detach().cpu().numpy(); pred_np = pred[0].detach().cpu().numpy()
    fde1 = float(np.linalg.norm(pred_np[9] - gt_np[9]))
    d01 = float(np.linalg.norm(pred_np[0] - gt_np[0]))
    pts=[gt_np[:,:2], pred_np[:,:2], np.zeros((1,2),dtype=np.float32)]
    allxy=np.concatenate(pts,axis=0)
    lim=max(30.0, float(np.nanmax(np.abs(allxy)))+8.0)
    fig, ax = plt.subplots(1,1,figsize=(8,8),dpi=130)
    try:
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
        ax.grid(True, alpha=0.25); ax.axhline(0,color='k',lw=0.5); ax.axvline(0,color='k',lw=0.5)
        ax.set_title(f'{city}/{man}/sample_{slot} step=145000 sampler={steps} | FDE1={fde1:.2f}m | d0.1={d01:.2f}m')
        ax.set_xlabel('x meters'); ax.set_ylabel('y meters')
        # lanes boundaries
        for lane in lanes:
            x=lane[:,0]; y=lane[:,1]
            if np.allclose(x,0) and np.allclose(y,0): continue
            lx=x+lane[:,4]; ly=y+lane[:,5]; rx=x+lane[:,6]; ry=y+lane[:,7]
            ml=(lx!=0)|(ly!=0); mr=(rx!=0)|(ry!=0)
            ax.plot(lx[ml], ly[ml], 'b--', lw=0.8, alpha=0.35)
            ax.plot(rx[mr], ry[mr], 'r--', lw=0.8, alpha=0.35)
        for i,rl in enumerate(route_lanes):
            x=rl[:,0]; y=rl[:,1]
            if np.allclose(x,0) and np.allclose(y,0): continue
            valid=(route_av[i]>0) if route_av is not None else ((x!=0)|(y!=0))
            if np.any(valid): ax.plot(x[valid], y[valid], '-', color='#D6C000', lw=2.0, alpha=0.75)
        # neighbors current/past
        for a in range(1, neighbor_past.shape[0]):
            past=neighbor_past[a,:,:2]; cur=past[-1]
            if abs(cur[0])<0.1 and abs(cur[1])<0.1: continue
            valid=(past[:,0]!=0)|(past[:,1]!=0)
            if np.any(valid): ax.plot(past[valid,0], past[valid,1], color='gray', lw=1.2, alpha=0.5)
            ax.scatter([cur[0]],[cur[1]],s=12,c='gray',alpha=0.7)
        # current ego
        ax.scatter([0],[0],s=160,c='red',edgecolors='darkred',zorder=5,label='current ego')
        ax.arrow(0,0,4,0,head_width=0.8,head_length=1.2,fc='red',ec='darkred',zorder=5)
        # trajectories
        ax.plot(gt_np[:,0], gt_np[:,1], color='blue', lw=3, alpha=0.75, label='GT full')
        ax.plot(pred_np[:,0], pred_np[:,1], color='orange', lw=3, alpha=0.85, label='pred full')
        ax.plot(gt_np[:10,0], gt_np[:10,1], color='cyan', lw=5, alpha=0.9, label='GT first 1s')
        ax.plot(pred_np[:10,0], pred_np[:10,1], color='magenta', lw=5, alpha=0.9, label='pred first 1s')
        ax.scatter([gt_np[9,0]],[gt_np[9,1]],s=260,c='cyan',edgecolors='black',zorder=10,label='GT @1s')
        ax.scatter([pred_np[9,0]],[pred_np[9,1]],s=260,c='magenta',edgecolors='black',zorder=10,label='pred @1s')
        ax.plot([gt_np[9,0], pred_np[9,0]],[gt_np[9,1], pred_np[9,1]], color='black', lw=2, ls=':', label=f'FDE1 {fde1:.2f}m')
        ax.annotate('GT 1s', xy=gt_np[9], xytext=(8,8), textcoords='offset points', color='blue', fontsize=10, weight='bold')
        ax.annotate('PRED 1s', xy=pred_np[9], xytext=(8,-16), textcoords='offset points', color='darkmagenta', fontsize=10, weight='bold')
        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()
        path=OUT / f'{city}_{man}_sample{slot}_sampler{steps}_fde1_{fde1:.2f}.png'
        fig.savefig(path)
        return path, fde1, d01, pred_np[9].tolist(), gt_np[9].tolist()
    finally:
        plt.close(fig)

def main():
    device=torch.device(q.DEVICE)
    torch.manual_seed(12345)
    samples=q.build_tb_samples(device)
    model=q.build_model(device)
    summary=[]
    for city,man,slot in CASES:
        batch=samples[city][man][slot]
        gt=q.gt_xy(batch).float()
        pred=get_pred(model,batch,10)
        path,fde1,d01,p1,g1=render(batch,pred,gt,city,man,slot,10)
        rec={'case':f'{city}/{man}/sample_{slot}','path':str(path),'fde1':fde1,'d0.1':d01,'pred_1s':p1,'gt_1s':g1}
        print(json.dumps(rec,ensure_ascii=False), flush=True)
        summary.append(rec)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
