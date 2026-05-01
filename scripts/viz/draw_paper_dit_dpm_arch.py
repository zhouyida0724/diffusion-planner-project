#!/usr/bin/env python3
"""Draw a high-level architecture diagram for our paper_dit_dpm Diffusion Planner.

This is a *structure* diagram (modules + data flow), not an autograd trace.
It explicitly shows the training vs inference branch in Decoder.

Usage:
  python3 scripts/viz/draw_paper_dit_dpm_arch.py \
    --out outputs/viz/paper_dit_dpm_arch.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def box(ax, x, y, w, h, text, *, fc="#ffffff", ec="#222222", lw=1.5, fontsize=10):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return p


def arrow(ax, x0, y0, x1, y1, *, text=None, fontsize=9):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=1.4, color="#222222"),
    )
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=fontsize)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="outputs/viz/paper_dit_dpm_arch.png")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9), dpi=140)
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Left: inputs
    b_in = box(
        ax,
        0.03,
        0.70,
        0.22,
        0.22,
        "Inputs (feature dict)\n\n- ego_current_state\n- neighbor_agents_past\n- static_objects\n- lanes / route_lanes\n- ...\n- meta.tags (for TB sampling only)",
        fc="#f7fbff",
        fontsize=9,
    )

    b_obs = box(
        ax,
        0.30,
        0.76,
        0.18,
        0.12,
        "ObservationNormalizer\n(key-wise mean/std)\n\nMANDATORY",
        fc="#fff5f5",
        ec="#b00020",
        fontsize=9,
    )

    arrow(ax, 0.25, 0.81, 0.30, 0.82)

    # Encoder
    b_enc = box(
        ax,
        0.52,
        0.74,
        0.18,
        0.16,
        "Encoder\n\n(agent/static/lane encoders\n+ mixer blocks)",
        fc="#f3fff5",
        fontsize=10,
    )
    arrow(ax, 0.48, 0.82, 0.52, 0.82)

    b_enc_out = box(
        ax,
        0.74,
        0.78,
        0.22,
        0.08,
        "encoder_outputs\n(encoding, ...)",
        fc="#f3fff5",
        fontsize=10,
    )
    arrow(ax, 0.70, 0.82, 0.74, 0.82)

    # Decoder (two branches)
    b_dec = box(
        ax,
        0.52,
        0.46,
        0.44,
        0.22,
        "Decoder (DiT + RouteEncoder)\n\nShared precompute:\n- neighbor_current_mask\n- current_states (ego + neighbors)",
        fc="#fffdf0",
        fontsize=10,
    )

    # Arrows into decoder: encoder_outputs and normalized inputs
    arrow(ax, 0.85, 0.78, 0.85, 0.68)
    arrow(ax, 0.48, 0.76, 0.55, 0.68, text="inputs_n")

    # Training branch
    b_train = box(
        ax,
        0.56,
        0.50,
        0.18,
        0.12,
        "TRAINING branch\n(self.training=True)\n\nDiT(sampled_trajectories,\n    diffusion_time,\n    cross_c, route_lanes,\n    mask)",
        fc="#eef7ff",
        fontsize=9,
    )

    # Inference branch
    b_inf = box(
        ax,
        0.76,
        0.50,
        0.18,
        0.12,
        "INFERENCE branch\n(self.training=False)\n\nStateNormalizer\n+ xT init\n+ DPM-Solver sampler\n+ inverse-normalize",
        fc="#eef7ff",
        fontsize=9,
    )

    # Output
    b_out = box(
        ax,
        0.62,
        0.20,
        0.30,
        0.14,
        "Outputs\n\n- prediction: [B,P,T,4]\n- (TB) denoise panels / sampler intermediates",
        fc="#f7fbff",
        fontsize=10,
    )

    arrow(ax, 0.65, 0.50, 0.72, 0.34)
    arrow(ax, 0.85, 0.50, 0.80, 0.34)

    ax.text(
        0.03,
        0.05,
        "Note: This diagram is a structural view. Autograd graphs depend on branch (train vs infer) and fixed shapes.",
        fontsize=9,
        color="#333333",
    )

    plt.tight_layout(pad=0.2)
    fig.savefig(out, bbox_inches="tight")
    print(str(out))


if __name__ == "__main__":
    main()
