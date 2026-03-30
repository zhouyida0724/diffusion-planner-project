"""Minimal PyTorch trainer used for diffusion planner smoke tests.

Design goals:
  - deterministic-ish (seed)
  - explicit shape + NaN/Inf assertions
  - lightweight checkpointing
  - writes to outputs/training/<exp_name>/

This is intentionally not PyTorch Lightning to keep the control surface small.

Perf instrumentation (added 2026-03):
  - step time (EMA + window p50/p90)
  - samples/s and frames/s (=batch_size/step_time)
  - GPU peak memory (torch.cuda.max_memory_allocated)
  - best-effort GPU utilization + power via nvidia-smi (throttled)
  - optional torch.profiler micro-run (first N steps) for approximate FLOPs

Metrics are written to outputs/training/<exp_name>/perf.json.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.methods.diffusion_planner.diffusion.losses import masked_mse
from src.methods.diffusion_planner.diffusion.scheduler import NoiseSchedule, q_sample

import torch
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    exp_name: str
    out_root: str = "outputs/training"
    steps: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 2
    log_every: int = 10
    ckpt_every: int = 200
    seed: int = 0
    device: str = "cuda"  # 'cuda' or 'cpu'

    # amp
    amp: str = "off"  # off|bf16|fp16

    # perf
    perf_window: int = 100
    perf_smi_every: int = 100
    profiler_steps: int = 0  # 0 disables torch.profiler micro-run

    # step-level breakdown profiling (cheap wall-clock timers; best-effort)
    profile_steps: int = 0  # if >0, collect per-step breakdown for the first N steps
    profile_every: int = 0  # if >0, print breakdown every N steps (only when profiling is enabled)


def seed_everything(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _assert_finite(t: torch.Tensor, name: str) -> None:
    if not torch.isfinite(t).all():
        bad = t[~torch.isfinite(t)]
        raise FloatingPointError(f"{name} has NaN/Inf; example bad values: {bad[:5]}")


def _percentile(sorted_vals: list[float], q: float) -> float:
    """q in [0,1]."""

    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    q = max(0.0, min(1.0, float(q)))
    idx = int(round(q * (len(sorted_vals) - 1)))
    return float(sorted_vals[idx])


def _safe_nvidia_smi() -> Dict[str, Any]:
    """Best-effort GPU stats via nvidia-smi.

    Returns empty dict on failure.
    """

    import subprocess

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,power.draw,memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        ).strip()
        if not out:
            return {}
        # If multiple GPUs, take the first line (we only train on one device).
        line = out.splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            return {"raw": line}
        name, util, power, mem = parts
        return {
            "name": name,
            "utilization_gpu_pct": float(util),
            "power_w": float(power),
            "memory_used_mib": float(mem),
        }
    except Exception:
        return {}


class _PerfTracker:
    def __init__(
        self,
        *,
        exp_dir: Path,
        cfg: TrainConfig,
        device: torch.device,
    ):
        self.exp_dir = exp_dir
        self.cfg = cfg
        self.device = device
        self.records: list[dict[str, Any]] = []

        self._ema_step_s: Optional[float] = None
        self._window: list[float] = []
        self._t_train_start = time.time()
        self._last_smi: Dict[str, Any] = {}

        # profiler
        self._profiler_total_flops: Optional[float] = None
        self._profiler_total_time_s: Optional[float] = None

    def on_step_end(self, *, step: int, step_s: float, loss: float) -> None:
        # window
        self._window.append(float(step_s))
        if len(self._window) > int(self.cfg.perf_window):
            self._window.pop(0)

        # ema
        alpha = 0.1
        if self._ema_step_s is None:
            self._ema_step_s = float(step_s)
        else:
            self._ema_step_s = alpha * float(step_s) + (1.0 - alpha) * float(self._ema_step_s)

        # GPU memory
        mem_max_bytes = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            mem_max_bytes = int(torch.cuda.max_memory_allocated(device=self.device))

        # best-effort nvidia-smi (throttled)
        if self.device.type == "cuda" and (step % int(self.cfg.perf_smi_every) == 0):
            self._last_smi = _safe_nvidia_smi()

        # compute throughput
        bs = int(self.cfg.batch_size)
        samples_s = bs / max(step_s, 1e-12)
        frames_s = bs / max(step_s, 1e-12)

        rec = {
            "step": int(step),
            "loss": float(loss),
            "step_s": float(step_s),
            "ema_step_s": float(self._ema_step_s) if self._ema_step_s is not None else None,
            "window_p50_step_s": _percentile(sorted(self._window), 0.50),
            "window_p90_step_s": _percentile(sorted(self._window), 0.90),
            "samples_s": float(samples_s),
            "frames_s": float(frames_s),
            "cuda_max_memory_allocated_bytes": mem_max_bytes,
            "nvidia_smi": dict(self._last_smi) if self._last_smi else {},
            "wall_s_since_start": float(time.time() - self._t_train_start),
        }
        self.records.append(rec)

    def set_profiler(self, *, total_flops: float | None, total_time_s: float | None) -> None:
        self._profiler_total_flops = None if total_flops is None else float(total_flops)
        self._profiler_total_time_s = None if total_time_s is None else float(total_time_s)

    def summary(self) -> Dict[str, Any]:
        last = self.records[-1] if self.records else {}
        ema = last.get("ema_step_s")
        if ema is None:
            ema = float("nan")

        # Use observed wall-clock average (includes data loading) for projections.
        avg_step_s = float("nan")
        if last:
            denom = float(int(last.get("step", 0)) + 1)
            if denom > 0:
                avg_step_s = float(last.get("wall_s_since_start", float("nan"))) / denom

        # projections
        proj_steps = {
            "n_steps": int(self.cfg.steps),
            "time_s": float(avg_step_s) * int(self.cfg.steps),
            "avg_step_s": float(avg_step_s),
            "ema_step_s": float(ema),
        }

        # Boston50w kept samples: ~404k
        kept = 404_000
        epoch_steps = int((kept + int(self.cfg.batch_size) - 1) // int(self.cfg.batch_size))
        proj_epoch = {
            "kept_samples": kept,
            "epoch_steps": epoch_steps,
            "time_s": float(avg_step_s) * epoch_steps,
            "avg_step_s": float(avg_step_s),
            "ema_step_s": float(ema),
        }

        # achieved tflops from profiler (if available)
        achieved_tflops = None
        if self._profiler_total_flops is not None and self._profiler_total_time_s and self._profiler_total_time_s > 0:
            achieved_tflops = (self._profiler_total_flops / self._profiler_total_time_s) / 1e12

        peak_tflops = None
        mfu = None
        env_peak = os.environ.get("GPU_PEAK_TFLOPS")
        if env_peak:
            try:
                peak_tflops = float(env_peak)
                if achieved_tflops is not None and peak_tflops > 0:
                    mfu = float(achieved_tflops / peak_tflops)
            except Exception:
                peak_tflops = None

        return {
            "last": last,
            "projections": {"steps": proj_steps, "epoch": proj_epoch},
            "profiler": {
                "micro_total_flops": self._profiler_total_flops,
                "micro_total_time_s": self._profiler_total_time_s,
                "achieved_tflops": achieved_tflops,
                "peak_tflops_env": peak_tflops,
                "mfu_estimate": mfu,
            },
        }

    def write_perf_json(self) -> None:
        payload = {
            "cfg": asdict(self.cfg),
            "records": self.records,
            "summary": self.summary(),
        }
        (self.exp_dir / "perf.json").write_text(json.dumps(payload, indent=2))


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device=device)


def train_loop(
    *,
    cfg: TrainConfig,
    model: torch.nn.Module,
    train_loader: DataLoader,
    max_grad_norm: Optional[float] = 1.0,
) -> Path:
    """Run training and return the experiment directory."""

    seed_everything(cfg.seed)

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(device)

    exp_dir = Path(cfg.out_root) / cfg.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    (exp_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    perf = _PerfTracker(exp_dir=exp_dir, cfg=cfg, device=device)

    model.train()
    t0 = time.time()
    step = 0

    loader_it = iter(train_loader)

    while step < cfg.steps:
        step_t0 = time.time()
        try:
            batch = next(loader_it)
        except StopIteration:
            loader_it = iter(train_loader)
            batch = next(loader_it)

        x = batch["x"].to(device=device, dtype=torch.float32)
        y = batch["y"].to(device=device, dtype=torch.float32)

        assert x.ndim == 2, f"x ndim {x.ndim}"
        assert y.ndim == 3 and y.shape[-1] == 3, f"y shape {tuple(y.shape)}"

        _assert_finite(x, "x")
        _assert_finite(y, "y")

        y_hat = model(x)
        assert y_hat.shape == y.shape, f"y_hat {tuple(y_hat.shape)} vs y {tuple(y.shape)}"
        _assert_finite(y_hat, "y_hat")

        loss = torch.mean((y_hat - y) ** 2)
        _assert_finite(loss, "loss")

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        opt.step()

        _maybe_sync(device)
        step_s = time.time() - step_t0

        # record perf every step (cheap) but only print/log at log_every
        perf.on_step_end(step=step, step_s=step_s, loss=float(loss.item()))
        if (step % cfg.log_every) == 0 or step == cfg.steps - 1:
            dt = time.time() - t0
            last = perf.records[-1]
            msg = (
                f"step {step:05d} | loss {loss.item():.6f} | "
                f"step_s {last['step_s']:.4f} (ema {last['ema_step_s']:.4f} "
                f"p50 {last['window_p50_step_s']:.4f} p90 {last['window_p90_step_s']:.4f}) | "
                f"samples/s {last['samples_s']:.1f} | {dt:.1f}s"
            )
            print(msg, flush=True)
            with (exp_dir / "train.log").open("a") as f:
                f.write(msg + "\n")
            # keep perf.json updated for long runs
            perf.write_perf_json()

        if (step % cfg.ckpt_every) == 0 or step == cfg.steps - 1:
            ckpt_path = exp_dir / f"checkpoint_step_{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "cfg": asdict(cfg),
                },
                ckpt_path,
            )

        step += 1

    # Convenience symlink/copy: latest
    latest = exp_dir / "checkpoint_latest.pt"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(os.path.basename(ckpt_path))
    except Exception:
        # fallback: copy
        import shutil

        shutil.copy2(ckpt_path, latest)

    perf.write_perf_json()
    _print_perf_summary(perf)

    return exp_dir


def _run_profiler_micro(
    *,
    fn_step: callable,
    steps: int,
    device: torch.device,
) -> tuple[Optional[float], Optional[float]]:
    """Run a short torch.profiler micro-benchmark.

    Returns (total_flops, total_time_s) if available.
    """

    if steps <= 0:
        return None, None

    try:
        import torch.profiler

        # Warmup a couple steps outside profiler for more stable numbers.
        for _ in range(min(2, steps)):
            fn_step()

        _maybe_sync(device)
        t0 = time.time()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                *( [torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else [] ),
            ],
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
            with_flops=True,
        ) as prof:
            for _ in range(steps):
                fn_step()
        _maybe_sync(device)
        dt = time.time() - t0

        total_flops = None
        try:
            # PyTorch exposes total_average().total_flops in newer versions.
            total_flops = float(getattr(prof.key_averages().total_average(), "total_flops", 0.0))
        except Exception:
            total_flops = None

        return total_flops, float(dt)
    except Exception:
        return None, None


def _print_perf_summary(perf: _PerfTracker) -> None:
    s = perf.summary()
    last = s.get("last", {})
    proj = s.get("projections", {})
    steps = proj.get("steps", {})
    epoch = proj.get("epoch", {})
    prof = s.get("profiler", {})

    def _fmt_hhmmss(sec: float) -> str:
        if sec != sec:  # nan
            return "nan"
        sec = max(0.0, float(sec))
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s2 = int(sec % 60)
        return f"{h:02d}:{m:02d}:{s2:02d}"

    print("=" * 60)
    print("Perf summary")
    if last:
        print(
            "last step: "
            f"step_s={last.get('step_s'):.4f} "
            f"ema={last.get('ema_step_s'):.4f} "
            f"p50={last.get('window_p50_step_s'):.4f} "
            f"p90={last.get('window_p90_step_s'):.4f} "
            f"samples/s={last.get('samples_s'):.1f}"
        )
        mem_b = last.get("cuda_max_memory_allocated_bytes")
        if mem_b is not None:
            print(f"cuda max_memory_allocated: {mem_b / (1024**3):.2f} GiB")
        smi = last.get("nvidia_smi") or {}
        if smi:
            print(
                f"nvidia-smi: {smi.get('name')} | util={smi.get('utilization_gpu_pct')}% | "
                f"power={smi.get('power_w')}W | mem_used={smi.get('memory_used_mib')}MiB"
            )

    if steps:
        print(
            f"avg_step_s (wall): {steps.get('avg_step_s', float('nan')):.4f} | "
            f"ema_step_s (compute-only-ish): {steps.get('ema_step_s', float('nan')):.4f}"
        )
        print(f"projected time for {steps.get('n_steps')} steps: {_fmt_hhmmss(steps.get('time_s', float('nan')))}")
    if epoch:
        print(
            f"projected time for 1 epoch (~{epoch.get('kept_samples')} kept): "
            f"{epoch.get('epoch_steps')} steps => {_fmt_hhmmss(epoch.get('time_s', float('nan')))}"
        )

    if prof.get("achieved_tflops") is not None:
        print(f"profiler achieved_tflops (micro): {prof.get('achieved_tflops'):.2f} TFLOP/s")
    if prof.get("mfu_estimate") is not None:
        print(f"MFU estimate (via $GPU_PEAK_TFLOPS): {prof.get('mfu_estimate'):.3f}")
    print("=" * 60, flush=True)


def train_loop_diffusion_eps(
    *,
    cfg: TrainConfig,
    model: torch.nn.Module,
    train_loader: DataLoader,
    schedule_cfg: NoiseSchedule = NoiseSchedule(),
    max_grad_norm: Optional[float] = 1.0,
) -> Path:
    """Train an eps-pred diffusion model.

    Expected model signature:
        eps_hat = model(x, y_t, t)

    Where:
        x: [B, x_dim]
        y_t: [B, T, 3]
        t: [B]

    Loss:
        MSE(eps_hat, eps) (optionally masked)

    This is a minimal training loop meant for smoke tests.
    """

    seed_everything(cfg.seed)

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(device)

    exp_dir = Path(cfg.out_root) / cfg.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    (exp_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    schedule = schedule_cfg.make(device=device)

    perf = _PerfTracker(exp_dir=exp_dir, cfg=cfg, device=device)

    model.train()
    t0 = time.time()
    step = 0
    loader_it = iter(train_loader)

    # Optional profiler micro-run: we define a single training step fn that
    # advances data iterator, runs forward/back/opt.
    def _one_step_for_profiler() -> None:
        nonlocal loader_it
        try:
            batch_p = next(loader_it)
        except StopIteration:
            loader_it = iter(train_loader)
            batch_p = next(loader_it)

        x_p = batch_p["x"].to(device=device, dtype=torch.float32)
        y0_p = batch_p["y"].to(device=device, dtype=torch.float32)
        mask_p = batch_p.get("mask")
        if mask_p is not None:
            mask_p = mask_p.to(device=device)

        Bp = x_p.shape[0]
        tp = torch.randint(low=0, high=schedule.num_steps, size=(Bp,), device=device, dtype=torch.int64)
        eps_p = torch.randn_like(y0_p)
        y_t_p = q_sample(schedule=schedule, x0=y0_p, t=tp, noise=eps_p)

        eps_hat_p = model(x_p, y_t_p, tp)
        loss_p = masked_mse(eps_hat_p, eps_p, mask=mask_p)

        opt.zero_grad(set_to_none=True)
        loss_p.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()

    if int(cfg.profiler_steps) > 0:
        flops, prof_dt = _run_profiler_micro(fn_step=_one_step_for_profiler, steps=int(cfg.profiler_steps), device=device)
        perf.set_profiler(total_flops=flops, total_time_s=prof_dt)
        # After profiler, reset CUDA peak memory stats so the run is representative.
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=device)

    # Main training loop
    while step < cfg.steps:
        step_t0 = time.time()
        try:
            batch = next(loader_it)
        except StopIteration:
            loader_it = iter(train_loader)
            batch = next(loader_it)

        x = batch["x"].to(device=device, dtype=torch.float32)
        y0 = batch["y"].to(device=device, dtype=torch.float32)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device=device)

        assert x.ndim == 2
        assert y0.ndim == 3 and y0.shape[-1] == 3

        _assert_finite(x, "x")
        _assert_finite(y0, "y0")

        B = x.shape[0]
        t = torch.randint(low=0, high=schedule.num_steps, size=(B,), device=device, dtype=torch.int64)
        eps = torch.randn_like(y0)
        y_t = q_sample(schedule=schedule, x0=y0, t=t, noise=eps)

        eps_hat = model(x, y_t, t)
        assert eps_hat.shape == eps.shape
        _assert_finite(eps_hat, "eps_hat")

        loss = masked_mse(eps_hat, eps, mask=mask)
        _assert_finite(loss, "loss")

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        opt.step()

        _maybe_sync(device)
        step_s = time.time() - step_t0

        perf.on_step_end(step=step, step_s=step_s, loss=float(loss.item()))

        if (step % cfg.log_every) == 0 or step == cfg.steps - 1:
            dt = time.time() - t0
            last = perf.records[-1]
            msg = (
                f"step {step:05d} | loss {loss.item():.6f} | "
                f"step_s {last['step_s']:.4f} (ema {last['ema_step_s']:.4f} "
                f"p50 {last['window_p50_step_s']:.4f} p90 {last['window_p90_step_s']:.4f}) | "
                f"samples/s {last['samples_s']:.1f} | {dt:.1f}s"
            )
            print(msg, flush=True)
            with (exp_dir / "train.log").open("a") as f:
                f.write(msg + "\n")
            perf.write_perf_json()

        if (step % cfg.ckpt_every) == 0 or step == cfg.steps - 1:
            ckpt_path = exp_dir / f"checkpoint_step_{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "cfg": asdict(cfg),
                },
                ckpt_path,
            )

        step += 1

    latest = exp_dir / "checkpoint_latest.pt"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(os.path.basename(ckpt_path))
    except Exception:
        import shutil

        shutil.copy2(ckpt_path, latest)

    perf.write_perf_json()
    _print_perf_summary(perf)

    return exp_dir
