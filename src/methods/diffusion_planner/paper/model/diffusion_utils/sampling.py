from __future__ import annotations

from typing import Dict

import torch

from . import dpm_solver_pytorch as dpm


def dpm_sampler(
    model: torch.nn.Module,
    x_T: torch.Tensor,
    other_model_params: Dict | None = None,
    diffusion_steps: int = 10,
    noise_schedule_params: Dict | None = None,
    model_wrapper_params: Dict | None = None,
    dpm_solver_params: Dict | None = None,
    sample_params: Dict | None = None,
) -> torch.Tensor:
    """DPM-Solver++ sampler (vendor-parity wrapper)."""

    other_model_params = other_model_params or {}
    noise_schedule_params = noise_schedule_params or {}
    model_wrapper_params = model_wrapper_params or {}
    dpm_solver_params = dpm_solver_params or {}
    sample_params = sample_params or {}

    with torch.no_grad():
        noise_schedule = dpm.NoiseScheduleVP(schedule="linear", **noise_schedule_params)

        model_fn = dpm.model_wrapper(
            model,
            noise_schedule,
            model_type=model.model_type,
            model_kwargs=other_model_params,
            **model_wrapper_params,
        )

        dpm_solver = dpm.DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++", **dpm_solver_params)

        sample_dpm = dpm_solver.sample(
            x_T,
            steps=diffusion_steps,
            order=2,
            skip_type="logSNR",
            method="multistep",
            denoise_to_zero=True,
            **sample_params,
        )

    return sample_dpm
