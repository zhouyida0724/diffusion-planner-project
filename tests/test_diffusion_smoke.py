import unittest

import torch

from src.methods.diffusion_planner.diffusion.scheduler import NoiseSchedule, q_sample
from src.methods.diffusion_planner.models.eps_mlp import EpsMLP


class TestDiffusionSmoke(unittest.TestCase):
    def test_q_sample_shapes(self):
        schedule = NoiseSchedule(num_steps=10).make(device=torch.device("cpu"))
        x0 = torch.zeros((4, 80, 3), dtype=torch.float32)
        t = torch.tensor([0, 1, 2, 9], dtype=torch.int64)
        eps = torch.randn_like(x0)
        xt = q_sample(schedule=schedule, x0=x0, t=t, noise=eps)
        self.assertEqual(tuple(xt.shape), tuple(x0.shape))
        self.assertTrue(torch.isfinite(xt).all().item())

    def test_eps_mlp_forward(self):
        B, x_dim, T = 2, 73, 80
        model = EpsMLP(x_dim=x_dim, T=T, hidden=64, t_embed_dim=32)
        x = torch.randn((B, x_dim))
        y_t = torch.randn((B, T, 3))
        t = torch.randint(0, 1000, (B,), dtype=torch.int64)
        eps_hat = model(x, y_t, t)
        self.assertEqual(tuple(eps_hat.shape), (B, T, 3))
        self.assertTrue(torch.isfinite(eps_hat).all().item())


if __name__ == "__main__":
    unittest.main()
