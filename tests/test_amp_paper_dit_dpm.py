import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from src.methods.diffusion_planner.train.trainer import TrainConfig
from src.methods.diffusion_planner.paper.model.diffusion_planner import PaperDiffusionPlanner, PaperModelConfig
from src.methods.diffusion_planner.paper.train.paper_trainer import train_loop_paper_dit_xstart


class _SyntheticPaperBatchDataset(Dataset):
    def __init__(
        self,
        *,
        n: int,
        cfg: PaperModelConfig,
        batch_size: int,
        seed: int = 0,
        include_ego_in_neighbors: bool = True,
    ):
        super().__init__()
        self.n = int(n)
        self.cfg = cfg
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.include_ego_in_neighbors = bool(include_ego_in_neighbors)

        # Shapes follow paper encoder/decoder expectations.
        self.P = int(cfg.predicted_neighbor_num + (1 if include_ego_in_neighbors else 0))

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        # Use per-index RNG for reproducibility.
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + int(idx))

        B = 1  # dataset returns single sample; DataLoader batches.
        Tf = int(self.cfg.future_len)
        Tl = int(self.cfg.time_len)

        # ego
        ego_current_state = torch.randn((B, 10), generator=g)
        ego_agent_future = torch.randn((B, Tf, 3), generator=g)

        # neighbors
        # neighbor_agents_past: [B,P,Tl,11] where last 3 dims are type one-hot-ish.
        neighbor_agents_past = torch.randn((B, self.P, Tl, 11), generator=g)
        # Make types roughly one-hot to avoid extreme values.
        t_raw = torch.randint(0, 3, (B, self.P), generator=g)
        neighbor_agents_past[..., 8:] = 0
        neighbor_agents_past[..., 8:].scatter_(-1, t_raw[..., None, None].expand(B, self.P, Tl, 1), 1.0)

        # neighbor_agents_future: [B,P,Tf,3] (x,y,heading)
        neighbor_agents_future = torch.randn((B, self.P, Tf, 3), generator=g)

        # static objects: [B,S,10]
        static_objects = torch.randn((B, int(self.cfg.static_objects_num), int(self.cfg.static_objects_state_dim)), generator=g)

        # lanes: [B,L,lane_len,12] (8 state + 4 traffic)
        lanes = torch.randn((B, int(self.cfg.lane_num), int(self.cfg.lane_len), 12), generator=g)

        lanes_speed_limit = torch.randn((B, int(self.cfg.lane_num), 1), generator=g)
        lanes_has_speed_limit = (torch.rand((B, int(self.cfg.lane_num), 1), generator=g) > 0.5).float()

        # route_lanes: [B,route_num,lane_len,4]
        route_lanes = torch.randn((B, int(self.cfg.route_num), int(self.cfg.lane_len), 4), generator=g)

        return {
            "ego_current_state": ego_current_state.squeeze(0),
            "ego_agent_future": ego_agent_future.squeeze(0),
            "neighbor_agents_past": neighbor_agents_past.squeeze(0),
            "neighbor_agents_future": neighbor_agents_future.squeeze(0),
            "static_objects": static_objects.squeeze(0),
            "lanes": lanes.squeeze(0),
            "lanes_speed_limit": lanes_speed_limit.squeeze(0),
            "lanes_has_speed_limit": lanes_has_speed_limit.squeeze(0),
            "route_lanes": route_lanes.squeeze(0),
        }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP")
@pytest.mark.parametrize("amp", ["bf16", "fp16"])
def test_paper_dit_dpm_amp_smoke(tmp_path, amp):
    # Keep the model tiny to make this fast.
    paper_cfg = PaperModelConfig(
        hidden_dim=32,
        num_heads=4,
        encoder_depth=1,
        decoder_depth=1,
        encoder_drop_path_rate=0.0,
        decoder_drop_path_rate=0.0,
        agent_num=5,
        static_objects_num=2,
        lane_num=6,
        route_num=3,
        time_len=4,
        future_len=6,
        lane_len=4,
        predicted_neighbor_num=4,
        static_objects_state_dim=10,
        diffusion_model_type="x_start",
        device="cuda",
    )

    model = PaperDiffusionPlanner(paper_cfg)

    ds = _SyntheticPaperBatchDataset(n=16, cfg=paper_cfg, batch_size=2, seed=123)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    cfg = TrainConfig(
        exp_name=f"pytest_paper_amp_{amp}",
        out_root=str(tmp_path),
        steps=3,
        batch_size=2,
        lr=1e-4,
        weight_decay=0.0,
        num_workers=0,
        log_every=1,
        ckpt_every=1000,
        seed=0,
        device="cuda",
        amp=str(amp),
        # Keep profiling off in tests.
        profile_steps=0,
        profile_every=0,
    )

    # Should run without dtype-mismatch crashes.
    train_loop_paper_dit_xstart(cfg=cfg, model=model, train_loader=loader, max_grad_norm=1.0)
