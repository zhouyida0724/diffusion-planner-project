import torch


def _make_batch(B: int = 4, *, device: str = "cpu"):
    dev = torch.device(device)
    # ego_current_state: [x,y,cos,sin,vx,vy,ax,ay,steer,yaw_rate]
    ego = torch.zeros((B, 10), device=dev, dtype=torch.float32)
    ego[:, 2] = 1.0  # cos(0)
    ego[:, 3] = 0.0  # sin(0)
    ego[:, 4] = 3.0  # vx so augmentation isn't gated by min_vx

    ego_fut = torch.zeros((B, 80, 3), device=dev, dtype=torch.float32)
    # simple straight future (avoid degenerate spline signals)
    t = torch.arange(80, device=dev, dtype=torch.float32) * 0.1
    ego_fut[..., 0] = t[None, :] * 3.0
    ego_fut[..., 1] = 0.0
    ego_fut[..., 2] = 0.0

    nb_past = torch.randn((B, 32, 21, 11), device=dev, dtype=torch.float32)
    nb_fut = torch.randn((B, 32, 81, 3), device=dev, dtype=torch.float32)

    lanes = torch.randn((B, 70, 20, 12), device=dev, dtype=torch.float32)
    route_lanes = torch.randn((B, 25, 20, 12), device=dev, dtype=torch.float32)
    static_objects = torch.randn((B, 5, 10), device=dev, dtype=torch.float32)

    # Add some all-zero rows to exercise masking behavior.
    nb_past[:, 0, :, :] = 0.0
    nb_fut[:, 0, :, :] = 0.0
    lanes[:, 0, :, :] = 0.0
    route_lanes[:, 0, :, :] = 0.0
    static_objects[:, 0, :] = 0.0

    return {
        "ego_current_state": ego,
        "ego_agent_future": ego_fut,
        "neighbor_agents_past": nb_past,
        "neighbor_agents_future": nb_fut,
        "static_objects": static_objects,
        "lanes": lanes,
        "lanes_speed_limit": torch.zeros((B, 70, 1), device=dev, dtype=torch.float32),
        "lanes_has_speed_limit": torch.zeros((B, 70, 1), device=dev, dtype=torch.float32),
        "route_lanes": route_lanes,
        "route_lanes_speed_limit": torch.zeros((B, 25, 1), device=dev, dtype=torch.float32),
        "route_lanes_has_speed_limit": torch.zeros((B, 25, 1), device=dev, dtype=torch.float32),
        "meta": {},
    }


def test_state_perturbation_prob0_is_identity():
    from src.methods.diffusion_planner.utils.state_perturbation import StatePerturbation, StatePerturbationConfig

    batch = _make_batch(B=2)
    batch_in = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}

    aug = StatePerturbation(StatePerturbationConfig(enabled=True, prob=0.0), device=torch.device("cpu"))
    out = aug(batch)

    for k, v in batch_in.items():
        if torch.is_tensor(v):
            assert torch.equal(v, out[k]), f"tensor changed for key={k}"


def test_state_perturbation_shapes_and_determinism():
    from src.methods.diffusion_planner.utils.state_perturbation import StatePerturbation, StatePerturbationConfig

    cfg = StatePerturbationConfig(enabled=True, prob=1.0, min_vx_mps=0.0)
    aug = StatePerturbation(cfg, device=torch.device("cpu"))

    torch.manual_seed(123)
    b1 = _make_batch(B=4)
    o1 = aug({k: (v.clone() if torch.is_tensor(v) else v) for k, v in b1.items()})

    torch.manual_seed(123)
    b2 = _make_batch(B=4)
    o2 = aug({k: (v.clone() if torch.is_tensor(v) else v) for k, v in b2.items()})

    # Shape invariants
    for k in ("ego_current_state", "ego_agent_future", "neighbor_agents_past", "neighbor_agents_future", "lanes", "route_lanes", "static_objects"):
        assert tuple(o1[k].shape) == tuple(b1[k].shape)

    # Deterministic given fixed seed + same inputs.
    for k, v in o1.items():
        if torch.is_tensor(v):
            assert torch.equal(v, o2[k]), f"non-deterministic for key={k}"


def test_state_perturbation_does_not_mutate_input_batch_inplace():
    """Augmentation should not mutate the caller-provided batch tensors in-place."""

    from src.methods.diffusion_planner.utils.state_perturbation import StatePerturbation, StatePerturbationConfig

    cfg = StatePerturbationConfig(enabled=True, prob=1.0, min_vx_mps=0.0)
    aug = StatePerturbation(cfg, device=torch.device("cpu"))

    torch.manual_seed(0)
    batch = _make_batch(B=2)
    # Keep copies of tensors we expect the augmentation to transform.
    saved = {k: v.clone() for k, v in batch.items() if torch.is_tensor(v)}

    _ = aug(batch)

    for k, v in saved.items():
        assert torch.equal(batch[k], v), f"input batch mutated in-place for key={k}"
