"""Tests for the latent world model module (predictor + LeJEPA losses)."""

import math

import pytest
import torch

from ssm_latent import (
    LatentPredictor,
    MlpEncoder,
    MlpEncoderConfig,
    MultiScaleSsmConfig,
    check_planning_consistency,
    check_stationarity,
    compute_exploration_quality,
    compute_identifiability_r2,
    curvature_loss,
    curvature_loss_with_dt,
    lejepa_loss,
    linear_latent_plan,
    normalize_projections,
    plan_path_cost,
    procrustes_alignment,
    random_projections,
    sigreg_loss,
    stability_loss,
    stability_loss_running,
)


def tiny_config(n_layers=2):
    return MultiScaleSsmConfig(
        d_model=16, d_state=8, expand=2, n_heads=2, mimo_rank=1, n_layers=n_layers
    )


# ---------------------------------------------------------------- predictor --
def test_predictor_forward_shapes():
    torch.manual_seed(0)
    model = LatentPredictor(tiny_config(), input_dim=6, action_dim=3).eval()
    x = torch.randn(2, 7, 6)
    a = torch.randn(2, 7, 3)

    z, pred_z, rec_x, pred_x = model(x, a)
    assert z.shape == (2, 7, 16)
    assert pred_z.shape == (2, 7, 16)
    assert rec_x.shape == (2, 7, 6)
    assert pred_x.shape == (2, 7, 6)
    assert torch.isfinite(pred_z).all()


def test_predictor_without_actions():
    torch.manual_seed(0)
    model = LatentPredictor(tiny_config(), input_dim=4, action_dim=0).eval()
    x = torch.randn(1, 5, 4)
    z, pred_z, rec_x, pred_x = model(x)
    assert z.shape == pred_z.shape == (1, 5, 16)
    assert model.action_encoder is None


def test_predictor_single_scale_backbone():
    torch.manual_seed(0)
    model = LatentPredictor(tiny_config(1), input_dim=4, action_dim=2, multi_scale=False).eval()
    z, pred_z, rec_x, _ = model(torch.randn(2, 6, 4), torch.randn(2, 6, 2))
    assert pred_z.shape == (2, 6, 16)


@pytest.mark.parametrize("multi_scale", [False, True])
def test_predictor_step_matches_forward(multi_scale):
    """Streaming inference reproduces the batched forward pass."""
    torch.manual_seed(0)
    config = tiny_config(1) if not multi_scale else tiny_config(3)
    model = LatentPredictor(config, input_dim=4, action_dim=2, multi_scale=multi_scale)
    model.eval()
    x = torch.randn(2, 6, 4)
    a = torch.randn(2, 6, 2)

    _, pred_z, _, _ = model(x, a)

    state = model.zero_state(2)
    outputs = []
    with torch.no_grad():
        for t in range(x.shape[1]):
            z_prev = model.encode(x[:, t])
            z_next, state = model.step(z_prev, state, a[:, t])
            outputs.append(z_next)
    streamed = torch.stack(outputs, dim=1)

    assert torch.allclose(pred_z, streamed, atol=1e-5)
    assert all(t.requires_grad == state.h[0].requires_grad for t in [state.h[0]])


def test_predictor_step_is_inference_only():
    """``step`` is the streaming-inference path and runs under ``no_grad``."""
    torch.manual_seed(0)
    model = LatentPredictor(tiny_config(2), input_dim=4, action_dim=0)
    state = model.zero_state(2)
    z_next, state = model.step(model.encode(torch.randn(2, 4)), state)
    assert not z_next.requires_grad
    assert not state.h[0].requires_grad
    assert state.detach().h[0].shape == state.h[0].shape


def test_predictor_gradients_flow():
    torch.manual_seed(0)
    model = LatentPredictor(tiny_config(2), input_dim=4, action_dim=2)
    x, a = torch.randn(2, 6, 4), torch.randn(2, 6, 2)
    z, pred_z, rec_x, pred_x = model(x, a)
    model.loss(z, pred_z, x, rec_x, pred_x).backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
    # The projection matrix is a fixed buffer, never trained.
    assert dict(model.named_buffers())["stability_projections"].requires_grad is False


def test_lejepa_gradients_reach_the_ssm_backbone():
    torch.manual_seed(0)
    model = LatentPredictor(tiny_config(2), input_dim=4, action_dim=2)
    z, pred_z, _, _ = model(torch.randn(2, 6, 4), torch.randn(2, 6, 2))
    model.lejepa(z, pred_z).backward()
    assert model.ssm.layers[0].a_re.grad is not None
    assert model.encoder.weight.grad is not None
    assert torch.isfinite(model.ssm.layers[0].a_re.grad).all()


def test_predictor_training_reduces_loss():
    torch.manual_seed(0)
    model = LatentPredictor(tiny_config(1), input_dim=4, action_dim=2)
    # Smooth circular-ish trajectories so that prediction is learnable.
    t = torch.linspace(0, 2 * math.pi, 32)
    x = torch.stack([torch.sin(t), torch.cos(t), torch.sin(2 * t), torch.cos(2 * t)], dim=-1)
    x = x.unsqueeze(0).repeat(8, 1, 1)
    a = torch.zeros(8, 32, 2)

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(30):
        z, pred_z, rec_x, pred_x = model(x, a)
        loss = model.loss(z, pred_z, x, rec_x, pred_x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_mlp_encoder():
    torch.manual_seed(0)
    linear = MlpEncoder(6, 8, MlpEncoderConfig(n_hidden=0))
    assert linear(torch.randn(2, 5, 6)).shape == (2, 5, 8)

    mlp = MlpEncoder(6, 8, MlpEncoderConfig(n_hidden=3, hidden_dim=16))
    assert mlp(torch.randn(2, 5, 6)).shape == (2, 5, 8)

    with pytest.raises(ValueError):
        MlpEncoderConfig(n_hidden=2, hidden_dim=0).validate()


# ------------------------------------------------------------------- losses --
def test_stability_loss_penalises_collapse():
    torch.manual_seed(0)
    w = random_projections(16, 8)
    assert torch.allclose(w.norm(dim=0), torch.ones(8), atol=1e-5)

    collapsed = stability_loss(torch.zeros(1, 10, 16), w)
    ideal = stability_loss(torch.randn(1, 200, 16), w)
    assert float(collapsed) > float(ideal)


def test_sigreg_loss_penalises_collapse():
    torch.manual_seed(0)
    w = random_projections(16, 8)
    collapsed = sigreg_loss(torch.zeros(1, 10, 16), w)
    ideal = sigreg_loss(torch.randn(1, 512, 16), w)
    assert float(collapsed) > float(ideal)
    assert float(ideal) < 0.1


def test_stability_loss_running_updates_ema():
    torch.manual_seed(0)
    w = random_projections(16, 4)
    mean = torch.zeros(4)
    var = torch.ones(4)
    loss, new_mean, new_var = stability_loss_running(
        torch.randn(2, 8, 16) * 3 + 1, w, mean, var, momentum=0.1
    )
    assert torch.isfinite(loss)
    assert not torch.allclose(new_mean, mean)   # EMA moved
    assert new_var.shape == (4,)


def test_lejepa_loss_is_finite_and_non_negative():
    torch.manual_seed(0)
    w = random_projections(32, 8)
    z = torch.randn(2, 16, 32)
    pred_z = torch.randn(2, 16, 32)
    loss = lejepa_loss(z, pred_z, w, sigreg_weight=1.0)
    assert torch.isfinite(loss) and float(loss) >= 0.0


def test_curvature_loss_straight_vs_jagged():
    straight = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1).repeat(1, 1, 16)
    assert float(curvature_loss(straight)) < 1e-6

    torch.manual_seed(0)
    jagged = torch.randn(1, 10, 16)
    assert float(curvature_loss(jagged)) > float(curvature_loss(straight))

    # Too short to define a second difference.
    assert float(curvature_loss(torch.randn(1, 2, 16))) == 0.0


def test_curvature_loss_with_dt():
    z = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1).repeat(1, 1, 4)
    dt = torch.full((10,), 0.5)
    assert float(curvature_loss_with_dt(z, dt)) < 1e-5

    # Non-linear time-warped trajectory has non-zero acceleration.
    z_warp = z.pow(2)
    assert float(curvature_loss_with_dt(z_warp, dt)) > 0.0


# ----------------------------------------------------- identifiability etc. --
def test_identifiability_r2_is_one_for_rotation():
    torch.manual_seed(0)
    z_true = torch.randn(2, 32, 8)
    q, _ = torch.linalg.qr(torch.randn(8, 8))
    z_hat = z_true @ q

    r2_rotated = float(compute_identifiability_r2(z_hat, z_true))
    r2_random = float(compute_identifiability_r2(torch.randn(2, 32, 8), z_true))
    assert r2_rotated > 0.99
    assert r2_random < r2_rotated


def test_procrustes_alignment_recovers_rotation():
    torch.manual_seed(0)
    z_true = torch.randn(2, 32, 6)
    q, _ = torch.linalg.qr(torch.randn(6, 6))
    z_hat = z_true @ q

    aligned, q_est = procrustes_alignment(z_hat, z_true)
    assert torch.allclose(aligned, z_true, atol=1e-4)
    assert torch.allclose(q_est.t() @ q_est, torch.eye(6), atol=1e-5)


def test_plans_and_path_cost():
    start = torch.zeros(1, 4)
    end = torch.ones(1, 4)
    plan = linear_latent_plan(start, end, 5)
    assert plan.shape == (5, 1, 4)
    assert torch.allclose(plan[0], start[0])
    assert torch.allclose(plan[-1], end[0])
    assert float(plan_path_cost(plan)) > 0.0


def test_planning_consistency_detects_consistent_plans():
    torch.manual_seed(0)
    plans = torch.cumsum(torch.randn(4, 8, 6), dim=1)
    identical = check_planning_consistency(plans, plans)
    assert identical.is_consistent
    assert identical.cost_consistency_r2 > 0.99

    inconsistent = check_planning_consistency(plans, torch.randn(4, 8, 6))
    assert not inconsistent.is_consistent


def test_exploration_quality_reports_risk():
    torch.manual_seed(0)
    isotropic = compute_exploration_quality(torch.randn(4, 64, 8))
    assert isotropic.risk_level.startswith("LOW")
    assert isotropic.effective_rank > 6.0

    # One dominant direction, seven nearly-constant ones -> anisotropic + low rank.
    scales = torch.cat([torch.ones(1), torch.full((7,), 1e-4)])
    bad = compute_exploration_quality(torch.randn(4, 64, 8) * scales)
    assert bad.anisotropy > 3.0
    assert bad.coverage < 0.2
    assert bad.risk_level.startswith("HIGH")


def test_stationarity_detects_trend():
    torch.manual_seed(0)
    z = torch.randn(2, 32, 8)
    stationary = check_stationarity(z, z + 0.01 * torch.randn(2, 32, 8), n_layers=3)
    assert stationary.risk_level.startswith("LOW")

    # Growing prediction error -> non-stationary drift.
    drift = z + torch.linspace(0, 50, 32).reshape(1, 32, 1)
    trending = check_stationarity(z, drift, n_layers=3)
    assert trending.residual_trend > 0.1
    assert trending.risk_level.startswith("HIGH")
    assert trending.dominant_layer == 2  # slowest layer blamed for the drift


def test_normalize_projections_columns_are_unit_norm():
    torch.manual_seed(0)
    w = normalize_projections(torch.randn(10, 6))
    assert torch.allclose(w.norm(dim=0), torch.ones(6), atol=1e-5)


def test_only_torch_is_required():
    """The library must depend on PyTorch only (no CUDA-only kernels)."""
    import sys
    from pathlib import Path

    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        pytest.skip("tomllib is only available on Python 3.11+")

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as fh:
        deps = tomllib.load(fh)["project"]["dependencies"]
    assert deps == ["torch>=2.0"]

    for name in ("mamba_ssm", "causal_conv1d"):
        assert name not in sys.modules
