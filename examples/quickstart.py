"""Minimal end-to-end example of the ssm-latent library.

Run with::

    python examples/quickstart.py

Trains a latent world model to predict the next latent state of a simple
phase-locked physical system (a point moving along a circle), then inspects the
learned latent space with the LeJEPA condition monitors and plans in it.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssm_latent import (  # noqa: E402
    LatentPredictor,
    MultiScaleSsmConfig,
    check_planning_consistency,
    health_check,
    linear_latent_plan,
)


def circle_world(batch: int = 16, seq_len: int = 64, seed: int = 0) -> torch.Tensor:
    """Observations of a point moving on a circle at a per-sample angular velocity."""
    g = torch.Generator().manual_seed(seed)
    omega = 0.05 + 0.1 * torch.rand(batch, 1, generator=g)
    t = torch.arange(seq_len).float().reshape(1, -1)
    phase = omega * t
    x = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)
    return x + 0.01 * torch.randn(x.shape, generator=g)


def main() -> None:
    torch.manual_seed(0)
    obs = circle_world()

    model = LatentPredictor(
        MultiScaleSsmConfig(
            d_model=32, d_state=8, expand=2, n_heads=2, mimo_rank=1, n_layers=3
        ),
        input_dim=2,
        action_dim=0,       # unsupervised: no actions required
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    for step in range(200):
        z, pred_z, rec_x, pred_x = model(obs)
        loss = model.loss(z, pred_z, obs, rec_x, pred_x)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        z, pred_z, _, _ = model(obs)

    # --- LeJEPA condition monitors ----------------------------------------- #
    exploration, stationarity = health_check(z, pred_z, n_layers=3)
    print("\nexploration:", {k: round(v, 3) for k, v in exploration.as_dict().items()})
    print("risk       :", exploration.risk_level)
    print("stationarity:", round(stationarity.residual_trend, 4), stationarity.risk_level)

    # --- Planning in latent space (Theorem 4) ------------------------------ #
    with torch.no_grad():
        z_start, z_end = model.encode(obs[:, 0]), model.encode(obs[:, -1])
        plan_z = linear_latent_plan(z_start, z_end, n_steps=10)            # [n, B, d]
        plan_x = linear_latent_plan(obs[:, 0], obs[:, -1], n_steps=10)     # [n, B, 2]
        z_from_x = model.encode(plan_x.transpose(0, 1)).transpose(0, 1)    # [n, B, d]

    consistency = check_planning_consistency(
        plan_z.transpose(0, 1), z_from_x.transpose(0, 1)                   # [B, n, d]
    )
    print("\nplan consistency R2 :", round(consistency.cost_consistency_r2, 4))
    print("directional error   :", round(consistency.directional_error, 4), "rad")
    print("consistent          :", consistency.is_consistent)


if __name__ == "__main__":  # pragma: no cover
    main()
