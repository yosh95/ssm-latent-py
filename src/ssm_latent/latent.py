"""Latent world model: JEPA-style prediction in latent space with an SSM backbone.

Ported from the Rust reference implementation (``ssm-latent-rs``, ``src/latent.rs``).

The central object is :class:`LatentPredictor`:

1. **Encoder** -- observation -> latent ``z``
2. **Action encoder** -- action -> latent (optional)
3. **Fusion** -- ``[z, a]`` -> dynamics input ``u``
4. **SSM** -- :class:`~ssm_latent.ssm.SsmBlock` predicting the next latent state
5. **Decoder** -- latent -> observation (auxiliary reconstruction signal only)

Following JEPA, prediction happens entirely in latent space; the decoder exists
only to keep the representation grounded.  Collapse is prevented without
stop-gradient, teacher-student or EMA schedules by the **LeJEPA** objective,
i.e. the SIGReg distribution-matching loss (:func:`sigreg_loss`).

Losses and diagnostics
----------------------
* :func:`latent_loss` -- prediction + reconstruction + stability + curvature
* :func:`lejepa_loss` -- next-step latent MSE + ``gamma * SIGReg``
* :func:`sigreg_loss` / :func:`stability_loss` -- collapse prevention
* :func:`curvature_loss` / :func:`curvature_loss_with_dt` -- temporal straightening
* :func:`compute_identifiability_r2`, :func:`procrustes_alignment` -- linear
  identifiability metrics
* :func:`compute_exploration_quality`, :func:`check_stationarity`,
  :func:`check_planning_consistency` -- condition monitors for the LeJEPA
  identifiability guarantees
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssm import (
    MultiScaleSsmBlock,
    MultiScaleSsmConfig,
    MultiScaleState,
    SsmBlock,
    SsmConfig,
)

# The recurrent state is shared by the single- and multi-scale backbones; the
# multi-scale container holds one entry per layer.
LatentState = MultiScaleState

__all__ = [
    "MlpEncoderConfig",
    "MlpEncoder",
    "LatentState",
    "LatentLossArgs",
    "LatentPredictor",
    "MultiScaleLatentPredictor",
    "normalize_projections",
    "latent_loss",
    "curvature_loss",
    "curvature_loss_with_dt",
    "stability_loss",
    "stability_loss_running",
    "sigreg_loss",
    "lejepa_loss",
    "compute_identifiability_r2",
    "procrustes_alignment",
    "linear_latent_plan",
    "plan_path_cost",
    "gennorm_identifiability_score",
    "compute_exploration_quality",
    "check_stationarity",
    "check_planning_consistency",
    "health_check",
]


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def normalize_projections(projections: torch.Tensor) -> torch.Tensor:
    """Normalize the columns of ``[d_model, n_projections]`` to unit L2 length."""
    return projections / (projections.pow(2).sum(dim=0, keepdim=True).sqrt() + 1e-6)


def random_projections(
    d_model: int, n_projections: int = 16, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Fixed (non-trainable) normalized Gaussian projection matrix."""
    w = torch.randn(d_model, n_projections, generator=generator)
    return normalize_projections(w)


def _as_batch_seq(z: torch.Tensor) -> torch.Tensor:
    """Flatten ``[..., d_model]`` to ``[N, d_model]``."""
    return z.reshape(-1, z.shape[-1])


# --------------------------------------------------------------------------- #
# MLP encoder (Theorem 1: a nonlinear encoder is still linearly identifiable)
# --------------------------------------------------------------------------- #
@dataclass
class MlpEncoderConfig:
    """Configuration of :class:`MlpEncoder` (``n_hidden = 0`` -> single Linear)."""

    n_hidden: int = 0
    hidden_dim: int = 256
    dropout: float = 0.0

    def validate(self) -> None:
        if self.n_hidden < 0:
            raise ValueError("n_hidden must be >= 0")
        if self.n_hidden > 0 and self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0 when n_hidden > 0")


class MlpEncoder(nn.Module):
    """``input_dim -> [hidden -> ReLU -> Dropout] * n_hidden -> d_model``."""

    def __init__(self, input_dim: int, d_model: int, config: Optional[MlpEncoderConfig] = None):
        super().__init__()
        config = config or MlpEncoderConfig()
        config.validate()
        self.config = config
        self.n_hidden = config.n_hidden

        if config.n_hidden == 0:
            self.input_proj = nn.Linear(input_dim, d_model)
            self.hidden = nn.ModuleList()
            self.output_proj = None
        else:
            self.input_proj = nn.Linear(input_dim, config.hidden_dim)
            self.hidden = nn.ModuleList(
                nn.Linear(config.hidden_dim, config.hidden_dim)
                for _ in range(config.n_hidden - 1)
            )
            self.output_proj = nn.Linear(config.hidden_dim, d_model)
            self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[batch, seq_len, input_dim]`` -> ``[batch, seq_len, d_model]``."""
        if self.n_hidden == 0:
            return self.input_proj(x)

        h = self.input_proj(x)
        for lin in self.hidden:
            h = lin(F.relu(h))
            if self.dropout is not None:
                h = self.dropout(h)
        return self.output_proj(F.relu(h))


# --------------------------------------------------------------------------- #
# Recurrent state
# --------------------------------------------------------------------------- #
@dataclass
class LatentLossArgs:
    """Inputs of :func:`latent_loss` (all tensors ``[batch, seq_len, dim]``)."""

    z: torch.Tensor                # current latents
    pred_z: torch.Tensor           # predicted next latents
    original_x: torch.Tensor       # observations
    stability_weight: float = 1.0
    curvature_weight: float = 0.1
    recon_weight: float = 1.0
    reconstructed_x: Optional[torch.Tensor] = None
    predicted_x: Optional[torch.Tensor] = None


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def stability_loss(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """VICReg-style moment matching on random projections (mean 0, variance 1)."""
    projections = _as_batch_seq(z) @ w                       # [N, n_projections]
    mean = projections.mean(dim=0)
    var = (projections - mean).pow(2).mean(dim=0)
    return mean.pow(2).mean() + (var - 1.0).pow(2).mean()


def stability_loss_running(
    z: torch.Tensor,
    w: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    momentum: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stability loss against EMA statistics (robust for tiny batches)."""
    projections = _as_batch_seq(z) @ w
    batch_mean = projections.mean(dim=0)
    batch_var = (projections - batch_mean).pow(2).mean(dim=0)

    new_mean = running_mean * (1.0 - momentum) + batch_mean * momentum
    new_var = running_var * (1.0 - momentum) + batch_var * momentum

    loss = batch_mean.pow(2).mean() + (batch_var - 1.0).pow(2).mean()
    return loss, new_mean, new_var


def sigreg_loss(
    z: torch.Tensor,
    w: torch.Tensor,
    freqs: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
) -> torch.Tensor:
    """Sketched Isotropic Gaussian Regularization (LeJEPA characteristic-function matching).

    Projects the latents onto ``M`` random 1-D directions, standardizes each
    projection, and penalizes the deviation of the empirical characteristic
    function from the standard-normal one ``phi(t) = exp(-t^2/2)`` at the
    frequencies ``freqs``.  Matching the whole CF (not only the first two
    moments) is what makes collapse prevention provable with a single
    hyperparameter.
    """
    projections = _as_batch_seq(z) @ w                        # [N, M]
    mean = projections.mean(dim=0, keepdim=True)
    std = projections.std(dim=0, keepdim=True).clamp_min(1e-6)
    u = (projections - mean) / std

    loss = z.new_zeros(())
    for freq in freqs:
        target = math.exp(-freq * freq / 2.0)
        cos_part = torch.cos(freq * u).mean(dim=0)
        sin_part = torch.sin(freq * u).mean(dim=0)
        loss = loss + ((cos_part - target).pow(2) + sin_part.pow(2)).mean()
    return loss / len(freqs)


def curvature_loss(z: torch.Tensor) -> torch.Tensor:
    """Second-order finite-difference (temporal straightening) penalty.

    ``mean ||z_t - 2 z_{t-1} + z_{t-2}||^2``; returns 0 for ``seq_len < 3``.
    """
    seq_len = z.shape[1]
    if seq_len < 3:
        return z.new_zeros(())
    accel = z[:, 2:] - 2.0 * z[:, 1:-1] + z[:, :-2]
    return accel.pow(2).mean()


def curvature_loss_with_dt(z: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Curvature loss for non-uniformly spaced sequences.

    ``dt`` is ``[seq_len]`` or ``[batch, seq_len]``; ``dt[t]`` is the time
    difference from step ``t-1`` to step ``t``.
    """
    seq_len = z.shape[1]
    if seq_len < 3:
        return z.new_zeros(())
    if dt.dim() == 1:
        dt = dt.unsqueeze(0).unsqueeze(-1)                    # [1, L, 1]
    else:
        dt = dt.unsqueeze(-1)                                 # [B, L, 1]

    dt_t = dt[:, 2:].clamp_min(1e-6)
    dt_t1 = dt[:, 1:-1].clamp_min(1e-6)
    first = (z[:, 2:] - z[:, 1:-1]) / dt_t
    second = (z[:, 1:-1] - z[:, :-2]) / dt_t1
    return (first - second).pow(2).mean()


def latent_loss(args: LatentLossArgs, projections: torch.Tensor) -> torch.Tensor:
    """Combined training loss for the latent predictors.

    ``L = L_latent + recon_weight * (L_recon + L_pred_x)
          + stability_weight * L_stability + curvature_weight * L_curvature``
    """
    target_z = args.z[:, 1:].detach()
    pred_z = args.pred_z[:, :-1]
    loss = F.mse_loss(pred_z, target_z)

    if args.reconstructed_x is not None:
        loss = loss + args.recon_weight * F.mse_loss(args.reconstructed_x, args.original_x)
    if args.predicted_x is not None:
        loss = loss + args.recon_weight * F.mse_loss(
            args.predicted_x[:, :-1], args.original_x[:, 1:].detach()
        )

    loss = loss + args.stability_weight * stability_loss(args.z, projections)
    loss = loss + args.curvature_weight * curvature_loss(args.z)
    return loss


def lejepa_loss(
    z: torch.Tensor,
    pred_z: torch.Tensor,
    projections: torch.Tensor,
    sigreg_weight: float = 1.0,
    freqs: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
) -> torch.Tensor:
    """LeJEPA objective: ``L_pred + sigreg_weight * SIGReg(z)``.

    No stop-gradient, no teacher-student, no EMA schedule: ``sigreg_weight`` is
    the only hyperparameter.
    """
    pred = F.mse_loss(pred_z[:, :-1], z[:, 1:].detach())
    return pred + sigreg_weight * sigreg_loss(z, projections, freqs)


# --------------------------------------------------------------------------- #
# Linear identifiability metrics
# --------------------------------------------------------------------------- #
def compute_identifiability_r2(z_hat: torch.Tensor, z_true: torch.Tensor) -> torch.Tensor:
    """Squared cross-correlation ratio between learned and true latents (0..1)."""
    a = _as_batch_seq(z_hat)
    b = _as_batch_seq(z_true)
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    n = a.shape[0]
    c = a.t() @ b / n
    auto = b.t() @ b / n
    return (c.pow(2).sum() / (auto.pow(2).sum() + 1e-8)).clamp(0.0, 1.0)


def procrustes_alignment(
    z_hat: torch.Tensor, z_true: torch.Tensor, n_iter: int = 30
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align ``z_hat`` to ``z_true`` by the nearest orthogonal map (polar factor).

    The orthogonal factor is computed with a Newton-Schulz iteration, which only
    needs matmuls (no SVD), and is differentiable.
    """
    d_model = z_hat.shape[-1]
    a = _as_batch_seq(z_hat)
    b = _as_batch_seq(z_true)
    a_mean = a.mean(dim=0, keepdim=True)
    b_mean = b.mean(dim=0, keepdim=True)
    a_c = a - a_mean
    b_c = b - b_mean

    c = a_c.t() @ b_c / a.shape[0]
    q = c / c.norm().clamp_min(1e-8) * math.sqrt(d_model)
    eye = torch.eye(d_model, device=q.device, dtype=q.dtype)
    for _ in range(n_iter):
        q = 0.5 * q @ (3.0 * eye - q.t() @ q)

    aligned = (a_c @ q + b_mean).reshape(z_hat.shape)
    return aligned, q


def linear_latent_plan(z_start: torch.Tensor, z_end: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Linear interpolation plan: ``[n_steps, batch, d_model]``."""
    alphas = torch.linspace(0.0, 1.0, n_steps, device=z_start.device, dtype=z_start.dtype)
    return (1.0 - alphas)[:, None, None] * z_start + alphas[:, None, None] * z_end


def plan_path_cost(plan: torch.Tensor) -> torch.Tensor:
    """Sum of squared step differences of a plan ``[n_steps, batch, d_model]``."""
    if plan.shape[0] < 2:
        return plan.new_zeros(())
    return (plan[1:] - plan[:-1]).pow(2).mean()


def gennorm_identifiability_score(z: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """How close the latent distribution is to a generalised normal of shape ``alpha``.

    Uses the kurtosis of the standardized latents; the score peaks at
    ``alpha = 2`` (Gaussian), the unique shape giving linear identifiability.
    """
    x = _as_batch_seq(z)
    x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-8)
    kurtosis = x.pow(4).mean(dim=0)
    targets = {0.5: 12.0, 1.0: 6.0, 2.0: 3.0, 5.0: 2.0}
    if alpha in targets:
        target = targets[alpha]
    elif alpha < 1.0:
        target = 12.0 + (6.0 - 12.0) * (alpha - 0.5) / 0.5
    elif alpha < 2.0:
        target = 6.0 + (3.0 - 6.0) * (alpha - 1.0)
    elif alpha < 5.0:
        target = 3.0 + (2.0 - 3.0) * (alpha - 2.0) / 3.0
    else:
        target = 1.8
    return torch.exp(-(kurtosis - target).pow(2) / 2.0).mean()


# --------------------------------------------------------------------------- #
# Condition monitors (Theorems 2-4 of Klindt, LeCun & Balestriero, 2026)
# --------------------------------------------------------------------------- #
@dataclass
class ExplorationSummary:
    """Isotropic-exploration diagnostics (Theorem 3: data coverage matters)."""

    coverage: float
    anisotropy: float
    effective_rank: float
    trajectory_narrowness: float
    gaussian_score: float
    risk_level: str = ""

    def as_dict(self) -> Dict[str, float]:
        return {
            "coverage": self.coverage,
            "anisotropy": self.anisotropy,
            "effective_rank": self.effective_rank,
            "trajectory_narrowness": self.trajectory_narrowness,
            "gaussian_score": self.gaussian_score,
        }


def compute_exploration_quality(z: torch.Tensor) -> ExplorationSummary:
    """Coverage / anisotropy / effective rank / narrowness / Gaussianity of latents."""
    d_model = z.shape[-1]
    x = _as_batch_seq(z)
    centered = x - x.mean(dim=0, keepdim=True)
    cov = centered.t() @ centered / x.shape[0]

    trace = cov.diagonal().sum().clamp_min(1e-8)
    frob_sq = cov.pow(2).sum().clamp_min(1e-8)
    effective_rank = float((trace * trace / frob_sq).clamp(1.0, float(d_model)))
    avg_eig = float(trace) / d_model

    # Dominant eigenvalue via power iteration (no SVD needed).
    v = torch.randn(d_model, 1, device=z.device, dtype=z.dtype)
    v = v / v.norm().clamp_min(1e-8)
    for _ in range(20):
        v = cov @ v
        v = v / v.norm().clamp_min(1e-8)
    lambda_max = float((v.t() @ cov @ v).sum())
    anisotropy = lambda_max / avg_eig if avg_eig > 1e-8 else 1.0

    # Trajectory narrowness: mean |cos| between consecutive displacements.
    narrowness = 0.0
    if z.shape[1] >= 3:
        d = z[:, 1:] - z[:, :-1]
        d1, d2 = d[:, :-1], d[:, 1:]
        cos_sim = (d1 * d2).sum(-1) / (
            d1.norm(dim=-1).clamp_min(1e-8) * d2.norm(dim=-1).clamp_min(1e-8)
        )
        narrowness = float(cos_sim.abs().mean())

    std = centered.std(dim=0) + 1e-8
    excess_kurtosis = float(((centered / std).pow(4).mean(dim=0) - 3.0).abs().mean())
    gaussian_score = float(math.exp(-excess_kurtosis * 0.5))

    coverage = min(max(effective_rank / d_model, 0.0), 1.0)
    if coverage < 0.2 or anisotropy > 3.0 or gaussian_score < 0.4:
        risk = "HIGH - identifiability guarantees likely violated; broaden data collection"
    elif coverage < 0.4 or anisotropy > 2.0 or gaussian_score < 0.6:
        risk = "MEDIUM - some risk; monitor the exploration strategy"
    else:
        risk = "LOW - exploration appears adequate for the LeJEPA guarantees"

    return ExplorationSummary(
        coverage=coverage,
        anisotropy=anisotropy,
        effective_rank=effective_rank,
        trajectory_narrowness=narrowness,
        gaussian_score=gaussian_score,
        risk_level=risk,
    )


@dataclass
class StationarityReport:
    """Stationarity diagnostics (Theorem 2: stationary dynamics required)."""

    residual_trend: float
    dominant_layer: int
    layer_entropy: float
    risk_level: str = ""


def check_stationarity(
    z: torch.Tensor, predicted_z: torch.Tensor, n_layers: int = 1
) -> StationarityReport:
    """Slope of the log prediction error over time (drift / phase transitions)."""
    seq_len = z.shape[1]
    trend = 0.0
    if seq_len >= 4:
        errors = (z[:, 1:] - predicted_z[:, :-1]).pow(2).sum(-1).mean(dim=0).detach()
        log_errors = (errors + 1e-10).log()
        t = torch.arange(1, log_errors.numel() + 1, device=z.device, dtype=z.dtype)
        if log_errors.numel() >= 4:
            t_c = t - t.mean()
            e_c = log_errors - log_errors.mean()
            denom = (t_c * t_c).sum()
            trend = float((t_c * e_c).sum() / denom) if denom > 0 else 0.0

    layer_entropy = 0.85 if n_layers > 1 else 1.0
    if trend > 0.05:
        dominant = max(n_layers - 1, 0)
    elif trend < -0.05:
        dominant = 0
    else:
        dominant = n_layers // 2

    if abs(trend) > 0.1:
        risk = "HIGH - dynamics may be non-stationary (prediction error is trending)"
    elif abs(trend) > 0.05:
        risk = "LOW-MEDIUM - mild trend detected; consider longer context windows"
    else:
        risk = "LOW - dynamics appear stationary"
    return StationarityReport(trend, dominant, layer_entropy, risk)


@dataclass
class PlanningConsistency:
    """Result of the latent-vs-observation planning consistency test (Theorem 4)."""

    cost_consistency_r2: float
    directional_error: float
    is_consistent: bool


def check_planning_consistency(
    z_plans: torch.Tensor, z_from_x_plans: torch.Tensor
) -> PlanningConsistency:
    """Compare path costs and directions in latent vs observation space."""
    plan_len = z_plans.shape[1]
    if plan_len < 2:
        return PlanningConsistency(0.0, math.pi / 2, False)

    z_steps = (z_plans[:, 1:] - z_plans[:, :-1]).pow(2).sum(-1)
    x_steps = (z_from_x_plans[:, 1:] - z_from_x_plans[:, :-1]).pow(2).sum(-1)
    zc = z_steps.flatten() - z_steps.mean()
    xc = x_steps.flatten() - x_steps.mean()
    r2 = float((zc @ xc).pow(2) / ((zc.pow(2).sum() * xc.pow(2).sum()).clamp_min(1e-8)))
    r2 = min(max(r2, 0.0), 1.0)

    dz = (z_plans[:, 1:] - z_plans[:, :-1]).reshape(-1, z_plans.shape[-1])
    dx = (z_from_x_plans[:, 1:] - z_from_x_plans[:, :-1]).reshape(-1, z_plans.shape[-1])
    cos_theta = (dz * dx).sum(-1) / (
        dz.norm(dim=-1).clamp_min(1e-8) * dx.norm(dim=-1).clamp_min(1e-8)
    )
    directional_error = float(cos_theta.mean().clamp(-1.0, 1.0).acos())
    return PlanningConsistency(r2, directional_error, r2 > 0.8 and directional_error < 0.5)


def health_check(
    z: torch.Tensor, predicted_z: torch.Tensor, n_layers: int = 1
) -> Tuple[ExplorationSummary, StationarityReport]:
    """Run the exploration and stationarity monitors together."""
    return (
        compute_exploration_quality(z),
        check_stationarity(z, predicted_z, n_layers),
    )


# --------------------------------------------------------------------------- #
# Predictors
# --------------------------------------------------------------------------- #
class LatentPredictor(nn.Module):
    """Latent world model with a single (optionally multi-scale) SSM backbone.

    ``forward`` returns ``(z, predicted_z, reconstructed_x, predicted_x)`` so
    that callers can compute :func:`latent_loss` or :func:`lejepa_loss`.
    ``step`` / ``step_no_action`` drive the ``O(1)`` recurrent path.
    """

    def __init__(
        self,
        config: Optional[MultiScaleSsmConfig] = None,
        input_dim: int = 8,
        action_dim: int = 0,
        n_projections: int = 16,
        multi_scale: bool = True,
    ):
        super().__init__()
        config = config or MultiScaleSsmConfig(d_model=64)
        config.validate()

        d_model = config.d_model
        self.d_model = d_model
        self.has_action = action_dim > 0

        self.encoder = nn.Linear(input_dim, d_model)
        self.decoder = nn.Linear(d_model, input_dim)
        if self.has_action:
            self.action_encoder = nn.Linear(action_dim, d_model)
            self.fusion = nn.Linear(d_model * 2, d_model)
        else:
            self.action_encoder = None
            self.fusion = nn.Linear(d_model, d_model)

        if multi_scale:
            self.ssm: nn.Module = MultiScaleSsmBlock(config)
        else:
            self.ssm = SsmBlock(
                SsmConfig(
                    d_model=d_model,
                    d_state=config.d_state,
                    expand=config.expand,
                    n_heads=config.n_heads,
                    mimo_rank=config.mimo_rank,
                    use_conv=config.use_conv,
                    conv_kernel=config.conv_kernel,
                )
            )

        # Fixed random projection buffer for the stability / SIGReg losses.
        self.register_buffer(
            "stability_projections", random_projections(d_model, n_projections)
        )

    # -- prediction -------------------------------------------------------- #
    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        """Observations -> latent ``z``."""
        return self.encoder(observations)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latent ``z`` -> observation space."""
        return self.decoder(z)

    def fuse(self, z: torch.Tensor, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Combine latents and (optional) encoded actions into the SSM input."""
        if self.action_encoder is not None and actions is not None:
            a = self.action_encoder(actions)
            return self.fusion(torch.cat([z, a], dim=-1))
        return self.fusion(z)

    def forward(
        self, observations: torch.Tensor, actions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``[B, L, input_dim]`` -> ``(z, predicted_z, reconstructed_x, predicted_x)``."""
        z = self.encode(observations)
        predicted_z = self.ssm(self.fuse(z, actions))
        return z, predicted_z, self.decode(z), self.decode(predicted_z)

    # -- autoregressive inference ------------------------------------------ #
    def zero_state(self, batch: int, device=None, dtype=None) -> MultiScaleState:
        """Zero-initialized recurrent state for :meth:`step`."""
        if isinstance(self.ssm, MultiScaleSsmBlock):
            return self.ssm.zero_state(batch, device=device, dtype=dtype)
        h, conv = self.ssm.zero_state(batch, device=device, dtype=dtype)
        return MultiScaleState(h=[h], prev_bx=[None], conv_state=[conv])

    @torch.no_grad()
    def step(
        self,
        z_prev: torch.Tensor,
        state: LatentState,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, MultiScaleState]:
        """One prediction step: ``z_prev`` ``[batch, d_model]`` -> next latent."""
        u = self.fuse(z_prev, action)
        if isinstance(self.ssm, MultiScaleSsmBlock):
            return self.ssm.forward_step(u, state)
        y, h, bx, conv = self.ssm.forward_step(
            u, state.h[0], state.prev_bx[0], state.conv_state[0]
        )
        return y, MultiScaleState(h=[h], prev_bx=[bx], conv_state=[conv])

    def step_no_action(
        self, z_prev: torch.Tensor, state: MultiScaleState
    ) -> Tuple[torch.Tensor, MultiScaleState]:
        """One prediction step without action conditioning."""
        return self.step(z_prev, state, action=None)

    # -- losses ------------------------------------------------------------ #
    def loss(
        self,
        z: torch.Tensor,
        pred_z: torch.Tensor,
        original_x: torch.Tensor,
        reconstructed_x: Optional[torch.Tensor] = None,
        predicted_x: Optional[torch.Tensor] = None,
        stability_weight: float = 1.0,
        curvature_weight: float = 0.1,
        recon_weight: float = 1.0,
    ) -> torch.Tensor:
        """Combined latent loss (see :func:`latent_loss`)."""
        return latent_loss(
            LatentLossArgs(
                z=z,
                pred_z=pred_z,
                original_x=original_x,
                stability_weight=stability_weight,
                curvature_weight=curvature_weight,
                recon_weight=recon_weight,
                reconstructed_x=reconstructed_x,
                predicted_x=predicted_x,
            ),
            self.stability_projections,
        )

    def lejepa(
        self,
        z: torch.Tensor,
        pred_z: torch.Tensor,
        sigreg_weight: float = 1.0,
        freqs: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
    ) -> torch.Tensor:
        """LeJEPA objective using the model's own projection buffer."""
        return lejepa_loss(z, pred_z, self.stability_projections, sigreg_weight, freqs)


# Backwards/forwards compatible alias kept explicit for readability.
MultiScaleLatentPredictor = LatentPredictor
