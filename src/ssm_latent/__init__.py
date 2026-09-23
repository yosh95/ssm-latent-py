"""ssm-latent-py -- Latent world models: Mamba-3 style SSM x JEPA in pure PyTorch.

Python port of the core of `ssm-latent-rs <https://github.com/yosh95/ssm-latent-rs>`_.
Only the PyTorch standard API is used (no CUDA-only ``mamba-ssm`` kernels), so the
library runs on CPU and GPU alike.  It is a library, not an application: there is
no ``main``.

Quick start
-----------
>>> import torch
>>> from ssm_latent import SsmConfig, SsmBlock, LatentPredictor, MultiScaleSsmConfig
>>> block = SsmBlock(SsmConfig(d_model=32, d_state=8, expand=2, n_heads=2))
>>> block(torch.randn(2, 16, 32)).shape
torch.Size([2, 16, 32])
>>> model = LatentPredictor(MultiScaleSsmConfig(d_model=32, d_state=8, n_heads=2, n_layers=3),
...                         input_dim=6, action_dim=2)
>>> z, pred_z, rec_x, pred_x = model(torch.randn(2, 16, 6), torch.randn(2, 16, 2))
"""

from .ssm import (
    MultiScaleSsmBlock,
    MultiScaleSsmConfig,
    MultiScaleState,
    SsmBlock,
    SsmConfig,
)
from .latent import (
    ExplorationSummary,
    LatentLossArgs,
    LatentPredictor,
    LatentState,
    MlpEncoder,
    MlpEncoderConfig,
    MultiScaleLatentPredictor,
    PlanningConsistency,
    StationarityReport,
    check_planning_consistency,
    check_stationarity,
    compute_exploration_quality,
    compute_identifiability_r2,
    curvature_loss,
    curvature_loss_with_dt,
    gennorm_identifiability_score,
    health_check,
    latent_loss,
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

__version__ = "0.1.0"

__all__ = [
    # ssm
    "SsmConfig",
    "SsmBlock",
    "MultiScaleSsmConfig",
    "MultiScaleSsmBlock",
    "MultiScaleState",
    # latent
    "LatentPredictor",
    "MultiScaleLatentPredictor",
    "LatentState",
    "LatentLossArgs",
    "MlpEncoder",
    "MlpEncoderConfig",
    "normalize_projections",
    "random_projections",
    "latent_loss",
    "lejepa_loss",
    "sigreg_loss",
    "stability_loss",
    "stability_loss_running",
    "curvature_loss",
    "curvature_loss_with_dt",
    "compute_identifiability_r2",
    "procrustes_alignment",
    "linear_latent_plan",
    "plan_path_cost",
    "gennorm_identifiability_score",
    "compute_exploration_quality",
    "check_stationarity",
    "check_planning_consistency",
    "health_check",
    "ExplorationSummary",
    "StationarityReport",
    "PlanningConsistency",
    "__version__",
]
