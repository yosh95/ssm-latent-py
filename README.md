# ssm-latent-py ⚡

**Latent world models in pure PyTorch** — a Mamba‑3 style selective SSM × JEPA (LeJEPA).

Python port of the core of [`ssm-latent-rs`](https://github.com/yosh95/ssm-latent-rs).

* 🧩 **Pure PyTorch** — only the standard API (`torch`, `torch.nn`, `torch.nn.functional`).
  No `mamba-ssm`, no `causal-conv1d`, no CUDA-only kernels: runs on CPU and GPU.
* 📚 **A library, not an app** — no `main`, no CLI; import it from your own trainer,
  agent, or simulator.
* 🪶 **Small and readable** — two modules (`ssm`, `latent`) and ~40 focused tests.

---

## Install

```bash
pip install -e .          # or: pip install torch pytest && pip install -e ".[test]"
pytest                    # run the test-suite (CPU only)
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0. The only runtime dependency is `torch`.

---

## What is implemented

### `ssm_latent.ssm` — Mamba‑3 style selective SSM

| Mamba‑3 ingredient | Where |
|---|---|
| **Exponential‑trapezoidal discretization** — `h_t = α_t h_{t-1} + β_t B_{t-1}x_{t-1} + γ_t B_t x_t` with a learned gate `λ_t` | `SsmBlock.selective_scan` |
| **Complex‑valued state transition** `A = a_re + i·a_im` with data‑dependent RoPE (`a_im·Δ + θ`) | `SsmBlock._rotation`, `rotate_state` |
| **MIMO** — matmul state update, `d_head / mimo_rank` values per rank | `SsmBlock.selective_scan` |
| **BCNorm** — RMSNorm on `B`/`C` plus learnable per‑head biases (makes short conv optional) | `SsmBlock._b_head`, `_c_head` |
| Optional causal depthwise `conv1d` | `SsmConfig(use_conv=True)` |
| **Parallel scan**, `O(L log L)` for training | `SsmBlock.forward` |
| **Recurrent step**, `O(1)` for inference | `SsmBlock.forward_step` |
| **Multi‑scale stack** (fast / medium / slow timescales) | `MultiScaleSsmBlock` |

The two execution modes are *mathematically equivalent*, which the test-suite
verifies numerically (`test_step_matches_forward_*`).

### `ssm_latent.latent` — JEPA / LeJEPA latent world model

* `LatentPredictor` — encoder → action encoder → fusion → SSM → decoder, with
  `forward` for training and `step` for streaming inference.
* Losses: `latent_loss` (prediction + reconstruction + stability + curvature),
  `lejepa_loss`, `sigreg_loss` (**S**ketched **I**sotropic **G**aussian
  **Reg**ularization — characteristic‑function matching, collapse prevention
  with a single hyperparameter and no stop‑gradient / EMA / teacher‑student),
  `stability_loss`, `stability_loss_running`, `curvature_loss`,
  `curvature_loss_with_dt` (temporal straightening).
* Metrics: `compute_identifiability_r2`, `procrustes_alignment` (differentiable
  polar decomposition via Newton–Schulz, no SVD), `linear_latent_plan`,
  `plan_path_cost`, `gennorm_identifiability_score`.
* Condition monitors for the LeJEPA identifiability guarantees:
  `compute_exploration_quality` (isotropic exploration),
  `check_stationarity` (stationary dynamics),
  `check_planning_consistency` (latent plans ≈ true plans), `health_check`.

---

## Quick start

```python
import torch
from ssm_latent import SsmBlock, SsmConfig, LatentPredictor, MultiScaleSsmConfig

# 1. The SSM block alone: [batch, seq, d_model]
block = SsmBlock(SsmConfig(d_model=32, d_state=8, expand=2, n_heads=2))
y = block(torch.randn(2, 16, 32))                      # -> [2, 16, 32]

# 2. The latent world model: observations (+ optional actions) -> next latent
model = LatentPredictor(
    MultiScaleSsmConfig(d_model=32, d_state=8, n_heads=2, n_layers=3),
    input_dim=6,      # observation size
    action_dim=2,     # set to 0 for unsupervised / action-free use
)
obs, act = torch.randn(4, 20, 6), torch.randn(4, 20, 2)
z, pred_z, rec_x, pred_x = model(obs, act)

loss = model.loss(z, pred_z, obs, rec_x, pred_x)       # or model.lejepa(z, pred_z)
loss.backward()

# 3. Streaming inference: O(1) per step, no re-encoding of the past
model.eval()
state = model.zero_state(batch=1)
with torch.no_grad():
    z_next, state = model.step(model.encode(obs[:1, 0]), state, act[:1, 0])
```

A runnable end-to-end example (training on a circular physical system, then
checking the latent space) lives in [`examples/quickstart.py`](examples/quickstart.py):

```bash
python examples/quickstart.py
```

---

## Design notes

* **Tensor layout** is `[batch, seq_len, feature]` everywhere; the step API uses
  `[batch, feature]`. `SsmBlock.forward_step` returns
  `(y, next_h, current_bx, next_conv_state)`.
* **State handling**: `MultiScaleState` holds one `(h, prev_bx, conv_state)`
  triple per SSM layer. `zero_state(batch)` creates it and `.detach()` cuts it
  from the autograd graph between truncated BPTT windows.
* **The stability/SIGReg projection matrix is a buffer**, not a parameter: it is
  drawn once from a Gaussian and never trained.
* **Complex rotation is stored as two halves**: `a_re`/`a_im` have shape
  `[n_heads, d_state/2]` and the state is laid out as `[h_re | h_im]`. The
  damping factor therefore has to be duplicated (`cat([da_re, da_re], -1)`) —
  not interleaved — to stay aligned with that layout.
* **No hidden dependencies**: everything degrades gracefully to CPU.

---

## Scope

This port intentionally covers only the **core** of `ssm-latent-rs`: the SSM
(`src/ssm.rs`) and the latent world model / JEPA-LeJEPA machinery
(`src/latent.rs`). The Rust project's peripheral modules — meta‑learning
(`meta/`), multimodal fusion (`multimodal.rs`), the standalone predictor
(`predictor.rs`), ONNX text preprocessing (`preprocess.rs`) and the demos
(`*-demo/`, `game-playing-wasm/`) — are **out of scope**.

## References

```bibtex
@article{lahoti2026mamba3,
  title={Mamba-3: Improved Sequence Modeling using State Space Principles},
  author={Lahoti, Aakash and others},
  journal={ICLR}, year={2026}
}
@article{balestriero2025lejepa,
  title={LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics},
  author={Balestriero, R. and LeCun, Y.}, journal={arXiv:2511.08544}, year={2025}
}
@article{klindt2026identifiability,
  title={When Does LeJEPA Learn a World Model?},
  author={Klindt, D. and LeCun, Y. and Balestriero, R.}, journal={arXiv:2605.26379}, year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
