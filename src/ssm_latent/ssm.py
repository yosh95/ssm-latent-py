"""Mamba-3 style selective state-space model in pure PyTorch.

Ported from the Rust reference implementation (``ssm-latent-rs``, ``src/ssm.rs``).
Only the PyTorch standard API is used: there is no dependency on the official
``mamba-ssm`` / ``causal-conv1d`` CUDA kernels, so everything runs on CPU as well
as GPU.

Core Mamba-3 ingredients implemented here
-----------------------------------------
* **Exponential-trapezoidal discretization** -- 3-term recurrence
  ``h_t = alpha_t h_{t-1} + beta_t B_{t-1} x_{t-1} + gamma_t B_t x_t``
  with a learned gate ``lambda_t`` selecting the convex combination of the
  interval endpoints.
* **Complex-valued state transition** -- ``A = a_re + i a_im``; the state is
  rotated by ``a_im * delta + theta`` (data-dependent RoPE) and damped by
  ``exp(a_re * delta)``.
* **MIMO** -- each head emits ``mimo_rank`` values through a matmul state update.
* **BCNorm** -- RMSNorm on the ``B`` and ``C`` projections, plus learnable
  per-head biases (makes the short convolution optional).
* Optional causal depthwise ``conv1d`` for local context.

Two mathematically equivalent execution modes are provided:

* :meth:`SsmBlock.forward` -- parallel (Blelloch/Hillis-Steele) associative scan,
  ``O(L log L)``, used for training.
* :meth:`SsmBlock.forward_step` -- ``O(1)`` recurrent state update, used for
  autoregressive / streaming inference.

Tensors are laid out as ``[batch, seq_len, d_model]`` (``[batch, d_model]`` for
the step API), matching the Rust implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SsmConfig",
    "SsmBlock",
    "MultiScaleSsmConfig",
    "MultiScaleSsmBlock",
    "MultiScaleState",
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class SsmConfig:
    """Configuration of :class:`SsmBlock`.

    Attributes
    ----------
    d_model:
        Input/output dimension of the block.
    d_state:
        State dimension. Must be even (complex rotation is stored as two
        ``d_state / 2`` real halves).
    expand:
        Inner expansion factor; ``d_inner = d_model * expand``.
    n_heads:
        Number of heads; ``d_inner`` must be divisible by ``n_heads``.
    mimo_rank:
        MIMO rank; ``d_head = d_inner / n_heads`` must be divisible by it.
    use_conv:
        Apply a causal depthwise convolution for local context.
    conv_kernel:
        Kernel size of that convolution.
    """

    d_model: int
    d_state: int = 8
    expand: int = 2
    n_heads: int = 1
    mimo_rank: int = 1
    use_conv: bool = False
    conv_kernel: int = 4

    def validate(self) -> None:
        """Raise :class:`ValueError` if the configuration is inconsistent."""
        if self.d_model <= 0:
            raise ValueError("d_model must be greater than 0")
        if self.d_state <= 0:
            raise ValueError("d_state must be greater than 0")
        if self.d_state % 2 != 0:
            raise ValueError(
                f"d_state must be even for complex rotation (got {self.d_state})"
            )
        if self.n_heads <= 0:
            raise ValueError("n_heads must be greater than 0")
        if self.mimo_rank <= 0:
            raise ValueError("mimo_rank must be greater than 0")
        if self.expand <= 0:
            raise ValueError("expand must be greater than 0")
        d_inner = self.d_model * self.expand
        if d_inner % self.n_heads != 0:
            raise ValueError(
                f"d_inner ({d_inner}) must be divisible by n_heads ({self.n_heads})"
            )
        d_head = d_inner // self.n_heads
        if d_head % self.mimo_rank != 0:
            raise ValueError(
                f"d_head ({d_head}) must be divisible by mimo_rank ({self.mimo_rank})"
            )
        if self.use_conv and self.conv_kernel <= 0:
            raise ValueError("conv_kernel must be greater than 0 when use_conv is True")

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand


# --------------------------------------------------------------------------- #
# Associative parallel scan over the complex rotation
# --------------------------------------------------------------------------- #
def _parallel_scan(
    a00: torch.Tensor,
    a01: torch.Tensor,
    a10: torch.Tensor,
    a11: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inclusive Hillis-Steele scan of state-transition / input pairs.

    Each element of the sequence is an affine map ``h_t = A_t h_{t-1} + w_t``
    where ``A_t`` is the 2x2 block ``[[a00, a01], [a10, a11]]`` operating on the
    real/imaginary pair ``(h_re, h_im)``.  The composition of two adjacent maps
    is again affine::

        A_new = A_right @ A_left
        w_new = A_right @ w_left + w_right

    All tensors have shape ``[batch, seq_len, ...]`` with the sequence on
    dimension 1; trailing dimensions are broadcast/batched freely.
    """
    seq_len = a00.shape[1]
    offset = 1
    while offset < seq_len:
        head = slice(0, offset)
        left = slice(0, seq_len - offset)
        right = slice(offset, seq_len)

        # Compose the pair (offset..seq) with the pair (0..seq-offset).
        n00 = a00[:, right] * a00[:, left] + a01[:, right] * a10[:, left]
        n01 = a00[:, right] * a01[:, left] + a01[:, right] * a11[:, left]
        n10 = a10[:, right] * a00[:, left] + a11[:, right] * a10[:, left]
        n11 = a10[:, right] * a01[:, left] + a11[:, right] * a11[:, left]
        nw0 = a00[:, right] * w0[:, left] + a01[:, right] * w1[:, left] + w0[:, right]
        nw1 = a10[:, right] * w0[:, left] + a11[:, right] * w1[:, left] + w1[:, right]

        a00 = torch.cat([a00[:, head], n00], dim=1)
        a01 = torch.cat([a01[:, head], n01], dim=1)
        a10 = torch.cat([a10[:, head], n10], dim=1)
        a11 = torch.cat([a11[:, head], n11], dim=1)
        w0 = torch.cat([w0[:, head], nw0], dim=1)
        w1 = torch.cat([w1[:, head], nw1], dim=1)

        offset *= 2

    return w0, w1


# --------------------------------------------------------------------------- #
# SSM block
# --------------------------------------------------------------------------- #
class SsmBlock(nn.Module):
    """Selective SSM block: Mamba-3 recurrence, complex rotation, MIMO, BCNorm."""

    def __init__(self, config: SsmConfig):
        super().__init__()
        config.validate()

        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.n_heads = config.n_heads
        self.mimo_rank = config.mimo_rank
        self.d_inner = config.d_inner
        self.d_head = self.d_inner // self.n_heads
        self.d_head_mimo = self.d_head // self.mimo_rank

        # Main branch (u) and evolution gate branch.
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model)

        self.conv1d: Optional[nn.Conv1d] = None
        if config.use_conv:
            self.conv1d = nn.Conv1d(
                self.d_inner,
                self.d_inner,
                kernel_size=config.conv_kernel,
                groups=self.d_inner,
                bias=True,
            )
            self.conv_kernel = config.conv_kernel

        # Input-dependent parameters (computed from the SiLU-activated input).
        self.dt_proj = nn.Linear(self.d_inner, self.n_heads)
        # Small initial step size (delta = softplus(dt_proj(u))): a plain
        # nn.Linear is created and its bias replaced, mirroring the Rust code.
        with torch.no_grad():
            self.dt_proj.bias.fill_(-3.1)
        self.lambda_proj = nn.Linear(self.d_inner, self.n_heads)
        self.theta_proj = nn.Linear(
            self.d_inner, self.n_heads * (self.d_state // 2)
        )
        self.b_proj = nn.Linear(
            self.d_inner, self.n_heads * self.mimo_rank * self.d_state
        )
        self.c_proj = nn.Linear(
            self.d_inner, self.n_heads * self.mimo_rank * self.d_state
        )

        # BCNorm (QK normalization): RMSNorm applied to the flattened B/C
        # projections before adding the learnable per-head biases.
        self.b_norm = nn.RMSNorm(self.n_heads * self.mimo_rank * self.d_state)
        self.c_norm = nn.RMSNorm(self.n_heads * self.mimo_rank * self.d_state)

        # Learnable B/C biases, shape [n_heads, mimo_rank, d_state].
        self.b_bias = nn.Parameter(torch.zeros(self.n_heads, self.mimo_rank, self.d_state))
        self.c_bias = nn.Parameter(torch.zeros(self.n_heads, self.mimo_rank, self.d_state))

        # Complex diagonal transition A = a_re + i a_im, stored as the
        # independent halves of size d_state / 2 (2x2 real blocks).
        # Damping: uniform in [-1, -0.1]; rotation: uniform in [0, 2*pi].
        a_re = torch.empty(self.n_heads, self.d_state // 2).uniform_(-1.0, -0.1)
        a_im = torch.empty(self.n_heads, self.d_state // 2).uniform_(0.0, 2.0 * math.pi)
        self.a_re = nn.Parameter(a_re)
        self.a_im = nn.Parameter(a_im)

        # Skip connection (D) and output normalization.
        self.d = nn.Parameter(torch.ones(self.d_inner))
        self.norm = nn.RMSNorm(self.d_inner)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _bias_view(self, bias: torch.Tensor, ndim: int) -> torch.Tensor:
        """Reshape a ``[H, mimo_rank, d_state]`` bias for broadcasting over ``ndim`` dims."""
        return bias.reshape(*(1,) * (ndim - 3), self.n_heads, self.mimo_rank, self.d_state)

    def _b_head(self, b: torch.Tensor) -> torch.Tensor:
        """BCNorm + bias on a projected B tensor.

        ``b``: ``[..., n_heads * mimo_rank * d_state]`` -> ``[..., H, d_state, mimo_rank]``.
        """
        b = self.b_norm(b)
        b = b.reshape(*b.shape[:-1], self.n_heads, self.mimo_rank, self.d_state)
        b = b + self._bias_view(self.b_bias, b.dim())
        return b.transpose(-1, -2)  # [.., H, d_state, mimo_rank]

    def _c_head(self, c: torch.Tensor) -> torch.Tensor:
        """BCNorm + bias on a projected C tensor.

        ``c``: ``[..., n_heads * mimo_rank * d_state]`` -> ``[..., H, mimo_rank, d_state]``.
        """
        c = self.c_norm(c)
        c = c.reshape(*c.shape[:-1], self.n_heads, self.mimo_rank, self.d_state)
        return c + self._bias_view(self.c_bias, c.dim())

    def _damping(self, delta: torch.Tensor) -> torch.Tensor:
        """``exp(a_re * delta)`` -- the per-step damping factor (``alpha``).

        Shape ``[..., n_heads, d_state / 2]``: one value per independent 2x2
        rotation block.
        """
        return torch.exp(self.a_re * delta.unsqueeze(-1))

    @staticmethod
    def _damping_full(da_re: torch.Tensor) -> torch.Tensor:
        """Expand the ``d_state / 2`` damping values to the full state layout.

        The real and imaginary halves of the state share the same damping, so
        the half is duplicated (not interleaved) to stay aligned with the
        ``[h_re, h_im]`` memory layout.
        """
        return torch.cat([da_re, da_re], dim=-1)

    def _angle(self, delta: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Rotation angle ``a_im * delta + theta`` (data-dependent RoPE)."""
        return self.a_im * delta.unsqueeze(-1) + theta

    def _rotation(
        self, delta: torch.Tensor, theta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-step scaled rotation blocks and the damping factor.

        ``delta``: ``[B, L, H]``, ``theta``: ``[B, L, H, d_state/2]``.  Returns
        ``(a00, a01, a10, a11, da_re)`` where ``A = da_re * [[cos, -sin], [sin, cos]]``
        is applied to the ``(h_re, h_im)`` pair of the complex state and ``da_re``
        is ``exp(a_re * delta)`` of shape ``[B, L, H, d_state/2]``.
        """
        da_re = self._damping(delta)                            # [B, L, H, d_state/2]
        angle = self._angle(delta, theta)                       # [B, L, H, d_state/2]
        cos = torch.cos(angle).unsqueeze(-1)                    # [.., d_state/2, 1]
        sin = torch.sin(angle).unsqueeze(-1)
        da = da_re.unsqueeze(-1)
        return da * cos, -(da * sin), da * sin, da * cos, da_re

    # ------------------------------------------------------------------ #
    # parallel (training) path
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[batch, seq_len, d_model]`` -> ``[batch, seq_len, d_model]``."""
        batch, _, _ = x.shape
        u_orig, evo_gate = self.in_proj(x).chunk(2, dim=-1)

        u = u_orig
        if self.conv1d is not None:
            # Causal convolution: left-pad by (kernel - 1).
            u = F.pad(u.transpose(1, 2), (self.conv_kernel - 1, 0))
            u = self.conv1d(u).transpose(1, 2)

        u_silu = F.silu(u)
        delta = F.softplus(self.dt_proj(u_silu))
        lam = torch.sigmoid(self.lambda_proj(u_silu))
        theta = self.theta_proj(u_silu)
        b = self.b_proj(u_silu)
        c = self.c_proj(u_silu)

        y_ssm = self.selective_scan(u_silu, delta, lam, theta, b, c)
        y = self.norm(y_ssm + u_silu * self.d)
        return self.out_proj(y * F.silu(evo_gate))

    def selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        lam: torch.Tensor,
        theta: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Parallel selective scan, ``O(L log L)``.

        ``u``: ``[B, L, d_inner]``, ``delta``/``lam``: ``[B, L, n_heads]``,
        ``theta``: ``[B, L, n_heads * d_state/2]``,
        ``b``/``c``: ``[B, L, n_heads * mimo_rank * d_state]``.

        Recurrence (Mamba-3, exponential-trapezoidal)::

            h_t = exp(delta_t * A_t) h_{t-1}
                + (1 - lam_t) delta_t exp(delta_t A_t) B_{t-1} x_{t-1}
                +      lam_t  delta_t                 B_t     x_t
            y_t = C_t^T h_t
        """
        batch, seq_len, _ = u.shape
        d_state, h_, m_ = self.d_state, self.n_heads, self.mimo_rank
        d_state_2 = d_state // 2

        u_mimo = u.reshape(batch, seq_len, h_, m_, self.d_head_mimo)
        b_step = self._b_head(b)                              # [B, L, H, d_state, m]
        c_step = self._c_head(c)                              # [B, L, H, m, d_state]

        # x_t = B_t @ u_t (MIMO): [B, L, H, d_state, d_head_mimo]
        bx = torch.einsum("blhds,blhsm->blhdm", b_step, u_mimo)
        bx_prev = torch.cat([torch.zeros_like(bx[:, :1]), bx[:, :-1]], dim=1)

        a00, a01, a10, a11, da_re = self._rotation(
            delta, theta.reshape(batch, seq_len, h_, d_state_2)
        )

        # gamma_t * bx_t + beta_t * bx_{t-1}
        dt = delta.unsqueeze(-1).unsqueeze(-1)                 # [B, L, H, 1, 1]
        gate = lam.unsqueeze(-1).unsqueeze(-1)
        gamma = dt * gate
        beta = dt * (1.0 - gate) * self._damping_full(da_re).unsqueeze(-1)
        w = gamma * bx + beta * bx_prev                        # [B, L, H, d_state, m]

        w0 = w[..., :d_state_2, :]
        w1 = w[..., d_state_2:, :]
        h_re, h_im = _parallel_scan(a00, a01, a10, a11, w0, w1)
        h = torch.cat([h_re, h_im], dim=-2)                    # [B, L, H, d_state, m]

        # y_t = C_t @ h_t -> [B, L, H, m, d_head_mimo] -> [B, L, d_inner]
        y = torch.einsum("blhmd,blhdj->blhmj", c_step, h)
        return y.reshape(batch, seq_len, self.d_inner)

    # ------------------------------------------------------------------ #
    # recurrent (inference) path
    # ------------------------------------------------------------------ #
    def rotate_state(self, h: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Rotate the complex state ``h`` by ``angle`` (data-dependent RoPE)."""
        d_state_2 = self.d_state // 2
        cos = torch.cos(angle).unsqueeze(-1)                   # [B, H, d_state/2, 1]
        sin = torch.sin(angle).unsqueeze(-1)
        h_re = h[..., :d_state_2, :]
        h_im = h[..., d_state_2:, :]
        return torch.cat([h_re * cos - h_im * sin, h_re * sin + h_im * cos], dim=-2)

    def forward_step(
        self,
        x: torch.Tensor,
        prev_h: torch.Tensor,
        prev_bx: Optional[torch.Tensor] = None,
        conv_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Single timestep with ``O(1)`` state update.

        ``x``: ``[batch, d_model]``; ``prev_h``: ``[batch, n_heads, d_state,
        d_head_mimo]``; ``prev_bx``: previous ``B x`` contribution (or ``None``
        on the first step); ``conv_state``: ``[batch, d_inner, kernel - 1]``.

        Returns ``(y, next_h, current_bx, next_conv_state)``.
        """
        batch = x.shape[0]
        u_orig, evo_gate = self.in_proj(x).chunk(2, dim=-1)

        next_conv_state = None
        u = u_orig
        if self.conv1d is not None:
            kernel = self.conv_kernel
            if conv_state is None:
                conv_state = torch.zeros(
                    batch, self.d_inner, kernel - 1, device=x.device, dtype=x.dtype
                )
            x_conv = torch.cat([conv_state, u_orig.unsqueeze(-1)], dim=-1)
            u = self.conv1d(x_conv).squeeze(-1)
            next_conv_state = x_conv[..., 1:kernel]

        u_silu = F.silu(u)
        delta = F.softplus(self.dt_proj(u_silu))               # [B, H]
        lam = torch.sigmoid(self.lambda_proj(u_silu))
        theta = self.theta_proj(u_silu)                        # [B, H * d_state/2]

        d_state_2 = self.d_state // 2
        theta_h = theta.reshape(batch, self.n_heads, d_state_2)
        da_re = self._damping(delta)                            # [B, H, d_state/2]
        h_rot = self.rotate_state(prev_h, self._angle(delta, theta_h))

        bx = torch.einsum(
            "bhdm,bhmj->bhdj",
            self._b_head(self.b_proj(u_silu)),
            u_silu.reshape(batch, self.n_heads, self.mimo_rank, self.d_head_mimo),
        )

        dt = delta.unsqueeze(-1).unsqueeze(-1)                 # [B, H, 1, 1]
        gate = lam.unsqueeze(-1).unsqueeze(-1)
        gamma = dt * gate
        beta = dt * (1.0 - gate) * self._damping_full(da_re).unsqueeze(-1)
        if prev_bx is None:
            prev_bx = torch.zeros_like(bx)

        h_next = (
            self._damping_full(da_re).unsqueeze(-1) * h_rot + gamma * bx + beta * prev_bx
        )

        c_step = self._c_head(self.c_proj(u_silu))            # [B, H, m, d_state]
        y_ssm = torch.einsum("bhmd,bhdj->bhmj", c_step, h_next)
        y = y_ssm.reshape(batch, self.d_inner)
        y = self.norm(y + u_silu * self.d)
        return self.out_proj(y * F.silu(evo_gate)), h_next, bx, next_conv_state

    def zero_state(
        self, batch: int, device=None, dtype=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Zero-initialized ``(h, conv_state)`` for :meth:`forward_step`."""
        device = device if device is not None else self.a_re.device
        dtype = dtype if dtype is not None else self.a_re.dtype
        h = torch.zeros(batch, self.n_heads, self.d_state, self.d_head_mimo,
                        device=device, dtype=dtype)
        conv = None
        if self.conv1d is not None:
            conv = torch.zeros(batch, self.d_inner, self.conv_kernel - 1,
                               device=device, dtype=dtype)
        return h, conv


# --------------------------------------------------------------------------- #
# Multi-scale stack
# --------------------------------------------------------------------------- #
@dataclass
class MultiScaleSsmConfig:
    """Configuration of :class:`MultiScaleSsmBlock`.

    ``n_layers`` SSM blocks are stacked with residual connections, each
    initialized with a different damping range so that the layers specialize in
    different timescales (fast / medium / slow).
    """

    d_model: int
    d_state: int = 8
    expand: int = 2
    n_heads: int = 1
    mimo_rank: int = 1
    n_layers: int = 3
    use_conv: bool = True
    conv_kernel: int = 4

    def validate(self) -> None:
        if self.n_layers <= 0:
            raise ValueError("n_layers must be greater than 0")
        SsmConfig(
            d_model=self.d_model,
            d_state=self.d_state,
            expand=self.expand,
            n_heads=self.n_heads,
            mimo_rank=self.mimo_rank,
            use_conv=self.use_conv,
            conv_kernel=self.conv_kernel,
        ).validate()


@dataclass
class MultiScaleState:
    """Recurrent state of a stack of SSM layers (one entry per layer).

    The same container is used for a single :class:`SsmBlock` (a one-element
    stack) by the higher-level predictors, so it is exported as the general
    recurrent-state type.
    """

    h: List[torch.Tensor]
    prev_bx: List[Optional[torch.Tensor]]
    conv_state: List[Optional[torch.Tensor]]

    def detach(self) -> "MultiScaleState":
        """Detach the state from the graph before feeding it back in."""
        return MultiScaleState(
            h=[t.detach() for t in self.h],
            prev_bx=[None if t is None else t.detach() for t in self.prev_bx],
            conv_state=[None if t is None else t.detach() for t in self.conv_state],
        )


class MultiScaleSsmBlock(nn.Module):
    """Stack of :class:`SsmBlock` layers with per-layer timescale initialization."""

    def __init__(self, config: MultiScaleSsmConfig):
        super().__init__()
        config.validate()

        self.n_layers = config.n_layers
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.n_heads = config.n_heads
        self.mimo_rank = config.mimo_rank
        self.d_inner = config.d_model * config.expand

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer_idx in range(config.n_layers):
            if layer_idx == 0:
                a_re_range = (-1.0, -0.3)     # fast transients
            elif layer_idx == 1:
                a_re_range = (-0.3, -0.05)    # medium / daily patterns
            else:
                a_re_range = (-0.05, -0.005)  # slow / seasonal patterns

            block = SsmBlock(
                SsmConfig(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    expand=config.expand,
                    n_heads=config.n_heads,
                    mimo_rank=config.mimo_rank,
                    # The convolution is applied on the first layer only: deeper
                    # layers already see convolved features through the residual.
                    use_conv=config.use_conv and layer_idx == 0,
                    conv_kernel=config.conv_kernel,
                )
            )
            with torch.no_grad():
                block.a_re.uniform_(*a_re_range)
            self.layers.append(block)
            self.norms.append(nn.RMSNorm(config.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-LN residual stack: ``x = x + layer(norm(x)) / n_layers``."""
        scale = 1.0 / self.n_layers
        out = x
        for layer, norm in zip(self.layers, self.norms):
            out = (layer(norm(out)) * scale + out).clamp(-100.0, 100.0)
        return out

    def forward_step(
        self, x: torch.Tensor, state: MultiScaleState
    ) -> Tuple[torch.Tensor, MultiScaleState]:
        """One timestep through every layer (``x``: ``[batch, d_model]``)."""
        scale = 1.0 / self.n_layers
        out = x
        hs, bxs, convs = [], [], []
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            y, h, bx, conv = layer.forward_step(
                norm(out), state.h[i], state.prev_bx[i], state.conv_state[i]
            )
            out = (y * scale + out).clamp(-100.0, 100.0)
            hs.append(h)
            bxs.append(bx)
            convs.append(conv)
        return out, MultiScaleState(h=hs, prev_bx=bxs, conv_state=convs)

    def zero_state(self, batch: int, device=None, dtype=None) -> MultiScaleState:
        """Zero-initialized :class:`MultiScaleState`."""
        hs, bxs, convs = [], [], []
        for layer in self.layers:
            h, conv = layer.zero_state(batch, device=device, dtype=dtype)
            hs.append(h)
            bxs.append(None)
            convs.append(conv)
        return MultiScaleState(h=hs, prev_bx=bxs, conv_state=convs)
