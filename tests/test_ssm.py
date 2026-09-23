"""Tests for the Mamba-3 style selective SSM block."""

import pytest
import torch

from ssm_latent import MultiScaleSsmBlock, MultiScaleSsmConfig, SsmBlock, SsmConfig


def make_block(**kwargs):
    torch.manual_seed(0)
    cfg = SsmConfig(**kwargs)
    block = SsmBlock(cfg)
    block.eval()
    return block


def test_config_validation():
    SsmConfig(d_model=16, d_state=8, expand=2, n_heads=2, mimo_rank=1).validate()

    with pytest.raises(ValueError, match="d_state must be even"):
        SsmConfig(d_model=16, d_state=7, n_heads=2).validate()
    with pytest.raises(ValueError, match="divisible by n_heads"):
        SsmConfig(d_model=16, d_state=8, expand=2, n_heads=7).validate()
    with pytest.raises(ValueError, match="divisible by mimo_rank"):
        SsmConfig(d_model=16, d_state=8, expand=2, n_heads=2, mimo_rank=3).validate()
    with pytest.raises(ValueError, match="conv_kernel"):
        SsmConfig(d_model=16, d_state=8, use_conv=True, conv_kernel=0).validate()


def test_forward_shapes():
    block = make_block(d_model=16, d_state=8, expand=2, n_heads=2, mimo_rank=1)
    for batch, seq_len in [(1, 1), (2, 5), (3, 16)]:
        x = torch.randn(batch, seq_len, 16)
        y = block(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()


def test_forward_with_mimo_and_conv():
    block = make_block(
        d_model=16, d_state=8, expand=2, n_heads=2, mimo_rank=2, use_conv=True
    )
    x = torch.randn(2, 7, 16)
    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_step_matches_forward_no_conv():
    """The O(1) recurrent path must reproduce the parallel scan exactly."""
    block = make_block(d_model=16, d_state=8, expand=2, n_heads=2, mimo_rank=2)
    x = torch.randn(2, 9, 16)

    y_parallel = block(x)

    h, conv_state = block.zero_state(x.shape[0])
    prev_bx = None
    outputs = []
    for t in range(x.shape[1]):
        y, h, prev_bx, conv_state = block.forward_step(
            x[:, t], h, prev_bx, conv_state
        )
        outputs.append(y)
    y_sequential = torch.stack(outputs, dim=1)

    assert torch.allclose(y_parallel, y_sequential, atol=1e-5)


def test_step_matches_forward_with_conv():
    block = make_block(d_model=16, d_state=4, expand=2, n_heads=2, use_conv=True, conv_kernel=3)
    block.eval()
    x = torch.randn(2, 8, 16)

    y_parallel = block(x)
    h, conv_state = block.zero_state(x.shape[0])
    prev_bx = None
    outputs = []
    for t in range(x.shape[1]):
        y, h, prev_bx, conv_state = block.forward_step(x[:, t], h, prev_bx, conv_state)
        outputs.append(y)
    y_sequential = torch.stack(outputs, dim=1)

    assert torch.allclose(y_parallel, y_sequential, atol=1e-5)


def test_causality():
    """Changing a future input must not change earlier outputs."""
    block = make_block(d_model=16, d_state=8, n_heads=2, use_conv=True, conv_kernel=4)
    x = torch.randn(1, 6, 16)
    x_perturbed = x.clone()
    x_perturbed[:, 4:] += 5.0

    y = block(x)
    y_perturbed = block(x_perturbed)
    assert torch.allclose(y[:, :4], y_perturbed[:, :4], atol=1e-6)


def test_gradients_flow_to_all_parameters():
    torch.manual_seed(0)
    block = SsmBlock(SsmConfig(d_model=16, d_state=8, expand=2, n_heads=2, use_conv=True))
    x = torch.randn(2, 6, 16, requires_grad=True)
    block(x).pow(2).mean().backward()

    for name, param in block.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_multiscale_stack_shapes_and_recurrence():
    torch.manual_seed(0)
    cfg = MultiScaleSsmConfig(d_model=16, d_state=8, n_heads=2, n_layers=3, use_conv=True)
    block = MultiScaleSsmBlock(cfg).eval()
    x = torch.randn(2, 6, 16)

    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    state = block.zero_state(2)
    outputs = []
    for t in range(x.shape[1]):
        y_t, state = block.forward_step(x[:, t], state)
        outputs.append(y_t)
    assert torch.allclose(y, torch.stack(outputs, dim=1), atol=1e-5)


def test_multiscale_layers_use_different_timescales():
    torch.manual_seed(0)
    block = MultiScaleSsmBlock(MultiScaleSsmConfig(d_model=8, d_state=4, n_heads=1, n_layers=3))
    # Layer 0 decays fastest (most negative a_re), later layers decay slowest.
    means = [layer.a_re.detach().mean().item() for layer in block.layers]
    assert means[0] < means[1] < means[2]
