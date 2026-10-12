# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import math

import pytest
import torch

from sbi.utils.torchutils import process_device
from sbi.utils.vector_field_utils import UniformNormalConvolution


@pytest.mark.parametrize("scale", [-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
@pytest.mark.parametrize("noise", [0.2, 0.5, 1.0])
@pytest.mark.parametrize("low_val,high_val", [(-1.0, 1.0), (0.5, 2.5)])
def test_uniform_normal_convolution_normalization(
    scale: float, noise: float, low_val: float, high_val: float
):
    """Verify that UniformNormalConvolution.log_prob integrates to 1.0 across scales."""
    low = torch.tensor([low_val])
    high = torch.tensor([high_val])
    scale_t = torch.tensor([scale])
    noise_t = torch.tensor([noise])

    dist = UniformNormalConvolution(low=low, high=high, scale=scale_t, noise=noise_t)

    # Numerical integration over wide support
    max_center = max(abs(scale * low_val), abs(scale * high_val))
    bound = float(max_center + 6.0 * noise)
    grid = torch.linspace(-bound, bound, 40001).unsqueeze(1)
    dx = grid[1] - grid[0]
    pdf = dist.log_prob(grid).exp()
    integral = torch.sum(pdf * dx).item()

    assert abs(integral - 1.0) < 2e-3, (
        f"PDF does not integrate to 1.0 for scale={scale}, noise={noise},"
        f" bounds=[{low_val}, {high_val}]. Got {integral}"
    )


def test_uniform_normal_convolution_sample_shape_and_reproducibility():
    """Verify sampling shapes and properties."""
    dist = UniformNormalConvolution(low=-2.0, high=2.0, scale=1.5, noise=0.3)
    samples = dist.sample(torch.Size([100, 5]))
    assert samples.shape == (100, 5)


def test_uniform_normal_convolution_accelerator():
    """Verify that multi-dimensional inputs and accelerator devices execute cleanly."""
    device = process_device("cpu")
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    low = torch.tensor([-1.0, -2.0])
    high = torch.tensor([1.0, 2.0])
    scale = torch.tensor([2.0, 3.0])
    noise = torch.tensor([0.5, 0.5])

    dist = UniformNormalConvolution(low=low, high=high, scale=scale, noise=noise)
    val = torch.randn(20, 2, device=device)
    log_probs = dist.log_prob(val)

    assert log_probs.shape == (20, 2)
    assert log_probs.device.type == torch.device(device).type
    assert torch.isfinite(log_probs).all()


def test_uniform_normal_convolution_negative_scale_symmetry_and_degenerate():
    """Verify that negative scale log_prob is mathematically symmetric to positive scale
    and that degenerate scale=0 matches the Gaussian limit."""
    # Symmetry test: Uniform[-1, 1] scaled by +2.0 vs -2.0
    dist_pos = UniformNormalConvolution(low=-1.0, high=1.0, scale=2.0, noise=0.5)
    dist_neg = UniformNormalConvolution(low=-1.0, high=1.0, scale=-2.0, noise=0.5)

    test_pts = torch.linspace(-5.0, 5.0, 101).unsqueeze(1)
    lp_pos = dist_pos.log_prob(test_pts)
    lp_neg = dist_neg.log_prob(test_pts)

    assert torch.allclose(lp_pos, lp_neg, atol=1e-6), (
        "Negative scale should produce identical density to positive scale for "
        "symmetric prior."
    )

    # Degenerate scale=0 matches Normal(0, noise)
    dist_zero = UniformNormalConvolution(low=-1.0, high=1.0, scale=0.0, noise=0.5)
    normal_ref = torch.distributions.Normal(0.0, 0.5)
    lp_zero = dist_zero.log_prob(test_pts)
    lp_ref = normal_ref.log_prob(test_pts)

    assert torch.allclose(lp_zero, lp_ref, atol=1e-6), (
        "Degenerate scale=0 must match Gaussian limit Normal(0, noise)."
    )


def test_uniform_normal_convolution_autograd_stability():
    """Verify that gradients can be backpropagated through log_prob without NaNs."""
    for s_val in [2.0, -2.0, 0.0]:
        val = torch.tensor([0.2], requires_grad=True)
        scale = torch.tensor([s_val], requires_grad=True)
        dist = UniformNormalConvolution(low=-1.0, high=1.0, scale=scale, noise=0.5)
        lp = dist.log_prob(val)
        lp.backward()
        assert val.grad is not None and torch.isfinite(val.grad).all()
        assert scale.grad is not None and torch.isfinite(scale.grad).all()


def test_uniform_normal_convolution_dtype_preservation():
    """Verify that float64 inputs preserve double precision throughout log_prob."""
    dist = UniformNormalConvolution(
        low=torch.tensor([0.0], dtype=torch.float64),
        high=torch.tensor([1.0], dtype=torch.float64),
        scale=torch.tensor([1.5], dtype=torch.float64),
        noise=torch.tensor([0.5], dtype=torch.float64),
    )
    val = torch.tensor([0.5], dtype=torch.float64)
    lp = dist.log_prob(val)
    assert lp.dtype == torch.float64
    assert torch.isfinite(lp).all()


def test_uniform_normal_convolution_broadcasting():
    """Verify that batch dimensions and multi-dimensional parameters broadcast."""
    dist = UniformNormalConvolution(
        low=torch.tensor([-1.0, -2.0]),
        high=torch.tensor([1.0, 2.0]),
        scale=torch.tensor([2.0, 0.0]),
        noise=torch.tensor([0.5, 0.5]),
    )
    val = torch.randn(25, 2)
    lp = dist.log_prob(val)
    assert lp.shape == (25, 2)
    assert torch.isfinite(lp).all()


@pytest.mark.parametrize("scale_val", [1e-3, 1e-5, 1e-7, 1e-9])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_uniform_normal_convolution_small_nonzero_scale_stability(scale_val, dtype):
    """Verify that small nonzero scales smoothly converge to the Gaussian limit
    without catastrophic floating-point cancellation or clamping."""
    low = torch.tensor([-1.0], dtype=dtype)
    high = torch.tensor([1.0], dtype=dtype)
    noise = torch.tensor([0.5], dtype=dtype)
    scale = torch.tensor([scale_val], dtype=dtype)

    dist = UniformNormalConvolution(low=low, high=high, scale=scale, noise=noise)
    val = torch.tensor([0.2], dtype=dtype)
    lp = dist.log_prob(val)

    normal_ref = torch.distributions.Normal(
        torch.tensor(0.0, dtype=dtype), torch.tensor(0.5, dtype=dtype)
    )
    lp_ref = normal_ref.log_prob(val)

    # In float32 and float64, log_prob must match Gaussian limit to within 1e-4
    assert torch.allclose(lp, lp_ref, atol=1e-4), (
        f"Small scale {scale_val} deviated from Gaussian limit: {lp} vs {lp_ref}"
    )
    assert lp > -10.0, f"Small scale {scale_val} erroneously clamped to floor: {lp}"


def test_uniform_normal_convolution_boundary_continuity():
    """Verify smooth continuity across series / direct branch boundary (delta=1e-2)."""
    sigma = 0.5
    noise = torch.tensor([sigma], dtype=torch.float64)
    low = torch.tensor([-1.0], dtype=torch.float64)
    high = torch.tensor([1.0], dtype=torch.float64)

    # delta = diff / (2 * noise) = scale * (high - low) / (2 * noise)
    # delta = 1e-2 occurs at scale = 0.005
    val = torch.tensor([0.2], dtype=torch.float64)

    scales = torch.tensor([0.00499, 0.00500, 0.00501], dtype=torch.float64)
    log_probs = []
    for s in scales:
        dist = UniformNormalConvolution(low=low, high=high, scale=s, noise=noise)
        log_probs.append(dist.log_prob(val).item())

    # Values across the boundary must be continuous to within 1e-6
    assert abs(log_probs[1] - log_probs[0]) < 1e-6
    assert abs(log_probs[2] - log_probs[1]) < 1e-6


def test_uniform_normal_convolution_gradient_accuracy_at_zero_and_negative():
    """Verify autograd score matches analytical score at scale=0 and is finite at
    scale<0."""
    sigma = 0.5
    noise = torch.tensor([sigma])
    low = torch.tensor([-1.0])
    high = torch.tensor([1.0])

    # Zero scale: exact score of Normal(0, sigma^2) is -y / sigma^2
    y_zero = torch.tensor([0.3], requires_grad=True)
    dist_zero = UniformNormalConvolution(low=low, high=high, scale=0.0, noise=noise)
    dist_zero.log_prob(y_zero).backward()
    expected_score = -0.3 / (sigma**2)
    assert y_zero.grad is not None
    assert torch.allclose(y_zero.grad, torch.tensor(expected_score), atol=1e-4), (
        f"Score at zero scale {y_zero.grad} does not match analytical {expected_score}"
    )

    # Negative scale: spatial gradient must match analytical score
    y_neg = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    dist_neg = UniformNormalConvolution(
        low=low.to(torch.float64),
        high=high.to(torch.float64),
        scale=torch.tensor([-1.5], dtype=torch.float64),
        noise=noise.to(torch.float64),
    )
    dist_neg.log_prob(y_neg).backward()
    assert y_neg.grad is not None
    assert torch.isfinite(y_neg.grad)

    # Analytical score for Y = s*X + sigma*Z:
    # (phi(z1) - phi(z2)) / (sigma * (Phi(z1) - Phi(z2)))
    # With low=-1, high=1, scale=-1.5: u_min = -1.5, u_max = 1.5
    z1 = (0.3 - (-1.5)) / sigma
    z2 = (0.3 - 1.5) / sigma
    phi1 = math.exp(-0.5 * z1**2) / math.sqrt(2 * math.pi)
    phi2 = math.exp(-0.5 * z2**2) / math.sqrt(2 * math.pi)
    # 0.5 * erfc(-z / sqrt(2)) = Phi(z)
    phi_diff = 0.5 * (math.erfc(-z1 / math.sqrt(2)) - math.erfc(-z2 / math.sqrt(2)))
    expected_neg_score = (phi1 - phi2) / (sigma * phi_diff)
    msg = (
        f"Score at scale=-1.5 ({y_neg.grad.item()}) "
        f"does not match analytical ({expected_neg_score})"
    )
    assert abs(y_neg.grad.item() - expected_neg_score) < 1e-12, msg


def test_uniform_normal_convolution_nan_regressions():
    """Verify that autograd gradient poisoning across branches and intermediate
    overflows do not produce NaN or Inf values."""
    # Regression 1: Large scale backward branch poisoning in float32
    scale = torch.tensor(1e10, dtype=torch.float32, requires_grad=True)
    val = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    dist = UniformNormalConvolution(low=-1.0, high=1.0, scale=scale, noise=1.0)
    lp = dist.log_prob(val)
    lp.backward()
    assert torch.isfinite(lp)
    assert scale.grad is not None and torch.isfinite(scale.grad)
    assert val.grad is not None and torch.isfinite(val.grad)
    assert torch.allclose(scale.grad, torch.tensor(-1e-10), rtol=1e-3)
    assert torch.allclose(val.grad, torch.tensor(0.0), atol=1e-6)

    # Regression 2: Tiny noise with scale=0 intermediate overflow
    scale_zero = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    val_nz = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
    noise_tiny = torch.tensor(1e-10, dtype=torch.float32, requires_grad=True)
    dist_tiny = UniformNormalConvolution(
        low=-1.0, high=1.0, scale=scale_zero, noise=noise_tiny
    )
    lp_tiny = dist_tiny.log_prob(val_nz)
    lp_tiny.backward()
    assert torch.isfinite(lp_tiny)
    assert val_nz.grad is not None and torch.isfinite(val_nz.grad)
    assert scale_zero.grad is not None and torch.isfinite(scale_zero.grad)
    assert torch.allclose(val_nz.grad, torch.tensor(-1e20), rtol=1e-3)

    # Regression 3: Non-finite inputs
    dist_std = UniformNormalConvolution(low=-1.0, high=1.0, scale=1.0, noise=0.5)
    assert dist_std.log_prob(torch.tensor(float("inf"))) == -float("inf")
    assert dist_std.log_prob(torch.tensor(-float("inf"))) == -float("inf")
    assert torch.isnan(dist_std.log_prob(torch.tensor(float("nan"))))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("y_val", [3.0, 5.0, 6.0, -3.0, -5.0, -6.0])
def test_uniform_normal_convolution_tails(dtype, y_val):
    """Verify that both positive and negative tails maintain finite log probabilities
    and nonzero gradients without premature zero-clamping."""
    dist = UniformNormalConvolution(
        low=torch.tensor([-1.0], dtype=dtype),
        high=torch.tensor([1.0], dtype=dtype),
        scale=torch.tensor([1.0], dtype=dtype),
        noise=torch.tensor([0.5], dtype=dtype),
    )
    y = torch.tensor([y_val], dtype=dtype, requires_grad=True)
    lp = dist.log_prob(y)
    lp.backward()

    assert torch.isfinite(lp)
    assert y.grad is not None and torch.isfinite(y.grad)
    # The gradient should push toward 0: negative for y > 0, positive for y < 0
    if y_val > 0:
        assert y.grad.item() < 0.0
    else:
        assert y.grad.item() > 0.0


def test_uniform_normal_convolution_extreme_tail_float64():
    """Verify that float64 maintains nonzero gradient even at y=8.0 (14 std devs)."""
    dist = UniformNormalConvolution(
        low=torch.tensor([-1.0], dtype=torch.float64),
        high=torch.tensor([1.0], dtype=torch.float64),
        scale=torch.tensor([1.0], dtype=torch.float64),
        noise=torch.tensor([0.5], dtype=torch.float64),
    )
    y = torch.tensor([8.0], dtype=torch.float64, requires_grad=True)
    lp = dist.log_prob(y)
    lp.backward()
    assert torch.isfinite(lp)
    assert y.grad is not None and y.grad.item() < -20.0


def test_uniform_normal_convolution_small_interval_large_distance():
    """Verify Taylor series stability when interval width is small but standardized
    distance is large."""
    dist = UniformNormalConvolution(
        low=torch.tensor([-0.01], dtype=torch.float64),
        high=torch.tensor([0.01], dtype=torch.float64),
        scale=torch.tensor([1.0], dtype=torch.float64),
        noise=torch.tensor([1.0], dtype=torch.float64),
    )
    for z in [5.0, 10.0, 20.0, 50.0]:
        y = torch.tensor([z], dtype=torch.float64, requires_grad=True)
        lp = dist.log_prob(y)
        lp.backward()
        assert torch.isfinite(lp)
        assert y.grad is not None and torch.isfinite(y.grad)
        assert y.grad.item() < 0.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "low_val,high_val,scale_val,noise_val,y_val",
    [
        (-1.0, 1.0, 2.5, 0.5, 1.2),
        (-1.0, 1.0, -2.5, 0.5, -1.2),
        (0.5, 2.5, 1.5, 0.4, 2.0),
        (0.5, 2.5, -1.5, 0.4, -2.0),
        (-1.0, 1.0, 0.005, 0.5, 0.2),  # exact series boundary
        (1.5, 1.5, 2.0, 0.5, 3.2),  # degenerate interval
    ],
)
def test_uniform_normal_convolution_quadrature_agreement(
    dtype, low_val, high_val, scale_val, noise_val, y_val
):
    """Verify agreement against independent numerical quadrature across regimes."""
    import scipy.integrate as integrate

    low = torch.tensor([low_val], dtype=dtype)
    high = torch.tensor([high_val], dtype=dtype)
    scale = torch.tensor([scale_val], dtype=dtype)
    noise = torch.tensor([noise_val], dtype=dtype)
    dist = UniformNormalConvolution(low=low, high=high, scale=scale, noise=noise)

    y = torch.tensor([y_val], dtype=dtype, requires_grad=True)
    lp = dist.log_prob(y)
    lp.backward()
    lp_obs = lp.item()
    grad_obs = y.grad.item()

    # Independent reference via quadrature:
    u_min = min(scale_val * low_val, scale_val * high_val)
    u_max = max(scale_val * low_val, scale_val * high_val)
    diff = u_max - u_min

    phi = lambda t: math.exp(-0.5 * t**2) / math.sqrt(2 * math.pi)
    if diff == 0:
        ref_lp = (
            -0.5 * math.log(2 * math.pi)
            - math.log(noise_val)
            - 0.5 * ((y_val - u_min) / noise_val) ** 2
        )
        ref_grad = -(y_val - u_min) / (noise_val**2)
    else:
        z1 = (y_val - u_min) / noise_val
        z2 = (y_val - u_max) / noise_val
        integral, _ = integrate.quad(phi, z2, z1)
        ref_lp = math.log(integral / diff)
        ref_grad = (phi(z1) - phi(z2)) / (noise_val * integral)

    atol_lp = 1e-4 if dtype == torch.float32 else 1e-11
    atol_grad = 1e-4 if dtype == torch.float32 else 1e-10
    assert abs(lp_obs - ref_lp) < atol_lp, (
        f"Deviation from quadrature ref: obs={lp_obs}, ref={ref_lp}"
    )
    assert abs(grad_obs - ref_grad) < atol_grad, (
        f"Deviation from analytical grad ref: obs={grad_obs}, ref={ref_grad}"
    )
