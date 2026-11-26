# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from torch.distributions import Normal, Uniform
from torch.special import log_ndtr

from sbi.utils.vector_field_utils import UniformNormalConvolution, denoise


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_uniform_normal_convolution_integrates_to_one(scale: float):
    dist = UniformNormalConvolution(
        torch.tensor(0.5, dtype=torch.float64),
        torch.tensor(2.5, dtype=torch.float64),
        torch.tensor(scale, dtype=torch.float64),
        torch.tensor(0.3, dtype=torch.float64),
    )
    grid = torch.linspace(-5.0, 10.0, 30001, dtype=torch.float64)
    integral = torch.trapezoid(dist.log_prob(grid).exp(), grid)
    assert torch.isclose(integral, torch.tensor(1.0, dtype=torch.float64), atol=1e-4)


@pytest.mark.parametrize("value", [-2.0, 2.0])
def test_uniform_normal_convolution_tail_score(value: float):
    """In float32, the score 10 noise stds outside the box must not vanish."""
    low, high, noise = -1.0, 1.0, 0.1
    dist = UniformNormalConvolution(low, high, 1.0, noise)
    theta = torch.tensor(value, requires_grad=True)
    score = torch.autograd.grad(dist.log_prob(theta), theta)[0]

    # Reference in float64 log space: on the right of the box the CDF difference
    # is Phi(-z_hi) - Phi(-z_lo); the left side follows by symmetry.
    theta64 = torch.tensor(abs(value), dtype=torch.float64, requires_grad=True)
    log_upper_hi = log_ndtr(-(theta64 - high) / noise)
    log_upper_lo = log_ndtr(-(theta64 - low) / noise)
    log_mass = log_upper_hi + torch.log1p(-torch.exp(log_upper_lo - log_upper_hi))
    expected = torch.autograd.grad(log_mass, theta64)[0] * (1 if value > 0 else -1)

    assert torch.isclose(score.double(), expected, rtol=1e-4)


@pytest.mark.parametrize("scale", [0.0, 1e-7])
def test_uniform_normal_convolution_narrow_box_is_gaussian(scale: float):
    """A box much narrower than the noise reduces to a Gaussian at its center."""
    dist = UniformNormalConvolution(-1.0, 1.0, scale, 1.0)
    theta = torch.tensor([0.75, 2.5], requires_grad=True)
    log_prob = dist.log_prob(theta)
    score = torch.autograd.grad(log_prob.sum(), theta)[0]

    expected = Normal(0.0, 1.0).log_prob(theta.detach())
    assert torch.allclose(log_prob.detach(), expected, atol=1e-6)
    assert torch.allclose(score, -theta.detach(), rtol=1e-5)


def test_uniform_normal_convolution_broadcast_batch_shape():
    dist = UniformNormalConvolution(
        low=0.0,
        high=1.0,
        scale=torch.tensor([[0.5], [2.0]]),
        noise=torch.tensor([0.1, 0.2, 0.3]),
    )

    assert dist.batch_shape == (2, 3)
    samples = dist.sample(torch.Size([5]))
    assert samples.shape == (5, 2, 3)
    assert dist.log_prob(samples).shape == samples.shape


# Reference moments and log density at the mean, from 50-digit quadrature.
@pytest.mark.parametrize(
    "high, x_t, s, mean, var, log_prob",
    [
        (1.0, 2.0, 0.1, 0.990190676604, 9.44537782566e-5, 3.6291882294),  # #2202
        (1.0, -1.0, 0.1, 0.00980932339625, 9.44537782566e-5, 3.6291882294),
        (1.0, 1.05, 1e-4, 0.999999800002, 3.99990400358e-14, 14.4249584703),
        (1.0, 0.2, 2.0, 0.493802499725, 0.0826180845751, 0.0106009446814),
        (1.0, 0.5, 0.01, 0.5, 1e-4, 3.68623165278),
        (1e-8, 1.0, 1.0, 5.00000000833e-9, 8.33333333333e-18, 18.420680744),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_uniform_denoise_matches_truncated_normal(
    high, x_t, s, mean, var, log_prob, dtype
):
    def t(v):
        return torch.tensor(v, dtype=dtype)

    dist = denoise(Uniform(t(0.0), t(high)), t(1.0), t(s), t(x_t))

    rel = 1e-4 if dtype == torch.float32 else 1e-8
    assert dist.mean.item() == pytest.approx(mean, rel=rel, abs=0)
    assert dist.variance.item() == pytest.approx(var, rel=rel, abs=0)
    if dtype == torch.float64:  # float32 cannot resolve x inside a 2e-7 wide mass
        assert dist.log_prob(dist.mean).item() == pytest.approx(
            log_prob, rel=rel, abs=0
        )
