import pytest
import torch

from sbi.inference import SnpeC, SRE, SNL
from torch import zeros, ones

import sbi.utils as utils

from sbi.simulators.linear_gaussian import (
    get_true_posterior_samples_linear_gaussian_uniform_prior,
    linear_gaussian,
)

from sbi.utils.sbiutils import find_nan_in_simulations


@pytest.mark.parametrize(
    "x_shape",
    (
        torch.Size((1, 1)),
        torch.Size((1, 10)),
        torch.Size((10, 1)),
        torch.Size((10, 10)),
    ),
)
def test_find_nan_in_simulations(x_shape, set_seed):

    x = torch.rand(x_shape)
    x[x < 0.5] = float("nan")

    x_is_nan = find_nan_in_simulations(x)

    assert torch.isfinite(x[~x_is_nan]).all()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("method", "handle_nans", "percent_nans"),
    (
        (SnpeC, True, 0.2),
        (SNL, True, 0.2),
        (SRE, True, 0.2),
        pytest.param(
            SNL,
            False,
            0.1,
            marks=pytest.mark.xfail(
                raises=AssertionError, reason="Handling NaNs is set False.",
            ),
        ),
        pytest.param(
            SNL,
            False,
            0.1,
            marks=pytest.mark.xfail(
                raises=AssertionError, reason="Handling NaNs is set False",
            ),
        ),
    ),
)
def test_inference_with_nan_simulator(method, handle_nans, percent_nans, set_seed):
    def linear_gaussian_nan(theta):
        x = linear_gaussian(theta)
        # Set nan randomly.
        x[torch.rand(x.shape) < (1.0 / x.shape[1]) * percent_nans] = float("nan")

        return x

    num_dim = 3
    x_o = zeros(1, num_dim)
    num_samples = 100

    prior = utils.BoxUniform(-1.0 * ones(num_dim), ones(num_dim))
    target_samples = get_true_posterior_samples_linear_gaussian_uniform_prior(
        x_o, num_samples=num_samples, prior=prior
    )

    if method == SnpeC:
        infer = method(simulator=linear_gaussian_nan, x_o=x_o, prior=prior)
    else:
        infer = method(
            simulator=linear_gaussian_nan, x_o=x_o, prior=prior, handle_nans=handle_nans
        )

    posterior = infer(num_rounds=2, num_simulations_per_round=500)
    samples = posterior.sample(num_samples)

    # Compute the mmd, and check if larger than expected
    mmd = utils.unbiased_mmd_squared(target_samples, samples)
    max_mmd = 0.03

    print(f"mmd for nan-linear-gaussian with SnpC is {mmd}.")

    assert (
        mmd < max_mmd
    ), f"MMD={mmd} is more than 2 stds above the average performance."
