import pytest
import torch
from torch import zeros, ones, eye
from torch.distributions import MultivariateNormal

import sbi.utils as utils
from sbi.inference.snpe.snpe_b import SnpeB
from sbi.inference.snl import SNL
from sbi.inference.snpe.snpe_c import SnpeC
from sbi.simulators.linear_gaussian import (
    get_true_posterior_samples_linear_gaussian_mvn_prior,
    get_true_posterior_samples_linear_gaussian_uniform_prior,
    diagonal_linear_gaussian,
)
from sbi.user_input.user_input_checks import prepare_sbi_problem

# Use cpu by default.
torch.set_default_tensor_type("torch.FloatTensor")


def test_snpe_on_linearGaussian_based_on_mmd(
    num_dim: int, prior_str: str, algorithm_str: str, simulation_batch_size: int,
):
    """Test whether SNPE B/C infer well a simple example with available round truth.

    This test is seeded using the set_seed fixture defined in tests/conftest.py.

    Args:
        set_seed: fixture for manual seeding, see tests/conftest.py
    """

    x_o = zeros(1, num_dim)
    num_samples = 100

    if prior_str == "gaussian":
        prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
        target_samples = get_true_posterior_samples_linear_gaussian_mvn_prior(
            x_o, num_samples=num_samples
        )
    else:
        prior = utils.BoxUniform(-1.0 * ones(num_dim), ones(num_dim))
        target_samples = get_true_posterior_samples_linear_gaussian_uniform_prior(
            x_o, num_samples=num_samples, prior=prior
        )

    simulator, prior, x_o = prepare_sbi_problem(diagonal_linear_gaussian, prior, x_o)

    snpe_common_args = dict(
        simulator=simulator,
        x_o=x_o,
        prior=prior,
        simulation_batch_size=simulation_batch_size,
    )

    if algorithm_str == "snpe_b":
        infer = SnpeB(**snpe_common_args)
    elif algorithm_str == "snpe_c":
        infer = SnpeC(
            show_progressbar=True, show_round_summary=True, **snpe_common_args,
        )

    posterior = infer(num_rounds=2, num_simulations_per_round=1000, max_num_epochs=None)  # type: ignore
    samples = posterior.sample(num_samples)

    # Compute the mmd, and check if larger than expected
    mmd = utils.unbiased_mmd_squared(target_samples, samples)
    max_mmd = 0.03

    print(f"mmd for {algorithm_str} is {mmd}.")

    assert (
        mmd < max_mmd
    ), f"MMD={mmd} is more than 2 stds above the average performance."


test_snpe_on_linearGaussian_based_on_mmd(
    1, "gaussian", "snpe_c", simulation_batch_size=50
)
