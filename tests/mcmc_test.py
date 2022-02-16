# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import Union

import numpy as np
import pytest
import torch
from torch import eye, ones, zeros

from sbi.samplers.mcmc.slice_numpy import (
    SliceSampler,
    SliceSamplerVectorized,
    slice_np_parallized,
)
from sbi.simulators.linear_gaussian import true_posterior_linear_gaussian_mvn_prior
from tests.test_utils import check_c2st


@pytest.mark.parametrize("num_dim", (1, 2))
def test_c2st_slice_np_on_Gaussian(num_dim: int):
    """Test MCMC on Gaussian, comparing to ground truth target via c2st.

    Args:
        num_dim: parameter dimension of the gaussian model

    """
    warmup = 100
    num_samples = 500

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    x_o = zeros((1, num_dim))
    target_distribution = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = target_distribution.sample((num_samples,))

    def lp_f(x):
        return target_distribution.log_prob(torch.as_tensor(x, dtype=torch.float32))

    sampler = SliceSampler(lp_f=lp_f, x=np.zeros((num_dim,)).astype(np.float32))
    _ = sampler.gen(warmup)
    samples = sampler.gen(num_samples)

    samples = torch.as_tensor(samples, dtype=torch.float32)

    check_c2st(samples, target_samples, alg=f"slice_np")


@pytest.mark.parametrize("num_dim", (1, 2))
def test_c2st_slice_np_vectorized_on_Gaussian(num_dim: int):
    """Test MCMC on Gaussian, comparing to ground truth target via c2st.

    Args:
        num_dim: parameter dimension of the gaussian model

    """
    num_samples = 500
    warmup = 500
    num_chains = 5

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    x_o = zeros((1, num_dim))
    target_distribution = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = target_distribution.sample((num_samples,))

    def lp_f(x):
        return target_distribution.log_prob(torch.as_tensor(x, dtype=torch.float32))

    sampler = SliceSamplerVectorized(
        log_prob_fn=lp_f,
        init_params=np.zeros(
            (
                num_chains,
                num_dim,
            )
        ).astype(np.float32),
        num_chains=num_chains,
    )
    samples = sampler.run(warmup + int(num_samples / num_chains))
    samples = samples[:, warmup:, :]
    samples = samples.reshape(-1, num_dim)

    samples = torch.as_tensor(samples, dtype=torch.float32)

    check_c2st(samples, target_samples, alg="slice_np_vectorized")


@pytest.mark.parametrize("vectorized", (False, True))
@pytest.mark.parametrize("num_workers", (1, 10))
@pytest.mark.parametrize("seed", (None, 42))
def test_c2st_slice_np_parallelized(
    vectorized: bool, num_workers: int, seed: Union[None, int]
):
    """Test MCMC on Gaussian, comparing to ground truth target via c2st.

    Args:
        num_dim: parameter dimension of the gaussian model

    """
    num_dim = 2
    num_samples = 500
    warmup = 500
    num_chains = 5

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    x_o = zeros((1, num_dim))
    target_distribution = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = target_distribution.sample((num_samples,))

    def lp_f(x):
        return target_distribution.log_prob(torch.as_tensor(x, dtype=torch.float32))

    initial_params = torch.zeros((num_chains, num_dim))

    # Maybe test seeding.
    if seed is not None:
        torch.manual_seed(seed)

    samples = slice_np_parallized(
        lp_f,
        initial_params,
        num_samples,
        thin=1,
        warmup_steps=warmup,
        vectorized=vectorized,
        num_workers=num_workers,
        show_progress_bars=False,
    )
    # Repeat to test seeding.
    if seed is not None:
        torch.manual_seed(seed)
        samples_2 = slice_np_parallized(
            lp_f,
            initial_params,
            num_samples,
            thin=1,
            warmup_steps=warmup,
            vectorized=vectorized,
            num_workers=num_workers,
        )
        # Test seeding.
        assert torch.allclose(samples, samples_2)

    samples = torch.as_tensor(samples, dtype=torch.float32).reshape(-1, num_dim)

    check_c2st(
        samples, target_samples, alg=f"slice_np {'vectorized' if vectorized else ''}"
    )
