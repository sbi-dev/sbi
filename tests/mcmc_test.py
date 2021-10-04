# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from math import ceil

import numpy as np
import pytest
import torch
from torch import eye, ones, zeros

from sbi.samplers.mcmc.slice_numpy import (
    SliceSampler,
    SliceSamplerSerial,
    SliceSamplerVectorized,
)
from sbi.simulators.linear_gaussian import true_posterior_linear_gaussian_mvn_prior
from sbi.utils import tensor2numpy
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

    sampler = SliceSampler(
        lp_f=lp_f,
        x=np.zeros((num_dim,)).astype(np.float32),
        tuning=warmup,
    )
    warmup_samples = sampler.gen(warmup)
    assert warmup_samples.shape == (warmup, num_dim)

    samples = sampler.gen(num_samples)
    assert samples.shape == (num_samples, num_dim)

    samples = torch.as_tensor(samples, dtype=torch.float32)

    check_c2st(samples, target_samples, alg="slice_np")


@pytest.mark.parametrize("num_dim", (1, 2))
@pytest.mark.parametrize("slice_sampler", (SliceSamplerVectorized, SliceSamplerSerial))
def test_c2st_slice_np_vectorized_on_Gaussian(num_dim: int, slice_sampler):
    """Test MCMC on Gaussian, comparing to ground truth target via c2st.

    Args:
        num_dim: parameter dimension of the gaussian model

    """
    num_samples = 500
    warmup = 500
    num_chains = 5
    thin = 2

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

    sampler = slice_sampler(
        log_prob_fn=lp_f,
        init_params=np.zeros(
            (
                num_chains,
                num_dim,
            )
        ).astype(np.float32),
        tuning=warmup,
        thin=thin,
        num_chains=num_chains,
    )
    samples = sampler.run(thin * (warmup + int(num_samples / num_chains)))
    assert samples.shape == (
        num_chains,
        warmup + int(num_samples / num_chains),
        num_dim,
    )
    samples = samples[:, warmup:, :]
    samples = samples.reshape(-1, num_dim)

    samples = torch.as_tensor(samples, dtype=torch.float32)

    alg = {
        SliceSamplerVectorized: "slice_np_vectorized",
        SliceSamplerSerial: "slice_np",
    }[slice_sampler]

    check_c2st(samples, target_samples, alg=alg)


@pytest.mark.parametrize("vectorized", (False, True))
@pytest.mark.parametrize("num_workers", (1, 10))
def test_c2st_slice_np_parallelized(vectorized: bool, num_workers: int):
    """Test MCMC on Gaussian, comparing to ground truth target via c2st.

    Args:
        num_dim: parameter dimension of the gaussian model

    """
    num_dim = 2
    num_samples = 500
    warmup = 100
    num_chains = 10
    thin = 2

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

    if not vectorized:
        SliceSamplerMultiChain = SliceSamplerSerial
    else:
        SliceSamplerMultiChain = SliceSamplerVectorized

    posterior_sampler = SliceSamplerMultiChain(
        init_params=tensor2numpy(initial_params),
        log_prob_fn=lp_f,
        num_chains=num_chains,
        thin=thin,
        verbose=False,
        num_workers=num_workers,
    )
    warmup_ = warmup * thin
    num_samples_ = ceil((num_samples * thin) / num_chains)
    samples = posterior_sampler.run(warmup_ + num_samples_)  # chains x samples x dim
    samples = samples[:, warmup:, :]  # discard warmup steps
    samples = torch.as_tensor(samples, dtype=torch.float32).reshape(-1, num_dim)

    check_c2st(
        samples, target_samples, alg=f"slice_np {'vectorized' if vectorized else ''}"
    )
