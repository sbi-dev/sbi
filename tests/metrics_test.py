# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.distributions import MultivariateNormal as tmvn

from sbi.utils.metrics import (
    biased_mmd_hypothesis_test,
    c2st,
    posterior_shrinkage,
    posterior_zscore,
    unbiased_mmd_squared_hypothesis_test,
    wasserstein_2_squared,
)

## c2st related:
## for a study about c2st see https://github.com/psteinb/c2st/

C2ST_TESTCASECONFIG = [
    (
        # both samples are identical, the mean accuracy should be around 0.5
        0.0,  # dist_sigma
        0.45,  # c2st_lowerbound
        0.55,  # c2st_upperbound
    ),
    (
        # both samples are rather close, the mean accuracy should be larger than 0.5 and
        # be lower than 1.
        1.0,
        0.85,
        1.0,
    ),
    (
        # both samples are very far apart, the mean accuracy should close to 1.
        20.0,
        0.98,
        1.0,
    ),
]


@pytest.mark.parametrize(
    "dist_sigma, c2st_lowerbound, c2st_upperbound,",
    C2ST_TESTCASECONFIG,
)
@pytest.mark.parametrize("classifier", ("rf", "mlp"))
def test_c2st_with_different_distributions(
    dist_sigma, c2st_lowerbound, c2st_upperbound, classifier: str
):
    ndim = 10
    nsamples = 1024

    refdist = tmvn(loc=torch.zeros(ndim), covariance_matrix=torch.eye(ndim))
    otherdist = tmvn(
        loc=dist_sigma + torch.zeros(ndim), covariance_matrix=torch.eye(ndim)
    )

    X = refdist.sample((nsamples,))
    Y = otherdist.sample((nsamples,))

    obs_c2st = c2st(X, Y, classifier=classifier)

    assert c2st_lowerbound < obs_c2st <= c2st_upperbound


@pytest.mark.parametrize("dims_constant", (1, 2))
def test_c2st_with_constant_features(dims_constant: int):
    num_dim = 2
    num_samples = 1024
    x = torch.randn(num_samples, num_dim)
    y = torch.randn(num_samples, num_dim)
    x[:, :dims_constant] = 1.0
    y[:, :dims_constant] = 1.0

    c2st(x, y)


@pytest.mark.slow
@pytest.mark.parametrize(
    "sigma",
    (0.0, 5, 20.0),
)
def test_wasserstein_2_distance(sigma):
    ndim = 10
    nsamples = 1024
    refdist = tmvn(loc=torch.zeros(ndim), covariance_matrix=torch.eye(ndim))
    X = refdist.sample((nsamples,))

    # As we are only dealing with a diagonal covariance,
    #  the residual terms coming from the covariance cancel out.
    analytical_wasserstein_2_squared = (
        torch.norm(sigma * torch.ones(ndim)) ** 2
    ).item()

    otherdist = tmvn(loc=sigma + torch.zeros(ndim), covariance_matrix=torch.eye(ndim))
    Y = otherdist.sample((nsamples - 1,))
    estimate = wasserstein_2_squared(X, Y, epsilon=5e-4).item()

    # Check if the wasserstein estimate is of the same order
    # as the analytically derived squared Wasserstein-2 distance
    exponent1 = (
        0
        if analytical_wasserstein_2_squared == 0
        else int(math.floor(math.log10(abs(analytical_wasserstein_2_squared))))
    )
    exponent2 = 0 if estimate == 0 else int(math.floor(math.log10(abs(estimate))))
    assert exponent1 == exponent2


@pytest.mark.slow
@pytest.mark.parametrize(
    "test", (unbiased_mmd_squared_hypothesis_test, biased_mmd_hypothesis_test)
)
@pytest.mark.parametrize("sigma", (0.0, 5.0))
def test_mmd_squared_distance(test, sigma):
    ndim = 10
    nsamples = 1024
    ref_sigma = 0.0
    refdist = tmvn(loc=ref_sigma * torch.ones(ndim), covariance_matrix=torch.eye(ndim))
    X = refdist.sample((nsamples,))

    otherdist = tmvn(loc=sigma + torch.zeros(ndim), covariance_matrix=torch.eye(ndim))
    Y = otherdist.sample((nsamples,))

    estimate, threshold = test(X, Y, alpha=0.05)

    if sigma == ref_sigma:
        assert estimate < threshold, "Rejecting 0-hypothesis even though q=p."
    else:
        assert estimate > threshold, "Accepting 0-hypothesis even though q!=p."


@pytest.mark.parametrize(
    "prior_samples, post_samples, expected_shrinkage, raises_exception",
    [
        (np.array([2]), np.array([3]), None, False),
        (
            np.array([[1, 2], [2, 3]]),
            np.array([[2, 3], [3, 4]]),
            torch.tensor([0.0, 0.0]),
            False,
        ),
        (
            torch.tensor([[1.0, 2.0], [2.0, 3.0]]),
            torch.tensor([[2.0, 3.0], [3.0, 4.0]]),
            torch.tensor([0.0, 0.0]),
            False,
        ),
        (np.array([]), np.array([]), None, True),
    ],
)
def test_posterior_shrinkage(
    prior_samples, post_samples, expected_shrinkage, raises_exception
):
    if raises_exception:
        with pytest.raises(ValueError):
            posterior_shrinkage(prior_samples, post_samples)
    else:
        if expected_shrinkage is not None:
            assert torch.allclose(
                posterior_shrinkage(prior_samples, post_samples), expected_shrinkage
            )
        else:
            assert torch.isnan(posterior_shrinkage(prior_samples, post_samples)[0])


@pytest.mark.parametrize(
    "true_theta, post_samples, expected_zscore, raises_exception",
    [
        (
            np.array([2, 3]),
            np.array([[1, 2], [2, 3], [3, 4]]),
            torch.tensor([0.0, 0.0]),
            False,
        ),
        (
            torch.tensor([2.0, 3.0]),
            torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
            torch.tensor([0.0, 0.0]),
            False,
        ),
        (np.array([]), np.array([]), None, True),
    ],
)
def test_posterior_zscore(true_theta, post_samples, expected_zscore, raises_exception):
    if raises_exception:
        with pytest.raises(ValueError):
            posterior_zscore(true_theta, post_samples)
    else:
        assert torch.allclose(
            posterior_zscore(true_theta, post_samples), expected_zscore
        )
