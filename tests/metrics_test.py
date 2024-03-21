# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.neural_network import MLPClassifier
from torch.distributions import MultivariateNormal as tmvn

from sbi.utils.metrics import c2st, c2st_scores, wasserstein_2_squared

## c2st related:
## for a study about c2st see https://github.com/psteinb/c2st/

TESTCASECONFIG = [
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
    TESTCASECONFIG,
)
def test_c2st_with_different_distributions(
    dist_sigma, c2st_lowerbound, c2st_upperbound
):
    ndim = 10
    nsamples = 1024

    refdist = tmvn(loc=torch.zeros(ndim), covariance_matrix=torch.eye(ndim))
    otherdist = tmvn(
        loc=dist_sigma + torch.zeros(ndim), covariance_matrix=torch.eye(ndim)
    )

    X = refdist.sample((nsamples,))
    Y = otherdist.sample((nsamples,))

    obs_c2st = c2st(X, Y)

    assert len(obs_c2st) > 0
    assert c2st_lowerbound < obs_c2st[0]
    assert obs_c2st[0] <= c2st_upperbound


@pytest.mark.slow
@pytest.mark.parametrize(
    "dist_sigma, c2st_lowerbound, c2st_upperbound,",
    TESTCASECONFIG,
)
def test_c2st_with_different_distributions_mlp(
    dist_sigma, c2st_lowerbound, c2st_upperbound
):
    ndim = 10
    nsamples = 1024

    refdist = tmvn(loc=torch.zeros(ndim), covariance_matrix=torch.eye(ndim))
    otherdist = tmvn(
        loc=dist_sigma + torch.zeros(ndim), covariance_matrix=torch.eye(ndim)
    )

    X = refdist.sample((nsamples,))
    Y = otherdist.sample((nsamples,))

    obs_c2st = c2st(X, Y, classifier="mlp")

    assert len(obs_c2st) > 0
    assert c2st_lowerbound < obs_c2st[0]
    assert obs_c2st[0] <= c2st_upperbound


@pytest.mark.slow
@pytest.mark.parametrize(
    "dist_sigma, c2st_lowerbound, c2st_upperbound,",
    TESTCASECONFIG,
)
def test_c2st_scores(dist_sigma, c2st_lowerbound, c2st_upperbound):
    ndim = 10
    nsamples = 1024

    xnormal = tmvn(loc=torch.zeros(ndim), covariance_matrix=torch.eye(ndim))
    ynormal = tmvn(
        loc=dist_sigma + torch.zeros(ndim), covariance_matrix=torch.eye(ndim)
    )

    X = xnormal.sample((nsamples,))
    Y = ynormal.sample((nsamples,))

    obs_c2st = c2st_scores(X, Y)

    assert hasattr(obs_c2st, "mean")
    assert c2st_lowerbound < obs_c2st.mean()
    assert obs_c2st.mean() <= c2st_upperbound

    clf_class = MLPClassifier
    clf_kwargs = {
        "activation": "relu",
        "hidden_layer_sizes": (8 * X.shape[1], X.shape[1]),
        "max_iter": 100,
        "solver": "adam",
        "early_stopping": True,
        "n_iter_no_change": 20,
    }

    obs2_c2st = c2st_scores(X, Y, clf_class=clf_class, clf_kwargs=clf_kwargs)

    assert hasattr(obs2_c2st, "mean")
    assert c2st_lowerbound < obs2_c2st.mean()
    assert obs2_c2st.mean() <= c2st_upperbound

    assert np.allclose(obs2_c2st, obs_c2st, atol=0.05)


def test_wasserstein_2_distance():
    ndim = 10
    nsamples = 1024
    refdist = tmvn(loc=torch.zeros(ndim), covariance_matrix=torch.eye(ndim))
    X = refdist.sample((nsamples,))

    dist_sigmas = [0.0, 1.0, 20.0]
    w2_prev = 0.0
    for dist_sigma in dist_sigmas:
        otherdist = tmvn(
            loc=dist_sigma + torch.zeros(ndim), covariance_matrix=torch.eye(ndim)
        )
        Y = otherdist.sample((nsamples - 1,))

        w2 = wasserstein_2_squared(X, Y)

        assert w2 > w2_prev
        w2_prev = w2
