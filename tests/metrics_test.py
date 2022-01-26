# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from torch import Tensor
from torch.distributions import MultivariateNormal as tmvn

from sbi.utils.metrics import c2st

## c2st related:
## for a study about c2st see https://github.com/psteinb/c2st/


def nn_c2st(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int = 1,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
    verbosity: int = 0,
) -> np.ndarray:

    ndim = X.shape[1]
    clf_class = MLPClassifier
    clf_kwargs = {
        "activation": "relu",
        "hidden_layer_sizes": (10 * ndim, 10 * ndim),
        "max_iter": 1000,
        "solver": "adam",
    }

    return c2st(
        X,
        Y,
        seed,
        n_folds,
        scoring,
        z_score,
        noise_scale,
        verbosity,
        clf_class,
        clf_kwargs,
    )


def test_same_distributions(set_seed):

    ndim = 10
    nsamples = 1024

    xnormal = tmvn(loc=torch.zeros(ndim), covariance_matrix=torch.eye(ndim))

    X = xnormal.sample((nsamples,))
    Y = xnormal.sample((nsamples,))

    obs_c2st = c2st(X, Y, seed=set_seed)

    assert obs_c2st != None
    assert 0.45 < obs_c2st[0] < 0.55  # only by chance we differentiate the 2 samples


def test_diff_distributions(set_seed):

    ndim = 10
    nsamples = 1024

    xnormal = tmvn(loc=torch.zeros(ndim), covariance_matrix=torch.eye(ndim))
    ynormal = tmvn(loc=20.0 * torch.ones(ndim), covariance_matrix=torch.eye(ndim))

    X = xnormal.sample((nsamples,))
    Y = ynormal.sample((nsamples,))

    obs_c2st = c2st(X, Y, seed=set_seed)

    assert obs_c2st != None
    assert (
        0.98 < obs_c2st[0]
    )  # distributions do not overlap, classifiers label with high accuracy
    print(obs_c2st)


def test_onesigma_apart_distributions(set_seed):

    ndim = 10
    nsamples = 1024

    xnormal = tmvn(loc=torch.zeros(ndim), covariance_matrix=torch.eye(ndim))
    ynormal = tmvn(loc=torch.ones(ndim), covariance_matrix=torch.eye(ndim))

    X = xnormal.sample((nsamples,))
    Y = ynormal.sample((nsamples,))

    obs_c2st = c2st(X, Y, seed=set_seed))

    assert obs_c2st != None
    print(obs_c2st)
    assert (
        0.85 < obs_c2st[0]
    )  # distributions do not overlap, classifiers label with high accuracy


@pytest.mark.slow
def test_same_distributions_mlp(set_seed):

    ndim = 10
    nsamples = 1024

    xnormal = tmvn(loc=torch.zeros(ndim), covariance_matrix=torch.eye(ndim))

    X = xnormal.sample((nsamples,))
    Y = xnormal.sample((nsamples,))

    obs_c2st = nn_c2st(X, Y, seed=set_seed))

    assert obs_c2st != None
    assert 0.45 < obs_c2st[0] < 0.55  # only by chance we differentiate the 2 samples


@pytest.mark.slow
def test_diff_distributions_flexible_mlp(set_seed):

    ndim = 10
    nsamples = 1024

    xnormal = tmvn(loc=torch.zeros(ndim), covariance_matrix=torch.eye(ndim))
    ynormal = tmvn(loc=20.0 * torch.ones(ndim), covariance_matrix=torch.eye(ndim))

    X = xnormal.sample((nsamples,))
    Y = ynormal.sample((nsamples,))

    obs_c2st = nn_c2st(X, Y, seed=set_seed)

    assert obs_c2st != None
    assert 0.95 < obs_c2st[0]

    clf_class = MLPClassifier
    clf_kwargs = {
        "activation": "relu",
        "hidden_layer_sizes": (10 * X.shape[1], 10 * X.shape[1]),
        "max_iter": 1000,
        "solver": "adam",
    }

    obs2_c2st = c2st(X, Y, seed=set_seed, clf_class=clf_class, clf_kwargs=clf_kwargs)

    assert obs2_c2st != None
    assert 0.95 < obs2_c2st[0]  # only by chance we differentiate the 2 samples
    assert np.allclose(obs2_c2st, obs_c2st)


@pytest.mark.slow
def test_onesigma_apart_distributions_mlp(set_seed):

    ndim = 10
    nsamples = 1024

    xnormal = tmvn(loc=torch.zeros(ndim), covariance_matrix=torch.eye(ndim))
    ynormal = tmvn(loc=torch.ones(ndim), covariance_matrix=torch.eye(ndim))

    X = xnormal.sample((nsamples,))
    Y = ynormal.sample((nsamples,))

    obs_c2st = nn_c2st(X, Y, seed=set_seed))

    assert obs_c2st != None
    print(obs_c2st)
    assert (
        0.8 < obs_c2st[0]
    )  # distributions do not overlap, classifiers label with high accuracy
