# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from torch import Tensor
from torch.distributions import MultivariateNormal as tmvn

from sbi.utils.metrics import c2st


def old_c2st(
    X: Tensor,
    Y: Tensor,
    seed: int = 1,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
) -> Tensor:
    """Return accuracy of classifier trained to distinguish samples from two distributions.

    Trains classifiers with N-fold cross-validation [1]. Scikit learn MLPClassifier are
    used, with 2 hidden layers of 10x dim each, where dim is the dimensionality of the
    samples X and Y.
    Args:
        X: Samples from one distribution.
        Y: Samples from another distribution.
        seed: Seed for sklearn
        n_folds: Number of folds
        z_score: Z-scoring using X
        noise_scale: If passed, will add Gaussian noise with std noise_scale to samples of X and of Y

    References:
        [1]: https://scikit-learn.org/stable/modules/cross_validation.html
    """
    if z_score:
        X_mean = torch.mean(X, axis=0)
        X_std = torch.std(X, axis=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X += noise_scale * torch.randn(X.shape)
        Y += noise_scale * torch.randn(Y.shape)

    X = tensor2numpy(X)
    Y = tensor2numpy(Y)

    ndim = X.shape[1]

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * ndim, 10 * ndim),
        max_iter=1000,
        solver="adam",
        random_state=seed,
    )

    data = np.concatenate((X, Y))
    target = np.concatenate((np.zeros((X.shape[0],)), np.ones((Y.shape[0],))))

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=scoring)

    scores = np.asarray(np.mean(scores)).astype(np.float32)
    return torch.from_numpy(np.atleast_1d(scores))


def test_same_distributions_alt():

    ndim = 5
    nsamples = 4048

    xnormal = tmvn(loc=torch.zeros(ndim), covariance_matrix=torch.eye(ndim))

    X = xnormal.sample((nsamples,))
    Y = xnormal.sample((nsamples,))

    obs_c2st = new_c2st(X, Y)

    assert obs_c2st != None
    assert 0.49 < obs_c2st[0] < 0.51  # only by chance we differentiate the 2 samples
    print(obs_c2st)


def test_diff_distributions_alt():

    ndim = 5
    nsamples = 4048

    xnormal = tmvn(loc=torch.zeros(ndim), covariance_matrix=torch.eye(ndim))
    ynormal = tmvn(loc=20.0 * torch.ones(ndim), covariance_matrix=torch.eye(ndim))

    X = xnormal.sample((nsamples,))
    Y = ynormal.sample((nsamples,))

    obs_c2st = new_c2st(X, Y)

    assert obs_c2st != None
    assert (
        0.98 < obs_c2st[0]
    )  # distributions do not overlap, classifiers label with high accuracy
    print(obs_c2st)


def test_distributions_overlap_by_two_sigma_alt():

    ndim = 5
    nsamples = 4048

    xnormal = tmvn(loc=torch.zeros(ndim), covariance_matrix=torch.eye(ndim))
    ynormal = tmvn(loc=1.0 * torch.ones(ndim), covariance_matrix=torch.eye(ndim))

    X = xnormal.sample((nsamples,))
    Y = ynormal.sample((nsamples,))

    obs_c2st = new_c2st(X, Y)

    assert obs_c2st != None
    print(obs_c2st)
    assert (
        0.8 < obs_c2st[0]
    )  # distributions do not overlap, classifiers label with high accuracy


def test_same_distributions_default():

    ndim = 5
    nsamples = 4048

    xnormal = tmvn(loc=torch.zeros(ndim), covariance_matrix=torch.eye(ndim))

    X = xnormal.sample((nsamples,))
    Y = xnormal.sample((nsamples,))

    obs_c2st = old_c2st(X, Y)

    assert obs_c2st != None
    assert 0.49 < obs_c2st[0] < 0.51  # only by chance we differentiate the 2 samples


def test_same_distributions_default_flexible_alt():

    ndim = 5
    nsamples = 4048

    xnormal = tmvn(loc=torch.zeros(ndim), covariance_matrix=torch.eye(ndim))

    X = xnormal.sample((nsamples,))
    Y = xnormal.sample((nsamples,))

    obs_c2st = old_c2st(X, Y, seed=42)

    assert obs_c2st != None
    assert 0.49 < obs_c2st[0] < 0.51  # only by chance we differentiate the 2 samples

    clf_class = MLPClassifier
    clf_kwargs = {
        "activation": "relu",
        "hidden_layer_sizes": (10 * X.shape[1], 10 * X.shape[1]),
        "max_iter": 1000,
        "solver": "adam",
    }

    obs2_c2st = new_c2st(X, Y, seed=42, clf_class=clf_class, clf_kwargs=clf_kwargs)

    assert obs2_c2st != None
    assert 0.49 < obs2_c2st[0] < 0.51  # only by chance we differentiate the 2 samples
    assert np.allclose(obs2_c2st, obs_c2st)


def test_diff_distributions_default():

    ndim = 5
    nsamples = 4048

    xnormal = tmvn(loc=torch.zeros(ndim), covariance_matrix=torch.eye(ndim))
    ynormal = tmvn(loc=20.0 * torch.ones(ndim), covariance_matrix=torch.eye(ndim))

    X = xnormal.sample((nsamples,))
    Y = ynormal.sample((nsamples,))

    obs_c2st = old_c2st(X, Y)

    assert obs_c2st != None
    print(obs_c2st)
    assert (
        0.98 < obs_c2st[0]
    )  # distributions do not overlap, classifiers label with high accuracy


def test_distributions_overlap_by_two_sigma_default():

    ndim = 5
    nsamples = 4048

    xnormal = tmvn(loc=torch.zeros(ndim), covariance_matrix=torch.eye(ndim))
    ynormal = tmvn(loc=1.0 * torch.ones(ndim), covariance_matrix=torch.eye(ndim))

    X = xnormal.sample((nsamples,))
    Y = ynormal.sample((nsamples,))

    obs_c2st = old_c2st(X, Y)

    assert obs_c2st != None
    print(obs_c2st)
    assert (
        0.8 < obs_c2st[0]
    )  # distributions do not overlap, classifiers label with high accuracy
