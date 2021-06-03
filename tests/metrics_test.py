# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import numpy as np

import torch
from torch.distributions import MultivariateNormal as tmvn
from sbi.utils.metrics import c2st

from sklearn.model_selection import KFold, cross_val_score
# from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from torch import Tensor


def rv_c2st(
    X: Tensor,
    Y: Tensor,
    seed: int = 1,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
        verbosity=0
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

    X = X.cpu().numpy()
    Y = Y.cpu().numpy()

    ndim = X.shape[1]

    clf = RandomForestClassifier(
        # n_estimators=100,
     #   max_depth=None,
        random_state=seed,
    )

    # clf = MLPClassifier(
    #     activation="relu",
    #     hidden_layer_sizes=(10 * ndim, 10 * ndim),
    #     max_iter=1000,
    #     solver="adam",
    #     random_state=seed,
    # )

    #prepare data
    data = np.concatenate((X, Y))
    #labels
    target = np.concatenate(
        (
            np.zeros((X.shape[0],)),
            np.ones((Y.shape[0],)),
        )
    )

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=scoring,
                             verbose=verbosity)

    scores = np.asarray(np.mean(scores)).astype(np.float32)
    return torch.from_numpy(np.atleast_1d(scores))


def test_same_distributions_rv():

    ndim = 5
    nsamples = 4048

    xnormal = tmvn(loc=torch.zeros(5),covariance_matrix=torch.eye(5))

    X = xnormal.sample((nsamples,))
    Y = xnormal.sample((nsamples,))

    obs_c2st = rv_c2st(X, Y, verbosity=2)

    assert obs_c2st != None
    print(obs_c2st)


def test_diff_distributions_rv():

    ndim = 5
    nsamples = 4048

    xnormal = tmvn(loc=torch.zeros(5),covariance_matrix=torch.eye(5))
    ynormal = tmvn(loc=20.*torch.ones(5),covariance_matrix=torch.eye(5))

    X = xnormal.sample((nsamples,))
    Y = ynormal.sample((nsamples,))

    obs_c2st = rv_c2st(X, Y)

    assert obs_c2st != None
    print(obs_c2st)

def test_distributions_overlap_by_two_sigma_rv():

    ndim = 5
    nsamples = 4048

    xnormal = tmvn(loc=torch.zeros(5),covariance_matrix=torch.eye(5))
    ynormal = tmvn(loc=1.*torch.ones(5),covariance_matrix=torch.eye(5))

    X = xnormal.sample((nsamples,))
    Y = ynormal.sample((nsamples,))

    obs_c2st = rv_c2st(X, Y)

    assert obs_c2st != None
    print(obs_c2st)
    assert .8 < obs_c2st[0] #distributions do not overlap, classifiers label with high accuracy

def test_same_distributions_default():

    ndim = 5
    nsamples = 4048

    xnormal = tmvn(loc=torch.zeros(5),covariance_matrix=torch.eye(5))

    X = xnormal.sample((nsamples,))
    Y = xnormal.sample((nsamples,))

    obs_c2st = c2st(X, Y)

    assert obs_c2st != None
    assert .49 < obs_c2st[0] < .51 #only by chance we differentiate the 2 samples


def test_diff_distributions_default():

    ndim = 5
    nsamples = 4048

    xnormal = tmvn(loc=torch.zeros(5),covariance_matrix=torch.eye(5))
    ynormal = tmvn(loc=20.*torch.ones(5),covariance_matrix=torch.eye(5))

    X = xnormal.sample((nsamples,))
    Y = ynormal.sample((nsamples,))

    obs_c2st = c2st(X, Y)

    assert obs_c2st != None
    print(obs_c2st)
    assert .98 < obs_c2st[0] #distributions do not overlap, classifiers label with high accuracy


def test_distributions_overlap_by_two_sigma_default():

    ndim = 5
    nsamples = 4048

    xnormal = tmvn(loc=torch.zeros(5),covariance_matrix=torch.eye(5))
    ynormal = tmvn(loc=1.*torch.ones(5),covariance_matrix=torch.eye(5))

    X = xnormal.sample((nsamples,))
    Y = ynormal.sample((nsamples,))

    obs_c2st = c2st(X, Y)

    assert obs_c2st != None
    print(obs_c2st)
    assert .8 < obs_c2st[0] #distributions do not overlap, classifiers label with high accuracy
