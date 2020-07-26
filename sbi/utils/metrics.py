# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import numpy as np
import torch

from typing import Callable, Optional, Union, Dict, Any, Tuple, Union, cast, List, Sequence, TypeVar
from torch import Tensor

from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier


def c2st(
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
        noise_scale: If passed, will add Gaussian noise with std noise_scale to samples

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

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * ndim, 10 * ndim),
        max_iter=1000,
        solver="adam",
        random_state=seed,
    )

    data = np.concatenate((X, Y))
    target = np.concatenate((np.zeros((X.shape[0],)), np.ones((Y.shape[0],)),))

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=scoring)

    scores = np.asarray(np.mean(scores)).astype(np.float32)
    return torch.from_numpy(np.atleast_1d(scores))


def unbiased_mmd_squared(x, y):
    nx, ny = x.shape[0], y.shape[0]

    def f(a, b, diag=False):
        if diag:
            return torch.sum((a[None, ...] - b[:, None, :]) ** 2, dim=-1).reshape(-1)
        else:
            m, n = a.shape[0], b.shape[0]
            ix = torch.tril_indices(m, n, offset=-1)
            return torch.sum(
                (a[None, ...] - b[:, None, :]) ** 2, dim=-1, keepdim=False
            )[ix[0, :], ix[1, :]].reshape(-1)

    xx = f(x, x)
    xy = f(x, y, diag=True)
    yy = f(y, y)

    scale = torch.median(torch.sqrt(torch.cat((xx, xy, yy))))
    c = -0.5 / (scale ** 2)

    k = lambda a: torch.sum(torch.exp(c * a))

    kxx = k(xx) / (nx * (nx - 1))
    kxy = k(xy) / (nx * ny)
    kyy = k(yy) / (ny * (ny - 1))
    del xx, xy, yy

    mmd_square = 2 * (kxx + kyy - kxy)
    del kxx, kxy, kyy

    return mmd_square


def biased_mmd(x, y):
    nx, ny = x.shape[0], y.shape[0]

    def f(a, b):
        return torch.sum((a[None, ...] - b[:, None, :]) ** 2, dim=-1).reshape(-1)

    xx = f(x, x)
    xy = f(x, y)
    yy = f(y, y)

    scale = torch.median(torch.sqrt(torch.cat((xx, xy, yy))))
    c = -0.5 / (scale ** 2)

    k = lambda a: torch.sum(torch.exp(c * a))

    kxx = k(xx) / nx ** 2
    del xx
    kxy = k(xy) / (nx * ny)
    del xy
    kyy = k(yy) / ny ** 2
    del yy

    mmd_square = kxx - 2 * kxy + kyy
    del kxx, kxy, kyy

    return torch.sqrt(mmd_square)


def biased_mmd_hypothesis_test(x, y, alpha=0.05):
    assert x.shape[0] == y.shape[0]
    mmd_biased = biased_mmd(x, y).item()
    threshold = np.sqrt(2 / x.shape[0]) * (1 + np.sqrt(-2 * np.log(alpha)))

    return mmd_biased, threshold


def unbiased_mmd_squared_hypothesis_test(x, y, alpha=0.05):
    assert x.shape[0] == y.shape[0]
    mmd_square_unbiased = unbiased_mmd_squared(x, y).item()
    threshold = (4 / np.sqrt(x.shape[0])) * np.sqrt(-np.log(alpha))

    return mmd_square_unbiased, threshold


def _test():
    n = 2500
    x, y = torch.randn(n, 5), torch.randn(n, 5)
    print(unbiased_mmd_squared(x, y), biased_mmd(x, y))
    # mmd(x, y), sq_maximum_mean_discrepancy(tensor2numpy(x), tensor2numpy(y))
    # mmd_hypothesis_test(x, y, alpha=0.0001)
    # unbiased_mmd_squared_hypothesis_test(x, y)


def main():
    _test()


if __name__ == "__main__":
    main()
