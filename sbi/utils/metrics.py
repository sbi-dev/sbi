# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Optional

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from torch import Tensor

from sbi.utils import tensor2numpy


def c2st(
    X: Tensor,
    Y: Tensor,
    seed: int = 1,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
    verbosity: int = 0,
    clf_class=RandomForestClassifier,
    clf_kwargs={},
) -> Tensor:
    """
    Return accuracy of classifier trained to distinguish samples from supposedly
    two distributions <X> and <Y>. For details on the method, see [1,2].
    If the returned accuracy is 0.5, <X> and <Y> are considered to be from the
    same generating PDF, i.e. they can not be differentiated.
    If the returned accuracy is around 1., <X> and <Y> are considered to be from
    two different generating PDFs.

    Training of the classifier with N-fold cross-validation [3] using sklearn.
    By default, a `RandomForestClassifier` by from `sklearn.ensemble` is used.
    This can be changed using <clf_class>.

    ``` py
    clf = clf_class(random_state=seed, **clf_kwargs)
    #...
    scores = cross_val_score(
        clf, data, target, cv=shuffle, scoring=scoring, verbose=verbosity
    )
    ```
    Further configuration of the classifier can be performed using <clf_kwargs>.

    Args:
        X: Samples from one distribution.
        Y: Samples from another distribution.
        seed: Seed for the sklearn classifier and the KFold cross-validation
        n_folds: Number of folds to use
        z_score: Z-scoring using X, i.e. mean and std deviation of X is used to normalize Y using (Y - mean)/std
        noise_scale: If passed, will add Gaussian noise with std noise_scale to samples of X and of Y
        verbosity: control the verbosity of sklearn.model_selection.cross_val_score
        clf_class: a scikit-learn classifier class
        clf_kwargs: key-value arguments dictionary to the class specified by clf_class, e.g. sklearn.ensemble.RandomForestClassifier

    Return:
        torch.tensor offering the mean accuracy score over the test sets
        from cross-validation

    Example:
    ``` py
    > c2st(X,Y)
    [0.51904464]
    #X and Y likely come from the same PDF or ensemble
    ```

    References:
        [1]: http://arxiv.org/abs/1610.06545
        [2]: https://www.osti.gov/biblio/826696/
        [3]: https://scikit-learn.org/stable/modules/cross_validation.html
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

    clf = clf_class(random_state=seed, **clf_kwargs)

    # prepare data
    data = np.concatenate((X, Y))
    # labels
    target = np.concatenate((np.zeros((X.shape[0],)), np.ones((Y.shape[0],))))

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(
        clf, data, target, cv=shuffle, scoring=scoring, verbose=verbosity
    )

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
