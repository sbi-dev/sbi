# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import inspect
import warnings
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from torch import Tensor


class MLPClassifierModule(nn.Module):
    """PyTorch implementation of a fully-connected multilayer perceptron (MLP)"""

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: Sequence[int],
        output_dim: int = 1,
    ) -> None:
        """Initialize the MLP classifier.

        Args:
            input_dim: Number of input features.
            hidden_layer_sizes: Sequence of the number of hidden units in each
                layer.
            output_dim: Size of the output layer. Defaults to 1.
        """
        super().__init__()
        layers = []
        current_dim = input_dim
        for hdim in hidden_layer_sizes:
            layers.append(nn.Linear(current_dim, hdim))
            layers.append(nn.ReLU())
            current_dim = hdim
        layers.append(nn.Linear(current_dim, output_dim))
        self.sequential = nn.Sequential(*layers)

    def forward(self, X: Tensor) -> Tensor:
        """Perform a forward pass through the MLP.

        Args:
            X: Input features tensor of shape `(batch, input_dim)`.

        Returns:
            Output logits of shape `(batch, output_dim)`.
        """
        return self.sequential(X.float())


def c2st(
    X: Tensor,
    Y: Tensor,
    seed: int = 1,
    n_folds: int = 5,
    metric: str = "accuracy",
    classifier: Union[str, Callable] = "rf",
    classifier_kwargs: Optional[Dict[str, Any]] = None,
    z_score: bool = True,
    noise_scale: Optional[float] = None,
    verbosity: int = 0,
) -> Tensor:
    """
    Return classifier based two-sample test accuracy between X and Y.

    For details on the method, see [1,2]. If the returned accuracy is 0.5, <X>
    and <Y> are considered to be from the same generating PDF, i.e. they can not
    be differentiated. If the returned accuracy is around 1., <X> and <Y> are
    considered to be from two different generating PDFs.

    Training of the classifier with N-fold cross-validation [3] using sklearn.
    By default, a `RandomForestClassifier` by from `sklearn.ensemble` is used
    (<classifier> = 'rf'). Alternatively, a multi-layer perceptron is available
    (<classifier> = 'mlp'). For a small study on the pros and cons for this
    choice see [4].

    Note: Both set of samples are normalized (z scored) using the mean and std
    of the samples in <X>. If <z_score> is set to False, no normalization is
    done. If features in <X> are close to constant with std close to zero, the
    std is set to 1 to avoud division by zero.

    If you need a more flexible interface which is able to take a sklearn
    compatible classifier and more, see the `c2st_` method in this module.

    Args:
        X: Samples from one distribution. Y: Samples from another distribution.
        seed: Seed for the sklearn classifier and the KFold cross-validation
        n_folds: Number of folds to use metric: sklearn compliant metric to use
        for the scoring parameter of
            cross_val_score
        classifier: classification architecture to use. Defaults to "rf" for a
            RandomForestClassifier. Should be a sklearn classifier, or a
            Callable that behaves like one.
        z_score: Z-scoring using X, i.e. mean and std deviation of X is
            used to normalize X and Y, i.e. Y=(Y - mean)/std
        noise_scale: If passed, will add Gaussian noise with standard deviation
            <noise_scale> to samples of X and of Y
        verbosity: control the verbosity of
        sklearn.model_selection.cross_val_score

    Return:
        torch.tensor containing the mean accuracy score over the test sets from
        cross-validation

    Example: ``` py > c2st(X,Y) [0.51904464] #X and Y likely come from the same
    PDF or ensemble > c2st(P,Q) [0.998456] #P and Q likely come from two
    different PDFs or ensembles ```

    References:
        [1]: http://arxiv.org/abs/1610.06545 [2]:
        https://www.osti.gov/biblio/826696/ [3]:
        https://scikit-learn.org/stable/modules/cross_validation.html [4]:
        https://github.com/psteinb/c2st/
    """

    # the default configuration
    if classifier == "rf":
        clf_class = RandomForestClassifier
        clf_kwargs = classifier_kwargs or {}  # use sklearn defaults
    elif classifier == "mlp":
        ndim = X.shape[-1]
        clf_class = MLPClassifier
        # set defaults for the MLP
        clf_kwargs = classifier_kwargs or {
            "activation": "relu",
            "hidden_layer_sizes": (10 * ndim, 10 * ndim),
            "max_iter": 1000,
            "solver": "adam",
            "early_stopping": True,
            "n_iter_no_change": 50,
        }

    if z_score:
        X_mean = torch.mean(X, dim=0)
        X_std = torch.std(X, dim=0)
        # Set std to 1 if it is close to zero.
        X_std[X_std < 1e-14] = 1
        assert not torch.any(torch.isnan(X_mean)), "X_mean contains NaNs"
        assert not torch.any(torch.isnan(X_std)), "X_std contains NaNs"
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X += noise_scale * torch.randn(X.shape)
        Y += noise_scale * torch.randn(Y.shape)

    clf = clf_class(random_state=seed, **clf_kwargs)

    # prepare data, convert to numpy
    data = np.concatenate((X.cpu().numpy(), Y.cpu().numpy()))
    # labels
    target = np.concatenate((np.zeros((X.shape[0],)), np.ones((Y.shape[0],))))

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(
        clf, data, target, cv=shuffle, scoring=metric, verbose=verbosity
    )

    return torch.from_numpy(scores).mean()


def check_c2st(x: Tensor, y: Tensor, alg: str, tol: float = 0.1) -> None:
    """Compute classification based two-sample test accuracy and assert it close to
    chance."""

    score = c2st(x, y).item()

    assert (0.5 - tol) <= score <= (0.5 + tol), (
        f"{alg}'s c2st={score:.2f} is too far from the desired near-chance performance."
    )


def unbiased_mmd_squared(x: Tensor, y: Tensor, scale: Optional[float] = None):
    """Unbiased approximation of the squared maximum-mean discrepancy (MMD) [1].
    The sample-based MMD relies on kernel evaluations between x_i and y_i. This
    implementation only features a Gaussian kernel with lengthscale `scale`.

    Args:
        x: Data of shape (m, d)
        y: Data of shape (n, d)
        scale: Lengthscale of the exponential kernel. If not specified,
            the lengthscale is chosen based on a median heuristic.

    Return:
        A single scalar for the squared MMD.

    References:
        [1] Gretton, A., et al. (2012). A kernel two-sample test.
    """
    nx, ny = x.shape[0], y.shape[0]
    assert nx != 1 and ny != 1, (
        "The unbiased MMD estimator is not defined "
        "for empirical distributions of size 1."
    )

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

    s = torch.median(torch.sqrt(torch.cat((xx, xy, yy)))) if scale is None else scale
    # Add small epsilon to prevent division by zero when distributions are identical
    s = torch.clamp(s, min=1e-8) if isinstance(s, torch.Tensor) else max(s, 1e-8)
    c = -0.5 / (s**2)

    k = lambda a: torch.sum(torch.exp(c * a))

    kxx = k(xx) / (nx * (nx - 1))
    kxy = k(xy) / (nx * ny)
    kyy = k(yy) / (ny * (ny - 1))
    del xx, xy, yy

    mmd_square = 2 * (kxx + kyy - kxy)
    del kxx, kxy, kyy

    return mmd_square


def biased_mmd(x: Tensor, y: Tensor, scale: Optional[float] = None):
    """Biased approximation of the squared maximum-mean discrepancy (MMD) [1].
    The sample-based MMD relies on kernel evaluations between x_i and y_i. This
    implementation only features a Gaussian kernel with lengthscale `scale`.

    Args:
        x: Data of shape (m, d)
        y: Data of shape (n, d)
        scale: Lengthscale of the exponential kernel. If not specified,
            the lengthscale is chosen based on a median heuristic.

    Return:
        A single scalar for the squared MMD.

    References:
        [1] Gretton, A., et al. (2012). A kernel two-sample test.
    """
    nx, ny = x.shape[0], y.shape[0]

    def f(a, b):
        return torch.sum((a[None, ...] - b[:, None, :]) ** 2, dim=-1).reshape(-1)

    xx = f(x, x)
    xy = f(x, y)
    yy = f(y, y)

    s = torch.median(torch.sqrt(torch.cat((xx, xy, yy)))) if scale is None else scale
    c = -0.5 / (s**2)

    k = lambda a: torch.sum(torch.exp(c * a))

    kxx = k(xx) / nx**2
    del xx
    kxy = k(xy) / (nx * ny)
    del xy
    kyy = k(yy) / ny**2
    del yy

    mmd_square = kxx - 2 * kxy + kyy
    del kxx, kxy, kyy

    return torch.sqrt(mmd_square)


def biased_mmd_hypothesis_test(x: Tensor, y: Tensor, alpha=0.05):
    assert x.shape[0] == y.shape[0]
    mmd_biased = biased_mmd(x, y).item()
    threshold = np.sqrt(2 / x.shape[0]) * (1 + np.sqrt(-2 * np.log(alpha)))

    return mmd_biased, threshold


def unbiased_mmd_squared_hypothesis_test(x: Tensor, y: Tensor, alpha=0.05):
    assert x.shape[0] == y.shape[0]
    mmd_square_unbiased = unbiased_mmd_squared(x, y).item()
    threshold = (4 / np.sqrt(x.shape[0])) * np.sqrt(-np.log(alpha))

    return mmd_square_unbiased, threshold


def wasserstein_2_squared(
    x: Tensor,
    y: Tensor,
    *,
    relative_epsilon: float = 1e-2,
    max_iter: int = 5000,
    tol: float = 1e-3,
):
    """Approximate the squared 2-Wasserstein distance
    using entropic regularized optimal transport [1]. In the limit
    'relative_epsilon' to 0, the squared Wasserstein-2 distance is recovered.

    Args:
        x: Data of shape (B, m, d) or (m, d)
        y: Data of shape (B, n, d) or (n, d)
        relative_epsilon: Entropic regularization, relative to the mean squared
            distance between the points of x and y. See `regularized_ot_dual`.
        max_iter: Maximum number of Sinkhorn iterations. See `regularized_ot_dual`.
        tol: Tolerance on the coupling's marginals. See `regularized_ot_dual`.

    Return:
        The squared 2-Wasserstein distance of shape (B, ) or ()

    References:
        [1] Peyré, G., & Cuturi, M. (2019). Computational optimal transport:
            With applications to data science.
    """
    assert x.ndim == y.ndim, (
        "Please make sure that 'x' and 'y' are both either batched or not."
    )
    if x.ndim == 2:
        nx, ny = x.shape[0], y.shape[0]
        a = torch.ones(nx, dtype=x.dtype, device=x.device) / nx
        b = torch.ones(ny, dtype=x.dtype, device=x.device) / ny
    elif x.ndim == 3:
        batch_size = x.shape[0]
        nx, ny = x.shape[1], y.shape[1]
        a = torch.ones((batch_size, nx), dtype=x.dtype, device=x.device) / nx
        b = torch.ones((batch_size, ny), dtype=x.dtype, device=x.device) / ny
    else:
        raise ValueError(
            "This implementation of Wasserstein is only implemented, "
            "if x.ndim=2 or x.ndim=3."
        )

    # Evaluate the cost matrix based on the default l2 cost
    cost_matrix = torch.cdist(x, y, 2) ** 2

    coupling = regularized_ot_dual(
        a,
        b,
        cost_matrix,
        relative_epsilon=relative_epsilon,
        max_iter=max_iter,
        tol=tol,
    )
    if a.ndim == 1:
        return torch.sum(coupling * cost_matrix)
    else:
        return torch.sum(coupling * cost_matrix, dim=(1, 2))


def regularized_ot_dual(
    a: Tensor,
    b: Tensor,
    cost: Tensor,
    *,
    relative_epsilon: float = 1e-2,
    max_iter: int = 5000,
    tol: float = 1e-3,
):
    """Implementation of regularized optimal transport based on
    the dual formulation of the regularized optimal transport problem.

    Args:
        a: Probability vector of the empirical distribution x,
            either in batched form (B, m) or as a single vector (m,).
        b: Probability vector of the empirical distribution y,
            either in batched form (B, n) or as a single vector (n,).
        cost: Cost-matrix between the empirical samples of x and y.
            Either in batched form (B, m, n) or as a matrix (m, n).
        relative_epsilon: Entropic regularization, relative to the mean of each
            cost matrix, so the result does not depend on the scale of the cost.
            The cost has to be nonnegative. Smaller values bias the result less
            but need more iterations.
        max_iter: Maximum number of iterations. A warning is raised if Sinkhorn
            has not converged by then.
        tol: Tolerance on the L1 error of the coupling's row marginals; the column
            marginals are exact after each update.

    Return:
        Optimal transport coupling of shape (B, m, n) or (m, n)
    """

    assert a.ndim == b.ndim, (
        "Please make sure that 'a' and 'b' are both either batched or not, "
        f"currently a.ndim={a.ndim} and b.ndim={b.ndim}"
    )

    batched = True
    if a.ndim == 1 and b.ndim == 1:
        batched = False
        a = torch.atleast_2d(a)
        b = torch.atleast_2d(b)
        cost = cost.unsqueeze(0)

    dtype = torch.promote_types(torch.promote_types(cost.dtype, a.dtype), b.dtype)
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    a, b, cost = a.to(dtype), b.to(dtype), cost.to(dtype)
    finfo = torch.finfo(dtype)

    epsilon = relative_epsilon * cost.mean(dim=(1, 2), keepdim=True)
    epsilon = epsilon.clamp(min=finfo.tiny)

    # Define potentials
    f, g = torch.zeros_like(a), torch.zeros_like(b)

    def s(f, g):
        return (cost - f.unsqueeze(2) - g.unsqueeze(1)) / epsilon

    terminated = torch.zeros(a.shape[0], dtype=torch.bool, device=a.device)
    for i in range(max_iter + 1):
        # The column marginals are exact after a g-update, so only the row
        # marginals are checked, from the first update on.
        log_row_marginal = torch.logsumexp(-s(f, g), dim=2)
        err = (log_row_marginal.exp() - a).abs().sum(dim=1)
        if i > 0:
            terminated = torch.logical_or(terminated, err < tol)
        if torch.all(terminated):
            break
        if i == max_iter:
            warnings.warn(
                f"Sinkhorn iterations did not converge within {max_iter} iterations. "
                "Consider a bigger 'relative_epsilon', increasing 'max_iter', "
                "or float64.",
                stacklevel=2,
            )
            break
        f_tmp = f + epsilon[..., 0] * (torch.log(a) - log_row_marginal)
        g_tmp = g + epsilon[..., 0] * (
            torch.log(b) - torch.logsumexp(-s(f_tmp, g), dim=1)
        )
        f = torch.where(terminated.unsqueeze(-1), f, f_tmp)
        g = torch.where(terminated.unsqueeze(-1), g, g_tmp)

    coupling = torch.exp(-s(f, g))

    if not batched:
        coupling = coupling.squeeze(0)

    return coupling


def posterior_shrinkage(
    prior_samples: Union[Tensor, np.ndarray], post_samples: Union[Tensor, np.ndarray]
) -> Tensor:
    """
    Calculate the posterior shrinkage, quantifying how much
    the posterior distribution contracts from the initial
    prior distribution.
    References:
    https://arxiv.org/abs/1803.08393

    Parameters
    ----------
    prior_samples : array_like or torch.Tensor [n_samples, n_params]
        Samples from the prior distribution.
    post_samples : array-like or torch.Tensor [n_samples, n_params]
        Samples from the posterior distribution.

    Returns
    -------
    shrinkage : torch.Tensor [n_params]
        The posterior shrinkage.
    """

    if len(prior_samples) == 0 or len(post_samples) == 0:
        raise ValueError("Input samples are empty")

    if not isinstance(prior_samples, torch.Tensor):
        prior_samples = torch.tensor(prior_samples, dtype=torch.float32)
    if not isinstance(post_samples, torch.Tensor):
        post_samples = torch.tensor(post_samples, dtype=torch.float32)

    if prior_samples.ndim == 1:
        prior_samples = prior_samples[:, None]
    if post_samples.ndim == 1:
        post_samples = post_samples[:, None]

    prior_std = torch.std(prior_samples, dim=0)
    post_std = torch.std(post_samples, dim=0)

    return 1 - (post_std / prior_std) ** 2


def posterior_zscore(
    true_theta: Union[Tensor, np.array, float], post_samples: Union[Tensor, np.array]
):
    """
    Calculate the posterior z-score, quantifying how much the posterior
    distribution of a parameter encompasses its true value.
    References:
    https://arxiv.org/abs/1803.08393

    Parameters
    ----------
    true_theta : float, array-like or torch.Tensor [n_params]
        The true value of the parameters.
    post_samples : array-like or torch.Tensor [n_samples, n_params]
        Samples from the posterior distributions.

    Returns
    -------
    z : Tensor [n_params]
        The z-score of the posterior distributions.
    """

    if len(post_samples) == 0:
        raise ValueError("Input samples are empty")

    if not isinstance(true_theta, torch.Tensor):
        true_theta = torch.tensor(true_theta, dtype=torch.float32)
    if not isinstance(post_samples, torch.Tensor):
        post_samples = torch.tensor(post_samples, dtype=torch.float32)

    true_theta = np.atleast_1d(true_theta)
    if post_samples.ndim == 1:
        post_samples = post_samples[:, None]

    post_mean = torch.mean(post_samples, dim=0)
    post_std = torch.std(post_samples, dim=0)

    return torch.abs((post_mean - true_theta) / post_std)


def l2(x: Tensor, y: Tensor, axis: int = -1) -> Tensor:
    """
    Calculates the L2 (Euclidean) distance between two tensors.

    Args:
        x (Tensor): The first tensor.
        y (Tensor): The second tensor.
        axis (int, optional): The axis along which to calculate the L2 distance.
            Defaults to -1.

    Returns:
        Tensor: A tensor containing the L2 distance between x and y along the
            specified axis.
    """
    return torch.sqrt(torch.sum((x - y) ** 2, dim=axis))


def l1(x: Tensor, y: Tensor, axis: int = -1) -> Tensor:
    """
    Calculates the L1 (Manhattan) distance between two tensors.

    Args:
        x (Tensor): The first tensor.
        y (Tensor): The second tensor.
        axis (int, optional): The axis along which to calculate the L1 distance.
            Defaults to -1.

    Returns:
        Tensor: A tensor containing the L1 distance between x and y along the
            specified axis.
    """
    return torch.sum(torch.abs(x - y), dim=axis)


class Distance:
    """A wrapper for distance functions to be used with ABC methods."""

    def __init__(
        self,
        distance: Union[str, Callable],
        requires_iid_data: Optional[bool] = None,
        distance_kwargs: Optional[Dict] = None,
        batch_size: int = -1,
    ):
        """
        Args:
            distance: Distance function to compare observed and simulated data. Can be
                a custom callable function or one of `l1`, `l2`, `mse`,
                `mmd`, `wasserstein`.
            requires_iid_data: Whether to allow conditioning on iid sampled data or not.
                Typically, this information is inferred by the choice of the distance,
                but in case a custom distance is used, this information is pivotal.
            distance_kwargs: Configurations parameters for the distances. In particular
                useful for the MMD and Wasserstein distance. Unknown keys raise a
                TypeError when the distance is created.
            batch_size: Number of simulations that the distance function
                evaluates against the reference observations at once. If -1, we evaluate
                all simulations at the same time.
        """
        self.distance_kwargs = distance_kwargs or {}
        self.batch_size = batch_size

        if callable(distance):
            if requires_iid_data is None:
                warnings.warn(
                    "Please specify if your the custom distance requires "
                    "iid data or is evaluated between single datapoints. "
                    "By default, we assume that `requires_iid_data=False`",
                    stacklevel=2,
                )
                requires_iid_data = False
            self.distance_fn = distance
            self._requires_iid_data = requires_iid_data
        else:
            implemented_pairwise_distances = ["l1", "l2", "mse"]
            implemented_statistical_distances = ["mmd", "wasserstein"]

            assert (
                distance
                in implemented_pairwise_distances + implemented_statistical_distances
            ), f"{distance} must be one of "
            f"{implemented_pairwise_distances + implemented_statistical_distances}."

            self._requires_iid_data = distance in implemented_statistical_distances

            distance_functions = {
                "mse": mse_distance,
                "l2": l2,
                "l1": l1,
                "mmd": mmd_distance,
                "wasserstein": wasserstein_distance,
            }
            distance_fn = distance_functions[distance]
            # partial only checks the keyword arguments when it is called.
            inspect.signature(distance_fn).bind(None, None, **self.distance_kwargs)
            self.distance_fn = partial(distance_fn, **self.distance_kwargs)

    def __call__(self, x_o: Tensor, x: Tensor) -> Tensor:
        """Distance evaluation between the reference data and the simulated data.

        Args:
            x_o: Reference data
            x: Simulated data
        """
        if self.requires_iid_data:
            assert x.ndim >= 3, "simulated data needs batch dimension"
            assert x_o.ndim + 1 == x.ndim
        else:
            assert x.ndim >= 2, "simulated data needs batch dimension"
        if self.batch_size == -1:
            return self.distance_fn(x_o, x)
        else:
            return self._batched_distance(x_o, x)

    def _batched_distance(self, x_o: Tensor, x: Tensor) -> Tensor:
        """Evaluate the distance is mini-batches.
        Especially for statistical distances, batching over two empirical
        datasets can lead to memory overflow. Batching can help to resolve
        the memory problems.

        Args:
            x_o: Reference data
            x: Simulated data
        """
        return torch.cat([
            self.distance_fn(x_o, x_batch).reshape(x_batch.shape[0])
            for x_batch in x.split(self.batch_size)
        ])

    @property
    def requires_iid_data(self):
        return self._requires_iid_data


def mse_distance(x_o: Tensor, x: Tensor) -> Tensor:
    """Calculates the Mean Squared Error (MSE) distance between two tensors.

    Args:
        x_o: The observed data tensor.
        x: The simulated data tensor.

    Returns:
        The MSE distance.
    """
    return torch.mean((x_o - x) ** 2, dim=-1)


def mmd_distance(x_o: Tensor, x: Tensor, scale: Optional[float] = None) -> Tensor:
    """Calculates the Maximum Mean Discrepancy (MMD) distance between two tensors.

    Args:
        x_o: The observed data tensor.
        x: The simulated data tensor.
        scale: Optional scale parameter for the MMD calculation.

    Returns:
        The MMD distance.
    """

    dist_fn = partial(unbiased_mmd_squared, scale=scale)
    return torch.vmap(dist_fn, in_dims=(None, 0))(x_o, x)


def wasserstein_distance(
    x_o: Tensor,
    x: Tensor,
    *,
    relative_epsilon: float = 1e-2,
    max_iter: int = 5000,
    tol: float = 1e-3,
) -> Tensor:
    """Calculates the squared Wasserstein-2 distance between two tensors.

    Args:
        x_o: The observed data tensor.
        x: The simulated data tensor.
        relative_epsilon: Entropic regularization, relative to the mean squared
            distance between the points of x_o and each simulated dataset. See
            `regularized_ot_dual`.
        max_iter: Maximum number of Sinkhorn iterations. See `regularized_ot_dual`.
        tol: Tolerance on the coupling's marginals. See `regularized_ot_dual`.

    Returns:
        The squared Wasserstein-2 distance.
    """
    batched_x_o = x_o.repeat((x.shape[0], *[1] * len(x_o.shape)))
    return wasserstein_2_squared(
        batched_x_o, x, relative_epsilon=relative_epsilon, max_iter=max_iter, tol=tol
    )
