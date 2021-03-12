# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from typing import Any, Callable, List, Optional, Tuple, Union
from warnings import warn

import torch
from torch import Tensor

from sbi.utils.torchutils import ensure_theta_batched


def eval_conditional_density(
    density: Any,
    condition: Tensor,
    limits: Tensor,
    dim1: int,
    dim2: int,
    resolution: int = 50,
    eps_margins1: Union[Tensor, float] = 1e-32,
    eps_margins2: Union[Tensor, float] = 1e-32,
    warn_about_deprecation: bool = True,
) -> Tensor:
    r"""
    Return the unnormalized conditional along `dim1, dim2` given parameters `condition`.

    We compute the unnormalized conditional by evaluating the joint distribution:
        $p(x1 | x2) = p(x1, x2) / p(x2) \propto p(x1, x2)$

    Args:
        density: Probability density function with `.log_prob()` method.
        condition: Parameter set that all dimensions other than dim1 and dim2 will be
            fixed to. Should be of shape (1, dim_theta), i.e. it could e.g. be
            a sample from the posterior distribution. The entries at `dim1` and `dim2`
            will be ignored.
        limits: Bounds within which to evaluate the density. Shape (dim_theta, 2).
        dim1: First dimension along which to evaluate the conditional.
        dim2: Second dimension along which to evaluate the conditional.
        resolution: Resolution of the grid along which the conditional density is
            evaluated.
        eps_margins1: We will evaluate the posterior along `dim1` at
            `limits[0]+eps_margins` until `limits[1]-eps_margins`. This avoids
            evaluations potentially exactly at the prior bounds.
        eps_margins2: We will evaluate the posterior along `dim2` at
            `limits[0]+eps_margins` until `limits[1]-eps_margins`. This avoids
            evaluations potentially exactly at the prior bounds.
        warn_about_deprecation: With sbi v0.15.0, we depracated the import of this
            function from `sbi.utils`. Instead, it should be imported from
            `sbi.analysis`.

    Returns: Conditional probabilities. If `dim1 == dim2`, this will have shape
        (resolution). If `dim1 != dim2`, it will have shape (resolution, resolution).
    """

    if warn_about_deprecation:
        warn(
            "Importing `eval_conditional_density` from `sbi.utils` is deprecated since "
            "sbi v0.15.0. Instead, use "
            "`from sbi.analysis import eval_conditional_density`."
        )

    condition = ensure_theta_batched(condition)

    theta_grid_dim1 = torch.linspace(
        float(limits[dim1, 0] + eps_margins1),
        float(limits[dim1, 1] - eps_margins1),
        resolution,
    )
    theta_grid_dim2 = torch.linspace(
        float(limits[dim2, 0] + eps_margins2),
        float(limits[dim2, 1] - eps_margins2),
        resolution,
    )

    if dim1 == dim2:
        repeated_condition = condition.repeat(resolution, 1)
        repeated_condition[:, dim1] = theta_grid_dim1

        log_probs_on_grid = density.log_prob(repeated_condition)
    else:
        repeated_condition = condition.repeat(resolution ** 2, 1)
        repeated_condition[:, dim1] = theta_grid_dim1.repeat(resolution)
        repeated_condition[:, dim2] = torch.repeat_interleave(
            theta_grid_dim2, resolution
        )

        log_probs_on_grid = density.log_prob(repeated_condition)
        log_probs_on_grid = torch.reshape(log_probs_on_grid, (resolution, resolution))

    # Subtract maximum for numerical stability.
    return torch.exp(log_probs_on_grid - torch.max(log_probs_on_grid))


def conditional_corrcoeff(
    density: Any,
    limits: Tensor,
    condition: Tensor,
    subset: Optional[List[int]] = None,
    resolution: int = 50,
    warn_about_deprecation: bool = True,
) -> Tensor:
    r"""
    Returns the conditional correlation matrix of a distribution.

    To compute the conditional distribution, we condition all but two parameters to
    values from `condition`, and then compute the Pearson correlation
    coefficient $\rho$ between the remaining two parameters under the distribution
    `density`. We do so for any pair of parameters specified in `subset`, thus
    creating a matrix containing conditional correlations between any pair of
    parameters.

    If `condition` is a batch of conditions, this function computes the conditional
    correlation matrix for each one of them and returns the mean.

    Args:
        density: Probability density function with `.log_prob()` function.
        limits: Limits within which to evaluate the `density`.
        condition: Values to condition the `density` on. If a batch of conditions is
            passed, we compute the conditional correlation matrix for each of them and
            return the average conditional correlation matrix.
        subset: Evaluate the conditional distribution only on a subset of dimensions.
            If `None` this function uses all dimensions.
        resolution: Number of grid points on which the conditional distribution is
            evaluated. A higher value increases the accuracy of the estimated
            correlation but also increases the computational cost.
        warn_about_deprecation: With sbi v0.15.0, we depracated the import of this
            function from `sbi.utils`. Instead, it should be imported from
            `sbi.analysis`.

    Returns: Average conditional correlation matrix of shape either `(num_dim, num_dim)`
    or `(len(subset), len(subset))` if `subset` was specified.
    """

    if warn_about_deprecation:
        warn(
            "Importing `conditional_corrcoeff` from `sbi.utils` is deprecated since "
            "sbi v0.15.0. Instead, use "
            "`from sbi.analysis import conditional_corrcoeff`."
        )

    condition = ensure_theta_batched(condition)

    if subset is None:
        subset = range(condition.shape[1])

    correlation_matrices = []
    for cond in condition:
        correlation_matrices.append(
            torch.stack(
                [
                    _compute_corrcoeff(
                        eval_conditional_density(
                            density,
                            cond,
                            limits,
                            dim1=dim1,
                            dim2=dim2,
                            resolution=resolution,
                            warn_about_deprecation=False,
                        ),
                        limits[[dim1, dim2]],
                    )
                    for dim1 in subset
                    for dim2 in subset
                    if dim1 < dim2
                ]
            )
        )

    average_correlations = torch.mean(torch.stack(correlation_matrices), dim=0)

    # `average_correlations` is still a vector containing the upper triangular entries.
    # Below, assemble them into a matrix:
    av_correlation_matrix = torch.zeros((len(subset), len(subset)))
    triu_indices = torch.triu_indices(row=len(subset), col=len(subset), offset=1)
    av_correlation_matrix[triu_indices[0], triu_indices[1]] = average_correlations

    # Make the matrix symmetric by copying upper diagonal to lower diagonal.
    av_correlation_matrix = torch.triu(av_correlation_matrix) + torch.tril(
        av_correlation_matrix.T
    )

    av_correlation_matrix.fill_diagonal_(1.0)
    return av_correlation_matrix


def _compute_corrcoeff(probs: Tensor, limits: Tensor):
    """
    Given a matrix of probabilities `probs`, return the correlation coefficient.

    Args:
        probs: Matrix of (unnormalized) evaluations of a 2D density.
        limits: Limits within which the entries of the matrix are evenly spaced.

    Returns: Pearson correlation coefficient.
    """

    normalized_probs = _normalize_probs(probs, limits)
    covariance = _compute_covariance(normalized_probs, limits)

    marginal_x, marginal_y = _calc_marginals(normalized_probs, limits)
    variance_x = _compute_covariance(marginal_x, limits[0], lambda x: x ** 2)
    variance_y = _compute_covariance(marginal_y, limits[1], lambda x: x ** 2)

    return covariance / torch.sqrt(variance_x * variance_y)


def _compute_covariance(
    probs: Tensor, limits: Tensor, f: Callable = lambda x, y: x * y
) -> Tensor:
    """
    Return the covariance between two RVs from evaluations of their pdf on a grid.

    The function computes the covariance as:
    Cov(X,Y) = E[X*Y] - E[X] * E[Y]

    In the more general case, when using a different function `f`, it returns:
    E[f(X,Y)] - f(E[X], E[Y])

    By using different function `f`, this function can be also deal with more than two
    dimensions, but this has not been tested.

    Lastly, this function can also compute the variance of a 1D distribution. In that
    case, `probs` will be a vector, and f would be: f = lambda x: x**2:
    Var(X,Y) = E[X**2] - E[X]**2

    Args:
        probs: Matrix of evaluations of a 2D density.
        limits: Limits within which the entries of the matrix are evenly spaced.
        f: The operation to be applied to the expected values, usually just the product.

    Returns: Covariance.
    """

    probs = ensure_theta_batched(probs)
    limits = ensure_theta_batched(limits)

    # Compute E[X*Y].
    expected_value_of_joint = _expected_value_f_of_x(probs, limits, f)

    # Compute E[X] * E[Y].
    expected_values_of_marginals = [
        _expected_value_f_of_x(prob.unsqueeze(0), lim.unsqueeze(0))
        for prob, lim in zip(_calc_marginals(probs, limits), limits)
    ]

    return expected_value_of_joint - f(*expected_values_of_marginals)


def _expected_value_f_of_x(
    probs: Tensor, limits: Tensor, f: Callable = lambda x: x
) -> Tensor:
    """
    Return the expected value of a function of random variable(s) E[f(X_i,...,X_k)].

    The expected value is computed from evaluations of the joint density on an evenly
    spaced grid, passed as `probs`.

    This function can not deal with functions `f` that have multiple outputs. They will
    simply be summed over.

    Args:
        probs: Matrix of evaluations of the density.
        limits: Limits within which the entries of the matrix are evenly spaced.
        f: The operation to be applied to the expected values.

    Returns: Expected value.
    """

    probs = ensure_theta_batched(probs)
    limits = ensure_theta_batched(limits)

    x_values_over_which_we_integrate = [
        torch.linspace(lim[0], lim[1], prob.shape[0])
        for lim, prob in zip(torch.flip(limits, [0]), probs)
    ]  # See #403 and #404 for flip().
    grids = list(torch.meshgrid(x_values_over_which_we_integrate))
    expected_val = torch.sum(f(*grids) * probs)

    limits_diff = torch.prod(limits[:, 1] - limits[:, 0])
    expected_val /= probs.numel() / limits_diff.item()

    return expected_val


def _calc_marginals(
    probs: Tensor, limits: Tensor
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Given a 2D matrix of probabilities, return the normalized marginal vectors.

    Args:
        probs: Matrix of evaluations of a 2D density.
        limits: Limits within which the entries of the matrix are evenly spaced.
    """

    if probs.shape[0] > 1:
        # Marginalize and normalize if multi-D distribution.
        marginal_x = torch.sum(probs, dim=0)
        marginal_y = torch.sum(probs, dim=1)

        marginal_x = _normalize_probs(marginal_x, limits[0].unsqueeze(0))
        marginal_y = _normalize_probs(marginal_y, limits[1].unsqueeze(0))
        return marginal_x, marginal_y
    else:
        # Only normalize if already a 1D distribution.
        return _normalize_probs(probs, limits)


def _normalize_probs(probs: Tensor, limits: Tensor) -> Tensor:
    """
    Given a matrix or a vector of probabilities, return the normalized matrix or vector.

    Args:
        probs: Matrix / vector of probabilities.
        limits: Limits within which the entries of the matrix / vector are evenly
            spaced. Must have a batch dimension if probs is a vector.

    Returns: Normalized probabilities.
    """
    limits_diff = torch.prod(limits[:, 1] - limits[:, 0])
    return probs * probs.numel() / limits_diff / torch.sum(probs)
