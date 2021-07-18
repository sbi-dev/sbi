# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Any, List, Optional, Union

from torch import Tensor

from sbi.utils import conditional_corrcoeff as utils_conditional_corrcoeff
from sbi.utils import eval_conditional_density as utils_eval_conditional_density


def eval_conditional_density(
    density: Any,
    condition: Tensor,
    limits: Tensor,
    dim1: int,
    dim2: int,
    resolution: int = 50,
    eps_margins1: Union[Tensor, float] = 1e-32,
    eps_margins2: Union[Tensor, float] = 1e-32,
    return_raw_log_prob: bool = False,
) -> Tensor:
    r"""
    Return the unnormalized conditional along `dim1, dim2` given parameters `condition`.

    We compute the unnormalized conditional by evaluating the joint distribution:
        $p(x1 | x2) = p(x1, x2) / p(x2) \propto p(x1, x2)$

    The joint distribution is evaluated on an evenly spaced grid defined by the
    `limits`.

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
        return_raw_log_prob: If `True`, return the log-probability evaluated on the
            grid. If `False`, return the probability, scaled down by the maximum value
            on the grid for numerical stability (i.e. exp(log_prob - max_log_prob)).

    Returns: Conditional probabilities. If `dim1 == dim2`, this will have shape
        (resolution). If `dim1 != dim2`, it will have shape (resolution, resolution).
    """
    return utils_eval_conditional_density(
        density=density,
        condition=condition,
        limits=limits,
        dim1=dim1,
        dim2=dim2,
        resolution=resolution,
        eps_margins1=eps_margins1,
        eps_margins2=eps_margins2,
        return_raw_log_prob=return_raw_log_prob,
        warn_about_deprecation=False,
    )


def conditional_corrcoeff(
    density: Any,
    limits: Tensor,
    condition: Tensor,
    subset: Optional[List[int]] = None,
    resolution: int = 50,
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

    Returns: Average conditional correlation matrix of shape either `(num_dim, num_dim)`
    or `(len(subset), len(subset))` if `subset` was specified.
    """
    return utils_conditional_corrcoeff(
        density=density,
        limits=limits,
        condition=condition,
        subset=subset,
        resolution=resolution,
        warn_about_deprecation=False,
    )
