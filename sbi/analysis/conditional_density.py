# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.distributions.transforms as torch_tf
from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from torch import Tensor, nn
from torch.distributions import Distribution

from sbi.types import Shape, TorchTransform
from sbi.utils.conditional_density_utils import (
    ConditionedPotential,
    RestrictedPriorForConditional,
    RestrictedTransformForConditional,
    compute_corrcoeff,
    condition_mog,
    extract_and_transform_mog,
)
from sbi.utils.torchutils import atleast_2d_float32_tensor, ensure_theta_batched


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
    r"""Return the unnormalized conditional along `dim1, dim2` given `condition`.

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

    condition = ensure_theta_batched(condition)

    theta_grid_dim1 = torch.linspace(
        float(limits[dim1, 0] + eps_margins1),
        float(limits[dim1, 1] - eps_margins1),
        resolution,
        device=condition.device,
    )
    theta_grid_dim2 = torch.linspace(
        float(limits[dim2, 0] + eps_margins2),
        float(limits[dim2, 1] - eps_margins2),
        resolution,
        device=condition.device,
    )

    if dim1 == dim2:
        repeated_condition = condition.repeat(resolution, 1)
        repeated_condition[:, dim1] = theta_grid_dim1

        log_probs_on_grid = density.log_prob(repeated_condition)
    else:
        repeated_condition = condition.repeat(resolution**2, 1)
        repeated_condition[:, dim1] = theta_grid_dim1.repeat(resolution)
        repeated_condition[:, dim2] = torch.repeat_interleave(
            theta_grid_dim2, resolution
        )

        log_probs_on_grid = density.log_prob(repeated_condition)
        log_probs_on_grid = torch.reshape(log_probs_on_grid, (resolution, resolution))

    if return_raw_log_prob:
        return log_probs_on_grid
    else:
        # Subtract maximum for numerical stability
        return torch.exp(log_probs_on_grid - torch.max(log_probs_on_grid))


def conditional_corrcoeff(
    density: Any,
    limits: Tensor,
    condition: Tensor,
    subset: Optional[List[int]] = None,
    resolution: int = 50,
) -> Tensor:
    r"""Returns the conditional correlation matrix of a distribution.

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

    device = density._device if hasattr(density, "_device") else "cpu"

    subset_ = subset if subset is not None else range(condition.shape[1])

    correlation_matrices = []
    for cond in condition:
        correlation_matrices.append(
            torch.stack(
                [
                    compute_corrcoeff(
                        eval_conditional_density(
                            density,
                            cond.to(device),
                            limits.to(device),
                            dim1=dim1,
                            dim2=dim2,
                            resolution=resolution,
                        ),
                        limits[[dim1, dim2]].to(device),
                    )
                    for dim1 in subset_
                    for dim2 in subset_
                    if dim1 < dim2
                ]
            )
        )

    average_correlations = torch.mean(torch.stack(correlation_matrices), dim=0)

    # `average_correlations` is still a vector containing the upper triangular entries.
    # Below, assemble them into a matrix:
    av_correlation_matrix = torch.zeros((len(subset_), len(subset_)), device=device)
    triu_indices = torch.triu_indices(
        row=len(subset_), col=len(subset_), offset=1, device=device
    )
    av_correlation_matrix[triu_indices[0], triu_indices[1]] = average_correlations

    # Make the matrix symmetric by copying upper diagonal to lower diagonal.
    av_correlation_matrix = torch.triu(av_correlation_matrix) + torch.tril(
        av_correlation_matrix.T
    )

    av_correlation_matrix.fill_diagonal_(1.0)
    return av_correlation_matrix


class ConditionedMDN:
    def __init__(
        self,
        net: nn.Module,
        x_o: Tensor,
        condition: Tensor,
        dims_to_sample: List[int],
    ) -> None:
        r"""Class that can sample and evaluate a conditional mixture-of-gaussians.

        Args:
            net: Mixture density network that models $p(\theta|x).
            x_o: The datapoint at which the `net` is evaluated.
            condition: Parameter set that all dimensions not specified in
                `dims_to_sample` will be fixed to. Should contain dim_theta elements,
                i.e. it could e.g. be a sample from the posterior distribution.
                The entries at all `dims_to_sample` will be ignored.
            dims_to_sample: Which dimensions to sample from. The dimensions not
                specified in `dims_to_sample` will be fixed to values given in
                `condition`.
        """
        condition = atleast_2d_float32_tensor(condition)

        logits, means, precfs, _ = extract_and_transform_mog(net=net, context=x_o)
        self.logits, self.means, self.precfs, self.sumlogdiag = condition_mog(
            condition, dims_to_sample, logits, means, precfs
        )
        self.prec = self.precfs.transpose(3, 2) @ self.precfs

    def sample(self, sample_shape: Shape = torch.Size()) -> Tensor:
        num_samples = torch.Size(sample_shape).numel()
        samples = mdn.sample_mog(num_samples, self.logits, self.means, self.precfs)
        return samples.detach().reshape((*sample_shape, -1))

    def log_prob(self, theta: Tensor) -> Tensor:
        batch_size, dim = theta.shape

        log_prob = mdn.log_prob_mog(
            theta,
            self.logits.repeat(batch_size, 1),
            self.means.repeat(batch_size, 1, 1),
            self.prec.repeat(batch_size, 1, 1, 1),
            self.sumlogdiag.repeat(batch_size, 1),
        )
        return log_prob


def conditonal_potential(
    potential_fn: Callable,
    theta_transform: TorchTransform,
    prior: Distribution,
    condition: Tensor,
    dims_to_sample: List[int],
) -> Tuple[Callable, torch_tf.Transform, Any]:
    r"""Returns potential function that can be used to sample the conditional potential.

    It also returns a transform and a prior to be used to sample the conditional
    potential.

    The conditional potential is $p(\theta_i | \theta_j, x_o) \propto p(\theta | x_o)$
    but is a function only of $\theta_i$.

    Args:
        potential_fn: The potential function to be conditioned.
        theta_transform: The parameter transformation that should be reduced (by
            ignoring dimensions not contained in `dims_to_sample`).
        prior: The prior distribution that should be reduced (by ignoring dimensions
            not contained in `dims_to_sample`).
        condition: Parameter set that all dimensions not specified in
            `dims_to_sample` will be fixed to. Should contain dim_theta elements,
            i.e. it could e.g. be a sample from the posterior distribution.
            The entries at all `dims_to_sample` will be ignored.
        dims_to_sample: Which dimensions to sample from. The dimensions not
            specified in `dims_to_sample` will be fixed to values given in
            `condition`.

    Returns:
        A conditioned potential function, conditioned parameter transformation, and
        a marginalised prior.
    """

    restricted_tf = RestrictedTransformForConditional(
        theta_transform, condition, dims_to_sample
    )

    condition = atleast_2d_float32_tensor(condition)

    # Transform the `condition` to unconstrained space.
    transformed_condition = theta_transform(condition)

    conditioned_potential_fn = ConditionedPotential(
        potential_fn, transformed_condition, dims_to_sample  # type: ignore
    )

    restricted_prior = RestrictedPriorForConditional(prior, dims_to_sample)

    return conditioned_potential_fn, restricted_tf, restricted_prior
