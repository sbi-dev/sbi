from typing import Any, Union

import torch
from torch import Tensor


def eval_conditional_density(
    pdf: Any,
    condition: Tensor,
    limits: Tensor,
    dim1: int,
    dim2: int,
    resolution: int = 20,
    eps_margins1: Union[Tensor, float] = 1e-32,
    eps_margins2: Union[Tensor, float] = 1e-32,
) -> Tensor:
    r"""
    Return the unnormalized conditional along `dim1, dim2` given parameters `condition`.

    We compute the unnormalized conditional by evaluating the joint distribution:
        $p(x1 | x2) = p(x1, x2) / p(x2) \propto p(x1, x2)$

    Args:
        pdf: Probability density function with `.log_prob()` method.
        condition: Parameter set that all dimensions other than dim1 and dim2 will be
            fixed to. The condition be of shape (1, dim_theta), i.e. it could e.g. be
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

    Returns: Conditional probabilities. If `dim1 == dim2`, this will have shape
        (resolution). If `dim1 != dim2`, it will have shape (resolution, resolution).
    """

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

        log_probs_on_grid = pdf.log_prob(repeated_condition)
    else:
        repeated_condition = condition.repeat(resolution ** 2, 1)
        repeated_condition[:, dim1] = theta_grid_dim1.repeat(resolution)
        repeated_condition[:, dim2] = torch.repeat_interleave(
            theta_grid_dim2, resolution
        )

        log_probs_on_grid = pdf.log_prob(repeated_condition)
        log_probs_on_grid = torch.reshape(log_probs_on_grid, (resolution, resolution))

    # Subtract maximum for numerical stability.
    return torch.exp(log_probs_on_grid - torch.max(log_probs_on_grid))
