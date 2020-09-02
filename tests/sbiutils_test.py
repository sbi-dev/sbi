import torch
from torch.distributions import MultivariateNormal
from torch import Tensor

from sbi.utils import eval_conditional_density
from typing import Tuple


def test_conditional_density_1d():
    """
    Test whether the conditional density matches analytical results for MVN.

    This uses a 3D joint and conditions on the last two values to generate a 1D
    conditional.
    """
    joint_mean = torch.zeros(3)
    joint_cov = torch.tensor([[1.0, 0.0, 0.7], [0.0, 1.0, 0.7], [0.7, 0.7, 1.0]])
    joint_dist = MultivariateNormal(joint_mean, joint_cov)

    condition_dim2 = torch.ones(2)
    full_condition = torch.ones(3)

    resolution = 100
    vals_to_eval_at = torch.linspace(-3, 3, resolution).unsqueeze(1)

    # Solution with sbi.
    probs = eval_conditional_density(
        pdf=joint_dist,
        condition=full_condition,
        limits=torch.tensor([[-3, 3], [-3, 3], [-3, 3]]),
        dim1=0,
        dim2=0,
        resolution=resolution,
    )
    probs_sbi = probs / torch.sum(probs)

    # Analytical solution.
    conditional_mean, conditional_cov = conditional_of_mvn(
        joint_mean, joint_cov, condition_dim2
    )
    conditional_dist = torch.distributions.MultivariateNormal(
        conditional_mean, conditional_cov
    )

    probs = torch.exp(conditional_dist.log_prob(vals_to_eval_at))
    probs_analytical = probs / torch.sum(probs)

    assert torch.all(torch.abs(probs_analytical - probs_sbi) < 1e-5)


def test_conditional_density_2d():
    """
    Test whether the conditional density matches analytical results for MVN.

    This uses a 3D joint and conditions on the last value to generate a 2D conditional.
    """
    joint_mean = torch.zeros(3)
    joint_cov = torch.tensor([[1.0, 0.0, 0.7], [0.0, 1.0, 0.7], [0.7, 0.7, 1.0]])
    joint_dist = MultivariateNormal(joint_mean, joint_cov)

    condition_dim2 = torch.ones(1)
    full_condition = torch.ones(3)

    resolution = 100
    vals_to_eval_at_dim1 = (
        torch.linspace(-3, 3, resolution).repeat(resolution).unsqueeze(1)
    )
    vals_to_eval_at_dim2 = torch.repeat_interleave(
        torch.linspace(-3, 3, resolution), resolution
    ).unsqueeze(1)
    vals_to_eval_at = torch.cat((vals_to_eval_at_dim1, vals_to_eval_at_dim2), axis=1)

    # Solution with sbi.
    probs = eval_conditional_density(
        pdf=joint_dist,
        condition=full_condition,
        limits=torch.tensor([[-3, 3], [-3, 3], [-3, 3]]),
        dim1=0,
        dim2=1,
        resolution=resolution,
    )
    probs_sbi = probs / torch.sum(probs)

    # Analytical solution.
    conditional_mean, conditional_cov = conditional_of_mvn(
        joint_mean, joint_cov, condition_dim2
    )
    conditional_dist = torch.distributions.MultivariateNormal(
        conditional_mean, conditional_cov
    )

    probs = torch.exp(conditional_dist.log_prob(vals_to_eval_at))
    probs = torch.reshape(probs, (resolution, resolution))
    probs_analytical = probs / torch.sum(probs)

    assert torch.all(torch.abs(probs_analytical - probs_sbi) < 1e-5)


def conditional_of_mvn(
    loc: Tensor, cov: Tensor, condition: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Return the mean and cov of a conditional Gaussian.

    We assume that we always condition on the last variables.

    Args:
        loc: Mean of the joint distribution.
        cov: Covariance matrix of the joint distribution.
        condition: Condition. Should have less entries than mean.
    """

    num_of_condition_dims = loc.shape[0] - condition.shape[0]

    mean_1 = loc[:num_of_condition_dims]
    mean_2 = loc[num_of_condition_dims:]
    cov_11 = cov[:num_of_condition_dims, :num_of_condition_dims]
    cov_12 = cov[:num_of_condition_dims, num_of_condition_dims:]
    cov_22 = cov[num_of_condition_dims:, num_of_condition_dims:]

    precision_observed = torch.inverse(cov_22)
    residual = condition - mean_2
    precision_weighted_residual = torch.einsum(
        "ij, i -> j", precision_observed, residual
    )
    mean_shift = torch.mv(cov_12, precision_weighted_residual)
    conditional_mean = mean_1 + mean_shift

    prec_cov = torch.einsum("ji, kj -> ik", precision_observed, cov_12)
    cov_prec_cov = torch.einsum("ij, jk -> ik", cov_12, prec_cov)
    conditional_cov = cov_11 - cov_prec_cov

    return conditional_mean, conditional_cov
