from typing import Tuple

import pytest
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal

from sbi.utils import (
    conditional_corrcoeff,
    conditional_pairplot,
    eval_conditional_density,
)


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
        density=joint_dist,
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
        density=joint_dist,
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


def test_conditional_pairplot():
    """
    This only tests whether `conditional.pairplot()` runs without errors. If does not
    test its correctness. See `test_conditional_density_2d` for a test on
    `eval_conditional_density`, which is the core building block of
    `conditional.pairplot()`
    """
    d = MultivariateNormal(
        torch.tensor([0.6, 5.0]), torch.tensor([[0.1, 0.99], [0.99, 10.0]])
    )
    _ = conditional_pairplot(
        density=d,
        condition=torch.ones(1, 2),
        limits=torch.tensor([[-1.0, 1.0], [-30, 30]]),
    )


def conditional_of_mvn(
    loc: Tensor, cov: Tensor, condition: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Return the mean and cov of a conditional Gaussian.

    We assume that we always condition on the last variables.

    Args:
        loc: Mean of the joint distribution.
        cov: Covariance matrix of the joint distribution.
        condition: Condition. Should have less entries than `loc`.
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


@pytest.mark.parametrize("corr", (0.99, 0.95, 0.0))
def test_conditional_corrcoeff(corr):
    """
    Test whether the conditional correlation coefficient is computed correctly.
    """
    d = MultivariateNormal(
        torch.tensor([0.6, 5.0]), torch.tensor([[0.1, corr], [corr, 10.0]])
    )
    estimated_corr = conditional_corrcoeff(
        density=d,
        condition=torch.ones(1, 2),
        limits=torch.tensor([[-2.0, 3.0], [-70, 90]]),
        resolution=500,
    )[0, 1]

    assert torch.abs(corr - estimated_corr) < 1e-3


def test_average_cond_coeff_matrix():
    d = MultivariateNormal(
        torch.tensor([10.0, 5, 1]),
        torch.tensor([[100.0, 30.0, 0], [30.0, 10.0, 0], [0, 0, 1.0]]),
    )
    cond_mat = conditional_corrcoeff(
        density=d,
        condition=torch.zeros(1, 3),
        limits=torch.tensor([[-60.0, 60.0], [-20, 20], [-7, 7]]),
        resolution=500,
    )
    corr_dim12 = torch.sqrt(torch.tensor(30.0 ** 2 / 100.0 / 10.0))
    gt_matrix = torch.tensor(
        [[1.0, corr_dim12, 0.0], [corr_dim12, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    assert (torch.abs(gt_matrix - cond_mat) < 1e-3).all()
