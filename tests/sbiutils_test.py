from typing import Tuple

import matplotlib.pyplot as plt
import pytest
import torch
from torch import Tensor, ones, eye, zeros
from torch.distributions import LogNormal, MultivariateNormal, Exponential

from sbi.inference import SNPE, SNPE_A
from sbi.inference.snpe.snpe_a import SNPE_A_MDN
from sbi.utils import (
    BoxUniform,
    conditional_corrcoeff,
    conditional_pairplot,
    eval_conditional_density,
    mcmc_transform,
    MultipleIndependent,
    posterior_nn,
)
from sbi.utils.user_input_checks import process_prior


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


@pytest.mark.parametrize("snpe_method", ("snpe_a", "snpe_c"))
def test_gaussian_transforms(snpe_method: str, plot_results: bool = False):
    """
    Tests whether the the product between proposal and posterior is computed correctly.

    For SNPE-C, this initializes two MoGs with two components each. It then evaluates
    their product by simply multiplying the probabilities of the two. The result is
    compared to the product of two MoGs as implemented in APT.

    For SNPE-A, it initializes a MoG with two compontents and one Gaussian (with one
    component). It then devices the MoG by the Gaussian and compares it to the
    transformation in SNPE-A.

    Args:
        snpe_method: String indicating whether to test snpe-a or snpe-c.
        plot_results: Whether to plot the products of the distributions.
    """

    class MoG:
        def __init__(self, means, preds, logits):
            self._means = means
            self._preds = preds
            self._logits = logits

        def log_prob(self, theta):
            probs = zeros(theta.shape[0])
            for m, p, l in zip(self._means, self._preds, self._logits):
                mvn = MultivariateNormal(m, p)
                weighted_prob = torch.exp(mvn.log_prob(theta)) * l
                probs += weighted_prob
            return probs

    # Build a grid on which to evaluate the densities.
    bound = 5.0
    theta_range = torch.linspace(-bound, bound, 100)
    theta1_grid, theta2_grid = torch.meshgrid(theta_range, theta_range)
    theta_grid = torch.stack([theta1_grid, theta2_grid])
    theta_grid_flat = torch.reshape(theta_grid, (2, 100 ** 2))

    # Generate two MoGs.
    means1 = torch.tensor([[2.0, 2.0], [-2.0, -2.0]])
    covs1 = torch.stack([0.5 * torch.eye(2), torch.eye(2)])
    weights1 = torch.tensor([0.3, 0.7])

    if snpe_method == "snpe_c":
        means2 = torch.tensor([[2.0, -2.2], [-2.0, 1.9]])
        covs2 = torch.stack([0.6 * torch.eye(2), 0.9 * torch.eye(2)])
        weights2 = torch.tensor([0.6, 0.4])
    elif snpe_method == "snpe_a":
        means2 = torch.tensor([[-0.2, -0.4]])
        covs2 = torch.stack([3.5 * torch.eye(2)])
        weights2 = torch.tensor([1.0])

    mog1 = MoG(means1, covs1, weights1)
    mog2 = MoG(means2, covs2, weights2)

    # Evaluate the product of their pdfs by evaluating them separately and multiplying.
    probs1_raw = mog1.log_prob(theta_grid_flat.T)
    probs1 = torch.reshape(probs1_raw, (100, 100))

    probs2_raw = mog2.log_prob(theta_grid_flat.T)
    probs2 = torch.reshape(probs2_raw, (100, 100))

    if snpe_method == "snpe_c":
        probs_mult = probs1 * probs2

        # Set up a SNPE object in order to use the
        # `_automatic_posterior_transformation()`.
        prior = BoxUniform(-5 * ones(2), 5 * ones(2))
        density_estimator = posterior_nn("mdn", z_score_theta=False, z_score_x=False)
        inference = SNPE(prior=prior, density_estimator=density_estimator)
        theta_ = torch.rand(100, 2)
        x_ = torch.rand(100, 2)
        _ = inference.append_simulations(theta_, x_).train(max_num_epochs=1)
        inference._set_state_for_mog_proposal()

        precs1 = torch.inverse(covs1)
        precs2 = torch.inverse(covs2)

        # `.unsqueeze(0)` is needed because the method requires a batch dimension.
        logits_pp, means_pp, _, covs_pp = inference._automatic_posterior_transformation(
            torch.log(weights1.unsqueeze(0)),
            means1.unsqueeze(0),
            precs1.unsqueeze(0),
            torch.log(weights2.unsqueeze(0)),
            means2.unsqueeze(0),
            precs2.unsqueeze(0),
        )

    elif snpe_method == "snpe_a":
        probs_mult = probs1 / probs2

        prior = BoxUniform(-5 * ones(2), 5 * ones(2))

        inference = SNPE_A(prior=prior)
        theta_ = torch.rand(100, 2)
        x_ = torch.rand(100, 2)
        density_estimator = inference.append_simulations(theta_, x_).train(
            max_num_epochs=1
        )
        wrapped_density_estimator = SNPE_A_MDN(
            flow=density_estimator, proposal=prior, prior=prior
        )

        precs1 = torch.inverse(covs1)
        precs2 = torch.inverse(covs2)

        # `.unsqueeze(0)` is needed because the method requires a batch dimension.
        (
            logits_pp,
            means_pp,
            _,
            covs_pp,
        ) = wrapped_density_estimator._proposal_posterior_transformation(
            torch.log(weights2.unsqueeze(0)),
            means2.unsqueeze(0),
            precs2.unsqueeze(0),
            torch.log(weights1.unsqueeze(0)),
            means1.unsqueeze(0),
            precs1.unsqueeze(0),
        )

    # Normalize weights.
    logits_pp_norm = logits_pp - torch.logsumexp(logits_pp, dim=-1, keepdim=True)
    weights_pp = torch.exp(logits_pp_norm)

    # Evaluate the product of the two distributions.
    mog_apt = MoG(means_pp[0], covs_pp[0], weights_pp[0])

    probs_apt_raw = mog_apt.log_prob(theta_grid_flat.T)
    probs_apt = torch.reshape(probs_apt_raw, (100, 100))

    # Compute the error between the two methods.
    norm_probs_mult = probs_mult / torch.max(probs_mult)
    norm_probs3_ = probs_apt / torch.max(probs_apt)
    error = torch.abs(norm_probs_mult - norm_probs3_)

    assert torch.max(error) < 1e-5

    if plot_results:
        _, ax = plt.subplots(1, 4, figsize=(16, 4))

        ax[0].imshow(probs1, extent=[-bound, bound, -bound, bound])
        ax[0].set_title("p_1")
        ax[1].imshow(probs2, extent=[-bound, bound, -bound, bound])
        ax[1].set_title("p_2")
        ax[2].imshow(probs_mult, extent=[-bound, bound, -bound, bound])
        ax[3].imshow(probs_apt, extent=[-bound, bound, -bound, bound])
        if snpe_method == "snpe_c":
            ax[2].set_title("p_1 * p_2")
            ax[3].set_title("APT")
        elif snpe_method == "snpe_a":
            ax[2].set_title("p_1 / p_2")
            ax[3].set_title("SNPE-A")

        plt.show()


@pytest.mark.parametrize(
    "prior, enable_transform",
    (
        (BoxUniform(zeros(5), ones(5)), True),
        (BoxUniform(zeros(1), ones(1)), True),
        (BoxUniform(zeros(5), ones(5)), False),
        (MultivariateNormal(zeros(5), eye(5)), True),
        (Exponential(rate=ones(1)), True),
        (LogNormal(zeros(1), ones(1)), True),
        (
            MultipleIndependent(
                [Exponential(rate=ones(1)), BoxUniform(zeros(5), ones(5))]
            ),
            True,
        ),
    ),
)
def test_mcmc_transform(prior, enable_transform):
    """
    Test whether the transform for MCMC returns the log_abs_det in the correct shape.
    """

    num_samples = 1000
    prior, _, _ = process_prior(prior)
    tf = mcmc_transform(prior, enable_transform=enable_transform)

    samples = prior.sample((num_samples,))
    unconstrained_samples = tf(samples)
    samples_original = tf.inv(unconstrained_samples)

    log_abs_det = tf.log_abs_det_jacobian(samples_original, unconstrained_samples)
    assert log_abs_det.shape == torch.Size([num_samples])
