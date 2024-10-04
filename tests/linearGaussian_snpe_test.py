# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import gaussian_kde
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import analysis as analysis
from sbi import utils as utils
from sbi.analysis import ConditionedMDN, conditional_potential
from sbi.inference import (
    NPE_A,
    NPE_B,
    NPE_C,
    DirectPosterior,
    MCMCPosterior,
    RejectionPosterior,
    posterior_estimator_based_potential,
)
from sbi.neural_nets import posterior_nn
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    samples_true_posterior_linear_gaussian_mvn_prior_different_dims,
    samples_true_posterior_linear_gaussian_uniform_prior,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils import RestrictedPrior, get_density_thresholder
from sbi.utils.user_input_checks import process_prior, process_simulator

from .sbiutils_test import conditional_of_mvn
from .test_utils import (
    check_c2st,
    get_dkl_gaussian_prior,
    get_normalization_uniform_prior,
    get_prob_outside_uniform_prior,
)


@pytest.mark.parametrize("npe_method", [NPE_A, NPE_C])
@pytest.mark.parametrize(
    "num_dim, prior_str",
    ((2, "gaussian"), (2, "uniform"), (1, "gaussian")),
)
def test_c2st_npe_on_linearGaussian(npe_method, num_dim: int, prior_str: str):
    """Test whether NPE infers well a simple example with available ground truth."""

    x_o = zeros(1, num_dim)
    num_samples = 1000
    num_simulations = 2500

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    if prior_str == "gaussian":
        prior_mean = zeros(num_dim)
        prior_cov = eye(num_dim)
        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
        gt_posterior = true_posterior_linear_gaussian_mvn_prior(
            x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
        )
        target_samples = gt_posterior.sample((num_samples,))
    else:
        prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))
        target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
            x_o,
            likelihood_shift,
            likelihood_cov,
            prior=prior,
            num_samples=num_samples,
        )

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inference = npe_method(prior, show_progress_bars=False)

    theta = prior.sample((num_simulations,))
    x = simulator(theta)

    posterior_estimator = inference.append_simulations(theta, x).train(
        training_batch_size=100
    )
    posterior = DirectPosterior(
        prior=prior, posterior_estimator=posterior_estimator
    ).set_default_x(x_o)
    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg="npe_c")

    map_ = posterior.map(num_init_samples=1_000, show_progress_bars=False)

    # Checks for log_prob()
    if prior_str == "gaussian":
        # For the Gaussian prior, we compute the KLd between ground truth and posterior.
        dkl = get_dkl_gaussian_prior(
            posterior,
            x_o[0],
            likelihood_shift,
            likelihood_cov,
            prior_mean,
            prior_cov,
        )

        max_dkl = 0.15

        assert (
            dkl < max_dkl
        ), f"D-KL={dkl} is more than 2 stds above the average performance."

        assert ((map_ - gt_posterior.mean) ** 2).sum() < 0.5

    elif prior_str == "uniform":
        # Check whether the returned probability outside of the support is zero.
        posterior_prob = get_prob_outside_uniform_prior(posterior, prior, num_dim)
        assert (
            posterior_prob == 0.0
        ), "The posterior probability outside of the prior support is not zero"

        # Check whether normalization (i.e. scaling up the density due
        # to leakage into regions without prior support) scales up the density by the
        # correct factor.
        (
            posterior_likelihood_unnorm,
            posterior_likelihood_norm,
            acceptance_prob,
        ) = get_normalization_uniform_prior(posterior, prior, x=x_o)
        # The acceptance probability should be *exactly* the ratio of the unnormalized
        # and the normalized likelihood. However, we allow for an error margin of 1%,
        # since the estimation of the acceptance probability is random (based on
        # rejection sampling).
        assert (
            acceptance_prob * 0.99
            < posterior_likelihood_unnorm / posterior_likelihood_norm
            < acceptance_prob * 1.01
        ), "Normalizing the posterior density using the acceptance probability failed."

        assert ((map_ - ones(num_dim)) ** 2).sum() < 0.5


@pytest.mark.slow
@pytest.mark.parametrize(
    "density_estimator",
    ["mdn", "maf", "maf_rqs", "nsf", "zuko_maf", "zuko_nsf"],
)
def test_density_estimators_on_linearGaussian(density_estimator):
    """Test NPE with different density estimators on linear Gaussian example."""

    theta_dim = 4
    x_dim = 4

    x_o = zeros(1, x_dim)
    num_samples = 1000
    num_simulations = 2500

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(x_dim)
    likelihood_cov = 0.3 * eye(x_dim)

    prior_mean = zeros(theta_dim)
    prior_cov = eye(theta_dim)

    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = gt_posterior.sample((num_samples,))

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inference = NPE_C(prior, density_estimator=density_estimator)

    theta = prior.sample((num_simulations,))
    x = simulator(theta)
    posterior_estimator = inference.append_simulations(theta, x).train(
        training_batch_size=100
    )
    posterior = DirectPosterior(
        prior=prior, posterior_estimator=posterior_estimator
    ).set_default_x(x_o)
    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg=f"npe_{density_estimator}")


def test_c2st_npe_on_linearGaussian_different_dims(density_estimator="maf"):
    """Test NPE on linear Gaussian with different theta and x dimensionality."""

    theta_dim = 3
    x_dim = 2
    discard_dims = theta_dim - x_dim

    x_o = zeros(1, x_dim)
    num_samples = 1000
    num_simulations = 2000

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(x_dim)
    likelihood_cov = 0.3 * eye(x_dim)

    prior_mean = zeros(theta_dim)
    prior_cov = eye(theta_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    target_samples = samples_true_posterior_linear_gaussian_mvn_prior_different_dims(
        x_o,
        likelihood_shift,
        likelihood_cov,
        prior_mean,
        prior_cov,
        num_discarded_dims=discard_dims,
        num_samples=num_samples,
    )

    def simulator(theta):
        return linear_gaussian(
            theta,
            likelihood_shift,
            likelihood_cov,
            num_discarded_dims=discard_dims,
        )

    # Test whether prior can be `None`.
    inference = NPE_C(
        prior=None,
        density_estimator=density_estimator,
        show_progress_bars=False,
    )

    theta = prior.sample((num_simulations,))
    x = simulator(theta)

    inference = inference.append_simulations(theta, x)
    posterior_estimator = inference.train(
        max_num_epochs=10
    )  # Test whether we can stop and resume.
    posterior_estimator = inference.train(
        resume_training=True, force_first_round_loss=True
    )
    posterior = DirectPosterior(
        prior=prior, posterior_estimator=posterior_estimator
    ).set_default_x(x_o)
    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg="snpe_c_different_dims")


# Test multi-round NPE.
@pytest.mark.slow
@pytest.mark.parametrize(
    "method_str",
    (
        "snpe_a",
        pytest.param(
            "snpe_b",
            marks=pytest.mark.xfail(
                raises=NotImplementedError, reason="""NPE-B not implemented"""
            ),
        ),
        "snpe_c",
        "snpe_c_non_atomic",
        "tsnpe_rejection",
        "tsnpe_sir",
    ),
)
def test_c2st_multi_round_snpe_on_linearGaussian(method_str: str):
    """Test whether NPE B/C infer well a simple example with available ground truth.
    .
    """

    num_dim = 2
    x_o = zeros((1, num_dim))
    num_samples = 1000

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)

    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = gt_posterior.sample((num_samples,))

    if method_str == "snpe_c_non_atomic":
        # Test whether NPE works properly with structured z-scoring.
        density_estimator = posterior_nn(
            "mdn", z_score_x="structured", num_components=5
        )
        method_str = "snpe_c"
    elif method_str == "snpe_a":
        density_estimator = "mdn_snpe_a"
    else:
        density_estimator = "maf"

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    creation_args = dict(
        prior=prior,
        density_estimator=density_estimator,
        show_progress_bars=False,
    )

    if method_str == "snpe_b":
        inference = NPE_B(**creation_args)
        theta = prior.sample((500,))
        x = simulator(theta)
        posterior_estimator = inference.append_simulations(theta, x).train()
        posterior1 = DirectPosterior(
            prior=prior, posterior_estimator=posterior_estimator
        ).set_default_x(x_o)
        theta = posterior1.sample((1000,))
        x = simulator(theta)
        posterior_estimator = inference.append_simulations(
            theta, x, proposal=posterior1
        ).train()
        posterior = DirectPosterior(
            prior=prior, posterior_estimator=posterior_estimator
        ).set_default_x(x_o)
    elif method_str == "snpe_c":
        inference = NPE_C(**creation_args)
        theta = prior.sample((900,))
        x = simulator(theta)
        posterior_estimator = inference.append_simulations(theta, x).train()
        posterior1 = DirectPosterior(
            prior=prior, posterior_estimator=posterior_estimator
        ).set_default_x(x_o)
        theta = posterior1.sample((1000,))
        x = simulator(theta)
        _ = inference.append_simulations(theta, x, proposal=posterior1).train()
        posterior = inference.build_posterior().set_default_x(x_o)
    elif method_str == "snpe_a":
        inference = NPE_A(**creation_args)
        proposal = prior
        final_round = False
        num_rounds = 3
        for r in range(num_rounds):
            if r == 2:
                final_round = True
            theta = proposal.sample((500,))
            x = simulator(theta)
            inference = inference.append_simulations(theta, x, proposal=proposal)
            _ = inference.train(max_num_epochs=200, final_round=final_round)
            posterior = inference.build_posterior().set_default_x(x_o)
            proposal = posterior
    elif method_str.startswith("tsnpe"):
        sample_method = "rejection" if method_str == "tsnpe_rejection" else "sir"
        inference = NPE_C(**creation_args)
        theta = prior.sample((1000,))
        x = simulator(theta)
        posterior_estimator = inference.append_simulations(theta, x).train()
        posterior1 = DirectPosterior(
            prior=prior, posterior_estimator=posterior_estimator
        ).set_default_x(x_o)
        accept_reject_fn = get_density_thresholder(posterior1, quantile=1e-4)
        proposal = RestrictedPrior(
            prior,
            accept_reject_fn,
            posterior=posterior1,
            sample_with=sample_method,
        )
        theta = proposal.sample((1000,))
        x = simulator(theta)
        _ = inference.append_simulations(theta, x).train(force_first_round_loss=True)
        posterior = inference.build_posterior().set_default_x(x_o)

    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg=method_str)


# Testing rejection and mcmc sampling methods.
@pytest.mark.slow
@pytest.mark.parametrize(
    "sample_with, mcmc_method, prior_str",
    (
        pytest.param("mcmc", "slice_np", "gaussian", marks=pytest.mark.mcmc),
        pytest.param("mcmc", "slice_np_vectorized", "gaussian", marks=pytest.mark.mcmc),
        ("rejection", "rejection", "uniform"),
    ),
)
def test_api_snpe_c_posterior_correction(
    sample_with, mcmc_method, prior_str, mcmc_params_fast: dict
):
    """Test that leakage correction applied to sampling works, with both MCMC and
    rejection.

    """

    num_dim = 2
    x_o = zeros(1, num_dim)

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    if prior_str == "gaussian":
        prior_mean = zeros(num_dim)
        prior_cov = eye(num_dim)
        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    else:
        prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inference = NPE_C(prior, show_progress_bars=False)

    theta = prior.sample((1000,))
    x = simulator(theta)
    posterior_estimator = inference.append_simulations(theta, x).train()
    potential_fn, theta_transform = posterior_estimator_based_potential(
        posterior_estimator, prior, x_o
    )
    if sample_with == "mcmc":
        posterior = MCMCPosterior(
            potential_fn=potential_fn,
            theta_transform=theta_transform,
            proposal=prior,
            method=mcmc_method,
            **mcmc_params_fast,
        )
    elif sample_with == "rejection":
        posterior = RejectionPosterior(
            potential_fn=potential_fn,
            proposal=prior,
            theta_transform=theta_transform,
        )

    # Posterior should be corrected for leakage even if num_rounds just 1.
    samples = posterior.sample((10,))

    # Evaluate the samples to check correction factor.
    _ = posterior.log_prob(samples)


# Testing rejection and mcmc sampling methods.
@pytest.mark.parametrize(
    "force_first_round_loss, pass_proposal_to_append",
    (
        (True, True),
        (True, False),
        (False, True),
        pytest.param(False, False, marks=pytest.mark.xfail),
    ),
)
def test_api_force_first_round_loss(
    force_first_round_loss: bool, pass_proposal_to_append: bool
):
    """Test that leakage correction applied to sampling works, with both MCMC and
    rejection.

    """

    num_dim = 2
    x_o = zeros(1, num_dim)
    num_simulations = 1000

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inference = NPE_C(prior, show_progress_bars=False)
    prior, _, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(simulator, prior, prior_returns_numpy)

    proposal = prior
    for _ in range(2):
        train_proposal = proposal if pass_proposal_to_append else None
        theta = proposal.sample((num_simulations,))
        x = simulator(theta)
        _ = inference.append_simulations(theta, x, proposal=train_proposal).train(
            force_first_round_loss=force_first_round_loss, max_num_epochs=2
        )
        posterior = inference.build_posterior().set_default_x(x_o)
        proposal = posterior


@pytest.mark.slow
@pytest.mark.mcmc
def test_sample_conditional(mcmc_params_accurate: dict):
    """
    Test whether sampling from the conditional gives the same results as
    evaluating.

    This compares samples that get smoothed with a Gaussian kde to evaluating
    the conditional log-probability with `eval_conditional_density`.

    `eval_conditional_density` is itself tested in `sbiutils_test.py`. Here, we
    use a bimodal posterior to test the conditional.

    NOTE: The comparison between conditional log_probs obtained from the MCMC
    posterior and from analysis.eval_conditional_density can be gamed by
    underfitting the posterior estimator, i.e., by using a small number of
    simulations.
    """

    num_dim = 3
    dim_to_sample_1 = 0
    dim_to_sample_2 = 2
    num_simulations = 5500
    num_conditional_samples = 1000
    num_conditions = 50

    x_o = zeros(1, num_dim)

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.05 * eye(num_dim)

    prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))

    # TODO: janfb does not see how this setup results in a bi-model posterior.
    def simulator(theta):
        batch_size, _ = theta.shape
        # create -1 1 mask for bimodality
        mask = torch.ones(batch_size, 1)
        # set mask to -1 randomly across the batch
        mask = mask * 2 * (torch.rand(batch_size, 1) > 0.5) - 1

        # Sample bi-modally by applying a 1-(-1) mask to the likelihood shift.
        return linear_gaussian(theta, mask * likelihood_shift, likelihood_cov)

    # Test whether NPE works properly with structured z-scoring.
    net = posterior_nn("maf", z_score_x="structured", hidden_features=20)

    inference = NPE_C(prior, density_estimator=net, show_progress_bars=True)

    # We need a pretty big dataset to properly model the bimodality.
    theta = prior.sample((num_simulations,))
    x = simulator(theta)
    posterior_estimator = inference.append_simulations(theta, x).train(
        training_batch_size=1000, max_num_epochs=60
    )

    # generate conditions
    posterior = DirectPosterior(
        prior=prior, posterior_estimator=posterior_estimator
    ).set_default_x(x_o)
    samples = posterior.sample((num_conditions,))

    # Evaluate the conditional density be drawing samples and smoothing with a Gaussian
    # kde.
    potential_fn, theta_transform = posterior_estimator_based_potential(
        posterior_estimator, prior=prior, x_o=x_o
    )
    (
        conditioned_potential_fn,
        restricted_tf,
        restricted_prior,
    ) = conditional_potential(
        potential_fn=potential_fn,
        theta_transform=theta_transform,
        prior=prior,
        condition=samples[0],
        dims_to_sample=[dim_to_sample_1, dim_to_sample_2],
    )
    conditioned_potential_fn.set_x(x_o, x_is_iid=False)
    mcmc_posterior = MCMCPosterior(
        potential_fn=conditioned_potential_fn,
        theta_transform=restricted_tf,
        proposal=restricted_prior,
        method="slice_np_vectorized",
        **mcmc_params_accurate,
    )
    cond_samples = mcmc_posterior.sample((num_conditional_samples,), x=x_o)

    limits = [[-2, 2], [-2, 2], [-2, 2]]

    # Fit a Gaussian KDE to the conditional samples and get log-probs.
    density = gaussian_kde(cond_samples.numpy().T, bw_method="scott")

    X, Y = np.meshgrid(
        np.linspace(limits[0][0], limits[0][1], 50),
        np.linspace(limits[1][0], limits[1][1], 50),
    )
    positions = np.vstack([X.ravel(), Y.ravel()])
    sample_kde_grid = np.reshape(density(positions).T, X.shape)

    # Get conditional log probs eval_conditional_density.
    eval_grid = analysis.eval_conditional_density(
        posterior,
        condition=samples[0],
        dim1=dim_to_sample_1,
        dim2=dim_to_sample_2,
        limits=torch.tensor([[-2, 2], [-2, 2], [-2, 2]]),
    )

    # Compare the two densities.
    sample_kde_grid = sample_kde_grid / np.sum(sample_kde_grid)
    eval_grid = eval_grid / torch.sum(eval_grid)

    error = np.abs(sample_kde_grid - eval_grid.numpy())

    max_err = np.max(error)
    assert max_err < 0.0027
    print(f"Max error: {max_err}")


def test_mdn_conditional_density(num_dim: int = 3, cond_dim: int = 1):
    """Test whether the conditional density infered from MDN parameters of a
    `DirectPosterior` matches analytical results for MVN. This uses a n-D joint and
    conditions on the last m values to generate a conditional.

    Gaussian prior used for easier ground truthing of conditional posterior.

    Args:
        num_dim: Dimensionality of the MVM.
        cond_dim: Dimensionality of the condition.
    """

    assert (
        num_dim > cond_dim
    ), "The number of dimensions needs to be greater than that of the condition!"

    x_o = zeros(1, num_dim)
    num_samples = 1000
    num_simulations = 2700
    condition = 0.1 * ones(1, num_dim)

    dims = list(range(num_dim))
    dims2condition = dims[:-cond_dim]

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)

    joint_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    joint_cov = joint_posterior.covariance_matrix
    joint_mean = joint_posterior.loc

    conditional_mean, conditional_cov = conditional_of_mvn(
        joint_mean, joint_cov, condition[0, dims2condition]
    )
    conditional_dist_gt = MultivariateNormal(conditional_mean, conditional_cov)

    conditional_samples_gt = conditional_dist_gt.sample((num_samples,))

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inference = NPE_C(density_estimator="mdn", show_progress_bars=False)

    theta = prior.sample((num_simulations,))
    x = simulator(theta)

    posterior_mdn = inference.append_simulations(theta, x).train(
        training_batch_size=100
    )
    conditioned_mdn = ConditionedMDN(
        posterior_mdn, x_o, condition=condition, dims_to_sample=[0]
    )
    conditional_samples_sbi = conditioned_mdn.sample((num_samples,))
    check_c2st(
        conditional_samples_sbi,
        conditional_samples_gt,
        alg="analytic_mdn_conditioning_of_direct_posterior",
    )


@pytest.mark.parametrize("npe_method", [NPE_A, NPE_C])
def test_example_posterior(npe_method: type):
    """Return an inferred `NeuralPosterior` for interactive examination."""
    num_dim = 2
    x_o = zeros(1, num_dim)
    num_simulations = 100

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)

    extra_kwargs = dict(final_round=True) if npe_method == NPE_A else dict()

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inference = npe_method(prior, show_progress_bars=False)
    theta = prior.sample((num_simulations,))
    x = simulator(theta)
    posterior_estimator = inference.append_simulations(theta, x).train(
        max_num_epochs=2, **extra_kwargs
    )
    if npe_method == NPE_A:
        posterior_estimator = inference.correct_for_proposal()
    posterior = DirectPosterior(
        prior=prior, posterior_estimator=posterior_estimator
    ).set_default_x(x_o)
    assert posterior is not None


@pytest.mark.slow
def test_multiround_mog_training():
    "Test whether multi-round training with MDNs is stable. See #669."

    def simulator(theta):
        return theta + torch.randn(theta.shape)

    dim = 15
    x_o = torch.zeros((1, dim))

    prior = utils.BoxUniform(-3 * torch.ones(dim), 3 * torch.ones(dim))

    proposal = prior
    inference = NPE_C(prior, density_estimator="mdn")

    for _ in range(3):
        theta = proposal.sample((200,))
        x = simulator(theta)
        _ = inference.append_simulations(theta, x, proposal=proposal).train()
        posterior = inference.build_posterior().set_default_x(x_o)
        proposal = posterior
