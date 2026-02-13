# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from dataclasses import asdict
from typing import List, Literal, cast

import numpy as np
import pytest
import torch
from scipy.stats import gaussian_kde
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import analysis as analysis
from sbi import utils as utils
from sbi.analysis import conditional_potential
from sbi.inference import (
    FMPE,
    NPSE,
    FlowMatchingSimformer,
    MCMCPosterior,
    Simformer,
    VectorFieldPosterior,
    simulate_for_sbi,
    vector_field_estimator_based_potential,
)
from sbi.inference.posteriors import MCMCPosteriorParameters
from sbi.inference.posteriors.posterior_parameters import VectorFieldPosteriorParameters
from sbi.neural_nets.factory import posterior_flow_nn
from sbi.simulators import linear_gaussian
from sbi.simulators.linear_gaussian import (
    samples_true_posterior_linear_gaussian_mvn_prior_different_dims,
    samples_true_posterior_linear_gaussian_uniform_prior,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils import BoxUniform
from sbi.utils.metrics import check_c2st
from sbi.utils.torchutils import process_device
from sbi.utils.user_input_checks import process_simulator

from .test_utils import get_dkl_gaussian_prior

# ------------------------------------------------------------------------------
# ------------------------------- FAST TESTS -----------------------------------
# ------------------------------------------------------------------------------


# We always test num_dim and sample_with with defaults and mark the rests as slow.
@pytest.mark.parametrize(
    "vf_estimator, vector_field_type, num_dim, prior_str, sample_with",
    [
        ("NeuralPosterior", "vp", 1, "gaussian", ["sde", "ode"]),
        ("NeuralPosterior", "vp", 3, "uniform", ["sde", "ode"]),
        ("NeuralPosterior", "vp", 3, "gaussian", ["sde", "ode"]),
        ("NeuralPosterior", "ve", 3, "uniform", ["sde", "ode"]),
        ("NeuralPosterior", "subvp", 3, "uniform", ["sde", "ode"]),
        ("NeuralPosterior", "flow", 1, "gaussian", ["sde", "ode"]),
        ("NeuralPosterior", "flow", 1, "uniform", ["sde", "ode"]),
        ("NeuralPosterior", "flow", 3, "gaussian", ["sde", "ode"]),
        ("NeuralPosterior", "flow", 3, "uniform", ["sde", "ode"]),
        ("Simformer", "ve", 1, "gaussian", ["sde", "ode"]),
        ("Simformer", "ve", 3, "gaussian", ["sde", "ode"]),
        ("Simformer", "flow", 3, "uniform", ["sde", "ode"]),
        pytest.param(
            "Simformer",
            "ve",
            3,
            "uniform",
            ["sde", "ode"],
            marks=[pytest.mark.slow],
        ),
        pytest.param(
            "Simformer",
            "vp",
            3,
            "uniform",
            ["sde", "ode"],
            marks=[pytest.mark.gpu, pytest.mark.slow],
        ),
        pytest.param(
            "Simformer",
            "subvp",
            3,
            "uniform",
            ["sde", "ode"],
            marks=[pytest.mark.gpu, pytest.mark.slow],
        ),
        pytest.param(
            "Simformer",
            "flow",
            1,
            "gaussian",
            ["sde", "ode"],
            marks=[pytest.mark.slow],
        ),
        pytest.param(
            "Simformer",
            "flow",
            1,
            "uniform",
            ["sde", "ode"],
            marks=[pytest.mark.slow],
        ),
        pytest.param(
            "Simformer",
            "flow",
            3,
            "gaussian",
            ["sde", "ode"],
            marks=[
                pytest.mark.slow,
            ],
        ),
    ],
)
def test_c2st_vector_field_on_linearGaussian(
    vf_estimator: str,
    vector_field_type,
    num_dim: int,
    prior_str: str,
    sample_with: List[Literal["sde", "ode"]],
):
    """
    Test whether NPSE and FMPE infer well a simple example with available ground truth.
    """

    num_samples = 1000
    num_simulations = 5000
    max_num_epochs = 100
    device = "cpu"
    tol = 0.15

    if vf_estimator == "Simformer":
        num_simulations = 5000
        max_num_epochs = 500

    device = process_device(device)
    x_o = zeros(1, num_dim, device=device)

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim, device=device)
    likelihood_cov = 0.3 * eye(num_dim, device=device)

    if prior_str == "gaussian":
        prior_mean = zeros(num_dim, device=device)
        prior_cov = eye(num_dim, device=device)
        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
        gt_posterior = true_posterior_linear_gaussian_mvn_prior(
            x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
        )
        target_samples = gt_posterior.sample((num_samples,))
    else:
        prior = utils.BoxUniform(
            -2.0 * ones(num_dim), 2.0 * ones(num_dim), device=device
        )
        target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
            x_o,
            likelihood_shift,
            likelihood_cov,
            prior=prior,
            num_samples=num_samples,
        )

    if vf_estimator == "NeuralPosterior":
        vf_params = {
            "prior": prior,
            "device": device,
        }
        if vector_field_type == "flow":
            inference = FMPE(**vf_params)
        else:
            inference = NPSE(sde_type=vector_field_type, **vf_params)  # type: ignore
    elif vf_estimator == "Simformer":
        vf_params = {
            "posterior_latent_idx": [0],
            "posterior_observed_idx": [1],
            "device": device,
        }
        if vector_field_type == "flow":
            inference = FlowMatchingSimformer(**vf_params)
        else:
            inference = Simformer(sde_type=vector_field_type, **vf_params)  # type: ignore

    theta = prior.sample((num_simulations,))
    x = linear_gaussian(theta, likelihood_shift, likelihood_cov)

    if vf_estimator == "Simformer":
        # Cast object to help pyright recognize the right type
        # This will avoid some warnings below
        inference = cast(Simformer | FlowMatchingSimformer, inference)
        # First unsqueeze to make the feature dimension appear (F), then concat
        inputs = torch.cat([theta.unsqueeze(1), x.unsqueeze(1)], dim=1)
        inference.append_simulations(inputs)
    else:
        inference = cast(NPSE | FMPE, inference)
        inference.append_simulations(theta, x)

    # Train for max_num_epochs
    _ = inference.train(max_num_epochs=max_num_epochs)

    # Amortize the training when testing sample_with.
    for method in sample_with:
        posterior = inference.build_posterior(
            prior=prior,
            sample_with=method,  # type: ignore
            posterior_parameters=VectorFieldPosteriorParameters(),
        )
        posterior.set_default_x(x_o)
        samples = posterior.sample((num_samples,))

        # Compute the c2st and assert it is near chance level of 0.5.
        check_c2st(
            samples,
            target_samples,
            alg=f"vector_field-{vector_field_type}-{prior_str}-{num_dim}D-{method}",
            tol=tol,
        )

    if prior_str == "gaussian" and vf_estimator == "NeuralPosterior":
        # For the Gaussian prior, we compute the KLd between ground truth and
        # posterior.

        # For type checking below.
        assert isinstance(posterior, VectorFieldPosterior)

        # Disable exact integration for the ODE solver to speed up the computation.
        # But this gives stochastic results -> increase max_dkl a bit

        posterior.potential_fn.neural_ode.update_params(
            exact=False,
            atol=1e-4,
            rtol=1e-4,
        )
        dkl = get_dkl_gaussian_prior(
            posterior,
            x_o[0],
            likelihood_shift,
            likelihood_cov,
            prior_mean,
            prior_cov,
        )

        max_dkl = 0.25

        assert dkl < max_dkl, (
            f"D-KL={dkl} is more than 2 stds above the average performance."
        )


# Simformer cannot be tested here as it requires a coherent input dimensionality
@pytest.mark.parametrize("vector_field_type", [NPSE, FMPE])
def test_c2st_vector_field_on_linearGaussian_different_dims(vector_field_type):
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
    inference = vector_field_type(prior=None)

    theta = prior.sample((num_simulations,))
    x = simulator(theta)

    # Test whether we can stop and resume.
    inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior().set_default_x(x_o)
    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(
        samples,
        target_samples,
        alg=f"{vector_field_type.__name__}_different_dims_and_resume_training",
    )


# Simformer cannot be tested here too, as it is always transformer-based
@pytest.mark.parametrize("vector_field_type", [NPSE, FMPE])
@pytest.mark.parametrize(
    "model", ["mlp", "ada_mlp", pytest.param("transformer", marks=[pytest.mark.slow])]
)
def test_vfinference_with_different_models(vector_field_type, model):
    """Test fmpe with different vector field estimators on linear Gaussian."""

    theta_dim = 3
    x_dim = 3

    x_o = zeros(1, x_dim)
    num_samples = 1000
    num_simulations = 2500

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(x_dim)
    likelihood_cov = 0.9 * eye(x_dim)

    prior_mean = zeros(theta_dim)
    prior_cov = eye(theta_dim)

    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = gt_posterior.sample((num_samples,))

    theta = prior.sample((num_simulations,))
    x = linear_gaussian(theta, likelihood_shift, likelihood_cov)

    estimator_build_fun = posterior_flow_nn(net=model)

    inference = vector_field_type(prior, vf_estimator=estimator_build_fun)

    inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior().set_default_x(x_o)
    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg=f"fmpe_{model}")


# ------------------------------------------------------------------------------
# -------------------------------- SLOW TESTS ----------------------------------
# ------------------------------------------------------------------------------


# NOTE: Using a function with explicit caching instead of a parametrized fixture here to
# make the test cases below more readable and maintainable.

_trained_models_cache = {}


def train_vector_field_model(vf_estimator, vector_field_type, prior_type):
    """Factory function that trains a score estimator for NPSE tests with caching."""
    cache_key = (vector_field_type, prior_type)

    # Return cached model if available
    if cache_key in _trained_models_cache:
        return _trained_models_cache[cache_key]

    # Train the model
    num_dim = 2
    num_simulations = 6000

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    # The likelihood covariance is increased to make the iid inference easier,
    # (otherwise the posterior gets too tight and the c2st is too high),
    # but it doesn't really improve the results for both FMPE and NPSE.
    likelihood_cov = 0.9 * eye(num_dim)

    if prior_type == "gaussian":
        prior_mean = zeros(num_dim)
        prior_cov = eye(num_dim)
        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    elif prior_type == "uniform":
        prior = BoxUniform(-2 * ones(num_dim), 2 * ones(num_dim))

    # This check that our method to handle "general" priors works.
    # i.e. if NPSE does not get a proper passed by the user.
    if vf_estimator == "NeuralPosterior":
        vf_params = {
            "prior": prior,
        }
        if vector_field_type == "flow":
            inference = FMPE(**vf_params)
        else:
            inference = NPSE(sde_type=vector_field_type, **vf_params)  # type: ignore
    elif vf_estimator == "Simformer":
        vf_params = {
            "posterior_latent_idx": [0],
            "posterior_observed_idx": [1],
        }
        if vector_field_type == "flow":
            inference = FlowMatchingSimformer(**vf_params)
        else:
            inference = Simformer(sde_type=vector_field_type, **vf_params)  # type: ignore

    theta = prior.sample((num_simulations,))
    x = linear_gaussian(theta, likelihood_shift, likelihood_cov)

    if vf_estimator == "Simformer":
        # Cast object to help pyright recognize the right type
        # This will avoid some warnings below
        inference = cast(Simformer | FlowMatchingSimformer, inference)
        # First unsqueeze to make the feature dimension appear (F), then concat
        inputs = torch.cat([theta.unsqueeze(1), x.unsqueeze(1)], dim=1)
        inference.append_simulations(inputs)
    else:
        inference = cast(NPSE | FMPE, inference)
        inference.append_simulations(theta, x)

    estimator = inference.train()

    result = {
        "estimator": estimator,
        "inference": inference,
        "prior": prior,
        "likelihood_shift": likelihood_shift,
        "likelihood_cov": likelihood_cov,
        "prior_mean": prior_mean if prior_type == "gaussian" else None,
        "prior_cov": prior_cov if prior_type == "gaussian" else None,
        "num_dim": num_dim,
        "vector_field_type": vector_field_type,
    }

    # Cache the result
    _trained_models_cache[cache_key] = result
    return result


@pytest.mark.slow
@pytest.mark.parametrize(
    "vf_estimator, vector_field_type, prior_type",
    [
        ("NeuralPosterior", "ve", "gaussian"),
        ("NeuralPosterior", "fmpe", "gaussian"),
        ("Simformer", "ve", "gaussian"),
        ("Simformer", "flow", "gaussian"),
    ],
)
def test_vector_field_sde_ode_sampling_equivalence(
    vf_estimator, vector_field_type, prior_type
):
    """
    Test whether SDE and ODE sampling are equivalent
    for FMPE, NPSE and Simformer.
    """
    vector_field_trained_model = train_vector_field_model(
        vf_estimator, vector_field_type, prior_type
    )

    num_samples = 1000
    x_o = zeros(1, vector_field_trained_model["num_dim"])

    inference = vector_field_trained_model["inference"]
    vector_field_type = vector_field_trained_model["vector_field_type"]
    sde_posterior = inference.build_posterior(sample_with="sde").set_default_x(x_o)
    ode_posterior = inference.build_posterior(sample_with="ode").set_default_x(x_o)

    sde_samples = sde_posterior.sample((num_samples,))
    ode_samples = ode_posterior.sample((num_samples,))

    check_c2st(
        sde_samples,
        ode_samples,
        alg=f"sample_methods_equivalence-{vector_field_type}",
        tol=0.07,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "vf_estimator, vector_field_type, prior_type",
    [
        ("NeuralPosterior", "ve", "gaussian"),
        ("NeuralPosterior", "vp", "gaussian"),
        ("NeuralPosterior", "subvp", "gaussian"),
        ("NeuralPosterior", "fmpe", "gaussian"),
        ("NeuralPosterior", "ve", "uniform"),
        ("NeuralPosterior", "vp", "uniform"),
        ("NeuralPosterior", "subvp", "uniform"),
        ("NeuralPosterior", "fmpe", "uniform"),
        # TODO: need to fine-tune hyper-parameters for Simformer
        # they all fail to converge properly with the current settings
        # TODO: need to handle bad shape management for Simformer
        pytest.param(
            "Simformer",
            "flow",
            "gaussian",
            marks=pytest.mark.skip(
                reason=(
                    "need to fine-tune hyper-parameters for Simformer, "
                    "they all fail to converge properly with the current settings; "
                    "need to handle bad shape management for Simformer"
                )
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "iid_method, num_trials",
    [
        pytest.param("fnpe", 5, id="fnpe-5trials"),
        pytest.param("gauss", 5, id="gauss-5trials"),
        pytest.param("auto_gauss", 5, id="auto_gauss-5trials"),
        pytest.param("jac_gauss", 5, id="jac_gauss-5trials"),
    ],
)
def test_vector_field_iid_inference(
    vf_estimator, vector_field_type, prior_type, iid_method, num_trials
):
    """
    Test whether NPSE and FMPE infers well a simple example with available ground truth.

    Args:
        vector_field_type: The type of vector field ("ve", "fmpe", etc.).
        prior_type: The type of prior distribution ("gaussian" or "uniform").
        iid_method: The IID method to use for sampling.
        num_trials: The number of trials to run.
    """

    vector_field_trained_model = train_vector_field_model(
        vf_estimator, vector_field_type, prior_type
    )

    # Extract data from the trained model
    estimator = vector_field_trained_model["estimator"]
    inference = vector_field_trained_model["inference"]
    prior = vector_field_trained_model["prior"]
    likelihood_shift = vector_field_trained_model["likelihood_shift"]
    likelihood_cov = vector_field_trained_model["likelihood_cov"]
    prior_mean = vector_field_trained_model["prior_mean"]
    prior_cov = vector_field_trained_model["prior_cov"]
    num_dim = vector_field_trained_model["num_dim"]

    num_samples = 1000

    if vf_estimator == "Simformer":
        # Simformer needs to differentiate between variables and features
        x_o = zeros(num_trials, 1, num_dim)
    else:
        x_o = zeros(num_trials, num_dim)

    posterior = inference.build_posterior(
        estimator,
        sample_with="sde",  # iid works only with score-based SDEs.
        posterior_parameters=VectorFieldPosteriorParameters(iid_method=iid_method),
    )
    posterior.set_default_x(x_o)
    samples = posterior.sample((num_samples,))

    if prior_type == "gaussian":
        gt_posterior = true_posterior_linear_gaussian_mvn_prior(
            x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
        )
        target_samples = gt_posterior.sample((num_samples,))
    elif prior_type == "uniform":
        target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
            x_o,
            likelihood_shift,
            likelihood_cov,
            prior,
        )
    else:
        raise ValueError(f"Invalid prior type: {prior_type}")

    # Compute the c2st and assert it is near chance level of 0.5.
    # Some degradation is expected, also because posterior get tighter which
    # usually makes the c2st worse.
    check_c2st(
        samples,
        target_samples,
        alg=(
            f"{vector_field_type}-{prior_type}-"
            f"{num_dim}-{iid_method}-{num_trials}iid-trials"
        ),
        tol=0.07 * max(num_trials, 2),
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "vector_field_type", ["npse", "fmpe", "simformer", "flow-simformer"]
)
def test_vector_field_map(vector_field_type):
    num_dim = 2
    x_o = zeros(num_dim)
    num_simulations = 3000
    max_num_epochs = 100

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )

    if vector_field_type == "npse":
        inference = NPSE(prior=prior)
    elif vector_field_type == "fmpe":
        inference = FMPE(prior=prior)
    elif vector_field_type == "simformer":
        inference = Simformer(
            posterior_latent_idx=[0],
            posterior_observed_idx=[1],
        )
    elif vector_field_type == "flow-simformer":
        inference = FlowMatchingSimformer(
            posterior_latent_idx=[0],
            posterior_observed_idx=[1],
        )
    else:
        raise ValueError(f"Invalid vector field type: {vector_field_type}")

    theta = prior.sample((num_simulations,))
    x = linear_gaussian(theta, likelihood_shift, likelihood_cov)

    if vector_field_type in {"simformer", "flow-simformer"}:
        inference = cast(Simformer | FlowMatchingSimformer, inference)
        inputs = torch.cat([theta.unsqueeze(1), x.unsqueeze(1)], dim=1)
        inference.append_simulations(inputs)
    else:
        inference = cast(NPSE | FMPE, inference)
        inference.append_simulations(theta, x)

    inference.train(max_num_epochs=max_num_epochs)

    posterior = inference.build_posterior().set_default_x(x_o)

    map_ = posterior.map(show_progress_bars=True, num_iter=5)

    assert ((map_ - gt_posterior.mean) ** 2).sum() < 0.5, "MAP is not close to GT."


# TODO: Need to add NPSE when the network builders are unified, but anyway
# this will only work after implementing additional methods for vector fields,
# so it is skipped for now.
@pytest.mark.slow
@pytest.mark.skip(reason="Potential evaluation is not implemented for iid yet.")
def test_sample_conditional():
    """
    Test whether sampling from the conditional gives the same results as evaluating.

    This compares samples that get smoothed with a Gaussian kde to evaluating the
    conditional log-probability with `eval_conditional_density`.

    `eval_conditional_density` is itself tested in `sbiutils_test.py`. Here, we use
    a bimodal posterior to test the conditional.
    """

    num_dim = 3
    dim_to_sample_1 = 0
    dim_to_sample_2 = 2
    num_simulations = 6000
    num_conditional_samples = 500

    mcmc_parameters = MCMCPosteriorParameters(
        method="slice_np_vectorized", num_chains=20, warmup_steps=50, thin=5
    )

    x_o = zeros(1, num_dim)

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.1 * eye(num_dim)

    prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))

    def simulator(theta):
        if torch.rand(1) > 0.5:
            return linear_gaussian(theta, likelihood_shift, likelihood_cov)
        else:
            return linear_gaussian(theta, -likelihood_shift, likelihood_cov)

    simulator = process_simulator(simulator, prior, False)

    # We need a pretty big dataset to properly model the bimodality.
    theta, x = simulate_for_sbi(
        simulator,
        prior,
        num_simulations,
        simulation_batch_size=10,  # choose small batch size to ensure bimoality.
    )

    # Test whether fmpe works properly with structured z-scoring.
    net = posterior_flow_nn(
        "mlp", z_score_x="structured", hidden_features=65, num_layers=5
    )

    inference = FMPE(prior, density_estimator=net, show_progress_bars=False)
    posterior_estimator = inference.append_simulations(theta, x).train(
        # max_num_epochs=60
    )

    posterior = VectorFieldPosterior(
        prior=prior, vector_field_estimator=posterior_estimator
    ).set_default_x(x_o)
    samples = posterior.sample((50,))

    # Evaluate the conditional density be drawing samples and smoothing with a Gaussian
    # kde.
    potential_fn, theta_transform = vector_field_estimator_based_potential(
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
        **asdict(mcmc_parameters),
    )
    mcmc_posterior.set_default_x(x_o)
    cond_samples = mcmc_posterior.sample((num_conditional_samples,))

    _ = analysis.pairplot(
        cond_samples,
        limits=[[-2, 2], [-2, 2], [-2, 2]],
        figsize=(2, 2),
        diag="kde",
        upper="kde",
    )

    limits = [[-2, 2], [-2, 2], [-2, 2]]

    density = gaussian_kde(cond_samples.numpy().T, bw_method="scott")

    X, Y = np.meshgrid(
        np.linspace(limits[0][0], limits[0][1], 50),
        np.linspace(limits[1][0], limits[1][1], 50),
    )
    positions = np.vstack([X.ravel(), Y.ravel()])
    sample_kde_grid = np.reshape(density(positions).T, X.shape)

    # Evaluate the conditional with eval_conditional_density.
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


@pytest.mark.slow
@pytest.mark.parametrize(
    "vf_estimator, vector_field_type, prior_type",
    [
        ("NeuralPosterior", "ve", "gaussian"),
        ("NeuralPosterior", "vp", "gaussian"),
        ("NeuralPosterior", "subvp", "gaussian"),
        ("NeuralPosterior", "fmpe", "gaussian"),
        # TODO: need to fine-tune hyper-parameters for Simformer
        # they all fail to converge properly with the current settings
        # TODO: need to handle bad shape management for Simformer
        pytest.param(
            "Simformer",
            "flow",
            "gaussian",
            marks=pytest.mark.skip(
                reason=(
                    "need to fine-tune hyper-parameters for Simformer, "
                    "they all fail to converge properly with the current settings; "
                    "need to handle bad shape management for Simformer"
                )
            ),
        ),
    ],
)
@pytest.mark.parametrize("iid_batch_size", [1, 2, 5])
def test_iid_log_prob(vf_estimator, vector_field_type, prior_type, iid_batch_size):
    '''
    Tests the log-probability computation of the score-based posterior.

    '''

    vector_field_trained_model = train_vector_field_model(
        vf_estimator, vector_field_type, prior_type
    )

    # Prior Gaussian
    prior = vector_field_trained_model["prior"]
    vf_estimator = vector_field_trained_model["estimator"]
    inference = vector_field_trained_model["inference"]
    likelihood_shift = vector_field_trained_model["likelihood_shift"]
    likelihood_cov = vector_field_trained_model["likelihood_cov"]
    prior_mean = vector_field_trained_model["prior_mean"]
    prior_cov = vector_field_trained_model["prior_cov"]
    num_dim = vector_field_trained_model["num_dim"]
    num_posterior_samples = 1000

    # Ground truth theta
    theta_o = zeros(num_dim)
    x_o = linear_gaussian(
        theta_o.repeat(iid_batch_size, 1),
        likelihood_shift=likelihood_shift,
        likelihood_cov=likelihood_cov,
    )
    true_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )

    approx_posterior = inference.build_posterior(vf_estimator, prior=prior)
    posterior_samples = true_posterior.sample((num_posterior_samples,))
    true_prob = true_posterior.log_prob(posterior_samples)
    approx_prob = approx_posterior.log_prob(posterior_samples, x=x_o)

    diff = torch.abs(true_prob - approx_prob)
    assert diff.mean() < 0.3 * iid_batch_size, (
        f"Probs diff: {diff.mean()} too big "
        f"for number of samples {num_posterior_samples}"
    )
