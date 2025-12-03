# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from dataclasses import asdict, dataclass
from itertools import product
from typing import List, Literal, Optional, Tuple

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
    MCMCPosterior,
    VectorFieldPosterior,
    simulate_for_sbi,
    vector_field_estimator_based_potential,
)
from sbi.inference.posteriors import MCMCPosteriorParameters
from sbi.neural_nets.factory import posterior_flow_nn
from sbi.simulators import linear_gaussian
from sbi.simulators.linear_gaussian import (
    samples_true_posterior_linear_gaussian_uniform_prior,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils import BoxUniform
from sbi.utils.metrics import check_c2st
from sbi.utils.user_input_checks import process_simulator

IID_METHODS = ["fnpe", "gauss", "auto_gauss", "jac_gauss"]
NUM_DIM = [1, 2, 3]
NUM_TRIAL = [1, 3, 8, 16]
PRIOR_TYPE = ["gaussian", "uniform", None]
SAMPLING_METHODS = ["sde", "ode"]
SDE_TYPE = ["vp", "ve", "subvp", "fmpe"]
VF_ESTIMATOR = ["mlp", "ada_mlp", "transformer"]


@dataclass(frozen=True)
class VectorFieldTrainingTestCase:
    """Defines a Vector Field training test case."""

    num_dim: int
    prior_type: Optional[str]
    vector_field_type: Literal["vp", "ve", "subvp", "fmpe"]
    vf_estimator: Literal["mlp", "ada_mlp", "transformer"]

    def __str__(self):
        return (
            f"{self.prior_type}_prior-dim_{self.num_dim}-{self.vector_field_type}"
            f"-{self.vf_estimator}"
        )

    @property
    def likelihood_shift(self):
        return -ones(self.num_dim)

    @property
    def likelihood_cov(self):
        return 0.3 * eye(self.num_dim)

    @property
    def prior_mean(self):
        if self.prior_type == "gaussian":
            return zeros(self.num_dim)
        else:
            return None

    @property
    def prior_cov(self):
        if self.prior_type == "gaussian":
            return eye(self.num_dim)
        else:
            return None

    @property
    def prior(self):
        if self.prior_type == "gaussian":
            prior = MultivariateNormal(
                loc=self.prior_mean, covariance_matrix=self.prior_cov
            )
        elif self.prior_type == "uniform":
            prior = utils.BoxUniform(
                -2.0 * ones(self.num_dim), 2.0 * ones(self.num_dim)
            )
        elif self.prior_type is None:
            prior = None
        else:
            raise NotImplementedError(f"Unsupported prior type: {self.prior_type}")
        return prior

    def _get_default_prior(self):
        if self.prior is not None:
            return self.prior
        else:
            return BoxUniform(-2.0 * ones(self.num_dim), 2.0 * ones(self.num_dim))

    def get_training_data(self, num_simulations: int):
        theta = self._get_default_prior().sample((num_simulations,))
        x = linear_gaussian(theta, self.likelihood_shift, self.likelihood_cov)
        return theta, x

    def get_target_samples(self, num_samples: int, x_o):
        if self.prior_type == "gaussian":
            gt_posterior = true_posterior_linear_gaussian_mvn_prior(
                x_o,
                self.likelihood_shift,
                self.likelihood_cov,
                self.prior_mean,  # type: ignore
                self.prior_cov,  # type: ignore
            )
            target_samples = gt_posterior.sample((num_samples,))
        elif self.prior_type in {"uniform", None}:
            prior: BoxUniform = self._get_default_prior()  # type: ignore
            target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
                x_o,
                self.likelihood_shift,
                self.likelihood_cov,
                prior=prior,
                num_samples=num_samples,
            )
        else:
            raise NotImplementedError(f"Unsupported prior type: {self.prior_type}")
        return target_samples


@dataclass(frozen=True)
class VectorFieldSamplingTestCase:
    iid_method: str
    sampling_method: str
    num_trials: int

    def __str__(self):
        return f"{self.iid_method}-{self.sampling_method}-trials_{self.num_trials}"


training_test_cases_gaussian = [
    VectorFieldTrainingTestCase(1, "gaussian", "vp", 'mlp'),
    VectorFieldTrainingTestCase(1, "gaussian", "ve", 'mlp'),
    VectorFieldTrainingTestCase(1, "gaussian", "fmpe", 'mlp'),
    VectorFieldTrainingTestCase(2, "gaussian", "ve", 'mlp'),
    VectorFieldTrainingTestCase(2, "gaussian", "vp", 'mlp'),
    VectorFieldTrainingTestCase(2, "gaussian", "subvp", 'mlp'),
    VectorFieldTrainingTestCase(2, "gaussian", "fmpe", 'mlp'),
]

training_test_cases_uniform = [
    VectorFieldTrainingTestCase(2, "uniform", "ve", 'mlp'),
    VectorFieldTrainingTestCase(2, "uniform", "vp", 'mlp'),
    VectorFieldTrainingTestCase(2, "uniform", "subvp", 'mlp'),
    VectorFieldTrainingTestCase(2, "uniform", "fmpe", 'mlp'),
    VectorFieldTrainingTestCase(3, "uniform", "ve", 'mlp'),
    VectorFieldTrainingTestCase(3, "uniform", "vp", 'mlp'),
    VectorFieldTrainingTestCase(3, "uniform", "subvp", 'mlp'),
    VectorFieldTrainingTestCase(3, "uniform", "fmpe", 'mlp'),
    VectorFieldTrainingTestCase(3, "uniform", "ve", 'ada_mlp'),
    VectorFieldTrainingTestCase(3, "uniform", "vp", 'ada_mlp'),
    VectorFieldTrainingTestCase(3, "uniform", "subvp", 'ada_mlp'),
    VectorFieldTrainingTestCase(3, "uniform", "fmpe", 'ada_mlp'),
    VectorFieldTrainingTestCase(3, "uniform", "ve", 'transformer'),
    VectorFieldTrainingTestCase(3, "uniform", "vp", 'transformer'),
    VectorFieldTrainingTestCase(3, "uniform", "subvp", 'transformer'),
    VectorFieldTrainingTestCase(3, "uniform", "fmpe", 'transformer'),
]

training_test_cases_all = training_test_cases_gaussian + training_test_cases_uniform

sampling_test_cases_1_trial = [
    VectorFieldSamplingTestCase(iid, sampling, 1)
    for iid, sampling in product(IID_METHODS, SAMPLING_METHODS)
]

sampling_test_cases_n_trials = [
    VectorFieldSamplingTestCase("fnpe", "sde", 3),
    VectorFieldSamplingTestCase("gauss", "sde", 3),
    VectorFieldSamplingTestCase("auto_gauss", "sde", 3),
    VectorFieldSamplingTestCase("jac_gauss", "sde", 3),
    VectorFieldSamplingTestCase("fnpe", "sde", 8),
    VectorFieldSamplingTestCase("gauss", "sde", 8),
    VectorFieldSamplingTestCase("auto_gauss", "sde", 8),
    VectorFieldSamplingTestCase("jac_gauss", "sde", 8),
]

sampling_test_cases_all = sampling_test_cases_1_trial + sampling_test_cases_n_trials


def _get_regression_cases() -> List[
    Tuple[VectorFieldTrainingTestCase, VectorFieldSamplingTestCase]
]:
    """
    # ToDO check if there is a bug for prior_type='uniform' and num_trial>1
    # ToDO validate bug for combination sde_type='ode' and num_trial>1
    # ToDO investigate non-determinism for prior_type=None, dim>= 2, ve, auto/jac_gauss
    To make the regression tests run fast enough, we exclude certain combinations
    :return: list of combinations of training and sampling test cases
    """
    all_train = [
        VectorFieldTrainingTestCase(num_dim, prior_type, sde_type, estimator)  # type: ignore
        for num_dim, prior_type, sde_type, estimator in product(
            NUM_DIM, PRIOR_TYPE, SDE_TYPE, VF_ESTIMATOR
        )
    ]
    all_sample = [
        VectorFieldSamplingTestCase(iid_method, sampling_method, num_trials)
        for iid_method, sampling_method, num_trials in product(
            IID_METHODS, SAMPLING_METHODS, NUM_TRIAL
        )
    ]
    all_combinations = product(all_train, all_sample)

    is_uniform = lambda t: t.prior_type == "uniform"
    too_many_trial = lambda s: s.num_trials > 1
    is_ode = lambda s: s.sampling_method == "ode"

    is_non_deterministic = (
        lambda t, s: t.prior_type is None
        and t.vector_field_type == "ve"
        and t.num_dim >= 2
        and s.iid_method in {"auto_gauss", "jac_gauss"}
        and s.num_trials >= 16
    )

    exclude_cond = lambda t, s: (
        (is_uniform(t) or is_ode(s)) and too_many_trial(s)
    ) or is_non_deterministic(t, s)
    combinations = []
    for train_case, sampling_case in all_combinations:
        if not exclude_cond(train_case, sampling_case):
            combinations.append((train_case, sampling_case))
    return combinations


def _train_vector_field(
    test_case: VectorFieldTrainingTestCase,
    num_simulations,
    stop_after_epochs=None,
    training_batch_size=None,
    max_num_epochs=None,
):
    if test_case.vector_field_type == "fmpe":
        inference = FMPE(prior=test_case.prior, show_progress_bars=True)
    else:
        inference = NPSE(
            prior=test_case.prior,
            show_progress_bars=True,
            sde_type=test_case.vector_field_type,
        )
    theta, x = test_case.get_training_data(num_simulations)

    kwargs = {}
    if stop_after_epochs is not None:
        kwargs["stop_after_epochs"] = stop_after_epochs
    if training_batch_size is not None:
        kwargs["training_batch_size"] = training_batch_size
    if max_num_epochs is not None:
        kwargs["max_num_epochs"] = max_num_epochs

    score_estimator = inference.append_simulations(theta, x).train(**kwargs)
    return inference, score_estimator


def _build_posterior(inference, score_estimator, x_o, prior=None, sample_with=None):
    kwargs = {}
    if prior is not None:
        kwargs["prior"] = prior
    if sample_with is not None:
        kwargs["sample_with"] = sample_with

    posterior = inference.build_posterior(score_estimator, **kwargs)
    posterior.set_default_x(x_o)
    return posterior


@pytest.fixture(scope="module")
def vector_field_trained_model(request):
    # TODO: Move those up to top of file as global constants for better visibility.
    num_simulations = 5_000
    stop_after_epochs = 200
    training_batch_size = 100
    test_case: VectorFieldTrainingTestCase = request.param
    if test_case.vector_field_type == "fmpe":
        inference = FMPE(prior=test_case.prior, show_progress_bars=True)
    else:
        inference = NPSE(
            prior=test_case.prior,
            show_progress_bars=True,
            sde_type=test_case.vector_field_type,
        )
    theta, x = test_case.get_training_data(num_simulations)
    inference.append_simulations(theta, x)
    inference, score_estimator = _train_vector_field(
        test_case, num_simulations, stop_after_epochs, training_batch_size
    )
    return inference, score_estimator, test_case


@pytest.mark.parametrize(
    "training_test_case, sampling_test_case", _get_regression_cases(), ids=str
)
def test_vector_field_snapshot(
    sampling_test_case: VectorFieldSamplingTestCase,
    training_test_case: VectorFieldTrainingTestCase,
    ndarrays_regression,
):
    num_simulations = 5
    num_samples = 2
    steps = 5

    inference, score_estimator = _train_vector_field(
        training_test_case,
        num_simulations,
        stop_after_epochs=1,
        max_num_epochs=1,
        training_batch_size=None,
    )
    x_o = zeros(sampling_test_case.num_trials, training_test_case.num_dim)
    posterior: VectorFieldPosterior = _build_posterior(
        inference,
        score_estimator,
        x_o,
        prior=training_test_case.prior,
        sample_with=sampling_test_case.sampling_method,
    )  # type: ignore
    samples = posterior.sample(
        (num_samples,),
        iid_method=sampling_test_case.iid_method,  # type: ignore
        steps=steps,
    )
    ndarrays_regression.check(
        {'values': samples.numpy()}, default_tolerance=dict(atol=1e-3, rtol=1e-2)
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "vector_field_trained_model", training_test_cases_all, indirect=True, ids=str
)
@pytest.mark.parametrize("sampling_test_case", sampling_test_cases_all, ids=str)
def test_c2st(
    vector_field_trained_model, sampling_test_case: VectorFieldSamplingTestCase
):
    num_samples = 1_000
    inference, score_estimator, test_case = vector_field_trained_model
    x_o = zeros(sampling_test_case.num_trials, test_case.num_dim)
    posterior = _build_posterior(
        inference,
        score_estimator,
        x_o,
        prior=test_case.prior,
        sample_with=sampling_test_case.sampling_method,
    )
    npse_samples = posterior.sample(
        (num_samples,), iid_method=sampling_test_case.iid_method
    )
    check_c2st(
        npse_samples,
        test_case.get_target_samples(npse_samples.shape[0], x_o),
        alg=f"vector_field-{test_case.vector_field_type or 'vp'}"
        f"-{test_case.vector_field_type}"
        f"-{test_case.prior_type}"
        f"-{test_case.num_dim}D"
        f"-{sampling_test_case.sampling_method}",
        tol=0.05 * min(sampling_test_case.num_trials, 8),
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "vector_field_trained_model", training_test_cases_gaussian, indirect=True, ids=str
)
def test_vector_field_map(vector_field_trained_model):
    inference, score_estimator, test_case = vector_field_trained_model
    x_o = zeros(1, test_case.num_dim)
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o,
        test_case.likelihood_shift,
        test_case.likelihood_cov,
        test_case.prior_mean,
        test_case.prior_cov,
    )

    map_ = (
        inference.build_posterior()
        .set_default_x(x_o)
        .map(show_progress_bars=True, num_iter=5)
    )
    assert ((map_ - gt_posterior.mean) ** 2).sum() < 0.5, "MAP is not close to GT."


# NOTE: Using a function with explicit caching instead of a parametrized fixture here to
# make the test cases below more readable and maintainable.

_trained_models_cache = {}


def train_vector_field_model(vector_field_type, prior_type):
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
        prior_npse = prior
    elif prior_type == "uniform":
        prior = BoxUniform(-2 * ones(num_dim), 2 * ones(num_dim))
        prior_npse = prior

    # This check that our method to handle "general" priors works.
    # i.e. if NPSE does not get a proper passed by the user.
    if vector_field_type == "fmpe":
        inference = FMPE(prior_npse, show_progress_bars=True)
    else:
        inference = NPSE(
            prior_npse, show_progress_bars=True, sde_type=vector_field_type
        )

    theta = prior.sample((num_simulations,))
    x = linear_gaussian(theta, likelihood_shift, likelihood_cov)

    estimator = inference.append_simulations(theta, x).train()

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
    "vector_field_type, prior_type", [("ve", "gaussian"), ("fmpe", "gaussian")]
)
def test_vector_field_sde_ode_sampling_equivalence(vector_field_type, prior_type):
    """
    Test whether SDE and ODE sampling are equivalent
    for FMPE and NPSE.
    """
    vector_field_trained_model = train_vector_field_model(vector_field_type, prior_type)

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
@pytest.mark.parametrize("vector_field_type", ["ve", "vp", "fmpe"])
@pytest.mark.parametrize("prior_type", ["gaussian"])
@pytest.mark.parametrize("iid_batch_size", [1, 2, 5])
def test_iid_log_prob(vector_field_type, prior_type, iid_batch_size):
    '''
    Tests the log-probability computation of the score-based posterior.

    '''

    vector_field_trained_model = train_vector_field_model(vector_field_type, prior_type)

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
