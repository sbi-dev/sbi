# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from torch import Tensor, eye, norm, ones, zeros
from torch.distributions import MultivariateNormal, biject_to

from sbi.inference import MCABC, SMC
from sbi.inference.abc.abc_base import ABCBASE
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    samples_true_posterior_linear_gaussian_uniform_prior,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils.kde import KDEWrapper
from sbi.utils.metrics import check_c2st
from sbi.utils.torchutils import BoxUniform


@pytest.mark.slow
@pytest.mark.parametrize("num_dim", (1, 2))
@pytest.mark.parametrize("distance", ("l1", lambda x, xo: norm((x - xo), dim=-1)))
@pytest.mark.parametrize("eps", (0.08, None))
def test_mcabc_performance(num_dim: int, distance: str, eps: float):
    """Main MCABC performance test with c2st validation."""
    x_o = zeros((1, num_dim))
    num_samples = 1000
    num_simulations = 120000
    dim_scaled_eps = eps * num_dim if eps is not None else None

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = gt_posterior.sample((num_samples,))

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inferer = MCABC(
        simulator, prior, simulation_batch_size=num_simulations, distance=distance
    )

    phat = inferer(
        x_o,
        num_simulations,
        eps=dim_scaled_eps,
        quantile=num_samples / num_simulations if eps is None else None,
        return_summary=False,
    )

    if not isinstance(phat, Tensor):
        raise TypeError("Expected phat to be a Tensor, got: {}".format(type(phat)))
    posterior_samples: Tensor = phat[:num_samples]
    print(posterior_samples.shape)
    check_c2st(posterior_samples, target_samples, alg=f"MCABC_{num_dim}d")


@pytest.mark.slow
@pytest.mark.parametrize(
    "prior_type, algorithm_variant, kernel",
    [
        ("uniform", "A", "gaussian"),
        ("gaussian", "B", "uniform"),
        ("gaussian", "C", "gaussian"),
    ],
)
def test_smcabc_performance(prior_type: str, algorithm_variant: str, kernel: str):
    """Main SMCABC performance test with c2st validation."""
    num_dim = 2
    x_o = zeros((1, num_dim))
    num_samples = 1000
    num_simulations = 5000  # Reduced for faster tests
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    if prior_type == "gaussian":
        prior_mean = zeros(num_dim)
        prior_cov = eye(num_dim)
        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
        gt_posterior = true_posterior_linear_gaussian_mvn_prior(
            x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
        )
        target_samples = gt_posterior.sample((num_samples,))
    elif prior_type == "uniform":
        prior = BoxUniform(-ones(num_dim), ones(num_dim))
        target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
            x_o, likelihood_shift, likelihood_cov, prior, num_samples
        )

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    infer = SMC(
        simulator,
        prior,
        distance="l2",
        simulation_batch_size=num_simulations,
        algorithm_variant=algorithm_variant,
        kernel=kernel,
    )

    phat = infer(
        x_o,
        num_particles=1000,
        num_initial_pop=5000,
        epsilon_decay=0.5,
        num_simulations=num_simulations,
        distance_based_decay=True,
        return_summary=False,
    )

    if not isinstance(phat, Tensor):
        raise TypeError("Expected phat to be a Tensor, got: {}".format(type(phat)))

    check_c2st(
        phat[:num_samples],
        target_samples,
        alg=f"SMCABC-{prior_type}-{algorithm_variant}",
    )


# Fast helper functions for coverage testing (no c2st)
def _fast_mcabc_test(**kwargs):
    """Fast MCABC test without c2st check for coverage testing."""
    defaults = {
        "num_dim": 1,
        "distance": "l2",
        "eps": None,
        "quantile": 0.1,
        "lra": False,
        "sass": False,
        "sass_expansion_degree": 1,
        "kde": False,
        "kde_bandwidth": "cv",
        "num_iid_samples": 1,
        "distance_kwargs": None,
        "num_simulations": 1000,
    }
    defaults.update(kwargs)

    x_o = zeros((defaults["num_iid_samples"], defaults["num_dim"]))
    likelihood_shift = -1.0 * ones(defaults["num_dim"])
    likelihood_cov = 0.3 * eye(defaults["num_dim"])
    prior = MultivariateNormal(zeros(defaults["num_dim"]), eye(defaults["num_dim"]))

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inferer = MCABC(
        simulator,
        prior,
        simulation_batch_size=500,
        distance=defaults["distance"],
        distance_kwargs=defaults["distance_kwargs"],
    )

    phat = inferer(
        x_o,
        defaults["num_simulations"],
        eps=defaults["eps"],
        quantile=defaults["quantile"],
        lra=defaults["lra"],
        sass=defaults["sass"],
        sass_expansion_degree=defaults["sass_expansion_degree"],
        sass_fraction=0.33,
        kde=defaults["kde"],
        kde_kwargs=dict(bandwidth=defaults["kde_bandwidth"]) if defaults["kde"] else {},
        return_summary=False,
        num_iid_samples=defaults["num_iid_samples"],
    )

    # Verify we got results
    if defaults["kde"]:
        assert hasattr(phat, 'sample')
        samples = phat.sample((10,))
        assert samples.shape[0] == 10
    else:
        assert phat.shape[0] > 0


def _fast_smcabc_test(**kwargs):
    """Fast SMCABC test without c2st check for coverage testing."""
    defaults = {
        "num_dim": 1,
        "prior_type": "gaussian",
        "algorithm_variant": "C",
        "kernel": "gaussian",
        "distance": "l2",
        "lra": False,
        "sass": False,
        "kde": False,
        "ess_min": None,
        "distance_based_decay": True,
        "num_iid_samples": 1,
        "distance_kwargs": None,
        "num_simulations": 1000,
    }
    defaults.update(kwargs)

    x_o = zeros((defaults["num_iid_samples"], defaults["num_dim"]))
    likelihood_shift = -1.0 * ones(defaults["num_dim"])
    likelihood_cov = 0.3 * eye(defaults["num_dim"])

    if defaults["prior_type"] == "gaussian":
        prior = MultivariateNormal(zeros(defaults["num_dim"]), eye(defaults["num_dim"]))
    else:  # uniform
        prior = BoxUniform(-ones(defaults["num_dim"]), ones(defaults["num_dim"]))

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    infer = SMC(
        simulator,
        prior,
        distance=defaults["distance"],
        simulation_batch_size=300,
        algorithm_variant=defaults["algorithm_variant"],
        kernel=defaults["kernel"],
        distance_kwargs=defaults["distance_kwargs"],
    )

    phat = infer(
        x_o,
        num_particles=50,
        num_initial_pop=200,
        epsilon_decay=0.5,
        num_simulations=defaults["num_simulations"],
        distance_based_decay=defaults["distance_based_decay"],
        return_summary=False,
        lra=defaults["lra"],
        sass=defaults["sass"],
        sass_fraction=0.5,
        kde=defaults["kde"],
        kde_kwargs=(
            dict(
                bandwidth="cv",
                transform=(
                    biject_to(prior.support)
                    if defaults["kde"] and defaults["prior_type"] == "uniform"
                    else None
                ),
            )
            if defaults["kde"]
            else {}
        ),
        num_iid_samples=defaults["num_iid_samples"],
        ess_min=defaults["ess_min"],
    )

    # Verify we got results
    if defaults["kde"]:
        if not isinstance(phat, KDEWrapper):
            raise TypeError(
                "Expected phat to be a KDEWrapper, got: {}".format(type(phat))
            )
        samples = phat.sample((10,))
        assert samples.shape[0] == 10
        phat.log_prob(samples)
    else:
        if not isinstance(phat, Tensor):
            raise TypeError("Expected phat to be a Tensor, got: {}".format(type(phat)))
        assert phat.shape[0] > 0


# Comprehensive coverage tests using fast helpers
@pytest.mark.parametrize(
    "test_params",
    [
        # Basic functionality
        pytest.param({}, id="mcabc-basic"),
        # SASS coverage
        pytest.param({"sass": True, "sass_expansion_degree": 1}, id="mcabc-sass-deg1"),
        pytest.param({"sass": True, "sass_expansion_degree": 2}, id="mcabc-sass-deg2"),
        # LRA coverage
        pytest.param({"lra": True}, id="mcabc-lra"),
        # Combined SASS+LRA
        pytest.param({"sass": True, "lra": True}, id="mcabc-sass-lra"),
        # KDE coverage
        pytest.param({"kde": True, "kde_bandwidth": "cv"}, id="mcabc-kde-cv"),
        pytest.param(
            {"kde": True, "kde_bandwidth": "silvermann"}, id="mcabc-kde-silvermann"
        ),
        pytest.param({"kde": True, "kde_bandwidth": 0.1}, id="mcabc-kde-numeric"),
        # IID inference
        pytest.param(
            {
                "distance": "mmd",
                "num_iid_samples": 3,
                "distance_kwargs": {"scale": 1.0},
            },
            id="mcabc-iid-mmd",
        ),
        # Custom distance
        pytest.param(
            {"distance": lambda x, xo: norm((x - xo), dim=-1)},
            id="mcabc-custom-distance",
        ),
        # Eps vs quantile
        pytest.param({"eps": 0.1, "quantile": None}, id="mcabc-eps"),
    ],
)
def test_mcabc_coverage(test_params):
    """Comprehensive MCABC coverage tests."""
    _fast_mcabc_test(**test_params)


@pytest.mark.parametrize(
    "test_params",
    [
        # Algorithm variants
        pytest.param({"algorithm_variant": "A"}, id="smcabc-algo-A"),
        pytest.param({"algorithm_variant": "B"}, id="smcabc-algo-B"),
        # Kernels
        pytest.param({"kernel": "uniform"}, id="smcabc-uniform-kernel"),
        # Decay types
        pytest.param({"distance_based_decay": False}, id="smcabc-constant-decay"),
        # ESS resampling
        pytest.param(
            {"algorithm_variant": "B", "ess_min": 0.5}, id="smcabc-ess-resample"
        ),
        # SASS/LRA
        pytest.param({"sass": True}, id="smcabc-sass"),
        pytest.param({"lra": True}, id="smcabc-lra"),
        pytest.param({"sass": True, "lra": True}, id="smcabc-sass-lra"),
        # KDE
        pytest.param({"kde": True, "prior_type": "uniform"}, id="smcabc-kde"),
        # IID inference
        pytest.param(
            {
                "distance": "mmd",
                "num_iid_samples": 3,
                "distance_kwargs": {"scale": 1.0},
            },
            id="smcabc-iid-mmd",
        ),
        pytest.param(
            {
                "distance": "wasserstein",
                "num_iid_samples": 2,
                "distance_kwargs": {"epsilon": 1.0, "tol": 1e-6, "max_iter": 100},
            },
            id="smcabc-iid-wasserstein",
        ),
    ],
)
def test_smcabc_coverage(test_params):
    """Comprehensive SMCABC coverage tests."""
    _fast_smcabc_test(**test_params)


# Error handling and edge cases
def test_mcabc_eps_too_small():
    """Test MCABC error handling for eps too small."""
    x_o = zeros((1, 1))
    prior = MultivariateNormal(zeros(1), eye(1))
    simulator = lambda theta: linear_gaussian(theta, -ones(1), 0.3 * eye(1))

    inferer = MCABC(simulator, prior)
    with pytest.raises(AssertionError):
        inferer(x_o, 100, eps=1e-12)


@pytest.mark.parametrize(
    "test_type, distance_name",
    [
        ("custom_warning", None),
        ("invalid_name", "invalid_distance"),
    ],
)
def test_distance_function_edge_cases(test_type: str, distance_name: str | None):
    """Test distance function edge cases."""
    from sbi.utils.metrics import Distance

    if test_type == "custom_warning":

        def custom_distance(x, y):
            return torch.norm(x - y)

        with pytest.warns(
            UserWarning, match="Please specify if your the custom distance"
        ):
            distance_metric = Distance(custom_distance, requires_iid_data=None)
        assert not distance_metric.requires_iid_data

    elif test_type == "invalid_name" and distance_name is not None:
        with pytest.raises(AssertionError, match="must be one of"):
            Distance(distance_name)


# Minimal mathematical correctness tests
def test_lra_correctness():
    """Test LRA mathematical correctness."""
    torch.manual_seed(42)
    n_samples, n_params, n_features = 50, 2, 3

    theta = torch.randn(n_samples, n_params)
    # Linear relationship: x = theta @ A.T + noise
    A = torch.tensor([[1.0, 2.0], [0.5, -1.0], [2.0, 1.0]])
    x = theta @ A.T + 0.1 * torch.randn(n_samples, n_features)
    observation = torch.tensor([[1.0, 0.0, 0.5]])

    theta_adjusted = ABCBASE._run_lra(theta, x, observation)

    # LRA should produce meaningful adjustment
    assert torch.norm(theta_adjusted - theta) > 0.01

    # Check that adjustment improves prediction
    error_original = torch.norm(theta @ A.T - observation, dim=1).mean()
    error_adjusted = torch.norm(theta_adjusted @ A.T - observation, dim=1).mean()
    assert error_adjusted < error_original


def test_sass_correctness():
    """Test SASS mathematical correctness."""
    torch.manual_seed(42)
    n_samples, n_params, n_features = 50, 2, 4

    theta = torch.randn(n_samples, n_params)
    A = torch.randn(n_params, n_features)
    x = theta @ A + 0.1 * torch.randn(n_samples, n_features)

    sass_transform = ABCBASE._get_sass_transform(theta, x, expansion_degree=1)
    x_test = torch.randn(10, n_features)
    x_transformed = sass_transform(x_test)

    # SASS should reduce to parameter dimensionality
    assert x_transformed.shape[1] == n_params

    # Should be deterministic
    x_transformed_2 = sass_transform(x_test)
    assert torch.allclose(x_transformed, x_transformed_2)
