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
from sbi.utils.metrics import check_c2st
from sbi.utils.torchutils import BoxUniform


@pytest.mark.slow
@pytest.mark.parametrize("num_dim", (1, 2))
@pytest.mark.parametrize("distance", ("l2", lambda x, xo: norm((x - xo), dim=-1)))
@pytest.mark.parametrize("eps", (0.05, None))
def test_mcabc_inference_on_linear_gaussian(
    num_dim: int,
    distance: str,
    eps: float,
    lra: bool = False,
    sass: bool = False,
    sass_expansion_degree: int = 1,
    kde: bool = False,
    kde_bandwidth: str = "cv",
    num_iid_samples: int = 1,
    distance_kwargs: dict = None,
):
    x_o = zeros((num_iid_samples, num_dim))
    num_samples = 1000
    num_simulations = 500000
    dim_scaled_eps = eps * num_dim if eps is not None else None

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

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inferer = MCABC(
        simulator,
        prior,
        simulation_batch_size=10000,
        distance=distance,
        distance_kwargs=distance_kwargs,
    )

    phat = inferer(
        x_o,
        num_simulations,
        eps=dim_scaled_eps,
        quantile=num_samples / num_simulations if eps is None else None,
        lra=lra,
        sass=sass,
        sass_expansion_degree=sass_expansion_degree,
        sass_fraction=0.33,
        kde=kde,
        kde_kwargs=dict(bandwidth=kde_bandwidth) if kde else {},
        return_summary=False,
        num_iid_samples=num_iid_samples,
    )

    if kde:
        posterior_samples: Tensor = phat.sample((num_samples,))
    else:
        posterior_samples: Tensor = phat[:num_samples]

    check_c2st(
        posterior_samples,
        target_samples,
        alg=f"MCABC_lra{lra}_sass{sass}_kde{kde}_{kde_bandwidth}",
    )


@pytest.mark.slow
def test_mcabc_inference_on_linear_gaussian_eps_too_small():
    num_dim = 1
    x_o = zeros((1, num_dim))
    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))

    def simulator(theta):
        return linear_gaussian(theta, -1.0 * ones(num_dim), 0.3 * eye(num_dim))

    inferer = MCABC(simulator, prior)
    with pytest.raises(AssertionError):
        inferer(x_o, 100, eps=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize(
    "num_dim, prior_type, algorithm_variant, kernel, distance, lra, sass, "
    "sass_expansion_degree, kde, kde_bandwidth, transform, num_simulations, "
    "num_iid_samples, distance_kwargs, ess_min",
    [
        pytest.param(
            1,
            "uniform",
            "A",
            "gaussian",
            "l2",
            False,
            False,
            1,
            False,
            "cv",
            False,
            40000,
            1,
            None,
            None,
            id="1d-uniform-A-gauss",
        ),
        pytest.param(
            2,
            "gaussian",
            "B",
            "uniform",
            "l2",
            False,
            False,
            1,
            False,
            "cv",
            False,
            80000,
            1,
            None,
            None,
            id="2d-gaussian-B-uniform",
        ),
        # Resampling test
        pytest.param(
            1,
            "gaussian",
            "B",
            "gaussian",
            "l2",
            False,
            False,
            1,
            False,
            "cv",
            False,
            200000,
            1,
            None,
            0.5,
            id="resampling",
        ),
        # SASS/LRA tests
        pytest.param(
            2,
            "gaussian",
            "C",
            "gaussian",
            "l2",
            True,
            True,
            1,
            False,
            "cv",
            False,
            40000,
            1,
            None,
            None,
            id="sass-lra-true-deg1",
        ),
        pytest.param(
            2,
            "gaussian",
            "C",
            "gaussian",
            "l2",
            False,
            True,
            2,
            False,
            "cv",
            False,
            40000,
            1,
            None,
            None,
            id="sass-false-lra-true-deg2",
        ),
        # KDE test
        pytest.param(
            2,
            "uniform",
            "C",
            "gaussian",
            "l2",
            False,
            False,
            1,
            True,
            "cv",
            True,
            40000,
            1,
            None,
            None,
            id="kde-cv",
        ),
        # IID inference tests
        pytest.param(
            2,
            "gaussian",
            "C",
            "gaussian",
            "mmd",
            False,
            False,
            1,
            False,
            "cv",
            False,
            40000,
            20,
            {"scale": 1.0},
            None,
            id="iid-mmd",
        ),
        pytest.param(
            2,
            "gaussian",
            "C",
            "gaussian",
            "wasserstein",
            False,
            False,
            1,
            False,
            "cv",
            False,
            40000,
            10,
            {"epsilon": 1.0, "tol": 1e-6, "max_iter": 1000},
            None,
            id="iid-wasserstein",
        ),
    ],
)
def test_smcabc_inference_on_linear_gaussian(
    num_dim: int,
    prior_type: str,
    algorithm_variant: str,
    kernel: str,
    distance: str,
    lra: bool,
    sass: bool,
    sass_expansion_degree: int,
    kde: bool,
    kde_bandwidth: str,
    transform: bool,
    num_simulations: int,
    num_iid_samples: int,
    distance_kwargs: dict,
    ess_min: float,
):
    x_o = zeros((num_iid_samples, num_dim))
    num_samples = 1000
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
    else:
        raise ValueError("Wrong prior string.")

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    infer = SMC(
        simulator,
        prior,
        distance=distance,
        simulation_batch_size=num_simulations,
        algorithm_variant=algorithm_variant,
        kernel=kernel,
        distance_kwargs=distance_kwargs,
    )

    phat = infer(
        x_o,
        num_particles=1000,
        num_initial_pop=5000,
        epsilon_decay=0.5,
        num_simulations=num_simulations,
        distance_based_decay=True,
        return_summary=False,
        lra=lra,
        sass=sass,
        sass_fraction=0.5,
        sass_expansion_degree=sass_expansion_degree,
        kde=kde,
        kde_kwargs=dict(
            bandwidth=kde_bandwidth,
            transform=biject_to(prior.support) if transform else None,
        ),
        num_iid_samples=num_iid_samples,
        ess_min=ess_min,
    )

    check_c2st(
        phat.sample((num_samples,)) if kde else phat[:num_samples],
        target_samples,
        alg=f"SMCABC-{prior_type}prior-lra{lra}-sass{sass}-kde{kde}-{kde_bandwidth}",
    )

    if kde:
        samples = phat.sample((10,))
        phat.log_prob(samples)


@pytest.mark.slow
@pytest.mark.parametrize("lra", (True, False))
@pytest.mark.parametrize("sass_expansion_degree", (1, 2))
def test_mcabc_sass_lra(lra, sass_expansion_degree):
    test_mcabc_inference_on_linear_gaussian(
        num_dim=2,
        lra=lra,
        sass=True,
        sass_expansion_degree=sass_expansion_degree,
        distance="l2",
        eps=None,
    )


@pytest.mark.slow
@pytest.mark.parametrize("kde_bandwidth", ("cv", "silvermann", "scott", 0.1))
def test_mcabc_kde(kde_bandwidth):
    test_mcabc_inference_on_linear_gaussian(
        num_dim=2,
        kde=True,
        kde_bandwidth=kde_bandwidth,
        distance="l2",
        eps=None,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "distance, num_iid_samples, distance_kwargs",
    (
        ["l2", 1, None],
        ["mmd", 10, {"scale": 1.5}],
    ),
)
def test_mc_abc_iid_inference(distance, num_iid_samples, distance_kwargs):
    test_mcabc_inference_on_linear_gaussian(
        num_dim=2,
        distance=distance,
        num_iid_samples=num_iid_samples,
        distance_kwargs=distance_kwargs,
        eps=None,
    )


def test_lra_mathematical_correctness():
    """Test LRA with synthetic data where we know the expected adjustment."""
    torch.manual_seed(42)

    # Create synthetic data with known linear relationship
    n_samples, n_params, n_features = 100, 2, 3

    # True linear relationship: x = A * theta + noise
    A = torch.tensor([[1.0, 2.0], [0.5, -1.0], [2.0, 1.0]])  # [n_features, n_params]

    theta = torch.randn(n_samples, n_params)
    x = theta @ A.T + 0.1 * torch.randn(n_samples, n_features)

    observation = torch.tensor([[1.0, 0.0, 0.5]])  # [1, n_features]

    theta_adjusted = ABCBASE._run_lra(theta, x, observation)

    mean_adjustment = (theta_adjusted - theta).mean(dim=0)

    # The adjustment should be non-zero (LRA is doing something)
    assert torch.norm(mean_adjustment) > 0.01, "LRA should produce non-zero adjustment"

    # Check that the adjusted parameters better predict the observation
    x_pred_original = theta @ A.T
    x_pred_adjusted = theta_adjusted @ A.T

    error_original = torch.norm(x_pred_original - observation, dim=1).mean()
    error_adjusted = torch.norm(x_pred_adjusted - observation, dim=1).mean()

    # LRA should reduce prediction error
    assert error_adjusted < error_original, "LRA should reduce prediction error"


def test_lra_with_weights():
    """Test that LRA with sample weights works correctly."""
    torch.manual_seed(42)

    n_samples, n_params, n_features = 50, 2, 2

    theta = torch.randn(n_samples, n_params)
    x = theta + 0.1 * torch.randn(n_samples, n_features)
    observation = torch.zeros(1, n_features)

    # Create weights that favor some samples
    weights = torch.exp(-torch.sum(theta**2, dim=1))  # Higher weight near origin

    # Run LRA with and without weights
    theta_unweighted = ABCBASE._run_lra(theta, x, observation)
    theta_weighted = ABCBASE._run_lra(theta, x, observation, sample_weight=weights)

    # Weighted and unweighted should give different results
    assert torch.norm(theta_weighted - theta_unweighted) > 0.01, (
        "Weights should affect LRA result"
    )


def test_sass_dimension_reduction():
    """Test SASS with high-dimensional data where only some dimensions matter."""
    torch.manual_seed(42)

    n_samples, n_params = 100, 2
    n_features_relevant, n_features_noise = 3, 5
    n_features_total = n_features_relevant + n_features_noise

    theta = torch.randn(n_samples, n_params)

    # Create features where only first few dimensions depend on theta
    x_relevant = theta @ torch.randn(n_params, n_features_relevant)
    x_noise = torch.randn(n_samples, n_features_noise)  # Pure noise
    x = torch.cat([x_relevant, x_noise], dim=1)

    sass_transform = ABCBASE._get_sass_transform(theta, x, expansion_degree=1)

    x_test = torch.randn(10, n_features_total)
    x_transformed = sass_transform(x_test)

    # SASS should reduce dimensionality while preserving information
    # The transformed features should have fewer dimensions than original
    assert x_transformed.shape[1] == n_params, (
        f"SASS should output {n_params} features, got {x_transformed.shape[1]}"
    )

    x_transformed_2 = sass_transform(x_test)
    assert torch.allclose(x_transformed, x_transformed_2), (
        "SASS transform should be deterministic"
    )


def test_sass_information_preservation():
    """Test that SASS preserves information about parameter-feature relationships."""
    torch.manual_seed(42)

    n_samples, n_params, n_features = 100, 2, 4

    A = torch.randn(n_params, n_features)
    theta = torch.randn(n_samples, n_params)
    x = theta @ A + 0.1 * torch.randn(n_samples, n_features)

    sass_transform = ABCBASE._get_sass_transform(theta, x, expansion_degree=1)

    x_transformed = sass_transform(x)

    # The transformed features should still correlate with parameters
    # Check that we can still predict parameters from transformed features
    correlation_transformed = torch.corrcoef(torch.cat([theta, x_transformed], dim=1))

    # SASS-transformed features should maintain some correlation with parameters
    param_feature_corr_transformed = (
        correlation_transformed[:n_params, n_params:].abs().mean()
    )

    # Should preserve significant correlation
    assert param_feature_corr_transformed > 0.1, (
        "SASS should preserve parameter-feature relationships"
    )


def test_sass_polynomial_expansion():
    """Test SASS with polynomial expansion."""
    torch.manual_seed(42)

    n_samples, n_params, n_features = 50, 2, 2

    theta = torch.randn(n_samples, n_params)
    x = theta + theta**2 + 0.1 * torch.randn(n_samples, n_features)

    # Get SASS transform with polynomial expansion
    sass_transform_linear = ABCBASE._get_sass_transform(theta, x, expansion_degree=1)
    sass_transform_quad = ABCBASE._get_sass_transform(theta, x, expansion_degree=2)

    x_test = torch.randn(10, n_features)
    x_transformed_linear = sass_transform_linear(x_test)
    x_transformed_quad = sass_transform_quad(x_test)

    assert x_transformed_linear.shape[1] == n_params, (
        "Linear SASS should output correct dimensions"
    )
    assert x_transformed_quad.shape[1] == n_params, (
        "Quadratic SASS should output correct dimensions"
    )

    # Quadratic expansion should give different results
    assert not torch.allclose(x_transformed_linear, x_transformed_quad, atol=1e-3), (
        "Different expansion degrees should give different results"
    )
