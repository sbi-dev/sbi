# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from scipy.stats import binomtest
from sklearn.neural_network import MLPClassifier
from torch import Tensor

from sbi.diagnostics.lc2st import LC2ST, LC2ST_NF, LC2STScores, LC2STState
from sbi.inference import NPE
from sbi.simulators.gaussian_mixture import (
    gaussian_mixture,
    uniform_prior_gaussian_mixture,
)

# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class SimulatorSetup:
    """Simulator configuration for LC2ST tests."""

    dim: int
    prior: torch.distributions.Distribution
    simulator: callable


@dataclass
class CalibrationData:
    """Calibration data for LC2ST tests."""

    thetas: Tensor
    xs: Tensor
    posterior_samples: Tensor


@pytest.fixture(scope="session")
def sim_setup() -> SimulatorSetup:
    """Basic simulator setup shared across LC2ST tests."""
    dim = 2
    return SimulatorSetup(
        dim=dim,
        prior=uniform_prior_gaussian_mixture(dim=dim),
        simulator=gaussian_mixture,
    )


@pytest.fixture(scope="session")
def badly_trained_npe(sim_setup):
    """A poorly trained NPE for testing LC2ST detection of bad posteriors."""
    theta_train = sim_setup.prior.sample((50,))
    x_train = sim_setup.simulator(theta_train)

    inference = NPE(sim_setup.prior, density_estimator="maf")
    inference = inference.append_simulations(theta=theta_train, x=x_train)
    return inference.train(max_num_epochs=1)


@pytest.fixture(scope="session")
def well_trained_npe(sim_setup):
    """A well-trained NPE for testing LC2ST false positive rate."""
    theta_train = sim_setup.prior.sample((5_000,))
    x_train = sim_setup.simulator(theta_train)

    inference = NPE(sim_setup.prior, density_estimator="maf")
    inference = inference.append_simulations(theta=theta_train, x=x_train)
    return inference.train(max_num_epochs=2**31 - 1)


@pytest.fixture(scope="session")
def cal_data(sim_setup, badly_trained_npe) -> CalibrationData:
    """Calibration data for LC2ST tests."""
    num_cal = 100
    thetas = sim_setup.prior.sample((num_cal,))
    xs = sim_setup.simulator(thetas)
    # Sample 1 posterior sample per observation; reshape (n, 1, dim) -> (n, dim)
    posterior_samples = (
        badly_trained_npe.sample((1,), xs).reshape(-1, thetas.shape[-1]).detach()
    )
    return CalibrationData(thetas=thetas, xs=xs, posterior_samples=posterior_samples)


@pytest.fixture
def lc2st_instance(cal_data) -> LC2ST:
    """A fresh LC2ST instance for testing."""
    return LC2ST(
        cal_data.thetas, cal_data.xs, cal_data.posterior_samples, num_trials_null=2
    )


@pytest.fixture
def theta_o(cal_data, badly_trained_npe) -> Tensor:
    """Evaluation samples from the posterior at a single observation."""
    # [None, :] adds batch dim for sample(); reshape flattens (1, n, dim) -> (n, dim)
    return (
        badly_trained_npe.sample((100,), condition=cal_data.xs[0][None, :])
        .reshape(-1, cal_data.thetas.shape[-1])
        .detach()
    )


@pytest.fixture
def x_o(cal_data) -> Tensor:
    """Single observation for evaluation."""
    return cal_data.xs[0]


# =============================================================================
# Core Functionality Tests
# =============================================================================


@pytest.mark.parametrize("method", [LC2ST, LC2ST_NF])
def test_lc2st_methods(method, cal_data, badly_trained_npe, theta_o, x_o):
    """Test that both LC2ST and LC2ST_NF complete full workflow."""
    if method == LC2ST:
        kwargs_init, kwargs_eval = {}, {"theta_o": theta_o}
    else:
        npe = badly_trained_npe
        kwargs_init = {
            "flow_inverse_transform": lambda t, x: npe.net._transform(t, context=x)[0],
            "flow_base_dist": torch.distributions.MultivariateNormal(
                torch.zeros(2), torch.eye(2)
            ),
            "num_eval": 100,
        }
        kwargs_eval = {}

    lc2st = method(
        cal_data.thetas,
        cal_data.xs,
        cal_data.posterior_samples,
        num_trials_null=2,
        **kwargs_init,
    )
    lc2st.train_under_null_hypothesis().train_on_observed_data()

    # All output methods should work
    assert lc2st.get_statistic_on_observed_data(x_o=x_o, **kwargs_eval) is not None
    null_stats = lc2st.get_statistics_under_null_hypothesis(x_o=x_o, **kwargs_eval)
    assert null_stats is not None
    assert lc2st.p_value(x_o=x_o, **kwargs_eval) is not None
    assert lc2st.reject_test(x_o=x_o, **kwargs_eval) is not None


@pytest.mark.parametrize(
    "method,classifier,z_score,cv_folds",
    [
        (LC2ST, "mlp", True, 2),
        (LC2ST, "random_forest", False, 1),
        (LC2ST, MLPClassifier, True, 1),  # Test class-based classifier
        (LC2ST_NF, "mlp", False, 2),
    ],
)
def test_lc2st_parameter_combinations(
    method, classifier, z_score, cv_folds, cal_data, badly_trained_npe
):
    """Test key combinations of method, classifier, z_score, and cv_folds."""
    num_eval = 100

    if method == LC2ST:
        theta_o = (
            badly_trained_npe.sample((num_eval,), condition=cal_data.xs[0][None, :])
            .reshape(-1, cal_data.thetas.shape[-1])
            .detach()
        )
        kwargs_init, kwargs_eval = {}, {"theta_o": theta_o}
    else:
        npe = badly_trained_npe
        kwargs_init = {
            "flow_inverse_transform": lambda t, x: npe.net._transform(t, context=x)[0],
            "flow_base_dist": torch.distributions.MultivariateNormal(
                torch.zeros(2), torch.eye(2)
            ),
            "num_eval": num_eval,
        }
        kwargs_eval = {}

    lc2st = method(
        cal_data.thetas,
        cal_data.xs,
        cal_data.posterior_samples,
        classifier=classifier,
        z_score=z_score,
        num_folds=cv_folds,
        num_trials_null=2,
        **kwargs_init,
    )
    lc2st.train_on_observed_data()

    # Verify classifier count matches cv_folds
    assert len(lc2st.trained_clfs) == cv_folds
    assert lc2st.get_statistic_on_observed_data(x_o=cal_data.xs[0], **kwargs_eval)


def test_lc2st_get_scores_returns_dataclass(lc2st_instance, theta_o, x_o):
    """Test that get_scores returns LC2STScores by default."""
    lc2st_instance.train_on_observed_data()
    result = lc2st_instance.get_scores(
        theta_o=theta_o, x_o=x_o, trained_clfs=lc2st_instance.trained_clfs
    )
    assert isinstance(result, LC2STScores)
    assert result.scores is not None
    assert result.probabilities is not None


def test_lc2st_get_scores_deprecated_return_probs(lc2st_instance, theta_o, x_o):
    """Test that return_probs=True emits deprecation warning."""
    lc2st_instance.train_on_observed_data()
    with pytest.warns(DeprecationWarning, match="return_probs"):
        probs, scores = lc2st_instance.get_scores(
            theta_o=theta_o,
            x_o=x_o,
            trained_clfs=lc2st_instance.trained_clfs,
            return_probs=True,
        )
    assert probs is not None and scores is not None


# =============================================================================
# State Machine Tests
# =============================================================================


def test_lc2st_state_transitions(lc2st_instance):
    """Test state machine transitions through training workflow."""
    assert lc2st_instance._state == LC2STState.INITIALIZED

    # Train observed first
    result = lc2st_instance.train_on_observed_data()
    assert result is lc2st_instance  # Method chaining
    assert lc2st_instance._state == LC2STState.OBSERVED_TRAINED

    # Then null
    lc2st_instance.train_under_null_hypothesis()
    assert lc2st_instance._state == LC2STState.READY


def test_lc2st_state_transitions_reverse_order(cal_data):
    """Test state transitions when training null first."""
    lc2st = LC2ST(
        cal_data.thetas, cal_data.xs, cal_data.posterior_samples, num_trials_null=2
    )
    assert lc2st._state == LC2STState.INITIALIZED

    lc2st.train_under_null_hypothesis()
    assert lc2st._state == LC2STState.NULL_TRAINED

    lc2st.train_on_observed_data()
    assert lc2st._state == LC2STState.READY


# =============================================================================
# Input Validation Tests
# =============================================================================


@pytest.mark.parametrize(
    "missing_arg",
    ["prior_samples", "xs", "posterior_samples"],
)
def test_lc2st_missing_required_input(missing_arg):
    """Test that missing required inputs raise ValueError."""
    kwargs = {
        "prior_samples": torch.randn(10, 2),
        "xs": torch.randn(10, 2),
        "posterior_samples": torch.randn(10, 2),
    }
    kwargs[missing_arg] = None

    with pytest.raises(ValueError, match=f"{missing_arg} is required"):
        LC2ST(**kwargs)


def test_lc2st_dimension_mismatch():
    """Test that dimension mismatch raises ValueError."""
    with pytest.raises(ValueError, match="Dimension mismatch"):
        LC2ST(
            prior_samples=torch.randn(10, 2),
            xs=torch.randn(10, 3),
            posterior_samples=torch.randn(10, 4),
        )


@pytest.mark.xfail(
    reason="Sample size validation happens after cleaning, causing IndexError",
    strict=True,
)
def test_lc2st_sample_size_mismatch():
    """Test that sample size mismatch raises ValueError."""
    with pytest.raises(ValueError, match="Sample size"):
        LC2ST(
            prior_samples=torch.randn(100, 2),
            xs=torch.randn(50, 3),
            posterior_samples=torch.randn(100, 2),
        )


@pytest.mark.parametrize(
    "num_folds,sample_size,match",
    [
        (0, 100, "num_folds must be >= 1"),
        (20, 10, "cannot exceed"),
    ],
)
def test_lc2st_invalid_num_folds(num_folds, sample_size, match):
    """Test that invalid num_folds raises ValueError."""
    with pytest.raises(ValueError, match=match):
        LC2ST(
            prior_samples=torch.randn(sample_size, 2),
            xs=torch.randn(sample_size, 3),
            posterior_samples=torch.randn(sample_size, 2),
            num_folds=num_folds,
        )


def test_lc2st_invalid_classifier():
    """Test that invalid classifier string raises ValueError."""
    with pytest.raises(ValueError, match="Invalid classifier"):
        LC2ST(
            prior_samples=torch.randn(10, 2),
            xs=torch.randn(10, 3),
            posterior_samples=torch.randn(10, 2),
            classifier="invalid",
        )


# =============================================================================
# State Error Tests
# =============================================================================


@pytest.mark.parametrize(
    "method_name,match",
    [
        ("p_value", "not ready"),
        ("get_statistic_on_observed_data", "train_on_observed_data"),
        ("get_statistics_under_null_hypothesis", "train_under_null_hypothesis"),
    ],
)
def test_lc2st_method_before_training(method_name, match, lc2st_instance, x_o, theta_o):
    """Test that calling methods before training raises RuntimeError."""
    method = getattr(lc2st_instance, method_name)
    with pytest.raises(RuntimeError, match=match):
        method(x_o=x_o, theta_o=theta_o)


def test_lc2st_null_training_requires_permutation_or_distribution(cal_data):
    """Test that null training without permutation or distribution raises ValueError."""
    lc2st = LC2ST(
        cal_data.thetas,
        cal_data.xs,
        cal_data.posterior_samples,
        permutation=False,
        num_trials_null=2,
    )
    with pytest.raises(ValueError, match="null_distribution"):
        lc2st.train_under_null_hypothesis()


# =============================================================================
# Deprecation Tests
# =============================================================================


def test_lc2st_thetas_parameter_deprecated(cal_data):
    """Test that 'thetas' parameter emits DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="thetas.*deprecated"):
        lc2st = LC2ST(
            thetas=cal_data.thetas,
            xs=cal_data.xs,
            posterior_samples=cal_data.posterior_samples,
        )
    assert lc2st.theta_q is not None


def test_lc2st_both_thetas_and_prior_samples_error(cal_data):
    """Test that specifying both thetas and prior_samples raises ValueError."""
    with pytest.raises(ValueError, match="Cannot specify both"):
        LC2ST(
            prior_samples=cal_data.thetas,
            thetas=cal_data.thetas,
            xs=cal_data.xs,
            posterior_samples=cal_data.posterior_samples,
        )


# =============================================================================
# Statistical Tests (Slow)
# =============================================================================


@pytest.mark.slow
@pytest.mark.parametrize("method", [LC2ST, LC2ST_NF])
def test_lc2st_true_positive_rate(method, sim_setup, badly_trained_npe):
    """For a bad estimator, LC2ST should reject the null hypothesis."""
    num_runs, num_cal, num_eval = 100, 1_000, 10_000
    confidence_level = 0.95

    thetas = sim_setup.prior.sample((num_cal,))
    xs = sim_setup.simulator(thetas)
    posterior_samples = (
        badly_trained_npe.sample((1,), xs).reshape(-1, thetas.shape[-1]).detach()
    )

    if method == LC2ST:
        kwargs_init, get_eval_kwargs = (
            {},
            lambda x: {
                "theta_o": badly_trained_npe.sample((num_eval,), condition=x)
                .reshape(-1, thetas.shape[-1])
                .detach()
            },
        )
    else:
        npe = badly_trained_npe
        kwargs_init = {
            "flow_inverse_transform": lambda t, x: npe.net._transform(t, context=x)[0],
            "flow_base_dist": torch.distributions.MultivariateNormal(
                torch.zeros(2), torch.eye(2)
            ),
            "num_eval": num_eval,
        }
        get_eval_kwargs = lambda x: {}

    lc2st = method(thetas, xs, posterior_samples, **kwargs_init)
    lc2st.train_under_null_hypothesis().train_on_observed_data()

    results = [
        lc2st.reject_test(
            x_o=sim_setup.simulator(sim_setup.prior.sample((1,))),
            alpha=1 - confidence_level,
            **get_eval_kwargs(sim_setup.simulator(sim_setup.prior.sample((1,)))),
        )
        for _ in range(num_runs)
    ]

    proportion_rejected = torch.tensor(results).float().mean()
    assert proportion_rejected > confidence_level, (
        f"LC2ST should reject at least {confidence_level * 100}% of the time, "
        f"but only rejected {proportion_rejected * 100}%"
    )


@pytest.mark.slow
@pytest.mark.parametrize("method", [LC2ST, LC2ST_NF])
def test_lc2st_false_positive_rate(method, sim_setup, well_trained_npe, set_seed):
    """For a good estimator, LC2ST should not reject the null hypothesis."""
    num_runs, num_cal, num_eval = 100, 1_000, 10_000
    confidence_level = 0.95
    expected_rate = 1 - confidence_level

    thetas = sim_setup.prior.sample((num_cal,))
    xs = sim_setup.simulator(thetas)
    posterior_samples = (
        well_trained_npe.sample((1,), xs).reshape(-1, thetas.shape[-1]).detach()
    )

    if method == LC2ST:
        kwargs_init, get_eval_kwargs = (
            {},
            lambda x: {
                "theta_o": well_trained_npe.sample((num_eval,), condition=x)
                .reshape(-1, thetas.shape[-1])
                .detach()
            },
        )
    else:
        npe = well_trained_npe
        kwargs_init = {
            "flow_inverse_transform": lambda t, x: npe.net._transform(t, context=x)[0],
            "flow_base_dist": torch.distributions.MultivariateNormal(
                torch.zeros(2), torch.eye(2)
            ),
            "num_eval": num_eval,
        }
        get_eval_kwargs = lambda x: {}

    lc2st = method(thetas, xs, posterior_samples, **kwargs_init)
    lc2st.train_under_null_hypothesis().train_on_observed_data()

    results = []
    for _ in range(num_runs):
        x = sim_setup.simulator(sim_setup.prior.sample((1,)))
        results.append(
            lc2st.reject_test(x_o=x, alpha=1 - confidence_level, **get_eval_kwargs(x))
        )

    num_rejections = sum(results)
    result = binomtest(num_rejections, num_runs, expected_rate, alternative="greater")

    assert result.pvalue > 0.01, (
        f"LC2ST rejection rate significantly higher than expected. "
        f"Expected: {expected_rate * 100:.1f}%, Observed: {num_rejections}%, "
        f"p-value: {result.pvalue:.4f}"
    )
