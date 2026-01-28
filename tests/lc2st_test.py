# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import pytest
import torch
from sklearn.neural_network import MLPClassifier

from sbi.diagnostics.lc2st import LC2ST, LC2ST_NF
from sbi.inference import NPE
from sbi.simulators.gaussian_mixture import (
    gaussian_mixture,
    uniform_prior_gaussian_mixture,
)


@pytest.fixture(scope="session")
def basic_setup():
    """Basic setup shared across LC2ST tests."""
    dim = 2
    prior = uniform_prior_gaussian_mixture(dim=dim)
    simulator = gaussian_mixture
    return {"dim": dim, "prior": prior, "simulator": simulator}


@pytest.fixture(scope="session")
def npe_factory(basic_setup):
    """Factory for creating NPE models with different training parameters."""

    def _create_npe(num_simulations, max_epochs=None):
        prior = basic_setup["prior"]
        simulator = basic_setup["simulator"]

        theta_train = prior.sample((num_simulations,))
        x_train = simulator(theta_train)

        inference = NPE(prior, density_estimator='maf')
        inference = inference.append_simulations(theta=theta_train, x=x_train)

        return inference.train(
            max_num_epochs=2**31 - 1 if max_epochs is None else max_epochs,
        )

    return _create_npe


@pytest.fixture(scope="session")
def badly_trained_npe(npe_factory):
    return npe_factory(num_simulations=50, max_epochs=1)


@pytest.fixture(scope="session")
def well_trained_npe(npe_factory):
    return npe_factory(num_simulations=5_000)


@pytest.fixture(scope="session")
def calibration_data(basic_setup, badly_trained_npe):
    """Calibration data for LC2ST tests."""
    prior = basic_setup["prior"]
    simulator = basic_setup["simulator"]
    npe = badly_trained_npe

    num_cal = 100  # Smaller for quick tests
    thetas = prior.sample((num_cal,))
    xs = simulator(thetas)
    posterior_samples = npe.sample((1,), xs).reshape(-1, thetas.shape[-1]).detach()

    return {"thetas": thetas, "xs": xs, "posterior_samples": posterior_samples}


@pytest.mark.parametrize("method", (LC2ST, LC2ST_NF))
@pytest.mark.parametrize("classifier", ('mlp', 'random_forest', MLPClassifier))
@pytest.mark.parametrize("cv_folds", (1, 2))
@pytest.mark.parametrize("num_ensemble", (1, 3))
@pytest.mark.parametrize("z_score", (True, False))
@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        "cuda" if torch.cuda.is_available() else "cpu",
        "mps" if torch.backends.mps.is_available() else "cpu",
    ),
)
def test_running_lc2st(
    method,
    classifier,
    cv_folds,
    num_ensemble,
    z_score,
    device,
    calibration_data,
    badly_trained_npe,
):
    """Tests running inference, LC2ST-(NF) and then getting test quantities."""

    num_eval = 100
    num_trials_null = 2

    # Get data from fixtures
    thetas = calibration_data["thetas"]
    xs = calibration_data["xs"]
    posterior_samples = calibration_data["posterior_samples"]
    npe = badly_trained_npe

    if method == LC2ST:
        theta_o = (
            npe.sample((num_eval,), condition=xs[0][None, :])
            .reshape(-1, thetas.shape[-1])
            .detach()
        )
        assert theta_o.shape == thetas.shape
        kwargs_test = {}
        kwargs_eval = {"theta_o": theta_o}
    else:
        flow_inverse_transform = lambda theta, x: npe.net._transform(theta, context=x)[
            0
        ]
        flow_base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(2), torch.eye(2)
        )
        kwargs_test = {
            "flow_inverse_transform": flow_inverse_transform,
            "flow_base_dist": flow_base_dist,
            "num_eval": num_eval,
        }
        kwargs_eval = {}
    kwargs_test["classifier"] = classifier
    kwargs_test["device"] = device

    lc2st = method(
        thetas,
        xs,
        posterior_samples,
        num_folds=cv_folds,
        num_trials_null=num_trials_null,
        num_ensemble=num_ensemble,
        z_score=z_score,
        **kwargs_test,
    )
    _ = lc2st.train_under_null_hypothesis()
    _ = lc2st.train_on_observed_data()

    _ = lc2st.get_scores(
        x_o=xs[0], trained_clfs=lc2st.trained_clfs, return_probs=True, **kwargs_eval
    )
    _ = lc2st.get_scores(
        x_o=xs[0], trained_clfs=lc2st.trained_clfs, return_probs=False, **kwargs_eval
    )
    _ = lc2st.get_statistic_on_observed_data(x_o=xs[0], **kwargs_eval)

    _ = lc2st.get_statistics_under_null_hypothesis(
        x_o=xs[0], return_probs=True, **kwargs_eval
    )
    _ = lc2st.get_statistics_under_null_hypothesis(
        x_o=xs[0], return_probs=False, **kwargs_eval
    )
    _ = lc2st.p_value(x_o=xs[0], **kwargs_eval)
    _ = lc2st.reject_test(x_o=xs[0], **kwargs_eval)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda is not available"
            ),
        ),
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ),
                reason="mps is not available",
            ),
        ),
    ],
)
def test_lc2st_runs_on_requested_device(calibration_data, device):
    """Test that LC2ST runs on cuda/mps (if available)."""

    thetas = calibration_data["thetas"]
    xs = calibration_data["xs"]
    posterior_samples = calibration_data["posterior_samples"]

    lc2st = LC2ST(thetas, xs, posterior_samples, classifier="mlp", device=device)
    lc2st.train_under_null_hypothesis()
    lc2st.train_on_observed_data()
    assert len(lc2st.trained_clfs) > 0


@pytest.mark.slow
@pytest.mark.parametrize("method", (LC2ST, LC2ST_NF))
def test_lc2st_true_positiv_rate(method, basic_setup, badly_trained_npe):
    """Tests the true positiv rate of the LC2ST-(NF) test:
    for a "bad" estimator, the LC2ST-(NF) should reject the null hypothesis."""
    num_runs = 100
    confidence_level = 0.95

    num_cal = 1_000
    num_eval = 10_000

    # Get data from fixtures
    prior = basic_setup["prior"]
    simulator = basic_setup["simulator"]
    npe = badly_trained_npe

    thetas = prior.sample((num_cal,))
    xs = simulator(thetas)
    posterior_samples = npe.sample((1,), xs).reshape(-1, thetas.shape[-1]).detach()

    if method == LC2ST:
        kwargs_test = {}
    else:
        flow_inverse_transform = lambda theta, x: npe.net._transform(theta, context=x)[
            0
        ]
        flow_base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(2), torch.eye(2)
        )
        kwargs_test = {
            "flow_inverse_transform": flow_inverse_transform,
            "flow_base_dist": flow_base_dist,
            "num_eval": num_eval,
        }

    lc2st = method(thetas, xs, posterior_samples, **kwargs_test)

    _ = lc2st.train_under_null_hypothesis()
    _ = lc2st.train_on_observed_data()

    results = []
    for _ in range(num_runs):
        x = simulator(prior.sample((1,)))
        if method == LC2ST:
            theta_o = (
                npe.sample((num_eval,), condition=x)
                .reshape(-1, thetas.shape[-1])
                .detach()
            )
            kwargs_eval = {"theta_o": theta_o}
        else:
            kwargs_eval = {}
        results.append(
            lc2st.reject_test(x_o=x, alpha=1 - confidence_level, **kwargs_eval)
        )

    proportion_rejected = torch.tensor(results).float().mean()

    assert proportion_rejected > confidence_level, (
        f"LC2ST p-values too big, test should be rejected \
        at least {confidence_level * 100}% of the time, but was rejected \
        only {proportion_rejected * 100}% of the time."
    )


@pytest.mark.slow
@pytest.mark.parametrize("method", (LC2ST, LC2ST_NF))
def test_lc2st_false_positiv_rate(method, basic_setup, well_trained_npe, set_seed):
    """Tests the false positiv rate of the LC2ST-(NF) test:
    for a "good" estimator, the LC2ST-(NF) should not reject the null hypothesis."""
    num_runs = 100
    confidence_level = 0.95

    num_cal = 1_000
    num_eval = 10_000

    # Get data from fixtures
    prior = basic_setup["prior"]
    simulator = basic_setup["simulator"]
    npe = well_trained_npe

    thetas = prior.sample((num_cal,))
    xs = simulator(thetas)
    posterior_samples = npe.sample((1,), xs).reshape(-1, thetas.shape[-1]).detach()

    if method == LC2ST:
        kwargs_test = {}
    else:
        flow_inverse_transform = lambda theta, x: npe.net._transform(theta, context=x)[
            0
        ]
        flow_base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(2), torch.eye(2)
        )
        kwargs_test = {
            "flow_inverse_transform": flow_inverse_transform,
            "flow_base_dist": flow_base_dist,
            "num_eval": num_eval,
        }

    lc2st = method(thetas, xs, posterior_samples, **kwargs_test)

    _ = lc2st.train_under_null_hypothesis()
    _ = lc2st.train_on_observed_data()

    results = []
    for _ in range(num_runs):
        x = simulator(prior.sample((1,)))
        if method == LC2ST:
            theta_o = (
                npe.sample((num_eval,), condition=x)
                .reshape(-1, thetas.shape[-1])
                .detach()
            )
            kwargs_eval = {"theta_o": theta_o}
        else:
            kwargs_eval = {}
        results.append(
            lc2st.reject_test(x_o=x, alpha=1 - confidence_level, **kwargs_eval)
        )

    proportion_rejected = torch.tensor(results).float().mean()

    assert proportion_rejected < (1 - confidence_level), (
        "LC2ST p-values too small, test should be rejected "
        f"less then {(1 - confidence_level) * 100.0:<.2f}% of the time, "
        f"but was rejected {proportion_rejected * 100.0:<.2f}% of the time."
    )


def test_lc2st_classifier_kwargs_defaults(calibration_data):
    """Test that sbi-specific defaults are applied when classifier_kwargs is None."""
    thetas = calibration_data["thetas"]
    xs = calibration_data["xs"]
    posterior_samples = calibration_data["posterior_samples"]

    lc2st = LC2ST(thetas, xs, posterior_samples, classifier="mlp", device="cpu")

    assert lc2st.clf_kwargs["activation"] == "relu"
    assert lc2st.clf_kwargs["max_iter"] == 1000
    assert lc2st.clf_kwargs["early_stopping"] is True


def test_lc2st_classifier_kwargs_override(calibration_data):
    """Test that user overrides merge with defaults correctly
    and do not mutate global state."""
    thetas = calibration_data["thetas"]
    xs = calibration_data["xs"]
    posterior_samples = calibration_data["posterior_samples"]

    custom_kwargs = {"max_iter": 50}
    lc2st_override = LC2ST(
        thetas,
        xs,
        posterior_samples,
        classifier="mlp",
        classifier_kwargs=custom_kwargs,
        device="cpu",
    )

    assert lc2st_override.clf_kwargs["max_iter"] == 50
    assert lc2st_override.clf_kwargs["activation"] == "relu"

    lc2st_clean = LC2ST(
        thetas,
        xs,
        posterior_samples,
        classifier="mlp",
        classifier_kwargs=None,
        device="cpu",
    )
    assert lc2st_clean.clf_kwargs["max_iter"] == 1000
