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


@pytest.mark.parametrize("method", (LC2ST, LC2ST_NF))
@pytest.mark.parametrize("classifier", ('mlp', 'random_forest', 'custom'))
@pytest.mark.parametrize("cv_folds", (1, 2))
@pytest.mark.parametrize("num_ensemble", (1, 3))
@pytest.mark.parametrize("z_score", (True, False))
def test_running_lc2st(method, classifier, cv_folds, num_ensemble, z_score):
    """Tests running inference, LC2ST-(NF) and then getting test quantities."""

    num_train = 100
    num_cal = 100
    num_eval = 100
    num_trials_null = 2

    # task
    dim = 2
    prior = uniform_prior_gaussian_mixture(dim=dim)
    simulator = gaussian_mixture

    # training data for the density estimator
    theta_train = prior.sample((num_train,))
    x_train = simulator(theta_train)

    # Train the neural posterior estimators
    inference = NPE(prior, density_estimator='maf')
    inference = inference.append_simulations(theta=theta_train, x=x_train)
    npe = inference.train(training_batch_size=100, max_num_epochs=1)

    # calibration data for the test
    thetas = prior.sample((num_cal,))
    xs = simulator(thetas)
    posterior_samples = (
        npe.sample((1,), condition=xs).reshape(-1, thetas.shape[-1]).detach()
    )
    assert posterior_samples.shape == thetas.shape

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
    if classifier == "custom":
        kwargs_test["clf_class"] = MLPClassifier
        kwargs_test["clf_kwargs"] = {"alpha": 0.0, "max_iter": 2500}
    kwargs_test["classifier"] = classifier

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


@pytest.mark.slow
@pytest.mark.parametrize("method", (LC2ST, LC2ST_NF))
def test_lc2st_true_positiv_rate(method):
    """Tests the true positiv rate of the LC2ST-(NF) test:
    for a "bad" estimator, the LC2ST-(NF) should reject the null hypothesis."""
    num_runs = 100
    confidence_level = 0.95

    # use small num_train and num_epochs to obtain "bad" estimator
    # (no convergence to the true posterior)
    num_train = 100
    num_epochs = 2

    num_cal = 1_000
    num_eval = 10_000

    # task
    dim = 2
    prior = uniform_prior_gaussian_mixture(dim=dim)
    simulator = gaussian_mixture

    # training data for the density estimator
    theta_train = prior.sample((num_train,))
    x_train = simulator(theta_train)

    # Train the neural posterior estimators
    inference = NPE(prior, density_estimator='maf')
    inference = inference.append_simulations(theta=theta_train, x=x_train)
    npe = inference.train(training_batch_size=100, max_num_epochs=num_epochs)

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

    assert (
        proportion_rejected > confidence_level
    ), f"LC2ST p-values too big, test should be rejected \
        at least {confidence_level * 100}% of the time, but was rejected \
        only {proportion_rejected * 100}% of the time."


@pytest.mark.slow
@pytest.mark.parametrize("method", (LC2ST, LC2ST_NF))
def test_lc2st_false_positiv_rate(method):
    """Tests the false positiv rate of the LC2ST-(NF) test:
    for a "good" estimator, the LC2ST-(NF) should not reject the null hypothesis."""
    num_runs = 100
    confidence_level = 0.95

    # use big num_train and num_epochs to obtain "good" estimator
    # (convergence of the estimator)
    num_train = 5_000
    num_epochs = 200

    num_cal = 1_000
    num_eval = 10_000

    # task
    dim = 2
    prior = uniform_prior_gaussian_mixture(dim=dim)
    simulator = gaussian_mixture

    # training data for the density estimator
    theta_train = prior.sample((num_train,))
    x_train = simulator(theta_train)

    # Train the neural posterior estimators
    inference = NPE(prior, density_estimator='maf')
    inference = inference.append_simulations(theta=theta_train, x=x_train)
    npe = inference.train(training_batch_size=100, max_num_epochs=num_epochs)

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

    assert proportion_rejected < (
        1 - confidence_level
    ), f"LC2ST p-values too small, test should be rejected \
        less then {(1 - confidence_level) * 100}% of the time, \
        but was rejected {proportion_rejected * 100}% of the time."
