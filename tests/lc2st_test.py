# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
import torch
from sklearn.neural_network import MLPClassifier

from sbi.diagnostics.lc2st import LC2ST, LC2ST_NF
from sbi.inference import SNPE
from sbi.simulators.gaussian_mixture import (
    gaussian_mixture,
    uniform_prior_gaussian_mixture,
)


@pytest.mark.parametrize("method", (LC2ST, LC2ST_NF))
@pytest.mark.parametrize("classifier", ('mlp', 'random_forest', 'custom'))
@pytest.mark.parametrize("cv_folds", (1, 2))
def test_running_lc2st(method, classifier, cv_folds):
    """Tests running inference and then LC2ST-(NF) and then getting test quantities."""

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
    inference = SNPE(prior, density_estimator='maf')
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
        theta_o = npe.sample((num_eval,), condition=xs[0]).detach()
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
        **kwargs_test,
    )
    _ = lc2st.train_null()
    _ = lc2st.train_data()

    _ = lc2st.scores_data(x_o=xs[0], return_probs=True, **kwargs_eval)
    _ = lc2st.scores_data(x_o=xs[0], return_probs=False, **kwargs_eval)
    _ = lc2st.statistic_data(x_o=xs[0], **kwargs_eval)

    _ = lc2st.statistics_null(x_o=xs[0], return_probs=True, **kwargs_eval)
    _ = lc2st.statistics_null(x_o=xs[0], return_probs=False, **kwargs_eval)
    _ = lc2st.p_value(x_o=xs[0], **kwargs_eval)
    _ = lc2st.reject(x_o=xs[0], **kwargs_eval)


@pytest.mark.parametrize("method", (LC2ST, LC2ST_NF))
def test_lc2st_tnr(method):
    num_runs = 10

    # small training and num_epochs = reject (no convergence of the estimator)
    num_train = 1_000
    num_epochs = 5

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
    inference = SNPE(prior, density_estimator='maf')
    inference = inference.append_simulations(theta=theta_train, x=x_train)
    npe = inference.train(training_batch_size=100, max_num_epochs=num_epochs)

    thetas = prior.sample((num_cal,))
    xs = simulator(thetas)
    posterior_samples = npe.sample((1,), xs)[:, 0, :].detach()

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

    _ = lc2st.train_null()
    _ = lc2st.train_data()

    results = []
    for _ in range(num_runs):
        x = simulator(prior.sample((1,)))[0]
        if method == LC2ST:
            theta_o = npe.sample((num_eval,), condition=x).detach()
            kwargs_eval = {"theta_o": theta_o}
        else:
            kwargs_eval = {}
        results.append(lc2st.reject(x_o=x, **kwargs_eval))

    assert (
        torch.tensor(results).sum() == num_runs
    ), "LC2ST p-values too big, test should be rejected."


@pytest.mark.slow
@pytest.mark.parametrize("method", (LC2ST, LC2ST_NF))
def test_lc2st_tpr(method):
    num_runs = 10
    # big training and num_epochs = accept (convergence of the estimator)
    num_train = 10_000
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
    inference = SNPE(prior, density_estimator='maf')
    inference = inference.append_simulations(theta=theta_train, x=x_train)
    npe = inference.train(training_batch_size=100, max_num_epochs=num_epochs)

    thetas = prior.sample((num_cal,))
    xs = simulator(thetas)
    posterior_samples = npe.sample((1,), xs)[:, 0, :].detach()

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

    _ = lc2st.train_null()
    _ = lc2st.train_data()

    results = []
    for _ in range(num_runs):
        x = simulator(prior.sample((1,)))[0]
        if method == LC2ST:
            theta_o = npe.sample((num_eval,), condition=x).detach()
            kwargs_eval = {"theta_o": theta_o}
        else:
            kwargs_eval = {}
        results.append(lc2st.reject(x_o=x, **kwargs_eval))

    assert (
        torch.tensor(results).sum() == 0
    ), "LC2ST p-values too small, test should be accepted."
