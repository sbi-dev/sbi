import pytest
import torch

from sbi.inference import NPE, NRE
from sbi.utils.metrics import c2st

from .mini_sbibm import get_task


@pytest.mark.benchmark
@pytest.mark.parametrize('task_name', ['two_moons'], ids=str)
@pytest.mark.parametrize('density_estimator', ["maf", "nsf"], ids=str)
def test_benchmark_npe_methods(
    task_name, density_estimator, results_bag, method=None, num_simulations=1000, seed=0
):
    torch.manual_seed(seed)
    task = get_task(task_name)
    thetas, xs = task.get_data(num_simulations)
    assert thetas.shape[0] == num_simulations
    assert xs.shape[0] == num_simulations

    inference = NPE(density_estimator=density_estimator)
    _ = inference.append_simulations(thetas, xs).train()

    posterior = inference.build_posterior()

    metrics = []
    for i in range(1, 2):  # Currently only one observation tested for speed
        x_o = task.get_observation(i)
        posterior_samples = task.get_reference_posterior_samples(i)
        approx_posterior_samples = posterior.sample((1000,), x=x_o)
        c2st_val = c2st(posterior_samples[:1000], approx_posterior_samples)
        metrics.append(c2st_val)

    mean_c2st = sum(metrics) / len(metrics)
    # Convert to float rounded to 3 decimal places
    mean_c2st = float(f"{mean_c2st:.3f}")

    results_bag.metric = mean_c2st
    results_bag.num_simulations = num_simulations
    results_bag.task_name = task_name
    results_bag.method = "NPE_" + density_estimator


@pytest.mark.benchmark
@pytest.mark.parametrize('task_name', ['two_moons'], ids=str)
def test_benchmark_nre_methods(task_name, results_bag, num_simulations=1000, seed=0):
    torch.manual_seed(seed)
    task = get_task(task_name)
    thetas, xs = task.get_data(num_simulations)
    prior = task.get_prior()
    assert thetas.shape[0] == num_simulations
    assert xs.shape[0] == num_simulations

    inference = NRE(prior)
    _ = inference.append_simulations(thetas, xs).train()

    posterior = inference.build_posterior()

    metrics = []
    for i in range(1, 2):
        x_o = task.get_observation(i)
        posterior_samples = task.get_reference_posterior_samples(i)
        approx_posterior_samples = posterior.sample((1000,), x=x_o)
        c2st_val = c2st(posterior_samples[:1000], approx_posterior_samples)
        metrics.append(c2st_val)

    mean_c2st = sum(metrics) / len(metrics)
    # Convert to float rounded to 3 decimal places
    mean_c2st = float(f"{mean_c2st:.3f}")

    results_bag.metric = mean_c2st
    results_bag.num_simulations = num_simulations
    results_bag.task_name = task_name
    results_bag.method = "NRE"


# Pytest harvest


# @pytest.mark.benchmark
# def test_synthesis(fixture_store):
#     """
#     In this test we inspect the contents of the fixture store so far, and
#     check that the 'results_bag' entry contains a dict <test_id>: <results_bag>
#     """
#     # print the keys in the store
#     print(dict(fixture_store))
#     results = fixture_store["results_bag"]

#     for k, v in results.items():
#         print(k)
#         for kk, vv in v.items():
#             print(kk, vv)
