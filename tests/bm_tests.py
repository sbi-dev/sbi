
import torch
from .mini_sbibm import get_task

import pytest

from sbi.inference import NPE
from sbi.utils.metrics import c2st


@pytest.fixture(params=["two_moons"])
def task(request):
    return get_task(request.param)


def test_benchmark_methods(task, method=None, num_simulations=1000, seed=0):
    torch.manual_seed(seed)
    thetas, xs = task.get_data(num_simulations)
    assert thetas.shape[0] == num_simulations
    assert xs.shape[0] == num_simulations

    inference = NPE()
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

    test_benchmark_methods.metric = mean_c2st
