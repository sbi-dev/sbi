import pytest
import torch

from sbi.inference import FMPE, NLE, NPE, NPSE, NRE
from sbi.utils.metrics import c2st

from .mini_sbibm import get_task

# The probably should be some user control on this
SEED = 0
TASKS = ["two_moons", "linear_mvg_2d", "gaussian_linear", "slcp"]
NUM_SIMULATIONS = 2000
EVALUATION_POINTS = 4  # Currently only 3 observation tested for speed

TRAIN_KWARGS = {
    # "training_batch_size": 200,  # To speed up training
}

# Amortized benchmarking


def standard_eval_c2st_loop(posterior, task):
    metrics = []
    for i in range(1, EVALUATION_POINTS):
        x_o = task.get_observation(i)
        posterior_samples = task.get_reference_posterior_samples(i)
        approx_posterior_samples = posterior.sample((1000,), x=x_o)
        if isinstance(approx_posterior_samples, tuple):
            approx_posterior_samples = approx_posterior_samples[0]
        c2st_val = c2st(posterior_samples[:1000], approx_posterior_samples)
        metrics.append(c2st_val)

    mean_c2st = sum(metrics) / len(metrics)
    # Convert to float rounded to 3 decimal places
    mean_c2st = float(f"{mean_c2st:.3f}")
    return mean_c2st


DENSITY_estimators = ["mdn", "made", "maf", "nsf", "maf_rqs"]  # "Kinda exhaustive"
DENSITY_estimators = ["maf", "nsf"]  # Fast


@pytest.mark.benchmark
@pytest.mark.parametrize('task_name', TASKS, ids=str)
@pytest.mark.parametrize('density_estimator', DENSITY_estimators, ids=str)
def test_benchmark_npe_methods(task_name, density_estimator, results_bag):
    torch.manual_seed(SEED)
    task = get_task(task_name)
    thetas, xs = task.get_data(NUM_SIMULATIONS)
    prior = task.get_prior()

    print(thetas.shape, xs.shape)

    inference = NPE(prior, density_estimator=density_estimator)
    _ = inference.append_simulations(thetas, xs).train(**TRAIN_KWARGS)

    posterior = inference.build_posterior()

    mean_c2st = standard_eval_c2st_loop(posterior, task)

    # Cache results
    results_bag.metric = mean_c2st
    results_bag.num_simulations = NUM_SIMULATIONS
    results_bag.task_name = task_name
    results_bag.method = "NPE_" + density_estimator


@pytest.mark.benchmark
@pytest.mark.parametrize('task_name', TASKS, ids=str)
def test_benchmark_nre_methods(task_name, results_bag):
    torch.manual_seed(SEED)
    task = get_task(task_name)
    thetas, xs = task.get_data(NUM_SIMULATIONS)
    prior = task.get_prior()

    inference = NRE(prior)
    _ = inference.append_simulations(thetas, xs).train(**TRAIN_KWARGS)

    posterior = inference.build_posterior()

    mean_c2st = standard_eval_c2st_loop(posterior, task)

    results_bag.metric = mean_c2st
    results_bag.num_simulations = NUM_SIMULATIONS
    results_bag.task_name = task_name
    results_bag.method = "NRE"


@pytest.mark.benchmark
@pytest.mark.parametrize('task_name', TASKS, ids=str)
def test_benchmark_nle_methods(task_name, results_bag):
    torch.manual_seed(SEED)
    task = get_task(task_name)
    thetas, xs = task.get_data(NUM_SIMULATIONS)
    prior = task.get_prior()

    inference = NLE(prior)
    _ = inference.append_simulations(thetas, xs).train(**TRAIN_KWARGS)

    posterior = inference.build_posterior()

    mean_c2st = standard_eval_c2st_loop(posterior, task)

    results_bag.metric = mean_c2st
    results_bag.num_simulations = NUM_SIMULATIONS
    results_bag.task_name = task_name
    results_bag.method = "NLE"


@pytest.mark.benchmark
@pytest.mark.parametrize('task_name', TASKS, ids=str)
def test_benchmark_fmpe_methods(task_name, results_bag):
    torch.manual_seed(SEED)
    task = get_task(task_name)
    thetas, xs = task.get_data(NUM_SIMULATIONS)
    prior = task.get_prior()

    inference = FMPE(prior)
    _ = inference.append_simulations(thetas, xs).train(**TRAIN_KWARGS)

    posterior = inference.build_posterior()

    mean_c2st = standard_eval_c2st_loop(posterior, task)

    results_bag.metric = mean_c2st
    results_bag.num_simulations = NUM_SIMULATIONS
    results_bag.task_name = task_name
    results_bag.method = "FMPE"


@pytest.mark.benchmark
@pytest.mark.parametrize('task_name', TASKS, ids=str)
def test_benchmark_npse_methods(task_name, results_bag):
    torch.manual_seed(SEED)
    task = get_task(task_name)
    thetas, xs = task.get_data(NUM_SIMULATIONS)
    prior = task.get_prior()

    inference = NPSE(prior)
    _ = inference.append_simulations(thetas, xs).train(**TRAIN_KWARGS)

    posterior = inference.build_posterior()

    mean_c2st = standard_eval_c2st_loop(posterior, task)

    results_bag.metric = mean_c2st
    results_bag.num_simulations = NUM_SIMULATIONS
    results_bag.task_name = task_name
    results_bag.method = "NPSE"
