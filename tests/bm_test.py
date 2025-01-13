import pytest
import torch

from sbi.inference import FMPE, NLE, NPE, NPSE, NRE
from sbi.utils.metrics import c2st

from .mini_sbibm import get_task

# Global settings
SEED = 0
TASKS = ["two_moons", "linear_mvg_2d", "gaussian_linear", "slcp"]
NUM_SIMULATIONS = 2000
EVALUATION_POINTS = 4  # Currently only 3 observation tested for speed
TRAIN_KWARGS = {}

# Density estimators to test
DENSITY_estimators = ["mdn", "made", "maf", "nsf", "maf_rqs"]  # "Kinda exhaustive"
CLASSIFIERS = ["linear", "mlp", "resnet"]
NNS = ["mlp", "resnet"]
SCORE_ESTIMATORS = ["mlp", "ada_mlp"]

# Benchmarking method groups
ALL_METHODS = [NPE, NRE, NLE, FMPE, NPSE]
METHOD_GROUPS = {
    "none": ALL_METHODS,
    "npe": [NPE],
    "nle": [NLE],
    "nre": [NRE],
    "fmpe": [FMPE],
    "npse": [NPSE],
}
METHOD_PARAMS = {
    "none": [{}],
    "npe": [{"density_estimator": de} for de in DENSITY_estimators],
    "nle": [{"density_estimator": de} for de in ["maf", "nsf"]],
    "nre": [{"classifier": cl} for cl in CLASSIFIERS],
    "fmpe": [{"density_estimator": nn} for nn in NNS],
    "npse": [
        {"score_estimator": nn, "sde_type": sde}
        for nn in SCORE_ESTIMATORS
        for sde in ["ve", "vp"]
    ],
}


@pytest.fixture
def method_list(benchmark_mode):
    name = str(benchmark_mode).lower()
    if name not in METHOD_GROUPS:
        raise ValueError(f"Benchmark mode '{benchmark_mode}' is not supported.")
    return METHOD_GROUPS[name]


@pytest.fixture
def kwargs_list(benchmark_mode):
    name = str(benchmark_mode).lower()
    if name not in METHOD_PARAMS:
        raise ValueError(f"Benchmark mode '{benchmark_mode}' is not supported.")
    return METHOD_PARAMS[name]


# Use pytest.mark.parametrize dynamically
# Generates a list of methods to test based on the benchmark mode
def pytest_generate_tests(metafunc):
    if "inference_method" in metafunc.fixturenames:
        method_list = metafunc.config.getoption("--bm-mode")
        method_group = METHOD_GROUPS.get(method_list, [])
        metafunc.parametrize("inference_method", method_group)
    if "extra_kwargs" in metafunc.fixturenames:
        kwargs_list = metafunc.config.getoption("--bm-mode")
        kwargs_group = METHOD_PARAMS.get(kwargs_list, [])
        metafunc.parametrize("extra_kwargs", kwargs_group)


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


def amortized_inference_eval(method, task_name, extra_kwargs, results_bag):
    torch.manual_seed(SEED)
    task = get_task(task_name)
    thetas, xs = task.get_data(NUM_SIMULATIONS)
    prior = task.get_prior()

    inference = method(prior, **extra_kwargs)
    _ = inference.append_simulations(thetas, xs).train(**TRAIN_KWARGS)

    posterior = inference.build_posterior()

    mean_c2st = standard_eval_c2st_loop(posterior, task)

    # Cache results
    results_bag.metric = mean_c2st
    results_bag.num_simulations = NUM_SIMULATIONS
    results_bag.task_name = task_name
    results_bag.method = method.__name__ + str(extra_kwargs)


@pytest.mark.benchmark
@pytest.mark.parametrize("task_name", TASKS, ids=str)
def test_benchmark_standard(
    inference_method,
    task_name,
    results_bag,
    extra_kwargs,
):
    amortized_inference_eval(inference_method, task_name, extra_kwargs, results_bag)
