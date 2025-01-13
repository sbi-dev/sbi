import pytest
import torch

from sbi.inference import FMPE, NLE, NPE, NPSE, NRE
from sbi.inference.trainers.npe import NPE_C
from sbi.inference.trainers.nre import BNRE, NRE_A, NRE_B, NRE_C
from sbi.utils.metrics import c2st

from .mini_sbibm import get_task

# NOTE: This might can be improved...
# Global settings
SEED = 0
TASKS = ["two_moons", "linear_mvg_2d", "gaussian_linear", "slcp"]
NUM_SIMULATIONS = 2000
EVALUATION_POINTS = 4  # Currently only 3 observation tested for speed
TRAIN_KWARGS = {}

# Density estimators to test
DENSITY_estimators = ["mdn", "made", "maf", "nsf", "maf_rqs"]  # "Kinda exhaustive"
CLASSIFIERS = ["mlp", "resnet"]
NNS = ["mlp", "resnet"]
SCORE_ESTIMATORS = ["mlp", "ada_mlp"]

# Benchmarking method groups
METHOD_GROUPS = {
    "none": [NPE, NRE, NLE, FMPE, NPSE],
    "npe": [NPE],
    "nle": [NLE],
    "nre": [NRE_A, NRE_B, NRE_C, BNRE],
    "fmpe": [FMPE],
    "npse": [NPSE],
    "snpe": [NPE_C],  # NPE_B not implemented, NPE_A need Gaussian prior
    "snle": [NLE],
    "snre": [NRE_A, NRE_B, NRE_C, BNRE],
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
    "snpe": [{}],
    "snle": [{}],
    "snre": [{}],
}


@pytest.fixture
def method_list(benchmark_mode: str) -> list:
    """
    Fixture to get the list of methods based on the benchmark mode.

    Args:
        benchmark_mode (str): The benchmark mode.

    Returns:
        list: List of methods for the given benchmark mode.
    """
    name = str(benchmark_mode).lower()
    if name not in METHOD_GROUPS:
        raise ValueError(f"Benchmark mode '{benchmark_mode}' is not supported.")
    return METHOD_GROUPS[name]


@pytest.fixture
def kwargs_list(benchmark_mode: str) -> list:
    """
    Fixture to get the list of kwargs based on the benchmark mode.

    Args:
        benchmark_mode (str): The benchmark mode.

    Returns:
        list: List of kwargs for the given benchmark mode.
    """
    name = str(benchmark_mode).lower()
    if name not in METHOD_PARAMS:
        raise ValueError(f"Benchmark mode '{benchmark_mode}' is not supported.")
    return METHOD_PARAMS[name]


# Use pytest.mark.parametrize dynamically
# Generates a list of methods to test based on the benchmark mode
def pytest_generate_tests(metafunc):
    """
    Dynamically generates a list of methods to test based on the benchmark mode.

    Args:
        metafunc: The metafunc object from pytest.
    """
    if "inference_method" in metafunc.fixturenames:
        method_list = metafunc.config.getoption("--bm-mode")
        name = str(method_list).lower()
        method_group = METHOD_GROUPS.get(name, [])
        metafunc.parametrize("inference_method", method_group)
    if "extra_kwargs" in metafunc.fixturenames:
        kwargs_list = metafunc.config.getoption("--bm-mode")
        name = str(kwargs_list).lower()
        kwargs_group = METHOD_PARAMS.get(name, [])
        metafunc.parametrize("extra_kwargs", kwargs_group)


def standard_eval_c2st_loop(posterior, task) -> float:
    """
    Evaluates the C2ST metric for the given posterior and task.

    Args:
        posterior: The posterior distribution.
        task: The task object.

    Returns:
        float: The mean C2ST value.
    """
    metrics = []
    for i in range(1, EVALUATION_POINTS):
        c2st_val = eval_c2st(posterior, task, i)
        metrics.append(c2st_val)

    mean_c2st = sum(metrics) / len(metrics)
    # Convert to float rounded to 3 decimal places
    mean_c2st = float(f"{mean_c2st:.3f}")
    return mean_c2st


def eval_c2st(posterior, task, i: int) -> float:
    """
    Evaluates the C2ST metric for a specific observation.

    Args:
        posterior: The posterior distribution.
        task: The task object.
        i (int): The observation index.

    Returns:
        float: The C2ST value.
    """
    x_o = task.get_observation(i)
    posterior_samples = task.get_reference_posterior_samples(i)
    approx_posterior_samples = posterior.sample((1000,), x=x_o)
    if isinstance(approx_posterior_samples, tuple):
        approx_posterior_samples = approx_posterior_samples[0]
    c2st_val = c2st(posterior_samples[:1000], approx_posterior_samples)
    return c2st_val


def amortized_inference_eval(
    method, task_name: str, extra_kwargs: dict, results_bag
) -> None:
    """
    Performs amortized inference evaluation.

    Args:
        method: The inference method.
        task_name (str): The name of the task.
        extra_kwargs (dict): Additional keyword arguments for the method.
        results_bag: The results bag to store evaluation results.
    """
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


def sequential_inference_eval(
    method, task_name: str, extra_kwargs: dict, results_bag
) -> None:
    """
    Performs sequential inference evaluation.

    Args:
        method: The inference method.
        task_name (str): The name of the task.
        extra_kwargs (dict): Additional keyword arguments for the method.
        results_bag: The results bag to store evaluation results.
    """
    torch.manual_seed(SEED)
    task = get_task(task_name)
    num_simulations1 = NUM_SIMULATIONS // 2
    thetas, xs = task.get_data(num_simulations1)
    prior = task.get_prior()
    idx_eval = 1

    # Round 1
    inference = method(prior, **extra_kwargs)
    _ = inference.append_simulations(thetas, xs).train(**TRAIN_KWARGS)

    proposal = inference.build_posterior().set_default_x(task.get_observation(idx_eval))
    num_simulations2 = NUM_SIMULATIONS - num_simulations1
    thetas2 = proposal.sample((num_simulations2,))
    xs2 = task.get_simulator()(thetas2)

    # Round 2
    if "npe" in method.__name__.lower():
        # NPE_C requires a Gaussian prior
        _ = inference.append_simulations(thetas2, xs2, proposal=proposal).train(
            **TRAIN_KWARGS
        )
    else:
        _ = inference.append_simulations(thetas2, xs2).train(**TRAIN_KWARGS)
    posterior = inference.build_posterior()

    c2st_val = eval_c2st(posterior, task, idx_eval)

    # Cache results
    results_bag.metric = c2st_val
    results_bag.num_simulations = NUM_SIMULATIONS
    results_bag.task_name = task_name
    results_bag.method = method.__name__ + str(extra_kwargs)


@pytest.mark.benchmark
@pytest.mark.parametrize("task_name", TASKS, ids=str)
def test_benchmark_standard(
    inference_method,
    task_name: str,
    results_bag,
    extra_kwargs: dict,
    benchmark_mode: str,
) -> None:
    """
    Benchmark test for standard and sequential inference methods.

    Args:
        inference_method: The inference method to test.
        task_name (str): The name of the task.
        results_bag: The results bag to store evaluation results.
        extra_kwargs (dict): Additional keyword arguments for the method.
        benchmark_mode (str): The benchmark mode.
    """
    if benchmark_mode in ["snpe", "snle", "snre"]:
        sequential_inference_eval(
            inference_method, task_name, extra_kwargs, results_bag
        )
    else:
        amortized_inference_eval(inference_method, task_name, extra_kwargs, results_bag)
