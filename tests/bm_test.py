# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from pytest_harvest import ResultsBag

from sbi.inference import FMPE, NLE, NPE, NPSE, NRE
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.trainers.npe import NPE_C
from sbi.inference.trainers.nre import BNRE, NRE_A, NRE_B, NRE_C
from sbi.utils.metrics import c2st

from .mini_sbibm import get_task
from .mini_sbibm.base_task import Task

# Global settings
SEED = 0
TASKS = ["two_moons", "linear_mvg_2d", "gaussian_linear", "slcp"]
NUM_SIMULATIONS = 2000
NUM_EVALUATION_OBS = 3  # Currently only 3 observation tested for speed
NUM_ROUNDS_SEQUENTIAL = 2
NUM_EVALUATION_OBS_SEQ = 1
TRAIN_KWARGS = {}

# Density estimators to test
DENSITY_ESTIMATORS = ["mdn", "made", "maf", "nsf", "maf_rqs"]  # "Kinda exhaustive"
CLASSIFIERS = ["mlp", "resnet"]
NNS = ["mlp", "resnet"]
SCORE_ESTIMATORS = ["mlp", "ada_mlp"]

# Benchmarking method groups i.e. what to run for different --bm-mode
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
    "npe": [{"density_estimator": de} for de in DENSITY_ESTIMATORS],
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
    if "inference_class" in metafunc.fixturenames:
        method_list = metafunc.config.getoption("--bm-mode")
        name = str(method_list).lower()
        method_group = METHOD_GROUPS.get(name, [])
        metafunc.parametrize("inference_class", method_group)
    if "extra_kwargs" in metafunc.fixturenames:
        kwargs_list = metafunc.config.getoption("--bm-mode")
        name = str(kwargs_list).lower()
        kwargs_group = METHOD_PARAMS.get(name, [])
        metafunc.parametrize("extra_kwargs", kwargs_group)


def standard_eval_c2st_loop(posterior: NeuralPosterior, task: Task) -> float:
    """
    Evaluates the C2ST metric for the given posterior and task.

    Args:
        posterior: The posterior distribution.
        task: The task object.

    Returns:
        float: The mean C2ST value.
    """
    c2st_scores = []
    for i in range(1, NUM_EVALUATION_OBS + 1):
        c2st_val = eval_c2st(posterior, task, i)
        c2st_scores.append(c2st_val)

    mean_c2st = sum(c2st_scores) / len(c2st_scores)
    # Convert to float rounded to 3 decimal places
    mean_c2st = float(f"{mean_c2st:.3f}")
    return mean_c2st


def eval_c2st(
    posterior: NeuralPosterior,
    task: Task,
    idx_observation: int,
    num_samples: int = 1000,
) -> float:
    """
    Evaluates the C2ST metric for a specific observation.

    Args:
        posterior: The posterior distribution.
        task: The task object.
        i (int): The observation index.

    Returns:
        float: The C2ST value.
    """
    x_o = task.get_observation(idx_observation)
    posterior_samples = task.get_reference_posterior_samples(idx_observation)
    approx_posterior_samples = posterior.sample((num_samples,), x=x_o)
    if isinstance(approx_posterior_samples, tuple):
        approx_posterior_samples = approx_posterior_samples[0]
    assert posterior_samples.shape[0] >= num_samples, "Not enough reference samples"
    c2st_val = c2st(posterior_samples[:num_samples], approx_posterior_samples)
    return float(c2st_val)


def train_and_eval_amortized_inference(
    inference_class, task_name: str, extra_kwargs: dict, results_bag: ResultsBag
) -> None:
    """
    Performs amortized inference evaluation.

    Args:
        method: The inference method.
        task_name: The name of the task.
        extra_kwargs: Additional keyword arguments for the method.
        results_bag: The results bag to store evaluation results. Subclass of dict, but
            allows item assignment with dot notation.
    """
    torch.manual_seed(SEED)
    task = get_task(task_name)
    thetas, xs = task.get_data(NUM_SIMULATIONS)
    prior = task.get_prior()

    inference = inference_class(prior, **extra_kwargs)
    _ = inference.append_simulations(thetas, xs).train(**TRAIN_KWARGS)

    posterior = inference.build_posterior()

    mean_c2st = standard_eval_c2st_loop(posterior, task)

    # Cache results
    results_bag.metric = mean_c2st
    results_bag.num_simulations = NUM_SIMULATIONS
    results_bag.task_name = task_name
    results_bag.method = inference_class.__name__ + str(extra_kwargs)


def train_and_eval_sequential_inference(
    inference_class, task_name: str, extra_kwargs: dict, results_bag: ResultsBag
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
    num_simulations = NUM_SIMULATIONS // NUM_ROUNDS_SEQUENTIAL
    thetas, xs = task.get_data(num_simulations)
    prior = task.get_prior()
    idx_eval = NUM_EVALUATION_OBS_SEQ
    x_o = task.get_observation(idx_eval)
    simulator = task.get_simulator()

    # Round 1
    inference = inference_class(prior, **extra_kwargs)
    _ = inference.append_simulations(thetas, xs).train(**TRAIN_KWARGS)

    for _ in range(NUM_ROUNDS_SEQUENTIAL - 1):
        proposal = inference.build_posterior().set_default_x(x_o)
        thetas_i = proposal.sample((num_simulations,))
        xs_i = simulator(thetas_i)
        if "npe" in inference_class.__name__.lower():
            # NPE_C requires a Gaussian prior
            _ = inference.append_simulations(thetas_i, xs_i, proposal=proposal).train(
                **TRAIN_KWARGS
            )
        else:
            inference.append_simulations(thetas_i, xs_i).train(**TRAIN_KWARGS)

    posterior = inference.build_posterior()

    c2st_val = eval_c2st(posterior, task, idx_eval)

    # Cache results
    results_bag.metric = c2st_val
    results_bag.num_simulations = NUM_SIMULATIONS
    results_bag.task_name = task_name
    results_bag.method = inference_class.__name__ + str(extra_kwargs)


@pytest.mark.benchmark
@pytest.mark.parametrize("task_name", TASKS, ids=str)
def test_run_benchmark(
    inference_class,
    task_name: str,
    results_bag,
    extra_kwargs: dict,
    benchmark_mode: str,
) -> None:
    """
    Benchmark test for amortized and sequential inference methods.

    Args:
        inference_class: The inference class to test i.e. NPE, NLE, NRE ...
        task_name: The name of the task.
        results_bag: The results bag to store evaluation results.
        extra_kwargs: Additional keyword arguments for the method.
        benchmark_mode: The benchmark mode. This is a fixture which based on user
            input, determines which type of methods should be run.
    """
    if benchmark_mode in ["snpe", "snle", "snre"]:
        train_and_eval_sequential_inference(
            inference_class, task_name, extra_kwargs, results_bag
        )
    else:
        train_and_eval_amortized_inference(
            inference_class, task_name, extra_kwargs, results_bag
        )
