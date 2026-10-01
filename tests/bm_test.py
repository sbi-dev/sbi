# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from functools import partial

import pytest
import torch
from pytest_harvest import ResultsBag

from sbi.inference import FMPE, MNLE, NLE, NPE, NPE_PFN, NPSE, NRE
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.trainers.npe import NPE_C
from sbi.inference.trainers.nre import BNRE, NRE_A, NRE_B, NRE_C
from sbi.utils.metrics import c2st

from .mini_sbibm import MixedData, get_task
from .mini_sbibm.base_task import Task

# Global settings
SEED = 0
TASKS = ["two_moons", "linear_mvg_2d", "gaussian_linear", "slcp"]
NUM_EVALUATION_OBS = 3  # Currently only 3 observation tested for speed
NUM_ROUNDS_SEQUENTIAL = 2
NUM_EVALUATION_OBS_SEQ = 1
TRAIN_KWARGS = {}

# Density estimators to test
DENSITY_ESTIMATORS = ["mdn", "made", "maf", "nsf", "maf_rqs"]  # "Kinda exhaustive"
CLASSIFIERS = ["mlp", "resnet"]
VF_ESTIMATORS = ["mlp", "ada_mlp", "transformer"]

# Benchmarking method groups i.e. what to run for different --bm-mode
METHOD_GROUPS = {
    "none": [NPE, NPE_PFN, NRE, NLE, FMPE, NPSE],
    "npe": [NPE],
    "npe_pfn": [NPE_PFN],
    "nle": [NLE],
    "nre": [NRE_A, NRE_B, NRE_C, BNRE],
    "fmpe": [FMPE],
    "npse": [NPSE],
    "vfpe": [FMPE, NPSE],
    "snpe": [NPE_C],  # NPE_B not implemented, NPE_A need Gaussian prior
    "snle": [NLE],
    "snre": [NRE_A, NRE_B, NRE_C, BNRE],
    "mnle": [MNLE],
}
METHOD_PARAMS = {
    "none": [{}],
    "npe": [{"density_estimator": de} for de in DENSITY_ESTIMATORS],
    "npe_pfn": [{}],
    "nle": [{"density_estimator": de} for de in ["maf", "nsf"]],
    "nre": [{"classifier": cl} for cl in CLASSIFIERS],
    "fmpe": [{"vf_estimator": nn} for nn in VF_ESTIMATORS],
    "npse": [
        {"vf_estimator": nn, "sde_type": sde}
        for nn in VF_ESTIMATORS
        for sde in ["ve", "vp"]
    ],
    "vfpe": [{"vf_estimator": nn} for nn in VF_ESTIMATORS],
    "snpe": [{}],
    "snle": [{}],
    "snre": [{}],
    "mnle": [{}],
}
ESTIMATOR_ARGUMENTS = {
    "mnle": "density_estimator",
    "npe": "density_estimator",
    "nle": "density_estimator",
    "nre": "classifier",
    "fmpe": "vf_estimator",
    "npse": "vf_estimator",
    "vfpe": "vf_estimator",
    "snpe": "density_estimator",
    "snle": "density_estimator",
    "snre": "classifier",
}


def _benchmark_mode(config) -> str:
    """Return and validate the selected benchmark mode."""
    mode = config.getoption("--bm-mode")
    name = "none" if mode is None else str(mode).lower()
    if name not in METHOD_GROUPS:
        supported = ", ".join(name for name in METHOD_GROUPS if name != "none")
        raise pytest.UsageError(
            f"Unknown benchmark mode '{mode}'. Supported modes: {supported}."
        )
    return name


def _benchmark_kwargs(config, mode: str) -> list[dict]:
    """Return method arguments, applying an optional estimator override."""
    estimator_option = config.getoption("--bm-estimators")
    if estimator_option is None:
        return METHOD_PARAMS[mode]

    estimator_argument = ESTIMATOR_ARGUMENTS.get(mode)
    if estimator_argument is None:
        raise pytest.UsageError(
            "--bm-estimators requires a benchmark mode whose methods use the same "
            "estimator argument."
        )

    estimators = [value.strip() for value in estimator_option.split(",")]
    if not estimators or any(not value for value in estimators):
        raise pytest.UsageError(
            "--bm-estimators requires one or more comma-separated estimator names."
        )

    remaining_options = []
    for parameters in METHOD_PARAMS[mode]:
        options = {
            key: value for key, value in parameters.items() if key != estimator_argument
        }
        if options not in remaining_options:
            remaining_options.append(options)

    return [
        {estimator_argument: estimator, **options}
        for estimator in estimators
        for options in remaining_options
    ]


def _kwargs_id(parameters: dict) -> str:
    """Return a readable identifier for one method configuration."""
    return "-".join(str(value) for value in parameters.values()) or "default"


@pytest.fixture
def task(request) -> Task:
    """Build the task selected by the benchmark mode."""
    return request.param()


# Use pytest.mark.parametrize dynamically
# Generates a list of methods to test based on the benchmark mode
def pytest_generate_tests(metafunc):
    """
    Dynamically generates a list of methods to test based on the benchmark mode.

    Args:
        metafunc: The metafunc object from pytest.
    """
    if not {"inference_class", "extra_kwargs", "task"}.intersection(
        metafunc.fixturenames
    ):
        return

    mode = _benchmark_mode(metafunc.config)
    if "inference_class" in metafunc.fixturenames:
        metafunc.parametrize("inference_class", METHOD_GROUPS[mode])
    if "extra_kwargs" in metafunc.fixturenames:
        kwargs_group = _benchmark_kwargs(metafunc.config, mode)
        metafunc.parametrize(
            "extra_kwargs", kwargs_group, ids=[_kwargs_id(p) for p in kwargs_group]
        )
    if "task" in metafunc.fixturenames:
        if mode == "mnle":
            num_trials = metafunc.config.getoption("--bm-num-iid-trials")
            tasks = [partial(MixedData, num_trials=num_trials)]
            task_ids = [f"mixed_data-{num_trials}trials"]
        else:
            tasks = [partial(get_task, name) for name in TASKS]
            task_ids = TASKS
        metafunc.parametrize("task", tasks, ids=task_ids, indirect=True)


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
    inference_class,
    task: Task,
    benchmark_num_simulations: int,
    extra_kwargs: dict,
    results_bag: ResultsBag,
) -> None:
    """
    Performs amortized inference evaluation.

    Args:
        method: The inference method.
        task: The benchmark task.
        benchmark_num_simulations: Number of training simulations.
        extra_kwargs: Additional keyword arguments for the method.
        results_bag: The results bag to store evaluation results. Subclass of dict, but
            allows item assignment with dot notation.
    """
    torch.manual_seed(SEED)
    thetas, xs = task.get_data(benchmark_num_simulations)
    prior = task.get_prior()

    inference = inference_class(prior, **extra_kwargs)
    _ = inference.append_simulations(thetas, xs).train(**TRAIN_KWARGS)

    posterior = inference.build_posterior()

    mean_c2st = standard_eval_c2st_loop(posterior, task)

    # Cache results
    results_bag.metric = mean_c2st
    results_bag.num_simulations = benchmark_num_simulations
    results_bag.task_name = task.name
    results_bag.method = inference_class.__name__ + str(extra_kwargs)


def train_and_eval_sequential_inference(
    inference_class,
    task: Task,
    benchmark_num_simulations: int,
    extra_kwargs: dict,
    results_bag: ResultsBag,
) -> None:
    """
    Performs sequential inference evaluation.

    Args:
        method: The inference method.
        task: The benchmark task.
        benchmark_num_simulations: Number of training simulations.
        extra_kwargs (dict): Additional keyword arguments for the method.
        results_bag: The results bag to store evaluation results.
    """
    torch.manual_seed(SEED)
    num_simulations = benchmark_num_simulations // NUM_ROUNDS_SEQUENTIAL
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
    results_bag.num_simulations = benchmark_num_simulations
    results_bag.task_name = task.name
    results_bag.method = inference_class.__name__ + str(extra_kwargs)


@pytest.mark.benchmark
def test_run_benchmark(
    inference_class,
    task: Task,
    results_bag,
    extra_kwargs: dict,
    benchmark_mode: str,
    benchmark_num_simulations: int,
) -> None:
    """
    Benchmark test for amortized and sequential inference methods.

    Args:
        inference_class: The inference class to test i.e. NPE, NLE, NRE ...
        task: The benchmark task.
        results_bag: The results bag to store evaluation results.
        extra_kwargs: Additional keyword arguments for the method.
        benchmark_mode: The benchmark mode. This is a fixture which based on user
            input, determines which type of methods should be run.
        benchmark_num_simulations: Number of training simulations.
    """
    if benchmark_mode in ["snpe", "snle", "snre"]:
        train_and_eval_sequential_inference(
            inference_class,
            task,
            benchmark_num_simulations,
            extra_kwargs,
            results_bag,
        )
    else:
        train_and_eval_amortized_inference(
            inference_class,
            task,
            benchmark_num_simulations,
            extra_kwargs,
            results_bag,
        )
