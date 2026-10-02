# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from pytest_harvest import ResultsBag

from sbi.inference import FMPE, NLE, NPE, NPE_PFN, NPSE, NRE
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.trainers.npe import NPE_C
from sbi.inference.trainers.nre import BNRE, NRE_A, NRE_B, NRE_C
from sbi.utils.metrics import c2st

from .mini_sbibm import get_task
from .mini_sbibm.base_task import Task

# Global settings
SEED = 0
TASKS = ["two_moons", "linear_mvg_2d", "gaussian_linear", "slcp"]
NUM_EVALUATION_OBS = 10
NUM_ROUNDS_SEQUENTIAL = 2
NUM_EVALUATION_OBS_SEQ = 1
SEQUENTIAL_MODES = {"snpe", "snle", "snre"}
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
}
ESTIMATOR_ARGUMENTS = {
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


def _class_id(inference_class, mode: str | None) -> str:
    """Return the class name, prefixed with "S" for multi-round runs."""
    prefix = "S" if mode in SEQUENTIAL_MODES else ""
    return prefix + inference_class.__name__


# Use pytest.mark.parametrize dynamically
# Generates a list of methods to test based on the benchmark mode
def pytest_generate_tests(metafunc):
    """
    Dynamically generates a list of methods to test based on the benchmark mode.

    Args:
        metafunc: The metafunc object from pytest.
    """
    if not {"inference_class", "extra_kwargs"}.intersection(metafunc.fixturenames):
        return

    mode = _benchmark_mode(metafunc.config)
    if "inference_class" in metafunc.fixturenames:
        classes = METHOD_GROUPS[mode]
        metafunc.parametrize(
            "inference_class", classes, ids=[_class_id(c, mode) for c in classes]
        )
    if "extra_kwargs" in metafunc.fixturenames:
        kwargs_group = _benchmark_kwargs(metafunc.config, mode)
        metafunc.parametrize(
            "extra_kwargs", kwargs_group, ids=[_kwargs_id(p) for p in kwargs_group]
        )
    num_seeds = metafunc.config.getoption("--bm-seeds")
    if num_seeds < 1:
        raise pytest.UsageError("--bm-seeds must be at least 1.")
    # One seed keeps the `benchmark_seed` fixture, and with it today's test ids.
    if "benchmark_seed" in metafunc.fixturenames and num_seeds > 1:
        seeds = range(SEED, SEED + num_seeds)
        metafunc.parametrize("benchmark_seed", seeds, ids=[f"seed{s}" for s in seeds])


@pytest.fixture
def benchmark_seed() -> int:
    """Training seed of a benchmark case. Parametrized by --bm-seeds."""
    return SEED


def eval_observations(posterior: NeuralPosterior, task: Task) -> dict[str, float]:
    """
    Evaluates the posterior on the first `NUM_EVALUATION_OBS` observations.

    Args:
        posterior: The posterior distribution.
        task: The task object.

    Returns:
        The metrics of `eval_observation`, averaged over the observations.
    """
    metrics = [
        eval_observation(posterior, task, i) for i in range(1, NUM_EVALUATION_OBS + 1)
    ]
    return {key: sum(m[key] for m in metrics) / len(metrics) for key in metrics[0]}


def eval_observation(
    posterior: NeuralPosterior,
    task: Task,
    idx_observation: int,
    num_samples: int = 1000,
) -> dict[str, float]:
    """
    Compares posterior samples with reference samples for one observation.

    The mean and std errors are absolute errors of the marginal means and standard
    deviations, divided by the reference standard deviation and averaged over
    parameter dimensions. Unlike C2ST, the std error says directly how much too wide
    or too narrow the posterior is.

    Args:
        posterior: The posterior distribution.
        task: The task object.
        idx_observation: The observation index.
        num_samples: The number of posterior samples.

    Returns:
        The C2ST value and the mean and std errors.
    """
    x_o = task.get_observation(idx_observation)
    reference_samples = task.get_reference_posterior_samples(idx_observation)
    samples = posterior.sample((num_samples,), x=x_o)
    if isinstance(samples, tuple):
        samples = samples[0]
    assert reference_samples.shape[0] >= num_samples, "Not enough reference samples"

    reference_std = reference_samples.std(0)
    mean_error = (samples.mean(0) - reference_samples.mean(0)).abs() / reference_std
    std_error = (samples.std(0) - reference_std).abs() / reference_std
    return {
        "c2st": float(c2st(reference_samples[:num_samples], samples)),
        "mean_err": float(mean_error.mean()),
        "std_err": float(std_error.mean()),
    }


def train_and_eval_amortized_inference(
    inference_class,
    task_name: str,
    benchmark_num_simulations: int,
    extra_kwargs: dict,
    seed: int,
) -> dict[str, float]:
    """
    Performs amortized inference evaluation.

    Args:
        inference_class: The inference class.
        task_name: The name of the task.
        benchmark_num_simulations: The number of training simulations.
        extra_kwargs: Additional keyword arguments for the method.
        seed: The training seed.

    Returns:
        The metrics, averaged over the evaluation observations.
    """
    torch.manual_seed(seed)
    task = get_task(task_name)
    thetas, xs = task.get_data(benchmark_num_simulations)
    prior = task.get_prior()

    inference = inference_class(prior, **extra_kwargs)
    _ = inference.append_simulations(thetas, xs).train(**TRAIN_KWARGS)

    posterior = inference.build_posterior()

    return eval_observations(posterior, task)


def train_and_eval_sequential_inference(
    inference_class,
    task_name: str,
    benchmark_num_simulations: int,
    extra_kwargs: dict,
    seed: int,
) -> dict[str, float]:
    """
    Performs sequential inference evaluation.

    Args:
        inference_class: The inference class.
        task_name: The name of the task.
        benchmark_num_simulations: The total number of training simulations.
        extra_kwargs: Additional keyword arguments for the method.
        seed: The training seed.

    Returns:
        The metrics for the evaluation observation.
    """
    task = get_task(task_name)
    idx_eval = NUM_EVALUATION_OBS_SEQ
    # Load x_o before seeding: the Gaussian tasks reseed torch in get_observation.
    x_o = task.get_observation(idx_eval)
    torch.manual_seed(seed)
    num_simulations = benchmark_num_simulations // NUM_ROUNDS_SEQUENTIAL
    thetas, xs = task.get_data(num_simulations)
    prior = task.get_prior()
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

    return eval_observation(posterior, task, idx_eval)


@pytest.mark.benchmark
@pytest.mark.parametrize("task_name", TASKS, ids=str)
def test_run_benchmark(
    inference_class,
    task_name: str,
    results_bag: ResultsBag,
    extra_kwargs: dict,
    benchmark_mode: str | None,
    benchmark_num_simulations: int,
    benchmark_seed: int,
) -> None:
    """
    Benchmark test for amortized and sequential inference methods.

    Args:
        inference_class: The inference class to test i.e. NPE, NLE, NRE ...
        task_name: The name of the task.
        results_bag: The results bag to store evaluation results. Subclass of dict,
            but allows item assignment with dot notation.
        extra_kwargs: Additional keyword arguments for the method.
        benchmark_mode: The benchmark mode. This is a fixture which based on user
            input, determines which type of methods should be run.
        benchmark_num_simulations: The number of training simulations.
        benchmark_seed: The training seed.
    """
    if benchmark_mode in SEQUENTIAL_MODES:
        train_and_eval = train_and_eval_sequential_inference
    else:
        train_and_eval = train_and_eval_amortized_inference
    metrics = train_and_eval(
        inference_class,
        task_name,
        benchmark_num_simulations,
        extra_kwargs,
        benchmark_seed,
    )

    for key, value in metrics.items():
        results_bag[key] = round(value, 3)
    results_bag.num_simulations = benchmark_num_simulations
    results_bag.task_name = task_name
    results_bag.seed = benchmark_seed
    results_bag.method = (
        f"{_class_id(inference_class, benchmark_mode)}-{_kwargs_id(extra_kwargs)}"
    )
