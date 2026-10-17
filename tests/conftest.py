# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import shutil
from pathlib import Path

import pytest
import torch
from pytest_harvest import get_session_results_df, is_main_process

from sbi.inference.posteriors.posterior_parameters import MCMCPosteriorParameters
from sbi.utils.sbiutils import seed_all_backends
from sbi.utils.torchutils import gpu_available

from .mini_sbibm.results import (
    RESULT_COLUMNS,
    benchmark_label,
    merge_results,
    write_results_tables,
)

# Seed for `set_seed` fixture. Change to random state of all seeded tests.
seed = 1
harvested_fixture_data = None

BENCHMARK_LABEL = pytest.StashKey[str]()


# Use seed automatically for every test function.
@pytest.fixture(autouse=True)
def set_seed():
    seed_all_backends(seed)


@pytest.fixture(autouse=True)
def guard_torch_validation_default():
    """Fail the test that mutates torch's global validation default.

    `set_default_validate_args` is a torch staticmethod: called on any instance, it
    changes validation for every distribution constructed afterwards, which makes
    test failures order-dependent. sbi sets validation per instance only.
    """
    default = torch.distributions.Distribution._validate_args
    yield
    polluted = torch.distributions.Distribution._validate_args is not default
    torch.distributions.Distribution._validate_args = default
    assert not polluted, (
        "This test changed torch's global validation default. Use "
        "`sbi.utils.torchutils.set_validate_args` to configure single instances."
    )


@pytest.fixture(scope="session", autouse=True)
def set_default_tensor_type():
    torch.set_default_dtype(torch.float32)


# Pytest hook to skip GPU tests if no devices are available.
def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if no CUDA or MPS device is available."""
    if not gpu_available():
        skip_gpu = pytest.mark.skip(reason="No GPU (CUDA or MPS) device available")

        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

    if not config.getoption("--bm"):
        # Skip marked benchmarking tests
        skip_bm = pytest.mark.skip(reason="Benchmarking disabled")
        for item in items:
            if "benchmark" in item.keywords:
                item.add_marker(skip_bm)
    else:
        # Filter tests to only those with the 'benchmark' marker
        filtered_items = []
        for item in items:
            if item.get_closest_marker("benchmark"):
                filtered_items.append(item)

        items[:] = filtered_items  # Inplace!


# Run mini-benchmark tests with `pytest --print-harvest`
def pytest_addoption(parser):
    parser.addoption(
        "--bm",
        action="store_true",
        default=False,
        help="Run mini-benchmark tests with specified mode",
    )
    parser.addoption(
        "--bm-mode",
        action="store",
        default=None,
        help="Run mini-benchmark tests with specified mode",
    )
    parser.addoption(
        "--bm-estimators",
        action="store",
        default=None,
        help="Comma-separated estimators for the selected mini-benchmark mode",
    )

    parser.addoption(
        "--bm-label",
        action="store",
        default=None,
        help="Name of this mini-benchmark run in the results table "
        "(default: the current git branch)",
    )
    parser.addoption(
        "--bm-results-dir",
        action="store",
        default=".bm_results",
        help="Folder of the mini-benchmark results files (default: .bm_results). "
        "Use the same folder to compare runs from different checkouts",
    )
    parser.addoption(
        "--bm-seeds",
        action="store",
        default=1,
        type=int,
        help="Number of training seeds per mini-benchmark case",
    )
    parser.addoption(
        "--bm-num-simulations",
        action="store",
        default=2000,
        type=int,
        help="Run mini-benchmark tests with specified number of simulations",
    )
    parser.addoption(
        "--bm-num-iid-trials",
        action="store",
        default=10,
        type=int,
        help="Number of independent trials for mixed data benchmark observations",
    )


def pytest_configure(config):
    if not config.getoption("--bm"):
        return
    # Parallel benchmark workers would otherwise each start one torch thread per core.
    if hasattr(config, "workerinput"):
        torch.set_num_threads(1)
    # Resolve the label at the start, so switching branches during a run does not
    # change where its results go.
    config.stash[BENCHMARK_LABEL] = benchmark_label(config.getoption("--bm-label"))


@pytest.fixture
def benchmark_num_simulations(request):
    """Fixture to access the --bm-num-simulations value in test files."""
    return int(request.config.getoption("--bm-num-simulations"))


@pytest.fixture(scope="session", autouse=True)
def finalize_fixture_store(request, fixture_store):
    # The code before `yield` runs at the start of the session (before tests).
    yield
    # The code after `yield` runs after all tests have completed.
    # At this point, fixture_store should have all the harvested data.
    global harvested_fixture_data
    harvested_fixture_data = dict(fixture_store)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the stored mini SBIBM results of all run labels."""
    if not config.getoption("--bm"):
        return
    write = terminalreporter.write_line
    terminal_width = shutil.get_terminal_size().columns
    write(f"\033[96m{' mini SBIBM results '.center(terminal_width, '=')}\033[0m")
    try:
        write_results_tables(
            write, Path(config.getoption("--bm-results-dir")).expanduser()
        )
    except Exception as e:
        write(f"Error processing results: {e}")


@pytest.fixture(scope="function")
def mcmc_params_accurate() -> MCMCPosteriorParameters:
    """Fixture for MCMC parameters for functional tests."""
    return MCMCPosteriorParameters(num_chains=20, thin=2, warmup_steps=50)


@pytest.fixture(scope="function")
def mcmc_params_fast() -> MCMCPosteriorParameters:
    """Fixture for MCMC parameters for fast tests."""
    return MCMCPosteriorParameters(num_chains=1, thin=1, warmup_steps=1)


def pytest_sessionfinish(session):
    """Merge the results of this run into the results file of its label.

    With xdist, the main process receives the results of all workers.
    """
    if not session.config.getoption("--bm") or not is_main_process(session):
        return

    results = get_session_results_df(session)
    if "c2st" not in results.columns:
        return
    results = results[(results["status"] == "passed") & results["c2st"].notna()]
    if results.empty:
        return
    label = session.config.stash[BENCHMARK_LABEL]
    results = results.assign(label=label)[RESULT_COLUMNS]
    merge_results(
        Path(session.config.getoption("--bm-results-dir")).expanduser(), results
    )
