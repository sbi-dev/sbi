# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
import pickle
import shutil
from logging import warning
from pathlib import Path
from shutil import rmtree

import pytest
import torch

from sbi.utils.sbiutils import seed_all_backends

# Seed for `set_seed` fixture. Change to random state of all seeded tests.
seed = 1
harvested_fixture_data = None


# Use seed automatically for every test function.
@pytest.fixture(autouse=True)
def set_seed():
    seed_all_backends(seed)


@pytest.fixture(scope="session", autouse=True)
def set_default_tensor_type():
    torch.set_default_dtype(torch.float32)


# Pytest hook to skip GPU tests if no devices are available.
def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if no devices are available."""
    gpu_device_available = (
        torch.cuda.is_available() or torch.backends.mps.is_available()
    )
    if not gpu_device_available:
        skip_gpu = pytest.mark.skip(reason="No devices available")
        skip_bm = pytest.mark.skip(reason="Benchmarking disabled")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

            if not config.getoption("--bm"):
                item.add_marker(skip_bm)

    # Filter tests to only those with the 'benchmark' marker
    if config.getoption("--bm"):
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


@pytest.fixture
def benchmark_mode(request):
    """Fixture to access the --bm value in test files."""
    return request.config.getoption("--bm-mode")


@pytest.fixture(scope="session", autouse=True)
def finalize_fixture_store(request, fixture_store):
    # The code before `yield` runs at the start of the session (before tests).
    yield
    # The code after `yield` runs after all tests have completed.
    # At this point, fixture_store should have all the harvested data.
    global harvested_fixture_data
    harvested_fixture_data = dict(fixture_store)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    Custom pytest terminal summary to display mini SBIBM results with relative coloring
    per task.

    This function is called after the test session ends and generates a summary
    of the results if the `--bm` option is specified. It displays the results
    in a formatted table with methods as rows and tasks as columns, applying
    relative coloring to metrics based on their performance within each task.

    Args:
        terminalreporter (TerminalReporter): The terminal reporter object for writing
            output.
        exitstatus (int): The exit status of the test session.
        config (Config): The pytest config object.
    """
    if config.getoption("--bm"):
        terminal_width = shutil.get_terminal_size().columns
        summary_text = " mini SBIBM results "
        centered_line = summary_text.center(terminal_width, '=')
        colored_line = f"\033[96m{centered_line}\033[0m"
        terminalreporter.write_line(colored_line)

        if harvested_fixture_data is not None:
            terminalreporter.write_line("Amortized inference:")

            results = harvested_fixture_data["results_bag"]

            # Extract relevant data (method, task, metric)
            methods = set()
            tasks = set()
            data = {}  # (method, task) -> metric

            for _, info in results.items():
                method = info.get('method')
                task = info.get('task_name')
                metric = info.get('metric')

                if method is not None and task is not None:
                    methods.add(method)
                    tasks.add(task)
                    data[(method, task)] = metric

            methods = sorted(methods)
            tasks = sorted(tasks)

            if not methods or not tasks:
                terminalreporter.write_line("No methods or tasks found.")
                return

            # Determine column widths
            method_col_width = max(len(m) for m in methods)
            task_col_widths = {t: max(len(t), 10) for t in tasks}

            # Print the header row
            header = " " * (method_col_width + 2)
            for t in tasks:
                header += t.center(task_col_widths[t] + 2)
            terminalreporter.write_line(header)

            # Print separator line
            sep_line = "-" * len(header)
            terminalreporter.write_line(sep_line)

            # Calculate min and max for each task
            min_max_per_task = {}
            for t in tasks:
                task_metrics = [data.get((m, t), float('inf')) for m in methods]
                min_max_per_task[t] = (min(task_metrics), max(task_metrics))

            # Print each row with colored values
            for m in methods:
                row = m.ljust(method_col_width + 2)
                for t in tasks:
                    val = data.get((m, t), "N/A")
                    if val == "N/A":
                        val_str = "N/A"
                        row += val_str.center(task_col_widths[t] + 2)
                    else:
                        val = float(val)
                        min_val, max_val = min_max_per_task[t]
                        normalized_val = (
                            (val - min_val) / (max_val - min_val)
                            if max_val > min_val
                            else 0.5
                        )

                        # Determine color based on normalized value
                        if normalized_val == 1.0:
                            color = "\033[92m"  # Green for best
                        elif normalized_val == 0.0:
                            color = "\033[91m"  # Red for worst
                        else:
                            color = f"\033[9{int(2 + normalized_val * 3)}m"

                        val_str = format(val, ".3f")
                        row += f"{color}{val_str}\033[0m".center(task_col_widths[t] + 2)
                terminalreporter.write_line(row)
        else:
            terminalreporter.write_line("No harvested fixture data found yet.")


@pytest.fixture(scope="function")
def mcmc_params_accurate() -> dict:
    """Fixture for MCMC parameters for functional tests."""
    return dict(num_chains=20, thin=2, warmup_steps=50)


@pytest.fixture(scope="function")
def mcmc_params_fast() -> dict:
    """Fixture for MCMC parameters for fast tests."""
    return dict(num_chains=1, thin=1, warmup_steps=1)


# Pytest harvest xdist support - not sure if we need it (for me xdist is always slower).


# Define the folder in which temporary worker's results will be stored
RESULTS_PATH = Path('./.xdist_results/')
RESULTS_PATH.mkdir(exist_ok=True)


def pytest_harvest_xdist_init():
    # reset the recipient folder
    if RESULTS_PATH.exists():
        rmtree(RESULTS_PATH)
    RESULTS_PATH.mkdir(exist_ok=False)
    return True


def pytest_harvest_xdist_worker_dump(worker_id, session_items, fixture_store):
    # persist session_items and fixture_store in the file system
    with open(RESULTS_PATH / ('%s.pkl' % worker_id), 'wb') as f:
        try:
            pickle.dump((session_items, fixture_store), f)
        except Exception as e:
            warning(
                "Error while pickling worker %s's harvested results: " "[%s] %s",
                (worker_id, e.__class__, e),
            )
    return True


def pytest_harvest_xdist_load():
    # restore the saved objects from file system
    workers_saved_material = dict()
    for pkl_file in RESULTS_PATH.glob('*.pkl'):
        wid = pkl_file.stem
        with pkl_file.open('rb') as f:
            workers_saved_material[wid] = pickle.load(f)
    return workers_saved_material


def pytest_harvest_xdist_cleanup():
    # delete all temporary pickle files
    rmtree(RESULTS_PATH)
    return True
