# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
import shutil

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
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

    # Filter tests to only those with the 'bm' marker
    if config.getoption("--bm"):
        # Filter tests to only those with the 'bm' marker
        filtered_items = []
        for item in items:
            # Using newer pytest versions (>=4.6):
            if item.get_closest_marker("benchmark"):
                filtered_items.append(item)

        items[:] = filtered_items  # Inplace!


# Run mini-benchmark tests with `pytest --print-harvest`
def pytest_addoption(parser):
    parser.addoption(
        "--bm",
        action="store_true",
        default=False,
        help="Print the harvest results at the end of the test session",
    )


@pytest.fixture(scope="session", autouse=True)
def finalize_fixture_store(request, fixture_store):
    # The code before `yield` runs at the start of the session (before tests).
    yield
    # The code after `yield` runs after all tests have completed.
    # At this point, fixture_store should have all the harvested data.
    global harvested_fixture_data
    harvested_fixture_data = dict(fixture_store)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if config.getoption("--bm"):
        terminal_width = shutil.get_terminal_size().columns
        summary_text = " mini SBIBM results "
        centered_line = summary_text.center(terminal_width, '=')
        colored_line = f"\033[96m{centered_line}\033[0m"
        terminalreporter.write_line(colored_line)

        if harvested_fixture_data is not None:
            terminalreporter.write_line("Harvested Fixture Data:")

            results = harvested_fixture_data["results_bag"]

            # Extract relevant data (method, task, metric)
            methods = set()
            tasks = set()
            data = {}  # (method, task) -> metric

            for _, info in results.items():
                method = info.get('method')
                task = info.get('task_name')
                metric = info.get('metric')
                # You can also choose another metric or value to display

                if method is not None and task is not None:
                    methods.add(method)
                    tasks.add(task)
                    data[(method, task)] = metric

            # Sort methods and tasks for consistent display
            methods = sorted(methods)
            tasks = sorted(tasks)

            if not methods or not tasks:
                terminalreporter.write_line("No methods or tasks found.")
                return

            # Determine column widths
            # We'll allow some spacing and ensure each column fits its longest entry
            method_col_width = max(len(m) for m in methods)
            task_col_widths = {}
            for t in tasks:
                task_col_widths[t] = max(len(t), 10)  # at least length 10

            # Print the header row: tasks
            header = " " * (method_col_width + 2)  # space for method column
            for t in tasks:
                header += t.center(task_col_widths[t] + 2)
            terminalreporter.write_line(header)

            # Print separator line
            sep_line = "-" * len(header)
            terminalreporter.write_line(sep_line)

            # Print each row: method followed by metrics for each task
            for m in methods:
                row = m.ljust(method_col_width + 2)
                for t in tasks:
                    val = data.get((m, t), "N/A")
                    # Convert metric to string with formatting if needed
                    # e.g. format(val, ".3f") if val is a float
                    val_str = str(val)
                    row += val_str.center(task_col_widths[t] + 2)
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
