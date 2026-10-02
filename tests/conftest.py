# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import torch
from pytest_harvest import get_session_results_df, is_main_process

from sbi.inference.posteriors.posterior_parameters import MCMCPosteriorParameters
from sbi.utils.sbiutils import seed_all_backends
from sbi.utils.torchutils import gpu_available

# Seed for `set_seed` fixture. Change to random state of all seeded tests.
seed = 1
harvested_fixture_data = None

# Mini SBIBM results. A new run replaces the stored rows with the same key.
RESULT_KEY = ["label", "method", "task_name", "seed"]
RESULT_COLUMNS = [*RESULT_KEY, "num_simulations", "c2st", "mean_err", "std_err"]
METRIC_TITLES = {
    "c2st": "C2ST (0.5 is best)",
    "mean_err": "Posterior mean error, in reference std (0 is best)",
    "std_err": "Posterior std error, in reference std (0 is best)",
}
MOVED_OLD_RESULTS = pytest.StashKey[bool]()


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
        help="Folder of the mini-benchmark results file (default: .bm_results). "
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


def pytest_configure(config):
    # Parallel benchmark workers would otherwise each start one torch thread per core.
    if config.getoption("--bm") and hasattr(config, "workerinput"):
        torch.set_num_threads(1)


@pytest.fixture
def benchmark_mode(request):
    """Fixture to access the --bm value in test files."""
    mode = request.config.getoption("--bm-mode")
    return None if mode is None else str(mode).lower()


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


def strip_ansi_escape_codes(text):
    ansi_escape = re.compile(r'\x1b\[.*?m')
    return ansi_escape.sub('', text)


# Function to center text with ANSI colors, adjusting for escape codes
def center_colored_text(text, width):
    visible_length = len(strip_ansi_escape_codes(text))
    padding = max(0, (width - visible_length) // 2)
    return " " * padding + text + " " * (width - visible_length - padding)


def _benchmark_label(config) -> str:
    """Return the run label: the --bm-label value, else the current git branch."""
    label = config.getoption("--bm-label")
    if label is not None:
        return label
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "default"
    return branch if branch and branch != "HEAD" else "default"


def _results_file(config) -> Path:
    """Return the path of the mini SBIBM results file."""
    return Path(config.getoption("--bm-results-dir")).expanduser() / "results_all.csv"


def _read_results(results_file: Path) -> pd.DataFrame | None:
    """Return the stored results, or None if there are none in the current format."""
    if not results_file.exists():
        return None
    try:
        results = pd.read_csv(results_file, dtype={"label": str})
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    if not set(RESULT_COLUMNS).issubset(results.columns):
        return None
    return results


def _write_metric_table(write, means: pd.DataFrame, spreads: pd.DataFrame) -> None:
    """Write one table with a row per method and label and a column per task.

    Within each task, the best (lowest) value is green and the worst red.
    """
    tasks = list(means.columns)
    texts = {}
    for row in means.index:
        for task in tasks:
            mean, spread = means.at[row, task], spreads.at[row, task]
            if pd.isna(mean):
                texts[row, task] = "N/A"
            elif pd.isna(spread):
                texts[row, task] = f"{mean:.3f}"
            else:
                texts[row, task] = f"{mean:.3f} ±{spread:.3f}"

    row_width = max(len(row) for row in means.index) + 2
    widths = {
        task: max(10, len(task), *(len(texts[row, task]) for row in means.index)) + 2
        for task in tasks
    }
    header = " " * row_width + "".join(task.center(widths[task]) for task in tasks)
    write(header)
    write("-" * len(header))

    for row in means.index:
        line = row.ljust(row_width)
        for task in tasks:
            mean = means.at[row, task]
            if pd.isna(mean):
                line += texts[row, task].center(widths[task])
                continue
            low, high = means[task].min(), means[task].max()
            normalized = (mean - low) / (high - low) if high > low else 0.5
            if normalized == 0.0:
                color = "\033[92m"  # Green for best
            elif normalized == 1.0:
                color = "\033[91m"  # Red for worst
            else:
                color = f"\033[9{int(2 + normalized * 3)}m"
            line += center_colored_text(
                f"{color}{texts[row, task]}\033[0m", widths[task]
            )
        write(line)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the stored mini SBIBM results, one table per metric.

    Rows are methods with their run label, columns are tasks. Several seeds of a
    case are shown as mean and standard deviation.
    """
    if not config.getoption("--bm"):
        return

    write = terminalreporter.write_line
    terminal_width = shutil.get_terminal_size().columns
    write(f"\033[96m{' mini SBIBM results '.center(terminal_width, '=')}\033[0m")
    if config.stash.get(MOVED_OLD_RESULTS, False):
        old_file = _results_file(config).with_name("results_all.old.csv")
        write(f"Moved results in an older format to {old_file}.")

    try:
        results = _read_results(_results_file(config))
        if results is None or results.empty:
            write("No results found.")
            return

        rows = results["method"] + " [" + results["label"]
        if results["num_simulations"].nunique() > 1:
            rows += ", " + results["num_simulations"].astype(str) + " sims"
        results = results.assign(row=rows + "]")

        for metric, title in METRIC_TITLES.items():
            stats = results.groupby(["row", "task_name"])[metric].agg(["mean", "std"])
            write(title)
            _write_metric_table(write, stats["mean"].unstack(), stats["std"].unstack())
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
    """Merge the results of this run into the mini SBIBM results file.

    With xdist, the main process receives the results of all workers. Rows with the
    same label, method, task and seed as a new result are replaced. A results file in an
    older format is moved to `results_all.old.csv`.
    """
    if not session.config.getoption("--bm") or not is_main_process(session):
        return

    results = get_session_results_df(session)
    if "c2st" not in results.columns:
        return
    results = results[(results["status"] == "passed") & results["c2st"].notna()]
    if results.empty:
        return
    results = results.assign(label=_benchmark_label(session.config))[RESULT_COLUMNS]

    results_file = _results_file(session.config)
    stored = _read_results(results_file)
    if stored is not None:
        replaced = stored.set_index(RESULT_KEY).index.isin(
            results.set_index(RESULT_KEY).index
        )
        results = pd.concat([stored[~replaced], results])
    elif results_file.exists():
        results_file.replace(results_file.with_name("results_all.old.csv"))
        session.config.stash[MOVED_OLD_RESULTS] = True

    results_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_file, index=False)
