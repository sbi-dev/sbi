# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch

from sbi.utils.sbiutils import seed_all_backends
from pytest_harvest import get_session_results_dct

# Seed for `set_seed` fixture. Change to random state of all seeded tests.
seed = 1


# Pytest harvest
@pytest.fixture(scope="session", autouse=True)
def collect_results(request):
    """
    Automatically runs after the test session.
    We'll collect the test results and store them for later use (e.g. in a dataframe).
    """
    print(request)
    yield  # Run all tests first
    # At this point, tests are all done. Let's harvest the results:
    session_results = get_session_results_dct(request.session)

    # Now `session_results` is a dict with entries for each test function.
    # You can transform it into a DataFrame if you want.
    try:
        import pandas as pd

        df = pd.DataFrame.from_dict(session_results, orient='index')
        print("\nHarvested Test Results:\n", df)
    except ImportError:
        print("\nHarvested Test Results (raw dictionary):\n", session_results)


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


@pytest.fixture(scope="function")
def mcmc_params_accurate() -> dict:
    """Fixture for MCMC parameters for functional tests."""
    return dict(num_chains=20, thin=2, warmup_steps=50)


@pytest.fixture(scope="function")
def mcmc_params_fast() -> dict:
    """Fixture for MCMC parameters for fast tests."""
    return dict(num_chains=1, thin=1, warmup_steps=1)
