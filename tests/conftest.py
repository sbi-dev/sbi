import pytest
import torch

from sbi.utils.sbiutils import seed_all_backends

# Seed for `set_seed` fixture. Change to random state of all seeded tests.
seed = 1


# Use seed automatically for every test function.
@pytest.fixture(autouse=True)
def set_seed():
    seed_all_backends(seed)


@pytest.fixture(scope="session", autouse=True)
def set_default_tensor_type():
    torch.set_default_tensor_type("torch.FloatTensor")
