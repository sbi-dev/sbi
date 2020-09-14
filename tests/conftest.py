import numpy
import pytest
import torch

# Seed for `set_seed` fixture. Change to change random state of all seeded tests.
seed = 1


@pytest.fixture(scope="session", autouse=True)
def set_default_tensor_type():
    torch.set_default_tensor_type("torch.FloatTensor")


@pytest.fixture(scope="function")
def set_seed():
    torch.manual_seed(seed)
    numpy.random.seed(seed)


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda: mark test that requires cuda.")
