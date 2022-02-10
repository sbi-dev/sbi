import numpy
import pytest
import torch

# Seed for `set_seed` fixture. Change to change random state of all seeded tests.
seed = 1


# Use seed automatically for every test function.
@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(seed)
    numpy.random.seed(seed)


@pytest.fixture(scope="session", autouse=True)
def set_default_tensor_type():
    torch.set_default_tensor_type("torch.FloatTensor")
