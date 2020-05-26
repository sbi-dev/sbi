import pytest
import torch
import numpy


# Seed for fixture. Change to change random state of all seeded tests.
seed = 1


# Fixture will be visible in all test files.
@pytest.fixture(scope="module")
def set_seed():
    torch.manual_seed(seed)
    numpy.random.seed(seed)
