from __future__ import annotations

import pytest
import torch
from torch import ones, zeros

from sbi.simulators.linear_gaussian import standard_linear_gaussian
from sbi.simulators.simutils import simulate_in_batches
from sbi.utils.torchutils import BoxUniform

# Use cpu by default.
torch.set_default_tensor_type("torch.FloatTensor")


@pytest.mark.parametrize(
    "num_sims", (pytest.param(0, marks=pytest.mark.xfail), 100, 1000)
)
@pytest.mark.parametrize("batch_size", (1, 100, 1000))
def test_simulate_in_batches(
    num_sims,
    batch_size,
    simulator=standard_linear_gaussian,
    prior=BoxUniform(zeros(5), ones(5)),
):
    """Test combinations of num_sims and simulation_batch_size. """

    theta = prior.sample((num_sims,))
    simulate_in_batches(simulator, theta, batch_size)
