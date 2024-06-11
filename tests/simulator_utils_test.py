# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import pytest
import torch
from torch import ones, zeros

from sbi.simulators.linear_gaussian import diagonal_linear_gaussian
from sbi.simulators.simutils import simulate_in_batches
from sbi.utils.torchutils import BoxUniform


@pytest.mark.parametrize("num_sims", (0, 10))
@pytest.mark.parametrize("batch_size", (1, 10))
@pytest.mark.parametrize(
    "simulator", (diagonal_linear_gaussian, lambda _: torch.randn((2,)))
)
@pytest.mark.parametrize("seed", (None, 42))
def test_simulate_in_batches(
    num_sims,
    batch_size,
    simulator,
    seed,
    prior=BoxUniform(zeros(5), ones(5)),
):
    """Test combinations of num_sims and simulation_batch_size."""

    theta = prior.sample((num_sims,))
    # run twice to check seeding.
    x1 = simulate_in_batches(simulator, theta, batch_size, seed=seed)
    x2 = simulate_in_batches(simulator, theta, batch_size, seed=seed)

    if seed is None and num_sims > 0:
        assert not torch.equal(x1, x2)
    else:
        assert torch.equal(x1, x2)
