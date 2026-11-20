# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import pytest
import torch
from torch import ones, zeros

from sbi.inference import simulate_for_sbi
from sbi.simulators.linear_gaussian import diagonal_linear_gaussian
from sbi.simulators.simutils import simulate_in_batches
from sbi.utils.sbiutils import seed_all_backends, seed_random_states
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


@pytest.fixture
def nondefault_cudnn_flags():
    """Set the cuDNN flags to values seeding must not overwrite, then restore them.

    The default of `deterministic=False, benchmark=False` cannot tell whether a call
    left the flags alone or set them, so we start from a non-default state.
    """
    flags = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    yield
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = flags


def test_seed_random_states_leaves_cudnn_flags(nondefault_cudnn_flags):
    seed_random_states(42)

    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True


def test_seed_all_backends_sets_cudnn_flags(nondefault_cudnn_flags):
    seed_all_backends(42)

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


@pytest.mark.parametrize("seed", (None, 42))
def test_simulating_leaves_cudnn_flags(seed, nondefault_cudnn_flags):
    """Simulating must not change global cuDNN settings, see #2159.

    With `num_workers > 1` the seeded simulator runs in a worker process, whose flags
    are not visible here. `seed_random_states` covers that call site directly.
    """
    prior = BoxUniform(zeros(5), ones(5))
    theta = prior.sample((4,))

    simulate_for_sbi(
        diagonal_linear_gaussian,
        prior,
        4,
        seed=seed,
        show_progress_bar=False,
    )
    simulate_in_batches(
        diagonal_linear_gaussian, theta, 2, seed=seed, show_progress_bars=False
    )

    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True
