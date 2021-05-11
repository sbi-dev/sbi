# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
from torch import ones, zeros

from sbi.simulators.linear_gaussian import diagonal_linear_gaussian
from sbi.simulators.simutils import simulate_in_batches
from sbi.utils.torchutils import BoxUniform
from sbi.utils.user_input_checks import prepare_for_sbi


@pytest.mark.parametrize(
    "num_sims",
    (0, 100, 1000),
)
@pytest.mark.parametrize("batch_size", (1, 100, 1000))
@pytest.mark.parametrize("simulator", (diagonal_linear_gaussian, lambda _: ones(2)))
def test_simulate_in_batches(
    num_sims,
    batch_size,
    simulator,
    prior=BoxUniform(zeros(5), ones(5)),
):
    """Test combinations of num_sims and simulation_batch_size."""

    simulator, prior = prepare_for_sbi(simulator, prior)
    theta = prior.sample((num_sims,))
    simulate_in_batches(simulator, theta, batch_size)
