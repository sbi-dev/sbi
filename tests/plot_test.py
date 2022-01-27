# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import pytest
import torch

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from torch.utils.tensorboard import SummaryWriter

from sbi.analysis import plot_summary
from sbi.inference import (
    SNLE,
    SNPE,
    SNRE,
    prepare_for_sbi,
    simulate_for_sbi,
)
from sbi.utils import BoxUniform


@pytest.mark.parametrize("method", (SNPE, SNLE, SNRE))
def test_plot_summary(method, tmp_path):
    num_dim = 1
    prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))

    summary_writer = SummaryWriter(tmp_path)

    def linear_gaussian(theta):
        return theta + 1.0 + torch.randn_like(theta) * 0.1

    simulator, prior = prepare_for_sbi(linear_gaussian, prior)

    inference = method(prior=prior, summary_writer=summary_writer)
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=6)
    train_kwargs = (
        dict(max_num_epochs=5, validation_fraction=0.5, num_atoms=2)
        if method == SNRE
        else dict(max_num_epochs=1)
    )
    _ = inference.append_simulations(theta, x).train(**train_kwargs)
    fig, axes = plot_summary(inference)
    assert isinstance(fig, Figure) and isinstance(axes[0], Axes)
