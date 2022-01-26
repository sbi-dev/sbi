# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from sbi.inference import (
    SNLE,
    SNPE,
    SNRE,
    prepare_for_sbi,
    simulate_for_sbi,
)
from sbi import utils
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def test_plot_summary(tmp_path):
    num_dim = 1
    prior = utils.BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))

    summary_writer = SummaryWriter(tmp_path)

    def linear_gaussian(theta):
        return theta + 1.0 + torch.randn_like(theta) * 0.1

    simulator, prior = prepare_for_sbi(linear_gaussian, prior)

    # SNPE
    inference = SNPE(prior=prior, summary_writer=summary_writer)
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=5)
    _ = inference.append_simulations(theta, x).train(max_num_epochs=1)
    fig, axes = utils.plot_summary(inference)
    assert isinstance(fig, Figure) and isinstance(axes[0], Axes)

    # SNLE
    inference = SNLE(prior=prior, summary_writer=summary_writer)
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=5)
    _ = inference.append_simulations(theta, x).train(max_num_epochs=1)
    fig, axes = utils.plot_summary(inference)
    assert isinstance(fig, Figure) and isinstance(axes[0], Axes)

    # SNRE
    inference = SNRE(prior=prior, summary_writer=summary_writer)
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=6)
    _ = inference.append_simulations(theta, x).train(
        num_atoms=2, max_num_epochs=5, validation_fraction=0.5
    )
    fig, axes = utils.plot_summary(inference)
    assert isinstance(fig, Figure) and isinstance(axes[0], Axes)
