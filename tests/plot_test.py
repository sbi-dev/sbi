# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import numpy as np
import pytest
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.analysis import pairplot, plot_summary, sbc_rank_plot
from sbi.inference import SNLE, SNPE, SNRE, simulate_for_sbi
from sbi.utils import BoxUniform


@pytest.mark.parametrize("samples", (torch.randn(100, 2), [torch.randn(100, 2)] * 2))
@pytest.mark.parametrize("labels", (None, ["a", "b"]))
@pytest.mark.parametrize("legend", (True, False))
@pytest.mark.parametrize("offdiag", ("hist", "scatter"))
@pytest.mark.parametrize("samples_labels", (["a", "b"], None))
@pytest.mark.parametrize("points_labels", (["a", "b"], None))
@pytest.mark.parametrize("points", (None, torch.ones(2)))
def test_pairplot(
    samples, labels, legend, offdiag, samples_labels, points_labels, points
):
    pairplot(**{k: v for k, v in locals().items() if v is not None})


@pytest.mark.parametrize("method", (SNPE, SNLE, SNRE))
def test_plot_summary(method, tmp_path):
    num_dim = 1
    prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))

    summary_writer = SummaryWriter(tmp_path)

    def simulator(theta):
        return theta + 1.0 + torch.randn_like(theta) * 0.1

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


@pytest.mark.parametrize("num_parameters", (2, 4, 10))
@pytest.mark.parametrize("num_cols", (2, 3, 4))
@pytest.mark.parametrize("custom_figure", (False, True))
@pytest.mark.parametrize("plot_type", ("hist", "cdf"))
def test_sbc_rank_plot(num_parameters, num_cols, custom_figure, plot_type):
    """Test sbc plots with different num_parameters, subplot shapes and plot types."""

    num_sbc_runs = 100
    num_posterior_samples = 100
    # Create uniform bins.
    ranks = np.random.randint(
        1, num_posterior_samples + 1, size=(num_sbc_runs, num_parameters)
    )

    # Test passing custom figure.
    if custom_figure:
        fig, ax = subplots(1, num_parameters, figsize=(5, 18))
    else:
        fig, ax = None, None

    fig, ax = sbc_rank_plot(
        ranks,
        num_posterior_samples,
        plot_type=plot_type,
        fig=fig,
        ax=ax,
        **dict(num_cols=num_cols, params_in_subplots=True),
    )
    if not custom_figure:
        if num_parameters > num_cols:
            assert ax.shape == (
                int(np.ceil(num_parameters / num_cols)),
                num_cols,
            )
        else:
            assert ax.shape == (num_parameters,)
