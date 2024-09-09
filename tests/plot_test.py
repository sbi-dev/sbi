# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import numpy as np
import pytest
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import close, subplots
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.analysis import pairplot, plot_summary, sbc_rank_plot
from sbi.inference import NLE, NPE, NRE
from sbi.utils import BoxUniform


@pytest.mark.parametrize("samples", (torch.randn(100, 1),))
@pytest.mark.parametrize("limits", ([(-1, 1)], None))
def test_pairplot1D(samples, limits):
    fig, axs = pairplot(**{k: v for k, v in locals().items() if v is not None})
    assert isinstance(fig, Figure)
    assert isinstance(axs, Axes)
    close()


@pytest.mark.parametrize("samples", (torch.randn(100, 2),))
@pytest.mark.parametrize("limits", ([(-1, 1)], None))
def test_nan_inf(samples, limits):
    samples[0, 0] = np.nan
    samples[5, 1] = np.inf
    samples[3, 0] = -np.inf
    fig, axs = pairplot(**{k: v for k, v in locals().items() if v is not None})
    assert isinstance(fig, Figure)
    assert isinstance(axs[0, 0], Axes)
    close()


@pytest.mark.parametrize("samples", (torch.randn(100, 2), [torch.randn(100, 3)] * 2))
@pytest.mark.parametrize("points", (torch.ones(1, 3),))
@pytest.mark.parametrize("limits", ([(-3, 3)], None))
@pytest.mark.parametrize("subset", (None, [0, 1]))
@pytest.mark.parametrize("upper", ("scatter",))
@pytest.mark.parametrize(
    "lower,lower_kwargs", [(None, None), ("scatter", {"mpl_kwargs": {"s": 10}})]
)
@pytest.mark.parametrize("diag", ("hist",))
@pytest.mark.parametrize("figsize", ((5, 5),))
@pytest.mark.parametrize("labels", (None, ["a", "b", "c"]))
@pytest.mark.parametrize("ticks", (None, [[-3, 0, 3], [-3, 0, 3], [0, 1, 2, 3]]))
@pytest.mark.parametrize("offdiag", (None,))
@pytest.mark.parametrize("diag_kwargs", (None, {"mpl_kwargs": {"bins": 10}}))
@pytest.mark.parametrize("upper_kwargs", (None,))
@pytest.mark.parametrize("fig_kwargs", (None, {"points_labels": ["a"], "legend": True}))
def test_pairplot(
    samples,
    points,
    limits,
    subset,
    upper,
    lower,
    diag,
    figsize,
    labels,
    ticks,
    offdiag,
    diag_kwargs,
    upper_kwargs,
    lower_kwargs,
    fig_kwargs,
):
    fig, axs = pairplot(**{k: v for k, v in locals().items() if v is not None})
    assert isinstance(fig, Figure)
    assert isinstance(axs, np.ndarray)
    assert isinstance(axs[0, 0], Axes)
    close()


@pytest.mark.parametrize("samples", (torch.randn(100, 2), [torch.randn(100, 2)] * 2))
@pytest.mark.parametrize("labels", (None, ["a", "b"]))
@pytest.mark.parametrize("legend", (True, False))
@pytest.mark.parametrize("offdiag", ("hist", "scatter"))
@pytest.mark.parametrize("samples_labels", (["a", "b"], None))
@pytest.mark.parametrize("points_labels", (["a", "b"], None))
@pytest.mark.parametrize("points", (None, torch.ones(2)))
def test_pairplot_dep(
    samples, labels, legend, offdiag, samples_labels, points_labels, points
):
    # uses old style keywords, checks backward compatibility
    fig, axs = pairplot(**{k: v for k, v in locals().items() if v is not None})

    assert isinstance(fig, Figure)
    assert isinstance(axs, np.ndarray)
    assert isinstance(axs[0, 0], Axes)
    close()


@pytest.mark.parametrize("method", (NPE, NLE, NRE))
def test_plot_summary(method, tmp_path):
    num_dim = 1
    prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))
    num_simulations = 6

    summary_writer = SummaryWriter(tmp_path)

    def simulator(theta):
        return theta + 1.0 + torch.randn_like(theta) * 0.1

    inference = method(prior=prior, summary_writer=summary_writer)

    theta = prior.sample((num_simulations,))
    x = simulator(theta)

    train_kwargs = (
        dict(max_num_epochs=5, validation_fraction=0.5, num_atoms=2)
        if method == NRE
        else dict(max_num_epochs=1)
    )
    _ = inference.append_simulations(theta, x).train(**train_kwargs)
    fig, axes = plot_summary(inference)
    assert isinstance(fig, Figure) and isinstance(axes[0], Axes)
    close()


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
    close()
