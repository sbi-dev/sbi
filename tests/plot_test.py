# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import get_args

import numpy as np
import pytest
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import close, subplots
from torch.utils.tensorboard.writer import SummaryWriter

import sbi.analysis.plot as plt
from sbi.analysis import pairplot, plot_summary, sbc_rank_plot
from sbi.analysis.plotting_classes import (
    FigOptions,
    HistDiagOptions,
    HistOffDiagOptions,
)
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


def test_plotting_dataclass_overrides_defaults(mocker):
    """
    Verify that custom keyword arguments in `HistOffDiagOptions` correctly
    override the default values when passed to `pairplot`.
    """

    posterior_samples = torch.randn(100, 3)
    hist_off_diag_options = HistOffDiagOptions(
        np_hist_kwargs=dict(bins=40), mpl_kwargs=dict(origin="upper")
    )

    spy = mocker.spy(plt, "plt_hist_2d")
    _ = pairplot(
        samples=posterior_samples,
        upper="hist",
        upper_kwargs=hist_off_diag_options,
    )

    assert spy.call_count > 0

    args, _ = spy.call_args
    off_diag_kwargs = args[5]

    # Default values set in HistOffDiagOptions dataclass
    assert off_diag_kwargs["mpl_kwargs"]["cmap"] == 'viridis'
    assert off_diag_kwargs["np_hist_kwargs"]["density"] is False

    # Updated values
    assert off_diag_kwargs["mpl_kwargs"]["origin"] == 'upper'
    assert off_diag_kwargs["np_hist_kwargs"]["bins"] == 40

    close()


def test_plotting_fails_for_insufficient_sample_label_length():
    """
    Ensure that `pairplot` raises a `ValueError` when the number of provided
    sample labels is fewer than the number of sample dimensions.
    """

    posterior_samples = torch.randn(100, 3)
    fig_options = FigOptions(legend=True, samples_labels=[])

    with pytest.raises(ValueError, match="Provide at least as many labels as samples."):
        _ = pairplot(samples=posterior_samples, fig_kwargs=fig_options)

    close()


def test_pairplot_warns_on_offdiag_argument():
    """
    Verify that `pairplot` raises a warning when using the `offdiag` argument.
    """

    posterior_samples = torch.randn(100, 3)

    with pytest.warns():
        _ = pairplot(samples=posterior_samples, offdiag="contour")

    close()


def test_pairplot_raises_error_on_offdiag_and_upper_conflict():
    """
    Verify that `pairplot` raises a ValueError when using the `offdiag`
    and `upper` argument together.
    """

    posterior_samples = torch.randn(100, 3)

    with pytest.raises(ValueError):
        _ = pairplot(samples=posterior_samples, offdiag="contour", upper="scatter")

    close()


@pytest.mark.parametrize("square_subplots", (True, False))
def test_plotting_subplot_aspect(square_subplots):
    """
    Verify that the subplot aspect ratio is set correctly based on the
    `square_subplots` option in `FigOptions`.
    """

    posterior_samples = torch.randn(100, 3)
    fig_options = FigOptions(square_subplots=square_subplots)

    _, axes = pairplot(samples=posterior_samples, fig_kwargs=fig_options)

    axes = axes.flatten()

    for ax in axes:
        aspect = ax.get_box_aspect()
        if square_subplots:
            assert aspect == 1.0
        else:
            assert aspect is None

    close()


valid_diag_kwargs = [{}, HistDiagOptions(), None, [{}, HistDiagOptions(), None]]
valid_off_diag_kwargs = [
    {},
    HistOffDiagOptions(),
    None,
    [{}, HistOffDiagOptions(), None],
]

invalid_kwargs_inputs = [False]


@pytest.mark.parametrize(
    "kwargs",
    [
        # invalid cases
        *[
            pytest.param(
                {"diag_kwargs": value}, marks=pytest.mark.xfail(raises=TypeError)
            )
            for value in invalid_kwargs_inputs
        ],
        *[
            pytest.param(
                {"upper_kwargs": value}, marks=pytest.mark.xfail(raises=TypeError)
            )
            for value in invalid_kwargs_inputs
        ],
        *[
            pytest.param(
                {"lower_kwargs": value}, marks=pytest.mark.xfail(raises=TypeError)
            )
            for value in invalid_kwargs_inputs
        ],
        # valid cases
        *[({"diag_kwargs": v}) for v in valid_diag_kwargs],
        *[({"lower_kwargs": v}) for v in valid_off_diag_kwargs],
        *[({"upper_kwargs": v}) for v in valid_off_diag_kwargs],
    ],
)
def test_plotting_kwargs_validation(kwargs):
    """
    Validate that `pairplot` correctly enforces the types of `diag_kwargs`,
    `upper_kwargs`, and `lower_kwargs`.
    """
    posterior_samples = torch.randn(100, 3)

    _ = pairplot(samples=posterior_samples, **kwargs)

    close()


valid_diag_options = get_args(plt.DiagLiteral) + (None, ["hist", None])
valid_upper_options = get_args(plt.UpperLiteral) + (None, ["scatter", None])
valid_lower_options = get_args(plt.LowerLiteral) + (None, ["hist", None])

invalid_inputs = ["", [""]]


@pytest.mark.parametrize(
    "kwargs",
    [
        # invalid inputs
        *[
            pytest.param({"diag": value}, marks=pytest.mark.xfail(raises=ValueError))
            for value in invalid_inputs
        ],
        *[
            pytest.param({"upper": value}, marks=pytest.mark.xfail(raises=ValueError))
            for value in invalid_inputs
        ],
        *[
            pytest.param({"lower": value}, marks=pytest.mark.xfail(raises=ValueError))
            for value in invalid_inputs
        ],
        # valid inputs
        *[({"diag": value}) for value in valid_diag_options],
        *[({"upper": value}) for value in valid_upper_options],
        *[({"lower": value}) for value in valid_lower_options],
    ],
)
def test_plotting_style_arguments_validation(kwargs):
    """
    Check that the `pairplot` function correctly validates plotting
    style arguments.
    """

    posterior_samples = torch.randn(100, 3)

    _ = pairplot(samples=posterior_samples, **kwargs)

    close()
