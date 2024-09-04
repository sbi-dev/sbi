# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import collections
import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
from warnings import warn

import matplotlib as mpl
import numpy as np
import six
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure, FigureBase
from matplotlib.patches import Rectangle
from scipy.stats import binom, gaussian_kde, iqr
from torch import Tensor

from sbi.analysis.conditional_density import eval_conditional_density
from sbi.utils.analysis_utils import pp_vals

try:
    collectionsAbc = collections.abc  # type: ignore
except AttributeError:
    collectionsAbc = collections


def hex2rgb(hex: str) -> List[int]:
    """Pass 16 to the integer function for change of base"""
    return [int(hex[i : i + 2], 16) for i in range(1, 6, 2)]


def rgb2hex(RGB: List[int]) -> str:
    """Components need to be integers for hex to make sense"""
    RGB = [int(x) for x in RGB]
    return "#" + "".join([
        "0{0:x}".format(v) if v < 16 else "{0:x}".format(v) for v in RGB
    ])


def to_list_string(
    x: Optional[Union[str, List[Optional[str]]]], len: int
) -> List[Optional[str]]:
    """If x is not a list, make it a list of strings of length `len`."""
    if not isinstance(x, list):
        return [x for _ in range(len)]
    return x


def to_list_kwargs(
    x: Optional[Union[Dict, List[Optional[Dict]]]], len: int
) -> List[Optional[Dict]]:
    """If x is not a list, make it a list of dicts of length `len`."""
    if not isinstance(x, list):
        return [x for _ in range(len)]
    return x


def _update(d: Dict, u: Optional[Dict]) -> Dict:
    """update dictionary with user input, see: https://stackoverflow.com/a/3233356"""
    if u is not None:
        for k, v in six.iteritems(u):
            dv = d.get(k, {})
            if not isinstance(dv, collectionsAbc.Mapping):  # type: ignore
                d[k] = v
            elif isinstance(v, collectionsAbc.Mapping):  # type: ignore
                d[k] = _update(dv, v)
            else:
                d[k] = v
    return d


# Plotting functions
def plt_hist_1d(
    ax: Axes,
    samples: np.ndarray,
    limits: torch.Tensor,
    diag_kwargs: Dict,
) -> None:
    """Plot 1D histogram."""
    hist_kwargs = copy.deepcopy(diag_kwargs["mpl_kwargs"])
    if "bins" not in hist_kwargs or hist_kwargs["bins"] is None:
        if diag_kwargs["bin_heuristic"] == "Freedman-Diaconis":
            # The Freedman-Diaconis heuristic
            binsize = 2 * iqr(samples) * len(samples) ** (-1 / 3)
            hist_kwargs["bins"] = np.arange(limits[0], limits[1] + binsize, binsize)
        else:
            # TODO: add more bin heuristics
            pass
    if isinstance(hist_kwargs["bins"], int):
        hist_kwargs["bins"] = np.linspace(limits[0], limits[1], hist_kwargs["bins"])
    ax.hist(samples, **hist_kwargs)


def plt_kde_1d(
    ax: Axes,
    samples: np.ndarray,
    limits: torch.Tensor,
    diag_kwargs: Dict,
) -> None:
    """Run 1D kernel density estimation on samples and plot it on a given axes."""
    density = gaussian_kde(samples, bw_method=diag_kwargs["bw_method"])
    xs = np.linspace(limits[0], limits[1], diag_kwargs["bins"])
    ys = density(xs)
    ax.plot(xs, ys, **diag_kwargs["mpl_kwargs"])


def plt_scatter_1d(
    ax: Axes,
    samples: np.ndarray,
    limits: torch.Tensor,
    diag_kwargs: Dict,
) -> None:
    """Plot vertical lines for each sample. Note: limits are not used."""
    for single_sample in samples:
        ax.axvline(single_sample, **diag_kwargs["mpl_kwargs"])


def plt_hist_2d(
    ax: Axes,
    samples_col: np.ndarray,
    samples_row: np.ndarray,
    limits_col: torch.Tensor,
    limits_row: torch.Tensor,
    offdiag_kwargs: Dict,
):
    hist_kwargs = copy.deepcopy(offdiag_kwargs)
    """Plot 2D histogram."""
    if (
        "bins" not in hist_kwargs["np_hist_kwargs"]
        or hist_kwargs["np_hist_kwargs"]["bins"] is None
    ):
        if hist_kwargs["bin_heuristic"] == "Freedman-Diaconis":
            # The Freedman-Diaconis heuristic applied to each direction
            binsize_col = 2 * iqr(samples_col) * len(samples_col) ** (-1 / 3)
            n_bins_col = int((limits_col[1] - limits_col[0]) / binsize_col)
            binsize_row = 2 * iqr(samples_row) * len(samples_row) ** (-1 / 3)
            n_bins_row = int((limits_row[1] - limits_row[0]) / binsize_row)
            hist_kwargs["np_hist_kwargs"]["bins"] = [n_bins_col, n_bins_row]
        else:
            # TODO: add more bin heuristics
            pass
    hist, xedges, yedges = np.histogram2d(
        samples_col,
        samples_row,
        range=[
            [limits_col[0], limits_col[1]],
            [limits_row[0], limits_row[1]],
        ],
        **hist_kwargs["np_hist_kwargs"],
    )
    ax.imshow(
        hist.T,
        extent=(
            xedges[0],
            xedges[-1],
            yedges[0],
            yedges[-1],
        ),
        **hist_kwargs["mpl_kwargs"],
    )


def plt_kde_2d(
    ax: Axes,
    samples_col: np.ndarray,
    samples_row: np.ndarray,
    limits_col: torch.Tensor,
    limits_row: torch.Tensor,
    offdiag_kwargs: Dict,
) -> None:
    """Run 2D Kernel Density Estimation and plot it on given axis."""
    X, Y, Z = get_kde(samples_col, samples_row, limits_col, limits_row, offdiag_kwargs)

    ax.imshow(
        Z,
        extent=(
            limits_col[0].item(),
            limits_col[1].item(),
            limits_row[0].item(),
            limits_row[1].item(),
        ),
        **offdiag_kwargs["mpl_kwargs"],
    )


def plt_contour_2d(
    ax: Axes,
    samples_col: np.ndarray,
    samples_row: np.ndarray,
    limits_col: torch.Tensor,
    limits_row: torch.Tensor,
    offdiag_kwargs: Dict,
) -> None:
    """2D Contour based on Kernel Density Estimation."""

    X, Y, Z = get_kde(samples_col, samples_row, limits_col, limits_row, offdiag_kwargs)

    ax.contour(
        X,
        Y,
        Z,
        extent=(
            limits_col[0],
            limits_col[1],
            limits_row[0],
            limits_row[1],
        ),
        levels=offdiag_kwargs["levels"],
        **offdiag_kwargs["mpl_kwargs"],
    )


def plt_scatter_2d(
    ax: Axes,
    samples_col: np.ndarray,
    samples_row: np.ndarray,
    limits_col: torch.Tensor,
    limits_row: torch.Tensor,
    offdiag_kwargs: Dict,
) -> None:
    """Scatter plot 2D. Note: limits are not used"""
    ax.scatter(
        samples_col,
        samples_row,
        **offdiag_kwargs["mpl_kwargs"],
    )


def plt_plot_2d(
    ax: Axes,
    samples_col: np.ndarray,
    samples_row: np.ndarray,
    limits_col: torch.Tensor,
    limits_row: torch.Tensor,
    offdiag_kwargs: Dict,
) -> None:
    """Plot 2D trajectory. Note: limits are not used."""

    ax.plot(
        samples_col,
        samples_row,
        **offdiag_kwargs["mpl_kwargs"],
    )


def get_kde(
    samples_col: np.ndarray,
    samples_row: np.ndarray,
    limits_col: torch.Tensor,
    limits_row: torch.Tensor,
    offdiag_kwargs: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D Kernel Density Estimation."""

    density = gaussian_kde(
        np.array([samples_col, samples_row]),
        bw_method=offdiag_kwargs["bw_method"],
    )
    X, Y = np.meshgrid(
        np.linspace(
            limits_col[0],
            limits_col[1],
            offdiag_kwargs["bins"],
        ),
        np.linspace(
            limits_row[0],
            limits_row[1],
            offdiag_kwargs["bins"],
        ),
    )
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(density(positions).T, X.shape)
    if "percentile" in offdiag_kwargs and "levels" in offdiag_kwargs:
        Z = probs2contours(Z, offdiag_kwargs["levels"])
    else:
        Z = (Z - Z.min()) / (Z.max() - Z.min())
    return X, Y, Z


def get_diag_funcs(
    diag_list: List[Optional[str]],
) -> List[
    Union[
        Callable[
            [
                Axes,
                np.ndarray,
                torch.Tensor,
                Dict,
            ],
            None,
        ],
        None,
    ]
]:
    """make a list of the functions for the diagonal plots."""
    diag_funcs = []
    for diag in diag_list:
        if diag == "hist":
            diag_funcs.append(plt_hist_1d)
        elif diag == "kde":
            diag_funcs.append(plt_kde_1d)
        elif diag == "scatter":
            diag_funcs.append(plt_scatter_1d)
        else:
            diag_funcs.append(None)

    return diag_funcs


def get_offdiag_funcs(
    off_diag_list: List[Optional[str]],
) -> List[
    Union[
        Callable[
            [
                Axes,
                np.ndarray,
                torch.Tensor,
                Dict,
            ],
            None,
        ],
        None,
    ]
]:
    """make a list of the functions for the off-diagonal plots."""
    offdiag_funcs = []
    for offdiag in off_diag_list:
        if offdiag == "hist" or offdiag == "hist2d":
            offdiag_funcs.append(plt_hist_2d)
        elif offdiag == "kde" or offdiag == "kde2d":
            offdiag_funcs.append(plt_kde_2d)
        elif offdiag == "contour" or offdiag == "contourf":
            offdiag_funcs.append(plt_contour_2d)
        elif offdiag == "scatter":
            offdiag_funcs.append(plt_scatter_2d)
        elif offdiag == "plot":
            offdiag_funcs.append(plt_plot_2d)
        else:
            offdiag_funcs.append(None)
    return offdiag_funcs


def _format_subplot(
    ax: Axes,
    current: str,
    limits: Union[List[List[float]], torch.Tensor],
    ticks: Optional[Union[List, torch.Tensor]],
    labels_dim: List[str],
    fig_kwargs: Dict,
    row: int,
    col: int,
    dim: int,
    flat: bool,
    excl_lower: bool,
) -> None:
    """
    Format subplot according to fig_kwargs and other arguments
    Args:
        ax: matplotlib axis
        current: str, 'diag','upper' or 'lower'
        limits: list of lists, limits for each dimension
        ticks: list of lists, ticks for each dimension
        labels_dim: list of strings, labels for each dimension
        fig_kwargs: dict, figure kwargs
        row: int, row index
        col: int, column index
        dim: int, number of dimensions
        flat: bool, whether the plot is flat (1 row)
        excl_lower: bool, whether lower triangle is empty

    """

    # Background color
    if (
        current in fig_kwargs["fig_bg_colors"]
        and fig_kwargs["fig_bg_colors"][current] is not None
    ):
        ax.set_facecolor(fig_kwargs["fig_bg_colors"][current])
    # Limits
    if isinstance(limits, Tensor):
        assert limits.dim() == 2, "Limits should be a 2D tensor."
        limits = limits.tolist()
    if current == "diag":
        eps = fig_kwargs["x_lim_add_eps"]
        ax.set_xlim((limits[col][0] - eps, limits[col][1] + eps))
    else:
        ax.set_xlim((limits[col][0], limits[col][1]))

    if current != "diag":
        ax.set_ylim((limits[row][0], limits[row][1]))

    # Ticks
    if ticks is not None:
        ax.set_xticks((ticks[col][0], ticks[col][1]))  # pyright: ignore[reportCallIssue]
        if current != "diag":
            ax.set_yticks((ticks[row][0], ticks[row][1]))  # pyright: ignore[reportCallIssue]

    # make square
    if fig_kwargs["square_subplots"]:
        ax.set_box_aspect(1)
    # Despine
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", fig_kwargs["despine"]["offset"]))

    # Formatting axes
    if current == "diag":  # diagonals
        if excl_lower or col == dim - 1 or flat:
            _format_axis(
                ax,
                xhide=False,
                xlabel=labels_dim[col],
                yhide=True,
                tickformatter=fig_kwargs["tickformatter"],
            )
        else:
            _format_axis(ax, xhide=True, yhide=True)
    else:  # off-diagonals
        if row == dim - 1:
            _format_axis(
                ax,
                xhide=False,
                xlabel=labels_dim[col],
                yhide=True,
                tickformatter=fig_kwargs["tickformatter"],
            )
        else:
            _format_axis(ax, xhide=True, yhide=True)
    if fig_kwargs["tick_labels"] is not None:
        ax.set_xticklabels((  # pyright: ignore[reportCallIssue]
            str(fig_kwargs["tick_labels"][col][0]),
            str(fig_kwargs["tick_labels"][col][1]),
        ))


def _format_axis(
    ax: Axes,
    xhide: bool = True,
    yhide: bool = True,
    xlabel: str = "",
    ylabel: str = "",
    tickformatter=None,
) -> Axes:
    """Format axis spines and ticks."""
    for loc in ["right", "top", "left", "bottom"]:
        ax.spines[loc].set_visible(False)
    if xhide:
        ax.set_xlabel("")
        ax.xaxis.set_ticks_position("none")
        ax.xaxis.set_tick_params(labelbottom=False)
    if yhide:
        ax.set_ylabel("")
        ax.yaxis.set_ticks_position("none")
        ax.yaxis.set_tick_params(labelleft=False)
    if not xhide:
        ax.set_xlabel(xlabel)
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_tick_params(labelbottom=True)
        if tickformatter is not None:
            ax.xaxis.set_major_formatter(tickformatter)
        ax.spines["bottom"].set_visible(True)  # pyright: ignore[reportCallIssue]
    if not yhide:
        ax.set_ylabel(ylabel)
        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_tick_params(labelleft=True)
        if tickformatter is not None:
            ax.yaxis.set_major_formatter(tickformatter)
        ax.spines["left"].set_visible(True)
    return ax


def probs2contours(
    probs: np.ndarray,
    levels: Union[List, torch.Tensor, np.ndarray],
) -> np.ndarray:
    """Takes an array of probabilities and produces an array of contours at specified
    percentile levels.
    Args:
        probs: Probability array. doesn't have to sum to 1, but it is assumed it
            contains all the mass
        levels: Percentile levels, have to be in [0.0, 1.0]. Specifies contour levels
            that include a given proportion of samples, i.e., 0.1 specifies where the
            top 10% of the density is.
    Returns:
        contours: Array of same shape as probs with percentile labels. Values in output
        array denote labels which percentile bin the probability mass belongs to.

    Example: for levels = [0.1, 0.5], output array will take on values [1.0, 0.5, 0.1],
    where elements labeled "0.1" correspond to the top 10% of the density, "0.5"
    corresponds to between top 50% to 10%, etc.
    """
    # make sure all contour levels are in [0.0, 1.0]
    levels = np.asarray(levels)
    assert np.all(levels <= 1.0) and np.all(levels >= 0.0)

    # flatten probability array
    shape = probs.shape
    probs = probs.flatten()

    # sort probabilities in descending order
    idx_sort = probs.argsort()[::-1]
    idx_unsort = idx_sort.argsort()
    probs = probs[idx_sort]

    # cumulative probabilities
    cum_probs = probs.cumsum()
    cum_probs /= cum_probs[-1]

    # create contours at levels
    contours = np.ones_like(cum_probs)
    levels = np.sort(levels)[::-1]
    for level in levels:
        contours[cum_probs <= level] = level

    # make sure contours have the order and the shape of the original
    # probability array
    contours = np.reshape(contours[idx_unsort], shape)

    return contours


def ensure_numpy(t: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Returns np.ndarray if torch.Tensor was provided.

    Used because samples_nd() can only handle np.ndarray.
    """
    if isinstance(t, torch.Tensor):
        return t.numpy()
    elif not isinstance(t, np.ndarray):
        return np.array(t)
    return t


def handle_nan_infs(samples: List[np.ndarray]) -> List[np.ndarray]:
    """Check if there are NaNs or Infs in the samples."""
    for i in range(len(samples)):
        if np.isnan(samples[i]).any():
            logging.warning("NaNs found in samples, omitting datapoints.")
        if np.isinf(samples[i]).any():
            logging.warning("Infs found in samples, omitting datapoints.")
            # cast inf to nan, so they are omitted in the next step
            np.nan_to_num(
                samples[i], copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan
            )
        samples[i] = samples[i][~np.isnan(samples[i]).any(axis=1)]
    return samples


def convert_to_list_of_numpy(
    arr: Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor],
) -> List[np.ndarray]:
    """Converts a list of torch.Tensor to a list of np.ndarray."""
    if not isinstance(arr, list):
        arr = ensure_numpy(arr)
        return [arr]
    return [ensure_numpy(a) for a in arr]


def infer_limits(
    samples: List[np.ndarray],
    dim: int,
    points: Optional[List[np.ndarray]] = None,
    eps: float = 0.1,
) -> List[List[float]]:
    """Infer limits for the plot.

    Args:
        samples: List of set of samples.
        dim: Dimension of the samples.
        points: List of points.
        eps: Relative margin for the limits.
    """
    limits = []
    for d in range(dim):
        # get min and max across all sets of samples
        min_val = min(np.min(sample[:, d]) for sample in samples)
        max_val = max(np.max(sample[:, d]) for sample in samples)
        # include points in the limits
        if points is not None:
            min_val = min(min_val, min(np.min(point[:, d]) for point in points))
            max_val = max(max_val, max(np.max(point[:, d]) for point in points))
        # add margin
        max_min_range = max_val - min_val
        epsilon_range = eps * max_min_range
        limits.append([min_val - epsilon_range, max_val + epsilon_range])
    return limits


def prepare_for_plot(
    samples: Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor],
    limits: Optional[Union[List, torch.Tensor, np.ndarray]] = None,
    points: Optional[
        Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ] = None,
) -> Tuple[List[np.ndarray], int, torch.Tensor]:
    """
    Ensures correct formatting for samples and limits, and returns dimension
    of the samples.
    """

    samples = convert_to_list_of_numpy(samples)
    if points is not None:
        points = convert_to_list_of_numpy(points)

    samples = handle_nan_infs(samples)

    dim = samples[0].shape[1]

    if limits is None or limits == []:
        limits = infer_limits(samples, dim, points)
    else:
        limits = [limits[0] for _ in range(dim)] if len(limits) == 1 else limits

    limits = torch.as_tensor(limits)
    return samples, dim, limits


def prepare_for_conditional_plot(condition, opts):
    """
    Ensures correct formatting for limits. Returns the margins just inside
    the domain boundaries, and the dimension of the samples.
    """
    # Dimensions
    dim = condition.shape[-1]

    # Prepare limits
    if len(opts["limits"]) == 1:
        limits = [opts["limits"][0] for _ in range(dim)]
    else:
        limits = opts["limits"]
    limits = torch.as_tensor(limits)

    # Infer the margin. This is to avoid that we evaluate the posterior **exactly**
    # at the boundary.
    limits_diffs = limits[:, 1] - limits[:, 0]
    eps_margins = limits_diffs / 1e5

    return dim, limits, eps_margins


def get_conditional_diag_func(opts, limits, eps_margins, resolution):
    """
    Returns the diag_func which returns the 1D marginal conditional plot for
    the parameter indexed by row.
    """

    def diag_func(row, **kwargs):
        p_vector = (
            eval_conditional_density(
                opts["density"],
                opts["condition"],
                limits,
                row,
                row,
                resolution=resolution,
                eps_margins1=eps_margins[row],
                eps_margins2=eps_margins[row],
            )
            .to("cpu")
            .numpy()
        )
        plt.plot(
            np.linspace(
                limits[row, 0],
                limits[row, 1],
                resolution,
            ),
            p_vector,
            c=opts["samples_colors"][0],
        )

    return diag_func


def pairplot(
    samples: Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor],
    points: Optional[
        Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ] = None,
    limits: Optional[Union[List, torch.Tensor]] = None,
    subset: Optional[List[int]] = None,
    upper: Optional[Union[List[Optional[str]], str]] = "hist",
    lower: Optional[Union[List[Optional[str]], str]] = None,
    diag: Optional[Union[List[Optional[str]], str]] = "hist",
    figsize: Tuple = (10, 10),
    labels: Optional[List[str]] = None,
    ticks: Optional[Union[List, torch.Tensor]] = None,
    offdiag: Optional[Union[List[Optional[str]], str]] = None,
    diag_kwargs: Optional[Union[List[Optional[Dict]], Dict]] = None,
    upper_kwargs: Optional[Union[List[Optional[Dict]], Dict]] = None,
    lower_kwargs: Optional[Union[List[Optional[Dict]], Dict]] = None,
    fig_kwargs: Optional[Dict] = None,
    fig: Optional[FigureBase] = None,
    axes: Optional[Axes] = None,
    **kwargs: Optional[Any],
) -> Tuple[FigureBase, Axes]:
    """
    Plot samples in a 2D grid showing marginals and pairwise marginals.

    Each of the diagonal plots can be interpreted as a 1D-marginal of the distribution
    that the samples were drawn from. Each upper-diagonal plot can be interpreted as a
    2D-marginal of the distribution.

    Args:
        samples: Samples used to build the histogram.
        points: List of additional points to scatter.
        limits: Array containing the plot xlim for each parameter dimension. If None,
            just use the min and max of the passed samples
        subset: List containing the dimensions to plot. E.g. subset=[1,3] will plot
            plot only the 1st and 3rd dimension but will discard the 0th and 2nd (and,
            if they exist, the 4th, 5th and so on).
        upper: Plotting style for upper diagonal, {hist, scatter, contour, kde,
            None}.
        lower: Plotting style for upper diagonal, {hist, scatter, contour, kde,
            None}.
        diag: Plotting style for diagonal, {hist, scatter, kde}.
        figsize: Size of the entire figure.
        labels: List of strings specifying the names of the parameters.
        ticks: Position of the ticks.
        offdiag: deprecated, use upper instead.
        diag_kwargs: Additional arguments to adjust the diagonal plot,
            see the source code in `_get_default_diag_kwarg()`
        upper_kwargs: Additional arguments to adjust the upper diagonal plot,
            see the source code in `_get_default_offdiag_kwarg()`
        lower_kwargs: Additional arguments to adjust the lower diagonal plot,
            see the source code in `_get_default_offdiag_kwarg()`
        fig_kwargs: Additional arguments to adjust the overall figure,
            see the source code in `_get_default_fig_kwargs()`
        fig: matplotlib figure to plot on.
        axes: matplotlib axes corresponding to fig.
        **kwargs: Additional arguments to adjust the plot (deprecated).

    Returns: figure and axis of posterior distribution plot
    """

    # Backwards compatibility
    if len(kwargs) > 0:
        warn(
            f"you passed deprecated arguments **kwargs: {[key for key in kwargs]}, use "
            "fig_kwargs instead. We continue calling the deprecated pairplot function",
            DeprecationWarning,
            stacklevel=2,
        )
        fig, axes = pairplot_dep(
            samples,
            points,
            limits,
            subset,
            offdiag,
            diag,
            figsize,
            labels,
            ticks,
            upper,
            fig,
            axes,
            **kwargs,
        )
        return fig, axes

    samples, dim, limits = prepare_for_plot(samples, limits, points)

    # prepate figure kwargs
    fig_kwargs_filled = _get_default_fig_kwargs()
    # update the defaults dictionary with user provided values
    fig_kwargs_filled = _update(fig_kwargs_filled, fig_kwargs)

    # checks.
    if fig_kwargs_filled["legend"]:
        assert len(fig_kwargs_filled["samples_labels"]) >= len(
            samples
        ), "Provide at least as many labels as samples."
    if offdiag is not None:
        warn("offdiag is deprecated, use upper or lower instead.", stacklevel=2)
        upper = offdiag

    # Prepare diag
    diag_list = to_list_string(diag, len(samples))
    diag_kwargs_list = to_list_kwargs(diag_kwargs, len(samples))
    diag_func = get_diag_funcs(diag_list)
    diag_kwargs_filled = []
    for i, (diag_i, diag_kwargs_i) in enumerate(zip(diag_list, diag_kwargs_list)):
        diag_kwarg_filled_i = _get_default_diag_kwargs(diag_i, i)
        # update the defaults dictionary with user provided values
        diag_kwarg_filled_i = _update(diag_kwarg_filled_i, diag_kwargs_i)
        diag_kwargs_filled.append(diag_kwarg_filled_i)

    # Prepare upper
    upper_list = to_list_string(upper, len(samples))
    upper_kwargs_list = to_list_kwargs(upper_kwargs, len(samples))
    upper_func = get_offdiag_funcs(upper_list)
    upper_kwargs_filled = []
    for i, (upper_i, upper_kwargs_i) in enumerate(zip(upper_list, upper_kwargs_list)):
        upper_kwarg_filled_i = _get_default_offdiag_kwargs(upper_i, i)
        # update the defaults dictionary with user provided values
        upper_kwarg_filled_i = _update(upper_kwarg_filled_i, upper_kwargs_i)
        upper_kwargs_filled.append(upper_kwarg_filled_i)

    # Prepare lower
    lower_list = to_list_string(lower, len(samples))
    lower_kwargs_list = to_list_kwargs(lower_kwargs, len(samples))
    lower_func = get_offdiag_funcs(lower_list)
    lower_kwargs_filled = []
    for i, (lower_i, lower_kwargs_i) in enumerate(zip(lower_list, lower_kwargs_list)):
        lower_kwarg_filled_i = _get_default_offdiag_kwargs(lower_i, i)
        # update the defaults dictionary with user provided values
        lower_kwarg_filled_i = _update(lower_kwarg_filled_i, lower_kwargs_i)
        lower_kwargs_filled.append(lower_kwarg_filled_i)

    return _arrange_grid(
        diag_func,
        upper_func,
        lower_func,
        diag_kwargs_filled,
        upper_kwargs_filled,
        lower_kwargs_filled,
        samples,
        points,
        limits,
        subset,
        figsize,
        labels,
        ticks,
        fig,
        axes,
        fig_kwargs_filled,
    )


def marginal_plot(
    samples: Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor],
    points: Optional[
        Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ] = None,
    limits: Optional[Union[List, torch.Tensor]] = None,
    subset: Optional[List[int]] = None,
    diag: Optional[Union[List[Optional[str]], str]] = "hist",
    figsize: Optional[Tuple] = (10, 2),
    labels: Optional[List[str]] = None,
    ticks: Optional[Union[List, torch.Tensor]] = None,
    diag_kwargs: Optional[Union[List[Optional[Dict]], Dict]] = None,
    fig_kwargs: Optional[Dict] = None,
    fig: Optional[FigureBase] = None,
    axes: Optional[Axes] = None,
    **kwargs: Optional[Any],
) -> Tuple[FigureBase, Axes]:
    """
    Plot samples in a row showing 1D marginals of selected dimensions.

    Each of the plots can be interpreted as a 1D-marginal of the distribution
    that the samples were drawn from.

    Args:
        samples: Samples used to build the histogram.
        points: List of additional points to scatter.
        limits: Array containing the plot xlim for each parameter dimension. If None,
            just use the min and max of the passed samples
        subset: List containing the dimensions to plot. E.g. subset=[1,3] will plot
            plot only the 1st and 3rd dimension but will discard the 0th and 2nd (and,
            if they exist, the 4th, 5th and so on).
        diag: Plotting style for 1D marginals, {hist, kde cond, None}.
        figsize: Size of the entire figure.
        labels: List of strings specifying the names of the parameters.
        ticks: Position of the ticks.
        diag_kwargs: Additional arguments to adjust the diagonal plot,
            see the source code in `_get_default_diag_kwarg()`
        fig_kwargs: Additional arguments to adjust the overall figure,
            see the source code in `_get_default_fig_kwargs()`
        fig: matplotlib figure to plot on.
        axes: matplotlib axes corresponding to fig.
        **kwargs: Additional arguments to adjust the plot (deprecated)
    Returns: figure and axis of posterior distribution plot
    """

    # backwards compatibility
    if len(kwargs) > 0:
        warn(
            "**kwargs are deprecated, use fig_kwargs instead. "
            "calling the to be deprecated marginal_plot function",
            DeprecationWarning,
            stacklevel=2,
        )
        fig, axes = marginal_plot_dep(
            samples,
            points,
            limits,
            subset,
            diag,
            figsize,
            labels,
            ticks,
            fig,
            axes,
            **kwargs,
        )
        return fig, axes

    samples, dim, limits = prepare_for_plot(samples, limits)

    # prepare kwargs and functions of the subplots
    diag_list = to_list_string(diag, len(samples))
    diag_kwargs_list = to_list_kwargs(diag_kwargs, len(samples))
    diag_func = get_diag_funcs(diag_list)
    diag_kwargs_filled = []
    for i, (diag_i, diag_kwargs_i) in enumerate(zip(diag_list, diag_kwargs_list)):
        diag_kwarg_filled_i = _get_default_diag_kwargs(diag_i, i)
        diag_kwarg_filled_i = _update(diag_kwarg_filled_i, diag_kwargs_i)
        diag_kwargs_filled.append(diag_kwarg_filled_i)

    # prepare fig_kwargs
    fig_kwargs_filled = _get_default_fig_kwargs()
    fig_kwargs_filled = _update(fig_kwargs_filled, fig_kwargs)

    # generate plot
    return _arrange_grid(
        diag_func,
        [None],
        [None],
        diag_kwargs_filled,
        [None],
        [None],
        samples,
        points,
        limits,
        subset,
        figsize,
        labels,
        ticks,
        fig,
        axes,
        fig_kwargs_filled,
    )


def _get_default_offdiag_kwargs(offdiag: Optional[str], i: int = 0) -> Dict:
    """Get default offdiag kwargs."""

    if offdiag == "kde" or offdiag == "kde2d":
        offdiag_kwargs = {
            "bw_method": "scott",
            "bins": 50,
            "mpl_kwargs": {"cmap": "viridis", "origin": "lower", "aspect": "auto"},
        }

    elif offdiag == "hist" or offdiag == "hist2d":
        offdiag_kwargs = {
            "bin_heuristic": None,  # "Freedman-Diaconis",
            "mpl_kwargs": {"cmap": "viridis", "origin": "lower", "aspect": "auto"},
            "np_hist_kwargs": {"bins": 50, "density": False},
        }

    elif offdiag == "scatter":
        offdiag_kwargs = {
            "mpl_kwargs": {
                "color": plt.rcParams["axes.prop_cycle"].by_key()["color"][i * 2],  # pyright: ignore[reportOptionalMemberAccess]
                "edgecolor": "white",
                "alpha": 0.5,
                "rasterized": False,
            }
        }
    elif offdiag == "contour" or offdiag == "contourf":
        offdiag_kwargs = {
            "bw_method": "scott",
            "bins": 50,
            "levels": [0.68, 0.95, 0.99],
            "percentile": True,
            "mpl_kwargs": {
                "colors": plt.rcParams["axes.prop_cycle"].by_key()["color"][i * 2],  # pyright: ignore[reportOptionalMemberAccess]
            },
        }
    elif offdiag == "plot":
        offdiag_kwargs = {
            "mpl_kwargs": {
                "color": plt.rcParams["axes.prop_cycle"].by_key()["color"][i * 2],  # pyright: ignore[reportOptionalMemberAccess]
                "aspect": "auto",
            }
        }
    else:
        offdiag_kwargs = {}
    return offdiag_kwargs


def _get_default_diag_kwargs(diag: Optional[str], i: int = 0) -> Dict:
    """Get default diag kwargs."""
    if diag == "kde":
        diag_kwargs = {
            "bw_method": "scott",
            "bins": 50,
            "mpl_kwargs": {
                "color": plt.rcParams["axes.prop_cycle"].by_key()["color"][i * 2]  # pyright: ignore[reportOptionalMemberAccess]
            },
        }

    elif diag == "hist":
        diag_kwargs = {
            "bin_heuristic": "Freedman-Diaconis",
            "mpl_kwargs": {
                "color": plt.rcParams["axes.prop_cycle"].by_key()["color"][i * 2],  # pyright: ignore[reportOptionalMemberAccess]
                "density": False,
                "histtype": "step",
            },
        }
    elif diag == "scatter":
        diag_kwargs = {
            "mpl_kwargs": {
                "color": plt.rcParams["axes.prop_cycle"].by_key()["color"][i * 2]  # pyright: ignore[reportOptionalMemberAccess]
            }
        }
    else:
        diag_kwargs = {}
    return diag_kwargs


def _get_default_fig_kwargs() -> Dict:
    """Get default figure kwargs."""
    return {
        "legend": None,
        "legend_kwargs": {},
        # labels
        "points_labels": [f"points_{idx}" for idx in range(10)],  # for points
        "samples_labels": [f"samples_{idx}" for idx in range(10)],  # for samples
        # colors: take even colors for samples, odd colors for points
        "samples_colors": plt.rcParams["axes.prop_cycle"].by_key()["color"][0::2],  # pyright: ignore[reportOptionalMemberAccess]
        "points_colors": plt.rcParams["axes.prop_cycle"].by_key()["color"][1::2],  # pyright: ignore[reportOptionalMemberAccess]
        # ticks
        "tickformatter": mpl.ticker.FormatStrFormatter("%g"),  # type: ignore
        "tick_labels": None,
        # formatting points (scale, markers)
        "points_diag": {},
        "points_offdiag": {
            "marker": ".",
            "markersize": 10,
        },
        # other options
        "fig_bg_colors": {"offdiag": None, "diag": None, "lower": None},
        "fig_subplots_adjust": {
            "top": 0.9,
        },
        "subplots": {},
        "despine": {
            "offset": 5,
        },
        "title": None,
        "title_format": {"fontsize": 16},
        "x_lim_add_eps": 1e-5,
        "square_subplots": True,
    }


def conditional_marginal_plot(
    density: Any,
    condition: torch.Tensor,
    limits: Union[List, torch.Tensor],
    points: Optional[
        Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ] = None,
    subset: Optional[List[int]] = None,
    resolution: int = 50,
    figsize: Tuple = (10, 10),
    labels: Optional[List[str]] = None,
    ticks: Optional[Union[List, torch.Tensor]] = None,
    fig=None,
    axes=None,
    **kwargs,
):
    r"""
    Plot conditional distribution given all other parameters.

    The conditionals can be interpreted as slices through the `density` at a location
    given by `condition`.

    For example:
    Say we have a 3D density with parameters $\theta_0$, $\theta_1$, $\theta_2$ and
    a condition $c$ passed by the user in the `condition` argument.
    For the plot of $\theta_0$ on the diagonal, this will plot the conditional
    $p(\theta_0 | \theta_1=c[1], \theta_2=c[2])$. All other diagonals and are built in
    the corresponding way.

    Args:
        density: Probability density with a `log_prob()` method.
        condition: Condition that all but the one/two regarded parameters are fixed to.
            The condition should be of shape (1, dim_theta), i.e. it could e.g. be
            a sample from the posterior distribution.
        limits: Limits in between which each parameter will be evaluated.
        points: Additional points to scatter.
        subset: List containing the dimensions to plot. E.g. subset=[1,3] will plot
            plot only the 1st and 3rd dimension but will discard the 0th and 2nd (and,
            if they exist, the 4th, 5th and so on)
        resolution: Resolution of the grid at which we evaluate the `pdf`.
        figsize: Size of the entire figure.
        labels: List of strings specifying the names of the parameters.
        ticks: Position of the ticks.
        points_colors: Colors of the `points`.

        fig: matplotlib figure to plot on.
        axes: matplotlib axes corresponding to fig.
        **kwargs: Additional arguments to adjust the plot, e.g., `samples_colors`,
            `points_colors` and many more, see the source code in `_get_default_opts()`
            in `sbi.analysis.plot` for details.

    Returns: figure and axis of posterior distribution plot
    """

    # Setting these is required because _marginal will check if opts['diag'] is
    # `None`. This would break if opts has no key 'diag'.
    diag = "cond"

    opts = _get_default_opts()
    # update the defaults dictionary by the current values of the variables (passed by
    # the user)
    opts = _update(opts, locals())
    opts = _update(opts, kwargs)

    dim, limits, eps_margins = prepare_for_conditional_plot(condition, opts)

    diag_func = get_conditional_diag_func(opts, limits, eps_margins, resolution)

    return _arrange_plots(
        diag_func, None, dim, limits, points, opts, fig=fig, axes=axes
    )


def conditional_pairplot(
    density: Any,
    condition: torch.Tensor,
    limits: Union[List, torch.Tensor],
    points: Optional[
        Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ] = None,
    subset: Optional[List[int]] = None,
    resolution: int = 50,
    figsize: Tuple = (10, 10),
    labels: Optional[List[str]] = None,
    ticks: Optional[Union[List, torch.Tensor]] = None,
    fig=None,
    axes=None,
    **kwargs,
):
    r"""
    Plot conditional distribution given all other parameters.

    The conditionals can be interpreted as slices through the `density` at a location
    given by `condition`.

    For example:
    Say we have a 3D density with parameters $\theta_0$, $\theta_1$, $\theta_2$ and
    a condition $c$ passed by the user in the `condition` argument.
    For the plot of $\theta_0$ on the diagonal, this will plot the conditional
    $p(\theta_0 | \theta_1=c[1], \theta_2=c[2])$. For the upper
    diagonal of $\theta_1$ and $\theta_2$, it will plot
    $p(\theta_1, \theta_2 | \theta_0=c[0])$. All other diagonals and upper-diagonals
    are built in the corresponding way.

    Args:
        density: Probability density with a `log_prob()` method.
        condition: Condition that all but the one/two regarded parameters are fixed to.
            The condition should be of shape (1, dim_theta), i.e. it could e.g. be
            a sample from the posterior distribution.
        limits: Limits in between which each parameter will be evaluated.
        points: Additional points to scatter.
        subset: List containing the dimensions to plot. E.g. subset=[1,3] will plot
            plot only the 1st and 3rd dimension but will discard the 0th and 2nd (and,
            if they exist, the 4th, 5th and so on)
        resolution: Resolution of the grid at which we evaluate the `pdf`.
        figsize: Size of the entire figure.
        labels: List of strings specifying the names of the parameters.
        ticks: Position of the ticks.
        points_colors: Colors of the `points`.

        fig: matplotlib figure to plot on.
        axes: matplotlib axes corresponding to fig.
        **kwargs: Additional arguments to adjust the plot, e.g., `samples_colors`,
            `points_colors` and many more, see the source code in `_get_default_opts()`
            in `sbi.analysis.plot` for details.

    Returns: figure and axis of posterior distribution plot
    """
    device = density._device if hasattr(density, "_device") else "cpu"

    # Setting these is required because _pairplot_scaffold will check if opts['diag'] is
    # `None`. This would break if opts has no key 'diag'. Same for 'upper'.
    diag = "cond"
    offdiag = "cond"

    opts = _get_default_opts()
    # update the defaults dictionary by the current values of the variables (passed by
    # the user)
    opts = _update(opts, locals())
    opts = _update(opts, kwargs)
    opts["lower"] = None

    dim, limits, eps_margins = prepare_for_conditional_plot(condition, opts)
    diag_func = get_conditional_diag_func(opts, limits, eps_margins, resolution)

    def offdiag_func(row, col, **kwargs):
        p_image = (
            eval_conditional_density(
                opts["density"],
                opts["condition"].to(device),
                limits.to(device),
                row,
                col,
                resolution=resolution,
                eps_margins1=eps_margins[row],
                eps_margins2=eps_margins[col],
            )
            .to("cpu")
            .numpy()
        )
        plt.imshow(
            p_image.T,
            origin="lower",
            extent=(
                limits[col, 0].item(),
                limits[col, 1].item(),
                limits[row, 0].item(),
                limits[row, 1].item(),
            ),
            aspect="auto",
        )

    return _arrange_plots(
        diag_func, offdiag_func, dim, limits, points, opts, fig=fig, axes=axes
    )


def _arrange_grid(
    diag_funcs: List[Optional[Callable]],
    upper_funcs: List[Optional[Callable]],
    lower_funcs: List[Optional[Callable]],
    diag_kwargs: List[Optional[Dict]],
    upper_kwargs: List[Optional[Dict]],
    lower_kwargs: List[Optional[Dict]],
    samples: List[np.ndarray],
    points: Optional[
        Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ],
    limits: torch.Tensor,
    subset: Optional[List[int]],
    figsize: Optional[Tuple],
    labels: Optional[List[str]],
    ticks: Optional[Union[List, torch.Tensor]],
    fig: Optional[FigureBase],
    axes: Optional[Axes],
    fig_kwargs: Dict,
) -> Tuple[FigureBase, Axes]:
    """
    Arranges the plots for any function that plots parameters either in a row of 1D
    marginals or a pairplot setting.

    Args:
        diag_funcs: List of plotting function that will be executed for the diagonal
            elements of the plot (or the columns of a row of 1D marginals).
        upper_funcs: List of plotting function that will be executed for the
            upper-diagonal elements of the plot. None if we are in a 1D setting.
        lower_funcs: List of plotting function that will be executed for the
            lower-diagonal elements of the plot. None if we are in a 1D setting.
        diag_kwargs: Additional arguments to adjust the diagonal plot,
            see the source code in `_get_default_diag_kwarg()`
        upper_kwargs: Additional arguments to adjust the upper diagonal plot,
            see the source code in `_get_default_offdiag_kwarg()`
        lower_kwargs: Additional arguments to adjust the lower diagonal plot,
            see the source code in `_get_default_offdiag_kwarg()`
        samples: List of samples given to the plotting functions
        points: List of additional points to scatter.
        limits: Limits for each dimension / axis.
        subset: List containing the dimensions to plot. E.g. subset=[1,3] will plot
            plot only the 1st and 3rd dimension
        figsize: Size of the entire figure.
        labels: List of strings specifying the names of the parameters.
        ticks: Position of the ticks.
        fig: matplotlib figure to plot on.
        axes: matplotlib axes corresponding to fig.
        fig_kwargs: Additional arguments to adjust the overall figure,
            see the source code in `_get_default_fig_kwargs()`

    Returns:
        Fig: matplotlib figure
        Axes: matplotlib axes
    """
    dim = samples[0].shape[1]
    # Prepare points
    if points is None:
        points = []
    if not isinstance(points, list):
        points = ensure_numpy(points)  # type: ignore
        points = [points]
    points = [np.atleast_2d(p) for p in points]
    points = [np.atleast_2d(ensure_numpy(p)) for p in points]
    # TODO: add asserts checking compatibility of dimensions

    # Prepare labels
    if labels == [] or labels is None:
        labels = ["dim {}".format(i + 1) for i in range(dim)]

    # Prepare ticks
    if ticks is not None:
        if len(ticks) == 1:
            ticks = [ticks[0] for _ in range(dim)]
        elif ticks == []:
            ticks = None

    # Figure out if we subset the plot
    if subset is None:
        rows = cols = dim
        subset = [i for i in range(dim)]
    else:
        if isinstance(subset, int):
            subset = [subset]
        elif isinstance(subset, list):
            pass
        else:
            raise NotImplementedError
        rows = cols = len(subset)

    # check which subplots are empty
    excl_lower = all(v is None for v in lower_funcs)
    excl_upper = all(v is None for v in upper_funcs)
    excl_diag = all(v is None for v in diag_funcs)
    flat = excl_lower and excl_upper
    one_dim = dim == 1
    # select the subset of rows and cols to be plotted
    if flat:
        rows = 1
        subset_rows = [1]
    else:
        subset_rows = subset
    subset_cols = subset

    # Create fig and axes if they were not passed.
    if fig is None or axes is None:
        fig, axes = plt.subplots(rows, cols, figsize=figsize, **fig_kwargs["subplots"])  # pyright: ignore reportAssignmenttype
    else:
        assert axes.shape == (  # pyright: ignore reportAttributeAccessIssue
            rows,
            cols,
        ), f"Passed axes must match subplot shape: {rows, cols}."

    # Style figure
    fig.subplots_adjust(**fig_kwargs["fig_subplots_adjust"])
    fig.suptitle(fig_kwargs["title"], **fig_kwargs["title_format"])

    # Main Loop through all subplots, style and create the figures
    for row_idx, row in enumerate(subset_rows):
        for col_idx, col in enumerate(subset_cols):
            if flat or row == col:
                current = "diag"
            elif row < col:
                current = "upper"
            else:
                current = "lower"

            if one_dim:
                ax = axes  # pyright: ignore reportIndexIssue
            elif flat:
                ax = axes[col_idx]  # pyright: ignore reportIndexIssue
            else:
                ax = axes[row_idx, col_idx]  # pyright: ignore reportIndexIssue
            # Diagonals
            _format_subplot(
                ax,  # pyright: ignore reportArgumentType
                current,
                limits,
                ticks,
                labels,
                fig_kwargs,
                row,
                col,
                dim,
                flat,
                excl_lower,
            )
            if current == "diag":
                if excl_diag:
                    ax.axis("off")  # pyright: ignore reportOptionalMemberAccess
                else:
                    for sample_ind, sample in enumerate(samples):
                        diag_f = diag_funcs[sample_ind]
                        if callable(diag_f):  # is callable:
                            diag_f(
                                ax, sample[:, row], limits[row], diag_kwargs[sample_ind]
                            )

                if len(points) > 0:
                    extent = ax.get_ylim()  # pyright: ignore reportOptionalMemberAccess
                    for n, v in enumerate(points):
                        ax.plot(  # pyright: ignore reportOptionalMemberAccess
                            [v[:, col], v[:, col]],
                            extent,
                            color=fig_kwargs["points_colors"][n],
                            **fig_kwargs["points_diag"],
                            label=fig_kwargs["points_labels"][n],
                        )
                if fig_kwargs["legend"] and col == 0:
                    ax.legend(**fig_kwargs["legend_kwargs"])  # pyright: ignore reportOptionalMemberAccess

            # Off-diagonals

            # upper
            elif current == "upper":
                if excl_upper:
                    ax.axis("off")  # pyright: ignore reportOptionalMemberAccess
                else:
                    for sample_ind, sample in enumerate(samples):
                        upper_f = upper_funcs[sample_ind]
                        if callable(upper_f):
                            upper_f(
                                ax,
                                sample[:, col],
                                sample[:, row],
                                limits[col],
                                limits[row],
                                upper_kwargs[sample_ind],
                            )
                    if len(points) > 0:
                        for n, v in enumerate(points):
                            ax.plot(  # pyright: ignore reportOptionalMemberAccess
                                v[:, col],
                                v[:, row],
                                color=fig_kwargs["points_colors"][n],
                                **fig_kwargs["points_offdiag"],
                            )
            # lower
            elif current == "lower":
                if excl_lower:
                    ax.axis("off")  # pyright: ignore reportOptionalMemberAccess
                else:
                    for sample_ind, sample in enumerate(samples):
                        lower_f = lower_funcs[sample_ind]
                        if callable(lower_f):
                            lower_f(
                                ax,
                                sample[:, row],
                                sample[:, col],
                                limits[row],
                                limits[col],
                                lower_kwargs[sample_ind],
                            )
                    if len(points) > 0:
                        for n, v in enumerate(points):
                            ax.plot(  # pyright: ignore reportOptionalMemberAccess
                                v[:, col],
                                v[:, row],
                                color=fig_kwargs["points_colors"][n],
                                **fig_kwargs["points_offdiag"],
                            )
    # Add dots if we subset
    if len(subset) < dim:
        if flat:
            ax = axes[len(subset) - 1]  # pyright: ignore[reportIndexIssue, reportOptionalSubscript]
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            text_kwargs = {"fontsize": plt.rcParams["font.size"] * 2.0}  # pyright: ignore[reportOptionalOperand]
            ax.text(x1 + (x1 - x0) / 8.0, (y0 + y1) / 2.0, "...", **text_kwargs)
        else:
            for row in range(len(subset)):
                ax = axes[row, len(subset) - 1]  # pyright: ignore[reportIndexIssue, reportOptionalSubscript]
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                text_kwargs = {"fontsize": plt.rcParams["font.size"] * 2.0}  # pyright: ignore[reportOptionalOperand]
                ax.text(x1 + (x1 - x0) / 8.0, (y0 + y1) / 2.0, "...", **text_kwargs)
                if row == len(subset) - 1:
                    ax.text(
                        x1 + (x1 - x0) / 12.0,
                        y0 - (y1 - y0) / 1.5,
                        "...",
                        rotation=-45,
                        **text_kwargs,
                    )

    return fig, axes  # pyright: ignore[reportReturnType]


def sbc_rank_plot(
    ranks: Union[Tensor, np.ndarray, List[Tensor], List[np.ndarray]],
    num_posterior_samples: int,
    num_bins: Optional[int] = None,
    plot_type: str = "cdf",
    parameter_labels: Optional[List[str]] = None,
    ranks_labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    figsize: Optional[tuple] = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot simulation-based calibration ranks as empirical CDFs or histograms.

    Additional options can be passed via the kwargs argument, see _sbc_rank_plot.

    Args:
        ranks: Tensor of ranks to be plotted shape (num_sbc_runs, num_parameters), or
            list of Tensors when comparing several sets of ranks, e.g., set of ranks
            obtained from different methods.
        num_bins: number of bins used for binning the ranks, default is
            num_sbc_runs / 20.
        plot_type: type of SBC plot, histograms ("hist") or empirical cdfs ("cdf").
        parameter_labels: list of labels for each parameter dimension.
        ranks_labels: list of labels for each set of ranks.
        colors: list of colors for each parameter dimension, or each set of ranks.

    Returns:
        fig, ax: figure and axis objects.

    """

    return _sbc_rank_plot(
        ranks,
        num_posterior_samples,
        num_bins,
        plot_type,
        parameter_labels,
        ranks_labels,
        colors,
        fig=fig,
        ax=ax,
        figsize=figsize,
        **kwargs,
    )


def _sbc_rank_plot(
    ranks: Union[Tensor, np.ndarray, List[Tensor], List[np.ndarray]],
    num_posterior_samples: int,
    num_bins: Optional[int] = None,
    plot_type: str = "cdf",
    parameter_labels: Optional[List[str]] = None,
    ranks_labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    num_repeats: int = 50,
    line_alpha: float = 0.8,
    show_uniform_region: bool = True,
    uniform_region_alpha: float = 0.3,
    xlim_offset_factor: float = 0.1,
    num_cols: int = 4,
    params_in_subplots: bool = False,
    show_ylabel: bool = False,
    sharey: bool = False,
    fig: Optional[FigureBase] = None,
    legend_kwargs: Optional[Dict] = None,
    ax=None,  # no type hint to avoid hassle with pyright. Should be `array(Axes).`
    figsize: Optional[tuple] = None,
) -> Tuple[Figure, Axes]:
    """Plot simulation-based calibration ranks as empirical CDFs or histograms.

    Args:
        ranks: Tensor of ranks to be plotted shape (num_sbc_runs, num_parameters), or
            list of Tensors when comparing several sets of ranks, e.g., set of ranks
            obtained from different methods.
        num_bins: number of bins used for binning the ranks, default is
            num_sbc_runs / 20.
        plot_type: type of SBC plot, histograms ("hist") or empirical cdfs ("cdf").
        parameter_labels: list of labels for each parameter dimension.
        ranks_labels: list of labels for each set of ranks.
        colors: list of colors for each parameter dimension, or each set of ranks.
        num_repeats: number of repeats for each empirical CDF step (resolution).
        line_alpha: alpha for cdf lines or histograms.
        show_uniform_region: whether to plot the region showing the cdfs expected under
            uniformity.
        uniform_region_alpha: alpha for region showing the cdfs expected under
            uniformity.
        xlim_offset_factor: factor for empty space left and right of the histogram.
        num_cols: number of subplot columns, e.g., when plotting ranks of many
            parameters.
        params_in_subplots: whether to show each parameter in a separate subplot, or
            all in one.
        show_ylabel: whether to show ylabels and ticks.
        sharey: whether to share the y-labels, ticks, and limits across subplots.
        fig: figure object to plot in.
        ax: axis object, must contain as many sublpots as parameters or len(ranks).
        figsize: dimensions of figure object, default (8, 5) or (len(ranks) * 4, 5).

    Returns:
        fig, ax: figure and axis objects.

    """

    if isinstance(ranks, (Tensor, np.ndarray)):
        ranks_list = [ranks]
    else:
        assert isinstance(ranks, List)
        ranks_list = ranks
    for idx, rank in enumerate(ranks_list):
        assert isinstance(rank, (Tensor, np.ndarray))
        if isinstance(rank, Tensor):
            ranks_list[idx]: np.ndarray = rank.numpy()  # type: ignore

    plot_types = ["hist", "cdf"]
    assert (
        plot_type in plot_types
    ), "plot type {plot_type} not implemented, use one in {plot_types}."

    if legend_kwargs is None:
        legend_kwargs = dict(loc="best", handlelength=0.8)

    num_sbc_runs, num_parameters = ranks_list[0].shape
    num_ranks = len(ranks_list)

    # For multiple methods, and for the hist plots plot each param in a separate subplot
    if num_ranks > 1 or plot_type == "hist":
        params_in_subplots = True

    for ranki in ranks_list:
        assert (
            ranki.shape == ranks_list[0].shape
        ), "all ranks in list must have the same shape."

    num_rows = int(np.ceil(num_parameters / num_cols))
    if figsize is None:
        figsize = (num_parameters * 4, num_rows * 5) if params_in_subplots else (8, 5)

    if parameter_labels is None:
        parameter_labels = [f"dim {i + 1}" for i in range(num_parameters)]
    if ranks_labels is None:
        ranks_labels = [f"rank set {i + 1}" for i in range(num_ranks)]
    if num_bins is None:
        # Recommendation from Talts et al.
        num_bins = num_sbc_runs // 20

    # Plot one row subplot for each parameter, different "methods" on top of each other.
    if params_in_subplots:
        if fig is None or ax is None:
            fig, ax = plt.subplots(
                num_rows,
                min(num_parameters, num_cols),
                figsize=figsize,
                sharey=sharey,
            )
            ax = np.atleast_1d(ax)  # type: ignore
        else:
            assert (
                ax.size >= num_parameters
            ), "There must be at least as many subplots as parameters."
            num_rows = ax.shape[0] if ax.ndim > 1 else 1
        assert ax is not None

        col_idx, row_idx = 0, 0
        for ii, ranki in enumerate(ranks_list):
            for jj in range(num_parameters):
                col_idx = jj if num_rows == 1 else jj % num_cols
                row_idx = jj // num_cols
                plt.sca(ax[col_idx] if num_rows == 1 else ax[row_idx, col_idx])

                if plot_type == "cdf":
                    _plot_ranks_as_cdf(
                        ranki[:, jj],  # type: ignore
                        num_bins,
                        num_repeats,
                        ranks_label=ranks_labels[ii],
                        color=f"C{ii}" if colors is None else colors[ii],
                        xlabel=f"posterior ranks {parameter_labels[jj]}",
                        # Show legend and ylabel only in first subplot.
                        show_ylabel=jj == 0,
                        alpha=line_alpha,
                    )
                    if ii == 0 and show_uniform_region:
                        _plot_cdf_region_expected_under_uniformity(
                            num_sbc_runs,
                            num_bins,
                            num_repeats,
                            alpha=uniform_region_alpha,
                        )
                elif plot_type == "hist":
                    _plot_ranks_as_hist(
                        ranki[:, jj],  # type: ignore
                        num_bins,
                        num_posterior_samples,
                        ranks_label=ranks_labels[ii],
                        color="firebrick" if colors is None else colors[ii],
                        xlabel=f"posterior rank {parameter_labels[jj]}",
                        # Show legend and ylabel only in first subplot.
                        show_ylabel=show_ylabel,
                        alpha=line_alpha,
                        xlim_offset_factor=xlim_offset_factor,
                    )
                    # Plot expected uniform band.
                    _plot_hist_region_expected_under_uniformity(
                        num_sbc_runs,
                        num_bins,
                        num_posterior_samples,
                        alpha=uniform_region_alpha,
                    )
                    # show legend only in first subplot.
                    if jj == 0 and ranks_labels[ii] is not None:
                        plt.legend(**legend_kwargs)

                else:
                    raise ValueError(
                        f"plot_type {plot_type} not defined, use one in {plot_types}"
                    )
                # Remove empty subplots.
        col_idx += 1
        while num_rows > 1 and col_idx < num_cols:
            ax[row_idx, col_idx].axis("off")
            col_idx += 1

    # When there is only one set of ranks show all params in a single subplot.
    else:
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        plt.sca(ax)
        ranki = ranks_list[0]
        for jj in range(num_parameters):
            _plot_ranks_as_cdf(
                ranki[:, jj],  # type: ignore
                num_bins,
                num_repeats,
                ranks_label=parameter_labels[jj],
                color=f"C{jj}" if colors is None else colors[jj],
                xlabel="posterior rank",
                # Plot ylabel and legend at last.
                show_ylabel=jj == (num_parameters - 1),
                alpha=line_alpha,
            )
        if show_uniform_region:
            _plot_cdf_region_expected_under_uniformity(
                num_sbc_runs,
                num_bins,
                num_repeats,
                alpha=uniform_region_alpha,
            )
        # show legend on the last subplot.
        plt.legend(**legend_kwargs)

    return fig, ax  # pyright: ignore[reportReturnType]


def _plot_ranks_as_hist(
    ranks: np.ndarray,
    num_bins: int,
    num_posterior_samples: int,
    ranks_label: Optional[str] = None,
    xlabel: Optional[str] = None,
    color: str = "firebrick",
    alpha: float = 0.8,
    show_ylabel: bool = False,
    num_ticks: int = 3,
    xlim_offset_factor: float = 0.1,
) -> None:
    """Plot ranks as histograms on the current axis.

    Args:
        ranks: SBC ranks in shape (num_sbc_runs, )
        num_bins: number of bins for the histogram, recommendation is num_sbc_runs / 20.
        num_posteriors_samples: number of posterior samples used for ranking.
        ranks_label: label for the ranks, e.g., when comparing ranks of different
            methods.
        xlabel: label for the current parameter.
        color: histogram color, default from Talts et al.
        alpha: histogram transparency.
        show_ylabel: whether to show y-label "counts".
        show_legend: whether to show the legend, e.g., when comparing multiple ranks.
        num_ticks: number of ticks on the x-axis.
        xlim_offset_factor: factor for empty space left and right of the histogram.
        legend_kwargs: kwargs for the legend.
    """
    xlim_offset = int(num_posterior_samples * xlim_offset_factor)
    plt.hist(
        ranks,
        bins=num_bins,
        label=ranks_label,
        color=color,
        alpha=alpha,
    )

    if show_ylabel:
        plt.ylabel("counts")
    else:
        plt.yticks([])

    plt.xlim(-xlim_offset, num_posterior_samples + xlim_offset)
    plt.xticks(np.linspace(0, num_posterior_samples, num_ticks))
    plt.xlabel("posterior rank" if xlabel is None else xlabel)


def _plot_ranks_as_cdf(
    ranks: np.ndarray,
    num_bins: int,
    num_repeats: int,
    ranks_label: Optional[str] = None,
    xlabel: Optional[str] = None,
    color: Optional[str] = None,
    alpha: float = 0.8,
    show_ylabel: bool = True,
    num_ticks: int = 3,
) -> None:
    """Plot ranks as empirical CDFs on the current axis.

    Args:
        ranks: SBC ranks in shape (num_sbc_runs, )
        num_bins: number of bins for the histogram, recommendation is num_sbc_runs / 20.
        num_repeats: number of repeats of each CDF step, i.e., resolution of the eCDF.
        ranks_label: label for the ranks, e.g., when comparing ranks of different
            methods.
        xlabel: label for the current parameter
        color: line color for the cdf.
        alpha: line transparency.
        show_ylabel: whether to show y-label "counts".
        show_legend: whether to show the legend, e.g., when comparing multiple ranks.
        num_ticks: number of ticks on the x-axis.
        legend_kwargs: kwargs for the legend.

    """
    # Generate histogram of ranks.
    hist, *_ = np.histogram(ranks, bins=num_bins, density=False)
    # Construct empirical CDF.
    histcs = hist.cumsum()
    # Plot cdf and repeat each stair step
    plt.plot(
        np.linspace(0, num_bins, num_repeats * num_bins),
        np.repeat(histcs / histcs.max(), num_repeats),
        label=ranks_label,
        color=color,
        alpha=alpha,
    )

    if show_ylabel:
        plt.yticks(np.linspace(0, 1, 3))
        plt.ylabel("empirical CDF")
    else:
        # Plot ticks only
        plt.yticks(np.linspace(0, 1, 3), [])

    plt.ylim(0, 1)
    plt.xlim(0, num_bins)
    plt.xticks(np.linspace(0, num_bins, num_ticks))
    plt.xlabel("posterior rank" if xlabel is None else xlabel)


def _plot_cdf_region_expected_under_uniformity(
    num_sbc_runs: int,
    num_bins: int,
    num_repeats: int,
    alpha: float = 0.2,
    color: str = "gray",
) -> None:
    """Plot region of empirical cdfs expected under uniformity on the current axis."""

    # Construct uniform histogram.
    uni_bins = binom(num_sbc_runs, p=1 / num_bins).ppf(0.5) * np.ones(num_bins)
    uni_bins_cdf = uni_bins.cumsum() / uni_bins.sum()
    # Decrease value one in last entry by epsilon to find valid
    # confidence intervals.
    uni_bins_cdf[-1] -= 1e-9

    lower = [binom(num_sbc_runs, p=p).ppf(0.005) for p in uni_bins_cdf]
    upper = [binom(num_sbc_runs, p=p).ppf(0.995) for p in uni_bins_cdf]

    # Plot grey area with expected ECDF.
    plt.fill_between(
        x=np.linspace(0, num_bins, num_repeats * num_bins),
        y1=np.repeat(lower / np.max(lower), num_repeats),
        y2=np.repeat(upper / np.max(upper), num_repeats),  # pyright: ignore[reportArgumentType]
        color=color,
        alpha=alpha,
        label="expected under uniformity",
    )


def _plot_hist_region_expected_under_uniformity(
    num_sbc_runs: int,
    num_bins: int,
    num_posterior_samples: int,
    alpha: float = 0.2,
    color: str = "gray",
) -> None:
    """Plot region of empirical cdfs expected under uniformity."""

    lower = binom(num_sbc_runs, p=1 / (num_bins + 1)).ppf(0.005)
    upper = binom(num_sbc_runs, p=1 / (num_bins + 1)).ppf(0.995)

    # Plot grey area with expected ECDF.
    plt.fill_between(
        x=np.linspace(0, num_posterior_samples, num_bins),
        y1=np.repeat(lower, num_bins),
        y2=np.repeat(upper, num_bins),  # pyright: ignore[reportArgumentType]
        color=color,
        alpha=alpha,
        label="expected under uniformity",
    )


# Diagnostics for hypothesis tests


def pp_plot(
    scores: Union[List[np.ndarray], Dict[Any, np.ndarray]],
    scores_null: Union[List[np.ndarray], Dict[Any, np.ndarray]],
    true_scores_null: np.ndarray,
    conf_alpha: float,
    n_alphas: int = 100,
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """Probability - Probability (P-P) plot for hypothesis tests
    to assess the validity of one (or several) estimator(s).

    See [here](https://en.wikipedia.org/wiki/P%E2%80%93P_plot) for more details.

    Args:
        scores: test scores estimated on observed data and evaluated on the test set,
            of shape (n_eval,). One array per estimator.
        scores_null: test scores estimated under the null hypothesis and evaluated on
            the test set, of shape (n_eval,). One array per null trial.
        true_scores_null: theoretical true scores under the null hypothesis,
            of shape (n_eval,).
        labels: labels for the estimators, defaults to None.
        colors: colors for the estimators, defaults to None.
        conf_alpha: significanecee level of the hypothesis test.
        n_alphas: number of cdf-values to compute the P-P plot, defaults to 100.
        ax: axis to plot on, defaults to None.
        kwargs: additional arguments for matplotlib plotting.

    Returns:
        ax: axes with the P-P plot.
    """
    if ax is None:
        ax = plt.gca()
    ax_: Axes = cast(Axes, ax)  # cast to fix pyright error

    alphas = np.linspace(0, 1, n_alphas)

    # pp_vals for the true null hypothesis
    pp_vals_true = pp_vals(true_scores_null, alphas)
    ax_.plot(alphas, pp_vals_true, "--", color="black", label="True Null (H0)")

    # pp_vals for the estimated null hypothesis over the multiple trials
    pp_vals_null = []
    for t in range(len(scores_null)):
        pp_vals_null.append(pp_vals(scores_null[t], alphas))
    pp_vals_null = np.array(pp_vals_null)

    # confidence region
    quantiles = np.quantile(pp_vals_null, [conf_alpha / 2, 1 - conf_alpha / 2], axis=0)
    ax_.fill_between(
        alphas,
        quantiles[0],
        quantiles[1],
        color="grey",
        alpha=0.2,
        label=f"{(1 - conf_alpha) * 100}% confidence region",
    )

    # pp_vals for the observed data
    for i, p_ in enumerate(scores):
        pp_vals_o = pp_vals(p_, alphas)
        if labels is not None:
            kwargs["label"] = labels[i]
        if colors is not None:
            kwargs["color"] = colors[i]
        ax_.plot(alphas, pp_vals_o, **kwargs)
    return ax_


def marginal_plot_with_probs_intensity(
    probs_per_marginal: dict,
    marginal_dim: int,
    n_bins: int = 20,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap_name: str = "Spectral_r",
    show_colorbar: bool = True,
    label: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot 1d or 2d marginal histogram of samples of the density estimator
    with probabilities as color intensity.

    Args:
        probs_per_marginal: dataframe with predicted class probabilities
            as obtained from `sbi.utils.analysis_utils.get_probs_per_marginal`.
        marginal_dim: dimension of the marginal histogram to plot.
        n_bins: number of bins for the histogram, defaults to 20.
        vmin: minimum value for the color intensity, defaults to 0.
        vmax: maximum value for the color intensity, defaults to 1.
        cmap: colormap for the color intensity, defaults to "Spectral_r".
        show_colorbar: whether to show the colorbar, defaults to True.
        label: label for the colorbar, defaults to None.
        ax (matplotlib.axes.Axes): axes to plot on, defaults to None.

    Returns:
        ax (matplotlib.axes.Axes): axes with the plot.
    """
    assert marginal_dim in [1, 2], "Only 1d or 2d marginals are supported."

    if ax is None:
        ax = plt.gca()
    ax_: Axes = cast(Axes, ax)  # cast to fix pyright error

    if label is None:
        label = "probability"

    # get colormap
    cmap = cm.get_cmap(cmap_name)

    # case of 1d marginal
    if marginal_dim == 1:
        # extract bins and patches
        _, bins, patches = ax_.hist(
            probs_per_marginal["s"], n_bins, density=True, color="green"
        )
        # create bins: all samples between bin edges are assigned to the same bin
        probs_per_marginal["bins"] = np.searchsorted(bins, probs_per_marginal["s"]) - 1
        probs_per_marginal["bins"][probs_per_marginal["bins"] < 0] = 0
        # get mean prob for each bin (same as pandas groupy method)
        array_probs = np.concatenate(
            [probs_per_marginal["bins"][:, None], probs_per_marginal["probs"][:, None]],
            axis=1,
        )
        array_probs = array_probs[array_probs[:, 0].argsort()]
        weights = np.split(
            array_probs[:, 1], np.unique(array_probs[:, 0], return_index=True)[1][1:]
        )
        weights = np.array([np.mean(w) for w in weights])
        # remove empty bins
        id = list(set(range(n_bins)) - set(probs_per_marginal["bins"]))
        patches = np.delete(np.array(patches), id)
        bins = np.delete(bins, id)

        # normalize color intensity
        norm = Normalize(vmin=vmin, vmax=vmax)
        # set color intensity
        for w, p in zip(weights, patches):
            p.set_facecolor(cmap(w))
        if show_colorbar:
            plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_, label=label)

    if marginal_dim == 2:
        # extract bin edges
        _, x, y = np.histogram2d(
            probs_per_marginal["s_1"], probs_per_marginal["s_2"], bins=n_bins
        )
        # create bins: all samples between bin edges are assigned to the same bin
        probs_per_marginal["bins_x"] = np.searchsorted(x, probs_per_marginal["s_1"]) - 1
        probs_per_marginal["bins_y"] = np.searchsorted(y, probs_per_marginal["s_2"]) - 1
        probs_per_marginal["bins_x"][probs_per_marginal["bins_x"] < 0] = 0
        probs_per_marginal["bins_y"][probs_per_marginal["bins_y"] < 0] = 0

        # extract unique bin pairs
        group_idx = np.concatenate(
            [
                probs_per_marginal["bins_x"][:, None],
                probs_per_marginal["bins_y"][:, None],
            ],
            axis=1,
        )
        unique_bins = np.unique(group_idx, return_counts=True, axis=0)[0]

        # get mean prob for each bin (same as pandas groupy method)
        mean_probs = np.zeros((len(unique_bins),))
        for i in range(len(unique_bins)):
            idx = np.where((group_idx == unique_bins[i]).all(axis=1))
            mean_probs[i] = np.mean(probs_per_marginal["probs"][idx])

        # create weight matrix with nan values for non-existing bins
        weights = np.zeros((n_bins, n_bins))
        weights[:] = np.nan
        weights[unique_bins[:, 0], unique_bins[:, 1]] = mean_probs

        # set color intensity
        norm = Normalize(vmin=vmin, vmax=vmax)
        for i in range(len(x) - 1):
            for j in range(len(y) - 1):
                facecolor = cmap(norm(weights.T[j, i]))
                # if no sample in bin, set color to white
                if weights.T[j, i] == np.nan:
                    facecolor = "white"
                rect = Rectangle(
                    (x[i], y[j]),
                    x[i + 1] - x[i],
                    y[j + 1] - y[j],
                    facecolor=facecolor,
                    edgecolor="none",
                )
                ax_.add_patch(rect)
        if show_colorbar:
            plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_, label=label)

    return ax_


# Customized plotting functions for LC2ST


def pp_plot_lc2st(
    probs: Union[List[np.ndarray], Dict[Any, np.ndarray]],
    probs_null: Union[List[np.ndarray], Dict[Any, np.ndarray]],
    conf_alpha: float,
    **kwargs: Any,
) -> Axes:
    """Probability - Probability (P-P) plot for LC2ST.

    Args:
        probs: predicted probability on observed data and evaluated on the test set,
            of shape (n_eval,). One array per estimator.
        probs_null: predicted probability under the null hypothesis and evaluated on
            the test set, of shape (n_eval,). One array per null trial.
        conf_alpha: significanecee level of the hypothesis test.
        kwargs: additional arguments for `pp_plot`.

    Returns:
        ax: axes with the P-P plot.
    """
    # probability at chance level (under the null) is 0.5
    true_probs_null = np.array([0.5] * len(probs))
    return pp_plot(
        scores=probs,
        scores_null=probs_null,
        true_scores_null=true_probs_null,
        conf_alpha=conf_alpha,
        **kwargs,
    )


def plot_tarp(
    ecp: Tensor, alpha: Tensor, title: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plots the expected coverage probability (ECP) against the credibility
    level,alpha, for a given alpha grid.

    Args:
        ecp : numpy.ndarray
            Array of expected coverage probabilities.
        alpha : numpy.ndarray
            Array of credibility levels.
        title : str, optional
            Title for the plot. The default is "".

     Returns
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.

    """

    fig = plt.figure(figsize=(6, 6))
    ax: Axes = plt.gca()

    ax.plot(alpha, ecp, color="blue", label="TARP")
    ax.plot(alpha, alpha, color="black", linestyle="--", label="ideal")
    ax.set_xlabel(r"Credibility Level $\alpha$")
    ax.set_ylabel(r"Expected Coverage Probility")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title or "")
    ax.legend()
    return fig, ax  # type: ignore


# TO BE DEPRECATED
# ----------------
def pairplot_dep(
    samples: Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor],
    points: Optional[
        Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ] = None,
    limits: Optional[Union[List, torch.Tensor]] = None,
    subset: Optional[List[int]] = None,
    offdiag: Optional[Union[List[Optional[str]], str]] = "hist",
    diag: Optional[Union[List[Optional[str]], str]] = "hist",
    figsize: Optional[Tuple] = (10, 10),
    labels: Optional[List[str]] = None,
    ticks: Optional[Union[List, torch.Tensor]] = None,
    upper: Optional[Union[List[Optional[str]], str]] = None,
    fig: Optional[FigureBase] = None,
    axes: Optional[Axes] = None,
    **kwargs: Optional[Any],
) -> Tuple[FigureBase, Axes]:
    """
    Plot samples in a 2D grid showing marginals and pairwise marginals.

    Each of the diagonal plots can be interpreted as a 1D-marginal of the distribution
    that the samples were drawn from. Each upper-diagonal plot can be interpreted as a
    2D-marginal of the distribution.

    Args:
        samples: Samples used to build the histogram.
        points: List of additional points to scatter.
        limits: Array containing the plot xlim for each parameter dimension. If None,
            just use the min and max of the passed samples
        subset: List containing the dimensions to plot. E.g. subset=[1,3] will plot
            plot only the 1st and 3rd dimension but will discard the 0th and 2nd (and,
            if they exist, the 4th, 5th and so on).
        offdiag: Plotting style for upper diagonal, {hist, scatter, contour, cond,
            None}.
        upper: deprecated, use offdiag instead.
        diag: Plotting style for diagonal, {hist, cond, None}.
        figsize: Size of the entire figure.
        labels: List of strings specifying the names of the parameters.
        ticks: Position of the ticks.
        fig: matplotlib figure to plot on.
        axes: matplotlib axes corresponding to fig.
        **kwargs: Additional arguments to adjust the plot, e.g., `samples_colors`,
            `points_colors` and many more, see the source code in `_get_default_opts()`
            in `sbi.analysis.plot` for details.

    Returns: figure and axis of posterior distribution plot
    """

    opts = _get_default_opts()
    # update the defaults dictionary by the current values of the variables (passed by
    # the user)

    opts = _update(opts, locals())
    opts = _update(opts, kwargs)

    samples, dim, limits = prepare_for_plot(samples, limits)

    # checks.
    if opts["legend"]:
        assert len(opts["samples_labels"]) >= len(
            samples
        ), "Provide at least as many labels as samples."
    if opts["upper"] is not None:
        opts["offdiag"] = opts["upper"]

    # Prepare diag/upper/lower
    if not isinstance(opts["diag"], list):
        opts["diag"] = [opts["diag"] for _ in range(len(samples))]
    if not isinstance(opts["offdiag"], list):
        opts["offdiag"] = [opts["offdiag"] for _ in range(len(samples))]
    # if type(opts['lower']) is not list:
    #    opts['lower'] = [opts['lower'] for _ in range(len(samples))]
    opts["lower"] = None

    diag_func = get_diag_func(samples, limits, opts, **kwargs)

    def offdiag_func(row, col, limits, **kwargs):
        if len(samples) > 0:
            for n, v in enumerate(samples):
                if opts["offdiag"][n] == "hist" or opts["offdiag"][n] == "hist2d":
                    hist, xedges, yedges = np.histogram2d(
                        v[:, col],
                        v[:, row],
                        range=[
                            [limits[col][0], limits[col][1]],
                            [limits[row][0], limits[row][1]],
                        ],
                        **opts["hist_offdiag"],
                    )
                    plt.imshow(
                        hist.T,
                        origin="lower",
                        extent=(
                            xedges[0],
                            xedges[-1],
                            yedges[0],
                            yedges[-1],
                        ),
                        aspect="auto",
                    )

                elif opts["offdiag"][n] in [
                    "kde",
                    "kde2d",
                    "contour",
                    "contourf",
                ]:
                    density = gaussian_kde(
                        v[:, [col, row]].T,
                        bw_method=opts["kde_offdiag"]["bw_method"],
                    )
                    X, Y = np.meshgrid(
                        np.linspace(
                            limits[col][0],
                            limits[col][1],
                            opts["kde_offdiag"]["bins"],
                        ),
                        np.linspace(
                            limits[row][0],
                            limits[row][1],
                            opts["kde_offdiag"]["bins"],
                        ),
                    )
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    Z = np.reshape(density(positions).T, X.shape)

                    if opts["offdiag"][n] == "kde" or opts["offdiag"][n] == "kde2d":
                        plt.imshow(
                            Z,
                            extent=(
                                limits[col][0],
                                limits[col][1],
                                limits[row][0],
                                limits[row][1],
                            ),
                            origin="lower",
                            aspect="auto",
                        )
                    elif opts["offdiag"][n] == "contour":
                        if opts["contour_offdiag"]["percentile"]:
                            Z = probs2contours(Z, opts["contour_offdiag"]["levels"])
                        else:
                            Z = (Z - Z.min()) / (Z.max() - Z.min())
                        plt.contour(
                            X,
                            Y,
                            Z,
                            origin="lower",
                            extent=[
                                limits[col][0],
                                limits[col][1],
                                limits[row][0],
                                limits[row][1],
                            ],
                            colors=opts["samples_colors"][n],
                            levels=opts["contour_offdiag"]["levels"],
                        )
                    else:
                        pass
                elif opts["offdiag"][n] == "scatter":
                    plt.scatter(
                        v[:, col],
                        v[:, row],
                        color=opts["samples_colors"][n],
                        **opts["scatter_offdiag"],
                    )
                elif opts["offdiag"][n] == "plot":
                    plt.plot(
                        v[:, col],
                        v[:, row],
                        color=opts["samples_colors"][n],
                        **opts["plot_offdiag"],
                    )
                else:
                    pass

    return _arrange_plots(
        diag_func, offdiag_func, dim, limits, points, opts, fig=fig, axes=axes
    )


def marginal_plot_dep(
    samples: Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor],
    points: Optional[
        Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ] = None,
    limits: Optional[Union[List, torch.Tensor]] = None,
    subset: Optional[List[int]] = None,
    diag: Optional[Union[List[Optional[str]], str]] = "hist",
    figsize: Optional[Tuple] = (10, 10),
    labels: Optional[List[str]] = None,
    ticks: Optional[Union[List, torch.Tensor]] = None,
    fig: Optional[FigureBase] = None,
    axes: Optional[Axes] = None,
    **kwargs: Optional[Any],
) -> Tuple[FigureBase, Axes]:
    """
    Plot samples in a row showing 1D marginals of selected dimensions.

    Each of the plots can be interpreted as a 1D-marginal of the distribution
    that the samples were drawn from.

    Args:
        samples: Samples used to build the histogram.
        points: List of additional points to scatter.
        limits: Array containing the plot xlim for each parameter dimension. If None,
            just use the min and max of the passed samples
        subset: List containing the dimensions to plot. E.g. subset=[1,3] will plot
            plot only the 1st and 3rd dimension but will discard the 0th and 2nd (and,
            if they exist, the 4th, 5th and so on).
        diag: Plotting style for 1D marginals, {hist, kde cond, None}.
        figsize: Size of the entire figure.
        labels: List of strings specifying the names of the parameters.
        ticks: Position of the ticks.
        points_colors: Colors of the `points`.
        fig: matplotlib figure to plot on.
        axes: matplotlib axes corresponding to fig.
        **kwargs: Additional arguments to adjust the plot, e.g., `samples_colors`,
            `points_colors` and many more, see the source code in `_get_default_opts()`
            in `sbi.analysis.plot` for details.

    Returns: figure and axis of posterior distribution plot
    """

    opts = _get_default_opts()
    # update the defaults dictionary by the current values of the variables (passed by
    # the user)

    opts = _update(opts, locals())
    opts = _update(opts, kwargs)

    samples, dim, limits = prepare_for_plot(samples, limits)

    # Prepare diag/upper/lower
    if not isinstance(opts["diag"], list):
        opts["diag"] = [opts["diag"] for _ in range(len(samples))]

    diag_func = get_diag_func(samples, limits, opts, **kwargs)

    return _arrange_plots(
        diag_func, None, dim, limits, points, opts, fig=fig, axes=axes
    )


def get_diag_func(samples, limits, opts, **kwargs):
    """
    Returns the diag_func which returns the 1D marginal plot for the parameter
    indexed by row.
    """
    warn(
        "get_diag_func will be deprecated, use get_diag_funcs instead",
        PendingDeprecationWarning,
        stacklevel=2,
    )

    def diag_func(row, **kwargs):
        if len(samples) > 0:
            for n, v in enumerate(samples):
                if opts["diag"][n] == "hist":
                    plt.hist(
                        v[:, row],
                        color=opts["samples_colors"][n],
                        label=opts["samples_labels"][n],
                        **opts["hist_diag"],
                    )
                elif opts["diag"][n] == "kde":
                    density = gaussian_kde(
                        v[:, row], bw_method=opts["kde_diag"]["bw_method"]
                    )
                    xs = np.linspace(
                        limits[row, 0], limits[row, 1], opts["kde_diag"]["bins"]
                    )
                    ys = density(xs)
                    plt.plot(
                        xs,
                        ys,
                        color=opts["samples_colors"][n],
                    )
                elif "offdiag" in opts and opts["offdiag"][n] == "scatter":
                    for single_sample in v:
                        plt.axvline(
                            single_sample[row],
                            color=opts["samples_colors"][n],
                            **opts["scatter_diag"],
                        )
                else:
                    pass

    return diag_func


def _arrange_plots(
    diag_func, offdiag_func, dim, limits, points, opts, fig=None, axes=None
):
    """
    Arranges the plots for any function that plots parameters either in a row of 1D
    marginals or a pairplot setting.

    Args:
        diag_func: Plotting function that will be executed for the diagonal elements of
            the plot (or the columns of a row of 1D marginals). It will be passed the
            current `row` (i.e. which parameter that is to be plotted) and the `limits`
            for all dimensions.
        offdiag_func: Plotting function that will be executed for the upper-diagonal
            elements of the plot. It will be passed the current `row` and `col` (i.e.
            which parameters are to be plotted and the `limits` for all dimensions. None
            if we are in a 1D setting.
        dim: The dimensionality of the density.
        limits: Limits for each parameter.
        points: Additional points to be scatter-plotted.
        opts: Dictionary built by the functions that call `_arrange_plots`. Must
            contain at least `labels`, `subset`, `figsize`, `subplots`,
            `fig_subplots_adjust`, `title`, `title_format`, ..
        fig: matplotlib figure to plot on.
        axes: matplotlib axes corresponding to fig.

    Returns: figure and axis
    """
    warn(
        "_arrange_plots will be deprecated, use _arrange_grid instead",
        PendingDeprecationWarning,
        stacklevel=2,
    )

    # Prepare points
    if points is None:
        points = []
    if not isinstance(points, list):
        points = ensure_numpy(points)  # type: ignore
        points = [points]
    points = [np.atleast_2d(p) for p in points]
    points = [np.atleast_2d(ensure_numpy(p)) for p in points]

    # TODO: add asserts checking compatibility of dimensions

    # Prepare labels
    if opts["labels"] == [] or opts["labels"] is None:
        labels_dim = ["dim {}".format(i + 1) for i in range(dim)]
    else:
        labels_dim = opts["labels"]

    # Prepare ticks
    if opts["ticks"] == [] or opts["ticks"] is None:
        ticks = None
    else:
        if len(opts["ticks"]) == 1:
            ticks = [opts["ticks"][0] for _ in range(dim)]
        else:
            ticks = opts["ticks"]

    # Figure out if we subset the plot
    subset = opts["subset"]
    if subset is None:
        rows = cols = dim
        subset = [i for i in range(dim)]
    else:
        if isinstance(subset, int):
            subset = [subset]
        elif isinstance(subset, list):
            pass
        else:
            raise NotImplementedError
        rows = cols = len(subset)
    flat = offdiag_func is None
    if flat:
        rows = 1
        opts["lower"] = None

    # Create fig and axes if they were not passed.
    if fig is None or axes is None:
        fig, axes = plt.subplots(
            rows, cols, figsize=opts["figsize"], **opts["subplots"]
        )
    else:
        assert axes.shape == (
            rows,
            cols,
        ), f"Passed axes must match subplot shape: {rows, cols}."
    # Cast to ndarray in case of 1D subplots.
    axes = np.array(axes).reshape(rows, cols)

    # Style figure
    fig.subplots_adjust(**opts["fig_subplots_adjust"])
    fig.suptitle(opts["title"], **opts["title_format"])

    # Style axes
    row_idx = -1
    for row in range(dim):
        if row not in subset:
            continue

        if not flat:
            row_idx += 1

        col_idx = -1
        for col in range(dim):
            if col not in subset:
                continue
            else:
                col_idx += 1

            if flat or row == col:
                current = "diag"
            elif row < col:
                current = "offdiag"
            else:
                current = "lower"

            ax = axes[row_idx, col_idx]
            plt.sca(ax)

            # Background color
            if (
                current in opts["fig_bg_colors"]
                and opts["fig_bg_colors"][current] is not None
            ):
                ax.set_facecolor(opts["fig_bg_colors"][current])

            # Axes
            if opts[current] is None:
                ax.axis("off")
                continue

            # Limits
            ax.set_xlim((limits[col][0], limits[col][1]))
            if current != "diag":
                ax.set_ylim((limits[row][0], limits[row][1]))

            # Ticks
            if ticks is not None:
                ax.set_xticks((ticks[col][0], ticks[col][1]))
                if current != "diag":
                    ax.set_yticks((ticks[row][0], ticks[row][1]))

            # Despine
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_position(("outward", opts["despine"]["offset"]))

            # Formatting axes
            if current == "diag":  # off-diagnoals
                if opts["lower"] is None or col == dim - 1 or flat:
                    _format_axis(
                        ax,
                        xhide=False,
                        xlabel=labels_dim[col],
                        yhide=True,
                        tickformatter=opts["tickformatter"],
                    )
                else:
                    _format_axis(ax, xhide=True, yhide=True)
            else:  # off-diagnoals
                if row == dim - 1:
                    _format_axis(
                        ax,
                        xhide=False,
                        xlabel=labels_dim[col],
                        yhide=True,
                        tickformatter=opts["tickformatter"],
                    )
                else:
                    _format_axis(ax, xhide=True, yhide=True)
            if opts["tick_labels"] is not None:
                ax.set_xticklabels((
                    str(opts["tick_labels"][col][0]),
                    str(opts["tick_labels"][col][1]),
                ))

            # Diagonals
            if current == "diag":
                diag_func(row=col, limits=limits)

                if len(points) > 0:
                    extent = ax.get_ylim()
                    for n, v in enumerate(points):
                        plt.plot(
                            [v[:, col], v[:, col]],
                            extent,
                            color=opts["points_colors"][n],
                            **opts["points_diag"],
                            label=opts["points_labels"][n],
                        )
                if opts["legend"] and col == 0:
                    plt.legend(**opts["legend_kwargs"])

            # Off-diagonals
            else:
                offdiag_func(
                    row=row,
                    col=col,
                    limits=limits,
                )

                if len(points) > 0:
                    for n, v in enumerate(points):
                        plt.plot(
                            v[:, col],
                            v[:, row],
                            color=opts["points_colors"][n],
                            **opts["points_offdiag"],
                        )

    if len(subset) < dim:
        if flat:
            ax = axes[0, len(subset) - 1]
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            text_kwargs = {"fontsize": plt.rcParams["font.size"] * 2.0}  # pyright: ignore[reportOptionalOperand]
            ax.text(x1 + (x1 - x0) / 8.0, (y0 + y1) / 2.0, "...", **text_kwargs)
        else:
            for row in range(len(subset)):
                ax = axes[row, len(subset) - 1]
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                text_kwargs = {"fontsize": plt.rcParams["font.size"] * 2.0}  # pyright: ignore[reportOptionalOperand]
                ax.text(x1 + (x1 - x0) / 8.0, (y0 + y1) / 2.0, "...", **text_kwargs)
                if row == len(subset) - 1:
                    ax.text(
                        x1 + (x1 - x0) / 12.0,
                        y0 - (y1 - y0) / 1.5,
                        "...",
                        rotation=-45,
                        **text_kwargs,
                    )

    return fig, axes


def _get_default_opts():
    warn(
        "_get_default_opts will be deprecated, use _get_default_fig_kwargs,"
        "get_default_diag_kwargs, get_default_offdiag_kwargs instead",
        PendingDeprecationWarning,
        stacklevel=2,
    )
    return {
        # title and legend
        "title": None,
        "legend": False,
        "legend_kwargs": {},
        # labels
        "points_labels": [f"points_{idx}" for idx in range(10)],  # for points
        "samples_labels": [f"samples_{idx}" for idx in range(10)],  # for samples
        # colors: take even colors for samples, odd colors for points
        "samples_colors": plt.rcParams["axes.prop_cycle"].by_key()["color"][0::2],  # pyright: ignore[reportOptionalMemberAccess]
        "points_colors": plt.rcParams["axes.prop_cycle"].by_key()["color"][1::2],  # pyright: ignore[reportOptionalMemberAccess]
        # ticks
        "ticks": [],
        "tickformatter": mpl.ticker.FormatStrFormatter("%g"),  # type: ignore
        "tick_labels": None,
        # options for hist
        "hist_diag": {
            "alpha": 1.0,
            "bins": 50,
            "density": False,
            "histtype": "step",
        },
        "hist_offdiag": {
            # 'edgecolor': 'none',
            # 'linewidth': 0.0,
            "bins": 50,
        },
        # options for kde
        "kde_diag": {"bw_method": "scott", "bins": 50, "color": "black"},
        "kde_offdiag": {"bw_method": "scott", "bins": 50},
        # options for contour
        "contour_offdiag": {"levels": [0.68], "percentile": True},
        # options for scatter
        "scatter_offdiag": {
            "alpha": 0.5,
            "edgecolor": "none",
            "rasterized": False,
        },
        "scatter_diag": {},
        # options for plot
        "plot_offdiag": {},
        # formatting points (scale, markers)
        "points_diag": {},
        "points_offdiag": {
            "marker": ".",
            "markersize": 10,
        },
        # other options
        "fig_bg_colors": {"offdiag": None, "diag": None, "lower": None},
        "fig_subplots_adjust": {
            "top": 0.9,
        },
        "subplots": {},
        "despine": {
            "offset": 5,
        },
        "title_format": {"fontsize": 16},
    }
