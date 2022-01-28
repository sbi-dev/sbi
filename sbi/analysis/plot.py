# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import collections
from typing import Any, List, Optional, Tuple, Union

import matplotlib as mpl
import numpy as np
import six
import torch
from matplotlib import pyplot as plt
from scipy.stats import binom, gaussian_kde
from torch import Tensor

from sbi.utils import eval_conditional_density

try:
    collectionsAbc = collections.abc
except:
    collectionsAbc = collections


def hex2rgb(hex):
    # Pass 16 to the integer function for change of base
    return [int(hex[i : i + 2], 16) for i in range(1, 6, 2)]


def rgb2hex(RGB):
    # Components need to be integers for hex to make sense
    RGB = [int(x) for x in RGB]
    return "#" + "".join(
        ["0{0:x}".format(v) if v < 16 else "{0:x}".format(v) for v in RGB]
    )


def _update(d, u):
    # https://stackoverflow.com/a/3233356
    for k, v in six.iteritems(u):
        dv = d.get(k, {})
        if not isinstance(dv, collectionsAbc.Mapping):
            d[k] = v
        elif isinstance(v, collectionsAbc.Mapping):
            d[k] = _update(dv, v)
        else:
            d[k] = v
    return d


def _format_axis(ax, xhide=True, yhide=True, xlabel="", ylabel="", tickformatter=None):
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
        ax.spines["bottom"].set_visible(True)
    if not yhide:
        ax.set_ylabel(ylabel)
        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_tick_params(labelleft=True)
        if tickformatter is not None:
            ax.yaxis.set_major_formatter(tickformatter)
        ax.spines["left"].set_visible(True)
    return ax


def probs2contours(probs, levels):
    """Takes an array of probabilities and produces an array of contours at specified
    percentile levels.
    Parameters
    ----------
    probs : array
        Probability array. doesn't have to sum to 1, but it is assumed it contains all the mass
    levels : list
        Percentile levels, have to be in [0.0, 1.0]. Specifies contour levels that
        include a given proportion of samples, i.e., 0.1 specifies where the top 10% of
        the density is.
    Return
    ------
    Array of same shape as probs with percentile labels. Values in output array
    denote labels which percentile bin the probability mass belongs to.

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
    else:
        return t


def prepare_for_plot(samples, limits):
    """
    Ensures correct formatting for samples and limits, and returns dimension
    of the samples.
    """

    # Prepare samples
    if type(samples) != list:
        samples = ensure_numpy(samples)
        samples = [samples]
    else:
        for i, sample_pack in enumerate(samples):
            samples[i] = ensure_numpy(samples[i])

    # Dimensionality of the problem.
    dim = samples[0].shape[1]

    # Prepare limits. Infer them from samples if they had not been passed.
    if limits == [] or limits is None:
        limits = []
        for d in range(dim):
            min = +np.inf
            max = -np.inf
            for sample in samples:
                min_ = sample[:, d].min()
                min = min_ if min_ < min else min
                max_ = sample[:, d].max()
                max = max_ if max_ > max else max
            limits.append([min, max])
    else:
        if len(limits) == 1:
            limits = [limits[0] for _ in range(dim)]
        else:
            limits = limits
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


def get_diag_func(samples, limits, opts, **kwargs):
    """
    Returns the diag_func which returns the 1D marginal plot for the parameter
    indexed by row.
    """

    def diag_func(row, **kwargs):
        if len(samples) > 0:
            for n, v in enumerate(samples):
                if opts["diag"][n] == "hist":
                    h = plt.hist(
                        v[:, row], color=opts["samples_colors"][n], **opts["hist_diag"]
                    )
                elif opts["diag"][n] == "kde":
                    density = gaussian_kde(
                        v[:, row], bw_method=opts["kde_diag"]["bw_method"]
                    )
                    xs = np.linspace(
                        limits[row, 0], limits[row, 1], opts["kde_diag"]["bins"]
                    )
                    ys = density(xs)
                    h = plt.plot(
                        xs,
                        ys,
                        color=opts["samples_colors"][n],
                    )
                elif "upper" in opts.keys() and opts["upper"][n] == "scatter":
                    for single_sample in v:
                        plt.axvline(
                            single_sample[row],
                            color=opts["samples_colors"][n],
                            **opts["scatter_diag"],
                        )
                else:
                    pass

    return diag_func


def get_conditional_diag_func(opts, limits, eps_margins, resolution):
    """
    Returns the diag_func which returns the 1D marginal conditional plot for
    the parameter indexed by row.
    """

    def diag_func(row, **kwargs):
        p_vector = eval_conditional_density(
            opts["density"],
            opts["condition"],
            limits,
            row,
            row,
            resolution=resolution,
            eps_margins1=eps_margins[row],
            eps_margins2=eps_margins[row],
            warn_about_deprecation=False,
        ).numpy()
        h = plt.plot(
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
    samples: Union[
        List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor
    ] = None,
    points: Optional[
        Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ] = None,
    limits: Optional[Union[List, torch.Tensor]] = None,
    subset: List[int] = None,
    upper: Optional[str] = "hist",
    diag: Optional[str] = "hist",
    figsize: Tuple = (10, 10),
    labels: Optional[List[str]] = None,
    ticks: Union[List, torch.Tensor] = [],
    points_colors: List[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"],
    fig=None,
    axes=None,
    **kwargs,
):
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
        upper: Plotting style for upper diagonal, {hist, scatter, contour, cond, None}.
        diag: Plotting style for diagonal, {hist, cond, None}.
        figsize: Size of the entire figure.
        labels: List of strings specifying the names of the parameters.
        ticks: Position of the ticks.
        points_colors: Colors of the `points`.
        fig: matplotlib figure to plot on.
        axes: matplotlib axes corresponding to fig.
        **kwargs: Additional arguments to adjust the plot, see the source code in
            `_get_default_opts()` in `sbi.utils.plot` for more details.

    Returns: figure and axis of posterior distribution plot
    """

    # TODO: add color map support
    # TODO: automatically determine good bin sizes for histograms
    # TODO: add legend (if legend is True)

    opts = _get_default_opts()
    # update the defaults dictionary by the current values of the variables (passed by
    # the user)

    opts = _update(opts, locals())
    opts = _update(opts, kwargs)

    samples, dim, limits = prepare_for_plot(samples, limits)

    # Prepare diag/upper/lower
    if type(opts["diag"]) is not list:
        opts["diag"] = [opts["diag"] for _ in range(len(samples))]
    if type(opts["upper"]) is not list:
        opts["upper"] = [opts["upper"] for _ in range(len(samples))]
    # if type(opts['lower']) is not list:
    #    opts['lower'] = [opts['lower'] for _ in range(len(samples))]
    opts["lower"] = None

    diag_func = get_diag_func(samples, limits, opts, **kwargs)

    def upper_func(row, col, limits, **kwargs):
        if len(samples) > 0:
            for n, v in enumerate(samples):
                if opts["upper"][n] == "hist" or opts["upper"][n] == "hist2d":
                    hist, xedges, yedges = np.histogram2d(
                        v[:, col],
                        v[:, row],
                        range=[
                            [limits[col][0], limits[col][1]],
                            [limits[row][0], limits[row][1]],
                        ],
                        **opts["hist_offdiag"],
                    )
                    h = plt.imshow(
                        hist.T,
                        origin="lower",
                        extent=[
                            xedges[0],
                            xedges[-1],
                            yedges[0],
                            yedges[-1],
                        ],
                        aspect="auto",
                    )

                elif opts["upper"][n] in [
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

                    if opts["upper"][n] == "kde" or opts["upper"][n] == "kde2d":
                        h = plt.imshow(
                            Z,
                            extent=[
                                limits[col][0],
                                limits[col][1],
                                limits[row][0],
                                limits[row][1],
                            ],
                            origin="lower",
                            aspect="auto",
                        )
                    elif opts["upper"][n] == "contour":
                        if opts["contour_offdiag"]["percentile"]:
                            Z = probs2contours(Z, opts["contour_offdiag"]["levels"])
                        else:
                            Z = (Z - Z.min()) / (Z.max() - Z.min())
                        h = plt.contour(
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
                elif opts["upper"][n] == "scatter":
                    h = plt.scatter(
                        v[:, col],
                        v[:, row],
                        color=opts["samples_colors"][n],
                        **opts["scatter_offdiag"],
                    )
                elif opts["upper"][n] == "plot":
                    h = plt.plot(
                        v[:, col],
                        v[:, row],
                        color=opts["samples_colors"][n],
                        **opts["plot_offdiag"],
                    )
                else:
                    pass

    return _arrange_plots(
        diag_func, upper_func, dim, limits, points, opts, fig=fig, axes=axes
    )


def marginal_plot(
    samples: Union[
        List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor
    ] = None,
    points: Optional[
        Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ] = None,
    limits: Optional[Union[List, torch.Tensor]] = None,
    subset: List[int] = None,
    diag: Optional[str] = "hist",
    figsize: Tuple = (10, 10),
    labels: Optional[List[str]] = None,
    ticks: Union[List, torch.Tensor] = [],
    points_colors: List[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"],
    fig=None,
    axes=None,
    **kwargs,
):
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
        **kwargs: Additional arguments to adjust the plot, see the source code in
            `_get_default_opts()` in `sbi.utils.plot` for more details.

    Returns: figure and axis of posterior distribution plot
    """

    opts = _get_default_opts()
    # update the defaults dictionary by the current values of the variables (passed by
    # the user)

    opts = _update(opts, locals())
    opts = _update(opts, kwargs)

    samples, dim, limits = prepare_for_plot(samples, limits)

    # Prepare diag/upper/lower
    if type(opts["diag"]) is not list:
        opts["diag"] = [opts["diag"] for _ in range(len(samples))]

    diag_func = get_diag_func(samples, limits, opts, **kwargs)

    return _arrange_plots(
        diag_func, None, dim, limits, points, opts, fig=fig, axes=axes
    )


def conditional_marginal_plot(
    density: Any,
    condition: torch.Tensor,
    limits: Union[List, torch.Tensor],
    points: Optional[
        Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ] = None,
    subset: List[int] = None,
    resolution: int = 50,
    figsize: Tuple = (10, 10),
    labels: Optional[List[str]] = None,
    ticks: Union[List, torch.Tensor] = [],
    points_colors: List[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"],
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
    $p(\theta_0 | \theta_1=c[1], \theta_2=c[2])$. All other diagonals and are built in the corresponding way.

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
        **kwargs: Additional arguments to adjust the plot, see the source code in
            `_get_default_opts()` in `sbi.utils.plot` for more details.

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
    subset: List[int] = None,
    resolution: int = 50,
    figsize: Tuple = (10, 10),
    labels: Optional[List[str]] = None,
    ticks: Union[List, torch.Tensor] = [],
    points_colors: List[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"],
    warn_about_deprecation: bool = True,
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
        warn_about_deprecation: With sbi v0.15.0, we depracated the import of this
            function from `sbi.utils`. Instead, it should be imported from
            `sbi.analysis`.
        fig: matplotlib figure to plot on.
        axes: matplotlib axes corresponding to fig.
        **kwargs: Additional arguments to adjust the plot, see the source code in
            `_get_default_opts()` in `sbi.utils.plot` for more details.

    Returns: figure and axis of posterior distribution plot
    """
    device = density._device if hasattr(density, "_device") else "cpu"

    # Setting these is required because _pairplot_scaffold will check if opts['diag'] is
    # `None`. This would break if opts has no key 'diag'. Same for 'upper'.
    diag = "cond"
    upper = "cond"

    opts = _get_default_opts()
    # update the defaults dictionary by the current values of the variables (passed by
    # the user)
    opts = _update(opts, locals())
    opts = _update(opts, kwargs)
    opts["lower"] = None

    dim, limits, eps_margins = prepare_for_conditional_plot(condition, opts)
    diag_func = get_conditional_diag_func(opts, limits, eps_margins, resolution)

    def upper_func(row, col, **kwargs):
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
                warn_about_deprecation=False,
            )
            .to("cpu")
            .numpy()
        )
        h = plt.imshow(
            p_image.T,
            origin="lower",
            extent=[
                limits[col, 0],
                limits[col, 1],
                limits[row, 0],
                limits[row, 1],
            ],
            aspect="auto",
        )

    return _arrange_plots(
        diag_func, upper_func, dim, limits, points, opts, fig=fig, axes=axes
    )


def _arrange_plots(
    diag_func, upper_func, dim, limits, points, opts, fig=None, axes=None
):
    """
    Arranges the plots for any function that plots parameters either in a row of 1D
    marginals or a pairplot setting.

    Args:
        diag_func: Plotting function that will be executed for the diagonal elements of
            the plot (or the columns of a row of 1D marginals). It will be passed the
            current `row` (i.e. which parameter that is to be plotted) and the `limits`
            for all dimensions.
        upper_func: Plotting function that will be executed for the upper-diagonal
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

    # Prepare points
    if points is None:
        points = []
    if type(points) != list:
        points = ensure_numpy(points)
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
        if type(subset) == int:
            subset = [subset]
        elif type(subset) == list:
            pass
        else:
            raise NotImplementedError
        rows = cols = len(subset)
    flat = upper_func is None
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
    for row in range(rows):
        if row not in subset and not flat:
            continue
        else:
            row_idx += 1

        col_idx = -1
        for col in range(dim):
            if col not in subset:
                continue
            else:
                col_idx += 1

            if flat:
                current = "diag"
            elif row == col:
                current = "diag"
            elif row < col:
                current = "upper"
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
                ax.set_xticklabels(
                    (
                        str(opts["tick_labels"][col][0]),
                        str(opts["tick_labels"][col][1]),
                    )
                )

            # Diagonals
            if current == "diag":
                diag_func(row=col, limits=limits)

                if len(points) > 0:
                    extent = ax.get_ylim()
                    for n, v in enumerate(points):
                        h = plt.plot(
                            [v[:, col], v[:, col]],
                            extent,
                            color=opts["points_colors"][n],
                            **opts["points_diag"],
                        )

            # Off-diagonals
            else:
                upper_func(
                    row=row,
                    col=col,
                    limits=limits,
                )

                if len(points) > 0:

                    for n, v in enumerate(points):
                        h = plt.plot(
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
            text_kwargs = {"fontsize": plt.rcParams["font.size"] * 2.0}
            ax.text(x1 + (x1 - x0) / 8.0, (y0 + y1) / 2.0, "...", **text_kwargs)
        else:
            for row in range(len(subset)):
                ax = axes[row, len(subset) - 1]
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                text_kwargs = {"fontsize": plt.rcParams["font.size"] * 2.0}
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
    """Return default values for plotting specs."""

    return {
        # 'lower': None,     # hist/scatter/None  # TODO: implement
        # title and legend
        "title": None,
        "legend": False,
        # labels
        "labels_points": [],  # for points
        "labels_samples": [],  # for samples
        # colors
        "samples_colors": plt.rcParams["axes.prop_cycle"].by_key()["color"],
        # ticks
        "tickformatter": mpl.ticker.FormatStrFormatter("%g"),
        "tick_labels": None,
        # options for hist
        "hist_diag": {"alpha": 1.0, "bins": 50, "density": False, "histtype": "step"},
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
            "markersize": 20,
        },
        # other options
        "fig_bg_colors": {"upper": None, "diag": None, "lower": None},
        "fig_subplots_adjust": {
            "top": 0.9,
        },
        "subplots": {},
        "despine": {
            "offset": 5,
        },
        "title_format": {"fontsize": 16},
    }


def sbc_rank_plot(
    ranks,
    num_posterior_samples,
    num_bins=100,
    num_repeats=50,
    plot_type="cdf",
    parameter_labels=None,
    ranks_labels=None,
    colors=None,
    line_alpha=0.8,
    show_uniform_region=True,
    uniform_region_alpha=0.2,
    num_cols=4,
    params_in_subplots=False,
    fig=None,
    ax=None,
    figsize=None,
):
    """Plot simulation-based calibration ranks as empirical CDFs.

    Args:
        ranks: Tensor of ranks to be plotted, or list of Tensors when comparing several sets
            of ranks, e.g., set of ranks obtained from different methods.
        num_bins: number of bins used for calculating empirical cdfs via histograms.
        num_repeats: number of repeats for each empirical CDF step (resolution).
        parameter_labels: list of labels for each parameter dimension.
        ranks_labels: list of labels for each set of ranks.
        colors: list of colors for each parameter dimension, or each set of ranks.
        line_alpha: alpha for cdf lines
        show_uniform_region: whether to plot the region showing the cdfs expected under uniformity
        uniform_region_alpha: alpha for region showing the cdfs expected under uniformity.
        fig: figure object to plot in.
        ax: axis object, must have shape (number of set of ranks,) or be a raw Axes object.
        figsize: dimensions of figure object, defaults to (8, 5) or (number of set of ranks * 4, 5).

    Returns:
        fig, ax: figure and axis objects.

    """

    if isinstance(ranks, (Tensor, np.ndarray)):
        ranks = [ranks]
    else:
        assert isinstance(ranks, List)
        assert isinstance(ranks[0], (Tensor, np.ndarray))

    num_sbc_runs, num_parameters = ranks[0].shape
    num_ranks = len(ranks)

    # For multiple methods, and for the hist plots plot each param in a separate subplot
    if num_ranks > 1 or plot_type == "hist":
        params_in_subplots = True

    for ranki in ranks:
        assert (
            ranki.shape == ranks[0].shape
        ), "all ranks in list must have the same shape."

    num_rows = int(np.ceil(num_parameters / num_cols))
    if figsize is None:
        figsize = (num_parameters * 4, num_rows * 5) if params_in_subplots else (8, 5)

    if parameter_labels is None:
        parameter_labels = [f"dim {i+1}" for i in range(num_parameters)]
    if ranks_labels is None:
        ranks_labels = [f"rank set {i+1}" for i in range(num_ranks)]

    # Plot one row subplot for each parameter, different "methods" on top of each other.
    if params_in_subplots:
        if fig is None or ax is None:
            fig, ax = plt.subplots(num_rows, num_parameters, figsize=figsize)
        else:
            assert ax.shape == (
                num_parameters,
            ), "Expecting subplots of shape 1, num_parameters."

        for ii, ranki in enumerate(ranks):
            for jj in range(num_parameters):
                col_idx = jj if num_rows == 1 else jj % num_rows
                plt.sca(ax[col_idx] if num_rows == 1 else ax[num_rows, col_idx])

                if plot_type == "cdf":
                    plot_ranks_as_cdf(
                        ranki[:, jj],
                        num_bins,
                        num_repeats,
                        label=ranks_labels[ii],
                        color=f"C{ii}" if colors is None else colors[ii],
                        xlabel=f"posterior rank {parameter_labels[jj]}",
                        # Show legend and ylabel only in first subplot.
                        show_ylabel=jj == 0,
                        show_legend=jj == 0,
                        alpha=line_alpha,
                    )
                    if ii == 0 and show_uniform_region:
                        plot_cdf_region_expected_under_uniformity(
                            num_sbc_runs, num_bins, num_repeats, alpha=0.1
                        )
                elif plot_type == "hist":
                    plot_ranks_as_hist(
                        ranki[:, jj],
                        num_bins,
                        num_posterior_samples,
                        label=ranks_labels[ii],
                        color=f"C{ii}" if colors is None else colors[ii],
                        xlabel=f"posterior rank {parameter_labels[jj]}",
                        # Show legend and ylabel only in first subplot.
                        show_ylabel=False,
                        show_legend=jj == 0,
                        alpha=line_alpha,
                    )
                    # TODO: fix uniform region plotting.
                #                     plot_hist_region_expected_under_uniformity(num_sbc_runs,
                #                                                                num_bins,
                #                                                                num_posterior_samples,
                #                                                                alpha=0.1)
                else:
                    pass
                    # TODO: raise ValueError(f"plot_type {plot_type} not defined, use one in {plot_types}")

    # When there is only one set of ranks show all params in a single subplot.
    else:
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        plt.sca(ax)
        ranki = ranks[0]
        for jj in range(num_parameters):
            plot_ranks_as_cdf(
                ranki[:, jj],
                num_bins,
                num_repeats,
                label=parameter_labels[jj],
                color=f"C{jj}" if colors is None else colors[jj],
                xlabel=f"posterior rank",
                # Plot ylabel and legend at last.
                show_ylabel=jj == (num_parameters - 1),
                show_legend=jj == (num_parameters - 1),
                alpha=line_alpha,
            )
        if show_uniform_region:
            plot_cdf_region_expected_under_uniformity(
                num_sbc_runs, num_bins, num_repeats, alpha=uniform_region_alpha
            )

    return fig, ax


def plot_ranks_as_hist(
    ranks: Tensor,
    num_bins: int,
    num_posterior_samples: int,
    label: str = None,
    color: str = None,
    alpha: float = 0.8,
    xlabel: str = None,
    show_ylabel: bool = False,
    show_legend: bool = False,
    num_ticks: int = 5,
    xlim_offset_factor: float = 0.1,
    legend_kwargs: dict = {},
) -> None:
    """Plot ranks as empirical CDFs.

    Args:
        ranks:
        num_bins:
        num_repeats:
        label:
        color:
        alpha:
        xlabel:
        show_ylabel:
        show_legend:
        num_ticks:
        legend_kwargs:

    """
    xlim_offset = int(num_posterior_samples * xlim_offset_factor)
    plt.hist(
        ranks,
        bins=num_bins,
        label=label,
        color=color,
        alpha=alpha,
        density=True,
    )

    if show_ylabel:
        plt.ylabel("counts")
    else:
        plt.yticks([])
    if show_legend and label:
        plt.legend(loc=2, handlelength=0.8, **legend_kwargs)

    plt.xlim(-xlim_offset, num_posterior_samples + xlim_offset)
    plt.xticks(np.linspace(0, num_posterior_samples, num_ticks))
    plt.xlabel("posterior rank" if xlabel is None else xlabel)


def plot_ranks_as_cdfs(
    ranks: Tensor,
    num_bins: int,
    num_repeats: int,
    label: str = None,
    color: str = None,
    alpha: float = 0.8,
    xlabel: str = None,
    show_ylabel: bool = False,
    show_legend: bool = False,
    num_ticks: int = 3,
    legend_kwargs: dict = {},
) -> None:
    """Plot ranks as empirical CDFs.

    Args:
        ranks:
        num_bins:
        num_repeats:
        label:
        color:
        alpha:
        xlabel:
        show_ylabel:
        show_legend:
        num_ticks:
        legend_kwargs:

    """
    # Generate histogram of ranks.
    hist, *_ = np.histogram(ranks, bins=num_bins, density=False)
    # Construct empirical CDF.
    histcs = hist.cumsum()
    # Plot cdf and repeat each stair step
    plt.plot(
        np.linspace(0, num_bins, num_repeats * num_bins),
        np.repeat(histcs / histcs.max(), num_repeats),
        label=label,
        color=color,
        alpha=alpha,
    )

    if show_ylabel:
        plt.yticks(np.linspace(0, 1, 3))
        plt.ylabel("empirical CDF")
    else:
        # Plot ticks only
        plt.yticks(np.linspace(0, 1, 3), [])
    if show_legend and label:
        plt.legend(loc=2, handlelength=0.8, **legend_kwargs)

    plt.ylim(0, 1)
    plt.xlim(0, num_bins)
    plt.xticks(np.linspace(0, num_bins, num_ticks))
    plt.xlabel("posterior rank" if xlabel is None else xlabel)


def plot_cdf_region_expected_under_uniformity(
    num_sbc_runs: int,
    num_bins: int,
    num_repeats: int,
    alpha: float = 0.1,
    color="grey",
):
    """Plot region of empirical cdfs expected under uniformity."""

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
        y2=np.repeat(upper / np.max(upper), num_repeats),
        color=color,
        alpha=alpha,
    )


def plot_hist_region_expected_under_uniformity(
    num_sbc_runs: int,
    num_bins: int,
    num_posterior_samples: int,
    alpha: float = 0.1,
    color="grey",
):
    """Plot region of empirical cdfs expected under uniformity."""

    lower = binom(num_sbc_runs, p=1 / (num_bins + 1)).ppf(0.005)
    upper = binom(num_sbc_runs, p=1 / (num_bins + 1)).ppf(0.995)
        
    # Plot grey area with expected ECDF.
    plt.fill_between(
        x=np.linspace(0, num_bins, num_posterior_samples),
        y1=np.repeat(lower, num_posterior_samples),
        y2=np.repeat(upper, num_posterior_samples),
        color=color,
        alpha=alpha,
    )

