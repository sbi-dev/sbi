# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import collections
import inspect
from typing import Callable, Optional, Union, Dict, Any, Tuple, Union, cast, List, Sequence, TypeVar

import torch
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import six
from scipy.stats import gaussian_kde

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
    """Takes an array of probabilities and produces an array of contours at specified percentile levels
    Parameters
    ----------
    probs : array
        Probability array. doesn't have to sum to 1, but it is assumed it contains all the mass
    levels : list
        Percentile levels, have to be in [0.0, 1.0]
    Return
    ------
    Array of same shape as probs with percentile labels
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


def pairplot(
    samples: Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor],
    points: Optional[
        Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ] = None,
    upper: Optional[str] = "hist",
    diag: Optional[str] = "hist",
    title: Optional[str] = None,
    legend: Optional[bool] = False,
    labels=None,
    labels_points=None,
    labels_samples=None,
    samples_colors=plt.rcParams["axes.prop_cycle"].by_key()["color"],
    points_colors=plt.rcParams["axes.prop_cycle"].by_key()["color"],
    subset=None,
    limits=None,
    ticks=None,
    tickformatter=mpl.ticker.FormatStrFormatter("%g"),
    tick_labels=None,
    hist_diag={"alpha": 1.0, "bins": 50, "density": False, "histtype": "step"},
    hist_offdiag={"bins": 50,},
    kde_diag={"bw_method": "scott", "bins": 50, "color": "black"},
    kde_offdiag={"bw_method": "scott", "bins": 50},
    contour_offdiag={"levels": [0.68], "percentile": True},
    scatter_offdiag={"alpha": 0.5, "edgecolor": "none", "rasterized": False},
    plot_offdiag={},
    points_diag={},
    points_offdiag={"marker": ".", "markersize": 20},
    fig_size: Tuple = (10, 10),
    fig_bg_colors={"upper": None, "diag": None, "lower": None},
    fig_subplots_adjust={"top": 0.9},
    subplots={},
    despine={"offset": 5},
    title_format={"fontsize": 16},
):
    """
    Plot samples and points.

    For developers: if you add arguments that expect dictionaries, make sure to access
    them via the opts dictionary instantiated below. E.g. if you want to access the dict
    stored in the input variable hist_diag, use opts[`hist_diag`].

    Args:
        samples: posterior samples used to build the histogram
        points: list of additional points to scatter
        upper: plotting style for upper diagonal, {hist, scatter, contour, None}
        diag: plotting style for diagonal, {hist, None}
        title: title string
        legend: whether to plot a legend for the points
        labels: np.ndarray of strings specifying the names of the parameters
        labels_points: np.ndarray of strings specifying the names of the passed points
        labels_samples: np.ndarray of strings specifying the names of the passed samples
        samples_colors: colors of the samples
        points_colors: colors of the points
        subset: List containing the dimensions to plot. E.g. subset=[1,3] will plot
            plot only the 1st and 3rd dimension but will discard the 0th and 2nd (and,
            if they exist, the 4th, 5th and so on)
        limits: array containing the plot xlim for each parameter dimension. If None,
            just use the min and max of the passed samples
        ticks: location of the ticks for each parameter. If None, just use the min and
            max along each parameter dimension
        tickformatter: passed to _format_axis()
        tick_labels: np.ndarray containing the ticklabels.
        hist_diag: dictionary passed to plt.hist() for diagonal plots
        hist_offdiag: dictionary passed to np.histogram2d() for off diagonal plots
        kde_diag: dictionary passed to gaussian_kde() for diagonal plots
        kde_offdiag: dictionary passed to gaussian_kde() for off diagonal plots
        contour_offdiag: dictionary that should contain `percentile` and `levels` keys.
            `percentile`: bool.
                If  `percentile`==True,
                the levels are made with respect to the max probability of the posterior
                If `percentile`==False,
                the levels are drawn at absolute positions
            `levels`: list or np.ndarray: specifies the location where the contours are
                drawn.
        scatter_offdiag: dictionary for plt.scatter() on off diagonal
        plot_offdiag: dictionary for plt.plot() on off diagonal
        points_diag: dictionary for plt.plot() used for plotting points on diagonal
        points_offdiag: dictionary for plt.plot() used for plotting points on off
            diagonal
        fig_size: size of the entire figure
        fig_bg_colors: Dictionary that contains `upper`, `diag`, `lower`, and specifies
            the respective background colors. Passed to ax.set_facecolor()
        fig_subplots_adjust: dictionary passed to fig.subplots_adjust()
        subplots: dictionary passed to plt.subplots()
        despine: dictionary passed to set_position() for axis position
        title_format: dictionary passed to plt.title()

    Returns: figure and axis of posterior distribution plot

    """

    # TODO: add color map support
    # TODO: automatically determine good bin sizes for histograms
    # TODO: add legend (if legend is True)

    # get default values of function arguments
    # https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    spec = inspect.getfullargspec(pairplot)

    # build a dict for the defaults
    # https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    # answer by gnr
    default_val_dict = dict(zip(spec.args[::-1], (spec.defaults or ())[::-1]))

    # update the defaults dictionary by the current values of the variables (passed by
    # the user)
    opts = _update(default_val_dict, locals())

    # Prepare samples
    if type(samples) != list:
        samples = ensure_numpy(samples)
        samples = [samples]
    else:
        for i, sample_pack in enumerate(samples):
            samples[i] = ensure_numpy(samples[i])

    # Prepare points
    if points is None:
        points = []
    if type(points) != list:
        points = ensure_numpy(points)
        points = [points]
    points = [np.atleast_2d(p) for p in points]
    points = [np.atleast_2d(ensure_numpy(p)) for p in points]

    # Dimensions
    dim = samples[0].shape[1]
    num_samples = samples[0].shape[0]

    # TODO: add asserts checking compatibility of dimensions

    # Prepare labels
    if opts["labels"] == [] or opts["labels"] is None:
        labels_dim = ["dim {}".format(i + 1) for i in range(dim)]
    else:
        labels_dim = opts["labels"]

    # Prepare limits
    if opts["limits"] == [] or opts["limits"] is None:
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
        if len(opts["limits"]) == 1:
            limits = [opts["limits"][0] for _ in range(dim)]
        else:
            limits = opts["limits"]

    # Prepare ticks
    if opts["ticks"] == [] or opts["ticks"] is None:
        ticks = None
    else:
        if len(opts["ticks"]) == 1:
            ticks = [opts["ticks"][0] for _ in range(dim)]
        else:
            ticks = opts["ticks"]

    # Prepare diag/upper/lower
    if type(opts["diag"]) is not list:
        opts["diag"] = [opts["diag"] for _ in range(len(samples))]
    if type(opts["upper"]) is not list:
        opts["upper"] = [opts["upper"] for _ in range(len(samples))]
    # if type(opts['lower']) is not list:
    #    opts['lower'] = [opts['lower'] for _ in range(len(samples))]
    opts["lower"] = None

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

    fig, axes = plt.subplots(rows, cols, figsize=opts["fig_size"], **opts["subplots"])
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
        else:
            row_idx += 1

        col_idx = -1
        for col in range(dim):
            if col not in subset:
                continue
            else:
                col_idx += 1

            if row == col:
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
            if limits is not None:
                ax.set_xlim((limits[col][0], limits[col][1]))
                if current != "diag":
                    ax.set_ylim((limits[row][0], limits[row][1]))
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            # Ticks
            if ticks is not None:
                ax.set_xticks((ticks[col][0], ticks[col][1]))
                if current != "diag":
                    ax.set_yticks((ticks[row][0], ticks[row][1]))

            # Despine
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_position(("outward", despine["offset"]))

            # Formatting axes
            if current == "diag":  # off-diagnoals
                if opts["lower"] is None or col == dim - 1:
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
                if len(samples) > 0:
                    for n, v in enumerate(samples):
                        if opts["diag"][n] == "hist":
                            h = plt.hist(
                                v[:, row],
                                color=opts["samples_colors"][n],
                                **opts["hist_diag"]
                            )
                        elif opts["diag"][n] == "kde":
                            density = gaussian_kde(
                                v[:, row], bw_method=opts["kde_diag"]["bw_method"]
                            )
                            xs = np.linspace(xmin, xmax, opts["kde_diag"]["bins"])
                            ys = density(xs)
                            h = plt.plot(xs, ys, color=opts["samples_colors"][n],)
                        else:
                            pass

                if len(points) > 0:
                    extent = ax.get_ylim()
                    for n, v in enumerate(points):
                        h = plt.plot(
                            [v[:, row], v[:, row]],
                            extent,
                            color=opts["points_colors"][n],
                            **opts["points_diag"]
                        )

            # Off-diagonals
            else:

                if len(samples) > 0:
                    for n, v in enumerate(samples):
                        if opts["upper"][n] == "hist" or opts["upper"][n] == "hist2d":
                            # h = plt.hist2d(
                            #     v[:, col], v[:, row],
                            #     range=(
                            #         [limits[col][0], limits[col][1]],
                            #         [limits[row][0], limits[row][1]]),
                            #     **opts['hist_offdiag']
                            #     )
                            hist, xedges, yedges = np.histogram2d(
                                v[:, col],
                                v[:, row],
                                range=[
                                    [limits[col][0], limits[col][1]],
                                    [limits[row][0], limits[row][1]],
                                ],
                                **opts["hist_offdiag"]
                            )
                            h = plt.imshow(
                                hist.T,
                                origin="lower",
                                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1],],
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
                                    Z = probs2contours(
                                        Z, opts["contour_offdiag"]["levels"]
                                    )
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
                                **opts["scatter_offdiag"]
                            )
                        elif opts["upper"][n] == "plot":
                            h = plt.plot(
                                v[:, col],
                                v[:, row],
                                color=opts["samples_colors"][n],
                                **opts["plot_offdiag"]
                            )
                        else:
                            pass

                if len(points) > 0:

                    for n, v in enumerate(points):
                        h = plt.plot(
                            v[:, col],
                            v[:, row],
                            color=opts["points_colors"][n],
                            **opts["points_offdiag"]
                        )

    if len(subset) < dim:
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
                    **text_kwargs
                )

    return fig, axes
