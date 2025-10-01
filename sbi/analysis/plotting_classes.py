# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import matplotlib as mpl
from matplotlib import pyplot as plt


@dataclass(frozen=True)
class DiagOptions:
    """
    Base class for keyword arguments used in diagonal plots.

    This class serves as a common parent for specific diagonal plot
    configuration classes such as KDE, Histogram, and Scatter.
    """

    mpl_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KdeDiagOptions(DiagOptions):
    bw_method: str = "scott"
    bins: int = 50


@dataclass(frozen=True)
class HistDiagOptions(DiagOptions):
    bin_heuristic: str = "Freedman-Diaconis"

    def __post_init__(self):
        mpl_kwargs_defaults = {
            "density": False,
            "histtype": "step",
        }
        updated = {**mpl_kwargs_defaults, **self.mpl_kwargs}
        object.__setattr__(self, "mpl_kwargs", updated)


@dataclass(frozen=True)
class ScatterDiagOptions(DiagOptions): ...


@dataclass(frozen=True)
class OffDiagOptions:
    """
    Base class for keyword arguments used in off-diagonal plots.

    This class serves as a common parent for off-diagonal plot types like KDE,
    scatter, histogram, contour, and line plots.
    """

    mpl_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KdeOffDiagOptions(OffDiagOptions):
    bw_method: str = "scott"
    bins: int = 50

    def __post_init__(self):
        mpl_kwargs_defaults = {"cmap": "viridis", "origin": "lower", "aspect": "auto"}
        updated = {**mpl_kwargs_defaults, **self.mpl_kwargs}
        object.__setattr__(self, "mpl_kwargs", updated)


@dataclass(frozen=True)
class HistOffDiagOptions(OffDiagOptions):
    bin_heuristic: Optional[str] = None
    np_hist_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        mpl_kwargs_defaults = {"cmap": "viridis", "origin": "lower", "aspect": "auto"}
        updated = {**mpl_kwargs_defaults, **self.mpl_kwargs}
        object.__setattr__(self, "mpl_kwargs", updated)

        np_hist_defaults = {"bins": 50, "density": False}
        updated = {**np_hist_defaults, **self.np_hist_kwargs}
        object.__setattr__(self, "np_hist_kwargs", updated)


@dataclass(frozen=True)
class ScatterOffDiagOptions(OffDiagOptions):
    def __post_init__(self):
        mpl_kwargs_defaults = {
            "edgecolor": "white",
            "alpha": 0.5,
            "rasterized": False,
        }
        updated = {**mpl_kwargs_defaults, **self.mpl_kwargs}
        object.__setattr__(self, "mpl_kwargs", updated)


@dataclass(frozen=True)
class ContourOffDiagOptions(OffDiagOptions):
    bw_method: str = "scott"
    bins: int = 50
    percentile: bool = True
    levels: list = field(default_factory=lambda: [0.68, 0.95, 0.99])


@dataclass(frozen=True)
class PlotOffDiagOptions(OffDiagOptions):
    def __post_init__(self):
        mpl_kwargs_defaults = {"aspect": "auto"}
        updated = {**mpl_kwargs_defaults, **self.mpl_kwargs}
        object.__setattr__(self, "mpl_kwargs", updated)


@dataclass(frozen=True)
class FigOptions:
    """
    Configuration options for customizing Matplotlib figures.

    Each attribute corresponds to a common Matplotlib API.
    """

    legend: bool = False
    """Whether to display the legend."""

    legend_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Keyword arguments passed to `Axes.legend`
    See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html"""

    # labels
    points_labels: List[str] = field(
        default_factory=lambda: [f"points_{idx}" for idx in range(10)]
    )
    """Default labels for plotted points (used with `label=` argument).
    See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html"""

    samples_labels: List[str] = field(
        default_factory=lambda: [f"samples_{idx}" for idx in range(10)]
    )
    """Default labels for plotted samples (used with `label=` argument).
    See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html"""

    samples_colors: List[str] = field(
        default_factory=lambda: plt.rcParams["axes.prop_cycle"].by_key()["color"][0::2]
    )
    """Colors for samples, taken from the even indices of the default color cycle.
    See: https://matplotlib.org/stable/tutorials/introductory/customizing.html#the-default-matplotlibrc-file"""

    points_colors: List[str] = field(
        default_factory=lambda: plt.rcParams["axes.prop_cycle"].by_key()["color"][1::2]
    )
    """Colors for points, taken from the odd indices of the default color cycle.
    See: https://matplotlib.org/stable/tutorials/introductory/customizing.html#the-default-matplotlibrc-file"""

    # ticks
    tickformatter: Any = field(
        default_factory=lambda: mpl.ticker.FormatStrFormatter("%g")  # type: ignore
    )
    """Set the formatter of the major ticker.
    See: https://matplotlib.org/stable/api/ticker_api.html#tick-formatting"""

    tick_labels: Optional[Any] = None
    """Optional custom tick labels for axes.
    See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xticklabels.html"""

    # formatting points (scale, markers)
    points_diag: Dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for formatting points on diagonal plots.
    See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html"""

    points_offdiag: Dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for formatting points on off-diagonal plots.

    Defaults: {"marker": ".", "markersize": 10}

    See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html"""

    # other options
    fig_bg_colors: Dict[str, Any] = field(default_factory=dict)
    """Background colors of the subplots.
      Keys should only be `upper`, `lower` or `diag`."""

    fig_subplots_adjust: Dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for adjusting the subplot layout.

    Defaults: {"top": 0.9}

    See: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots_adjust.html"""

    subplots: Dict[str, Any] = field(default_factory=dict)
    """Keyword arguments passed to `plt.subplots`, besides the `figsize` argument,
       which is set through the plotting functions.

    See: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib.pyplot.figure"""

    despine: Dict[str, int] = field(default_factory=lambda: dict(offset=5))
    """Options for adjusting the plot spines.

    Currently, only the ``{"offset": int}`` field is supported, which specifies how far
    to move the spine away outward from the bottom by the specified number of points.

    See: https://matplotlib.org/stable/api/spines_api.html#matplotlib.spines.Spine.set_position
    """

    title: Optional[str] = None
    """Title of the figure.
    See: https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.suptitle.html"""

    title_format: Dict[str, Any] = field(default_factory=dict)
    """Formatting options for the title.

    Defaults: {"fontsize": 16}

    See: https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.suptitle.html"""

    x_lim_add_eps: float = 1e-5
    """Extra margin decreased from the left and added to the right of x-axis limits for
        diagonal plots.
    See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlim.html"""

    square_subplots: bool = True
    """Whether to force subplots to be square."""

    def __post_init__(self):
        """Post-initialization method for setting default plotting options."""

        # sbi plotting figure default values
        fig_options_defaults = {
            "title_format": {"fontsize": 16},
            "fig_subplots_adjust": {"top": 0.9},
            "points_offdiag": {"marker": ".", "markersize": 10},
            "fig_bg_colors": {"upper": None, "diag": None, "lower": None},
        }

        for field_name in fig_options_defaults:
            default_value = fig_options_defaults[field_name]
            current_value = getattr(self, field_name)

            # Merge defaults with current values
            updated = {**default_value, **current_value}

            # Update fields
            object.__setattr__(self, field_name, updated)


def _set_color(i: int) -> str:
    """
    Returns a distinct color from Matplotlib's default color cycle.

    Args:
        i: Index used to select a color, spaced by 2.

    Returns:
        str: A color string from the Matplotlib color cycle.
    """

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # Take modulo to avoid IndexError for large i value
    new_color = colors[(i * 2) % len(colors)]
    return new_color


def get_default_offdiag_kwargs(offdiag: Optional[str], i: int = 0) -> Dict[str, Any]:
    """Get default offdiag kwargs."""

    if offdiag == "kde" or offdiag == "kde2d":
        offdiag_options = KdeOffDiagOptions()
    elif offdiag == "hist" or offdiag == "hist2d":
        offdiag_options = HistOffDiagOptions()
    elif offdiag == "scatter":
        offdiag_options = ScatterOffDiagOptions(mpl_kwargs=dict(color=_set_color(i)))
    elif offdiag == "contour" or offdiag == "contourf":
        offdiag_options = ContourOffDiagOptions(mpl_kwargs=dict(colors=_set_color(i)))
    elif offdiag == "plot":
        offdiag_options = PlotOffDiagOptions(mpl_kwargs=dict(color=_set_color(i)))
    else:
        return {}
    return asdict(offdiag_options)


def get_default_diag_kwargs(diag: Optional[str], i: int = 0) -> Dict[str, Any]:
    """Get default diag kwargs."""

    if diag == "kde":
        diag_options = KdeDiagOptions(mpl_kwargs=dict(color=_set_color(i)))
    elif diag == "hist":
        diag_options = HistDiagOptions(mpl_kwargs=dict(color=_set_color(i)))
    elif diag == "scatter":
        diag_options = ScatterDiagOptions(mpl_kwargs=dict(color=_set_color(i)))
    else:
        return {}
    return asdict(diag_options)
