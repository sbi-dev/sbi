from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional

import matplotlib as mpl
from matplotlib import pyplot as plt
from typing_extensions import Self


@dataclass(frozen=True)
class MplKwargs:
    color: str = field(
        default_factory=lambda: plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    )

    def set_color(self, i: int = 0) -> Self:
        new_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i * 2]
        return replace(self, color=new_color)


@dataclass(frozen=True)
class KdeMplKwargs(MplKwargs): ...


@dataclass(frozen=True)
class HistMplKwargs(MplKwargs):
    density: bool = False
    histtype: str = "step"


@dataclass(frozen=True)
class ScatterMplKwargs(MplKwargs): ...


@dataclass(frozen=True)
class KdeOffMplKwargs:
    cmap: str = "viridis"
    origin: str = "lower"
    aspect: str = "auto"


@dataclass(frozen=True)
class HistOffMplKwargs:
    cmap: str = "viridis"
    origin: str = "lower"
    aspect: str = "auto"


@dataclass(frozen=True)
class ScatterOffMplKwargs(MplKwargs):
    edgecolor: str = "white"
    alpha: float = 0.5
    rasterized: bool = False


@dataclass(frozen=True)
class ContourOffMplKwargs(MplKwargs): ...


@dataclass(frozen=True)
class PlotOffMplKwargs(MplKwargs):
    aspect: str = "auto"


@dataclass(frozen=True)
class DiagKwargs: ...


@dataclass(frozen=True)
class KdeDiagKwargs(DiagKwargs):
    bw_method: str = "scott"
    bins: int = 50
    mpl_kwargs: KdeMplKwargs = field(default_factory=lambda: KdeMplKwargs())


@dataclass(frozen=True)
class HistDiagKwargs(DiagKwargs):
    bin_heuristic: str = "Freedman-Diaconis"
    mpl_kwargs: HistMplKwargs = field(default_factory=lambda: HistMplKwargs())


@dataclass(frozen=True)
class ScatterDiagKwargs(DiagKwargs):
    mpl_kwargs: ScatterMplKwargs = field(default_factory=lambda: ScatterMplKwargs())


@dataclass(frozen=True)
class NpHistKwargs:
    bins: int = 50
    density: bool = False


@dataclass(frozen=True)
class OffDiagKwargs: ...


@dataclass(frozen=True)
class KdeOffDiagKwargs(OffDiagKwargs):
    bw_method: str = "scott"
    bins: int = 50
    mpl_kwargs: KdeOffMplKwargs = field(default_factory=lambda: KdeOffMplKwargs())


@dataclass(frozen=True)
class HistOffDiagKwargs(OffDiagKwargs):
    bin_heuristic = None
    np_hist_kwargs: NpHistKwargs = field(default_factory=lambda: NpHistKwargs())
    mpl_kwargs: HistOffMplKwargs = field(default_factory=lambda: HistOffMplKwargs())


@dataclass(frozen=True)
class ScatterOffDiagKwargs(OffDiagKwargs):
    mpl_kwargs: ScatterOffMplKwargs = field(
        default_factory=lambda: ScatterOffMplKwargs()
    )


@dataclass(frozen=True)
class ContourOffDiagKwargs(OffDiagKwargs):
    bw_method: str = "scott"
    bins: int = 50
    percentile: bool = True
    levels: list = field(default_factory=lambda: [0.68, 0.95, 0.99])
    mpl_kwargs: ContourOffMplKwargs = field(
        default_factory=lambda: ContourOffMplKwargs()
    )


@dataclass(frozen=True)
class PlotOffDiagKwargs(OffDiagKwargs):
    mpl_kwargs: PlotOffMplKwargs = field(default_factory=lambda: PlotOffMplKwargs())


@dataclass(frozen=True)
class Despine:
    offset: int = 5


@dataclass(frozen=True)
class TitleFormat:
    fontsize: int = 16


@dataclass(frozen=True)
class SubplotAdjust:
    top: float = 0.9


@dataclass(frozen=True)
class PointsOffDiag:
    marker: str = "."
    markersize: int = 10


@dataclass(frozen=True)
class FigBgColors:
    offdiag: Optional[Any] = None
    diag: Optional[Any] = None
    lower: Optional[Any] = None


@dataclass(frozen=True)
class FigKwargs:
    legend: Optional[str] = None
    legend_kwargs: Dict[str, Any] = field(default_factory=dict)
    # labels
    points_labels: List[str] = field(
        default_factory=lambda: [f"points_{idx}" for idx in range(10)]
    )  # for points
    samples_labels: List[str] = field(
        default_factory=lambda: [f"samples_{idx}" for idx in range(10)]
    )  # for samples
    # colors: take even colors for samples, odd colors for points
    samples_colors: List[str] = field(
        default_factory=lambda: plt.rcParams["axes.prop_cycle"].by_key()["color"][0::2]
    )  # pyright: ignore[reportOptionalMemberAccess]
    points_colors: List[str] = field(
        default_factory=lambda: plt.rcParams["axes.prop_cycle"].by_key()["color"][1::2]
    )  # pyright: ignore[reportOptionalMemberAccess]
    # ticks
    tickformatter: Any = field(
        default_factory=lambda: mpl.ticker.FormatStrFormatter("%g")  # type: ignore
    )
    tick_labels: Optional[Any] = None
    # formatting points (scale, markers)
    points_diag: Dict[str, Any] = field(default_factory=dict)
    points_offdiag: PointsOffDiag = field(default_factory=PointsOffDiag)
    # other options
    fig_bg_colors: FigBgColors = field(default_factory=FigBgColors)
    fig_subplots_adjust: SubplotAdjust = field(default_factory=SubplotAdjust)
    subplots: Dict[str, Any] = field(default_factory=dict)
    despine: Despine = field(default_factory=Despine)
    title: Optional[str] = None
    title_format: TitleFormat = field(default_factory=TitleFormat)
    x_lim_add_eps: float = 1e-5
    square_subplots: bool = True


def get_default_offdiag_kwargs(offdiag: Optional[str], i: int = 0) -> Dict:
    """Get default offdiag kwargs."""

    if offdiag == "kde" or offdiag == "kde2d":
        offdiag_kwargs = KdeOffDiagKwargs()
    elif offdiag == "hist" or offdiag == "hist2d":
        offdiag_kwargs = HistOffDiagKwargs()
    elif offdiag == "scatter":
        offdiag_kwargs = ScatterOffDiagKwargs(
            mpl_kwargs=ScatterOffMplKwargs().set_color(i)
        )
    elif offdiag == "contour" or offdiag == "contourf":
        offdiag_kwargs = ContourOffDiagKwargs(
            mpl_kwargs=ContourOffMplKwargs().set_color(i)
        )
    elif offdiag == "plot":
        offdiag_kwargs = PlotOffDiagKwargs(mpl_kwargs=PlotOffMplKwargs().set_color(i))
    else:
        return {}
    return asdict(offdiag_kwargs)


def get_default_diag_kwargs(diag: Optional[str], i: int = 0) -> Dict:
    """Get default diag kwargs."""

    if diag == "kde":
        diag_kwargs = KdeDiagKwargs(mpl_kwargs=KdeMplKwargs().set_color(i))
    elif diag == "hist":
        diag_kwargs = HistDiagKwargs(mpl_kwargs=HistMplKwargs().set_color(i))
    elif diag == "scatter":
        diag_kwargs = ScatterDiagKwargs(mpl_kwargs=ScatterMplKwargs().set_color(i))
    else:
        return {}
    return asdict(diag_kwargs)
