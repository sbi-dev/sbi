# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
"""Utils for processing tensorboard event data."""

import inspect
import logging
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure, FigureBase
from tensorboard.backend.event_processing.event_accumulator import (
    DEFAULT_SIZE_GUIDANCE,
    EventAccumulator,
)

from sbi.analysis.plot import _get_default_opts
from sbi.inference import NeuralInference
from sbi.utils.io import get_log_root


def plot_summary(
    trainer: Union[NeuralInference, Path, None] = None,
    tags: Optional[List[str]] = None,
    *,
    overlay: bool = False,
    colors: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    figsize: Sequence[int] = (20, 6),
    fontsize: float = 12,
    xlabel: str = "epochs_trained",
    ylabel: Optional[List[str]] = None,
    fig: Optional[FigureBase] = None,
    axes: Optional[Axes] = None,
    plot_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    tensorboard_scalar_limit: int = 10_000,
    inference: Union[NeuralInference, Path, None] = None,
    disable_tensorboard_prompt: Optional[bool] = None,
) -> Tuple[Figure, Axes]:
    """Plot scalar data logged by the TensorBoard tracker of a trainer.

    Args:
        trainer: trainer object (``NPE``/``NLE``/``NRE``/``NPSE``/``FMPE``/...)
            whose tracker exposes a ``log_dir``, or a ``Path`` to a tensorboard
            log directory.
        tags: tensorboard tags to visualize. Defaults to ``["validation_loss"]``.
        overlay: if True, plots all ``tags`` on a single axes (useful for
            comparing training vs validation loss). Otherwise one subplot per tag.
        colors: per-tag line colors. ``None`` uses matplotlib's defaults. Must
            have the same length as ``tags`` if provided.
        labels: per-tag legend labels. ``None`` uses tag names. Must have the
            same length as ``tags`` if provided.
        figsize: figure size in inches.
        fontsize: fontsize for axis ticks and labels.
        xlabel: x-axis label.
        ylabel: per-tag y-axis labels. ``None`` uses tag names. Must have the
            same length as ``tags`` if provided.
        fig: existing figure instance to plot into. If ``None``, creates one.
        axes: existing axes to plot into. If ``None``, creates them.
        plot_kwargs: forwarded to ``ax.plot()``. ``colors`` and ``labels``
            (when set) take precedence over ``"color"`` / ``"label"`` here.
        verbose: if True (default), log the tensorboard launch hint and the
            list of valid tags.
        tensorboard_scalar_limit: max number of scalars loaded per tag.
        inference: deprecated alias for ``trainer``. Will be removed in a
            future release.
        disable_tensorboard_prompt: deprecated, use ``verbose`` instead
            (note the polarity flip). Will be removed in a future release.

    Returns:
        ``(fig, axes)`` for further composition (e.g. ``axes[0].set_title(...)``,
        ``axes[0].grid(True)``).

    Examples:
        Default — plot validation loss::

            fig, axes = plot_summary(trainer)

        Compare training vs validation loss in a single panel::

            fig, axes = plot_summary(
                trainer,
                tags=["training_loss", "validation_loss"],
                overlay=True,
                colors=["C0", "C1"],
                labels=["train", "val"],
            )

        Plot from a log directory directly::

            fig, axes = plot_summary(Path("./sbi-logs/NPE_C/2026_04_26_12_00_00"))

        Compose with matplotlib after the call::

            fig, axes = plot_summary(trainer, overlay=True)
            axes[0].set_title("Training progress")
            axes[0].grid(True)
    """
    logger = logging.getLogger(__name__)

    # Deprecation shims for renamed kwargs.
    if inference is not None:
        if trainer is not None:
            raise TypeError(
                "Pass either `trainer` or `inference` (deprecated), not both."
            )
        warnings.warn(
            "`inference` is deprecated and will be removed in a future release; "
            "use `trainer` instead.",
            FutureWarning,
            stacklevel=2,
        )
        trainer = inference
    if trainer is None:
        raise TypeError("plot_summary() missing required argument: 'trainer'")
    if disable_tensorboard_prompt is not None:
        warnings.warn(
            "`disable_tensorboard_prompt` is deprecated and will be removed in a "
            "future release; use `verbose` instead (note the polarity flip).",
            FutureWarning,
            stacklevel=2,
        )
        verbose = not disable_tensorboard_prompt

    if tags is None:
        tags = ["validation_loss"]

    for name, vals in (("colors", colors), ("labels", labels), ("ylabel", ylabel)):
        if vals is not None and len(vals) != len(tags):
            raise ValueError(
                f"`{name}` must have the same length as `tags` "
                f"(got {len(vals)} vs {len(tags)})."
            )

    size_guidance = deepcopy(DEFAULT_SIZE_GUIDANCE)
    size_guidance.update(scalars=tensorboard_scalar_limit)

    if isinstance(trainer, NeuralInference):
        log_dir = getattr(trainer._tracker, "log_dir", None)
        if log_dir is None:
            raise ValueError(
                "Trainer's tracker does not expose a log_dir. "
                "Use a TensorBoard tracker or pass a log directory directly."
            )
    elif isinstance(trainer, Path):
        log_dir = trainer
    else:
        raise ValueError(f"trainer {trainer}")

    all_event_data = _get_event_data_from_log_dir(log_dir, size_guidance)
    scalars = all_event_data["scalars"]

    if verbose:
        logger.warning(
            (
                "For an interactive, detailed view of the summary, launch tensorboard "
                f" with 'tensorboard --logdir={log_dir}' from a"
                " terminal on your machine, visit http://127.0.0.1:6006 afterwards."
                " Requires port forwarding if tensorboard runs on a remote machine, as"
                " e.g. https://stackoverflow.com/a/42445070/7770835 explains.\n"
            )
        )
        logger.warning(f"Valid tags are: {sorted(list(scalars.keys()))}.")

    _check_tags(scalars, tags)

    if len(scalars[tags[0]]["step"]) == tensorboard_scalar_limit:
        logger.warning(
            (
                "Event data as large as the chosen limit for tensorboard scalars."
                "Tensorboard might be subsampling your data, as "
                "https://stackoverflow.com/a/65564389/7770835 explains."
                " Consider increasing tensorboard_scalar_limit to see all data.\n"
            )
        )

    plot_options = _get_default_opts()
    plot_options.update(figsize=figsize, fontsize=fontsize)

    if fig is None or axes is None:
        num_subplots = 1 if overlay else len(tags)
        fig, axes = plt.subplots(  # pyright: ignore[reportAssignmentType]
            1,
            num_subplots,
            figsize=plot_options["figsize"],
            **plot_options["subplots"],
        )
    axes = np.atleast_1d(axes)  # type: ignore
    assert fig is not None and axes is not None

    _labels = labels if labels is not None else tags
    _ylabel = ylabel if ylabel is not None else tags
    user_kwargs = plot_kwargs or {}

    # Build (axis, [(tag_index, tag), ...]) pairs so a single loop handles both
    # overlay and per-tag-subplot modes.
    if overlay:
        subplot_specs = [(axes[0], list(enumerate(tags)))]
    else:
        subplot_specs = [(axes[i], [(i, tag)]) for i, tag in enumerate(tags)]

    for ax, tag_items in subplot_specs:
        for i, tag in tag_items:
            # Precedence: explicit `colors`/`labels` > plot_kwargs > matplotlib default.
            kw = {**user_kwargs, "label": _labels[i]}
            if colors is not None:
                kw["color"] = colors[i]
            ax.plot(scalars[tag]["step"], scalars[tag]["value"], **kw)

        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(_resolve_ylabel(tag_items, _ylabel, overlay), fontsize=fontsize)
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)

        if labels is not None or (overlay and len(tags) > 1):
            ax.legend(fontsize=fontsize)

    plt.subplots_adjust(wspace=0.3)

    return fig, axes  # type: ignore


def _resolve_ylabel(
    tag_items: List[Tuple[int, str]], ylabels: List[str], overlay: bool
) -> str:
    """Return the ylabel for one subplot.

    Non-overlay: the single tag's ylabel. Overlay: distinct ylabels joined
    by ``" / "`` (or the single label if all tags share it).
    """
    labels_for_subplot = [ylabels[i] for i, _ in tag_items]
    if not overlay or len(set(labels_for_subplot)) == 1:
        return labels_for_subplot[0]
    return " / ".join(labels_for_subplot)


def list_all_logs(trainer: NeuralInference) -> List:
    """Returns a list of all log dirs for a trainer class."""
    method = trainer.__class__.__name__
    log_dir = Path(get_log_root()) / method
    return sorted(log_dir.iterdir())


def _get_event_data_from_log_dir(
    log_dir: Union[str, Path], size_guidance=DEFAULT_SIZE_GUIDANCE
) -> Dict[str, Dict[str, Dict[str, List[Any]]]]:
    """All event data stored by tensorboards summary writer as nested dictionary.

    The event data is stripped off from their native tensorboard event types and
    represented in a tabular way, i.e. Dict[str, List].

    The hierarchy of the dictionary is:
        1. tag type: event types that can be logged with tensorboard like 'scalars',
           'images', 'histograms', etc.
        2. tag: tag for the event type that the user of the SummaryWriter specifies.
        3. tag type attribute: attribute of the event.

    Args:
        log_dir: log dir of a tensorboard summary writer.
        size_guidance: to avoid causing out of memory errors by loading too much data at
            once into memory. Defaults to tensorboards default size_guidance.

    Returns a nested, exhaustive dictionary of all event data under log_dir.

    Based on: https://stackoverflow.com/a/45899735/7770835
    """

    event_acc = _get_event_accumulator(log_dir, size_guidance)

    all_event_data = {}
    # tensorboard logs different event types, like scalars, images, histograms etc.
    for tag_type, list_of_tags in event_acc.Tags().items():
        all_event_data[tag_type] = {}

        if list_of_tags:
            for tag in list_of_tags:
                all_event_data[tag_type][tag] = {}

                # to retrieve the data from the EventAccumulator as in
                # event_acc.Scalars('epochs_trained')
                _getter_fn = getattr(event_acc, tag_type.capitalize())
                data = _getter_fn(tag)

                # ScalarEvent has three attributes, wall_time, step, and value
                # a generic way to get data from all other EventType as for ScalarEvent,
                # we inspect their argument signature. These events are named tuples
                # that can be found here:
                # https://github.com/tensorflow/tensorboard/blob/b84f3738032277894c6f3fd3e011f032a89d002c/tensorboard/backend/event_processing/event_accumulator.py#L37
                # When looping over `inspect.getfullargspec()` there is also a `self`
                # attribute.
                _type = type(data[0])
                for attribute in inspect.getfullargspec(_type).args:
                    if not attribute.startswith("_") and attribute != "self":
                        if attribute not in all_event_data[tag_type][tag]:
                            all_event_data[tag_type][tag][attribute] = []
                        for datapoint in data:
                            all_event_data[tag_type][tag][attribute].append(
                                getattr(datapoint, attribute)
                            )
    return all_event_data


def _get_event_accumulator(
    log_dir: Union[str, Path], size_guidance: Dict = DEFAULT_SIZE_GUIDANCE
) -> EventAccumulator:
    """Returns the tensorboard EventAccumulator instance for a log dir."""
    event_acc = EventAccumulator(str(log_dir), size_guidance=size_guidance)
    event_acc.Reload()
    return event_acc


def _check_tags(adict: Dict, tags: List[str]) -> None:
    """Checks if tags are present in a dict."""
    for tag in tags:
        if tag not in adict:
            raise KeyError(
                f"'{tag}' is not a valid tag of the tensorboard SummaryWriter. "
                f"Valid tags are: {list(adict.keys())}."
            )


def _remove_all_logs(path: Path) -> None:
    """Removes all logs in path/sbi-logs."""
    if (path / "sbi-logs").exists():
        import shutil

        shutil.rmtree(path / "sbi-logs")
