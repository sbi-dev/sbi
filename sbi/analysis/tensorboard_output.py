# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
"""Utils for processing tensorboard event data."""

import inspect
import logging
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

# creating an alias for annotating, because sbi.inference.base.NeuralInference creates
# a circular import error
_NeuralInference = Any


def plot_summary(
    inference: Union[_NeuralInference, Path],
    tags: Optional[List[str]] = None,
    disable_tensorboard_prompt: bool = False,
    tensorboard_scalar_limit: int = 10_000,
    figsize: Sequence[int] = (20, 6),
    fontsize: float = 12,
    fig: Optional[FigureBase] = None,
    axes: Optional[Axes] = None,
    xlabel: str = "epochs_trained",
    ylabel: Optional[str] = None,                   #single ylabel for joint plot
    plot_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Figure, Axes]:
    """Plots data logged by the tensorboard summary writer of an inference object.

    Args:
        inference: inference object that holds a ._summary_writer.log_dir attribute.
            Optionally the log_dir itself.
        tags: list of summery writer tags to visualize.
        disable_tensorboard_prompt: flag to disable the logging of how to run
            tensorboard and valid tags. Default is False.
        tensorboard_scalar_limit: overriding DEFAULT_SIZE_GUIDANCE.
        figsize: determines the figure size. Defaults is [6, 6].
        fontsize: determines the fontsize of axes ticks and labels. Default is 12.
        fig: optional existing figure instance.
        axes: optional existing axes instance.
        xlabel: x-axis label describing 'steps' attribute of tensorboards ScalarEvent.
        ylabel: list of alternative ylabels for items in tags. Optional.
        plot_kwargs: will be passed to ax.plot.

    Returns a tuple of Figure and Axes objects.
    """
    logger = logging.getLogger(__name__)

    if tags is None:
        tags = ["validation_loss"]

    size_guidance = deepcopy(DEFAULT_SIZE_GUIDANCE)
    size_guidance.update(scalars=tensorboard_scalar_limit)

    
    # Resolve log directory
    if isinstance(inference, NeuralInference):
        log_dir = inference._summary_writer.log_dir
    elif isinstance(inference, Path):
        log_dir = inference
    else:
        raise ValueError(f"inference {inference}")

    all_event_data = _get_event_data_from_log_dir(log_dir, size_guidance)
    scalars = all_event_data["scalars"]

    if not disable_tensorboard_prompt:
        logger.warning(
            (
                "For an interactive, detailed view of the summary, launch tensorboard "
                f" with 'tensorboard --logdir={log_dir}'.\n"
            )
        )
        logger.warning(f"Valid tags are: {sorted(list(scalars.keys()))}.")

    _check_tags(scalars, tags)

    # single axis instead of multiple subplots
    if fig is None or axes is None:
        fig, axes = plt.subplots(figsize=figsize)

    ax = axes  # use a single axis

    plot_kwargs = plot_kwargs or {}

    for tag in tags:
        ax.plot(
            scalars[tag]["step"],
            scalars[tag]["value"],
            label=tag.replace("_", " ").title(),  # clean legend names
            **plot_kwargs,
        )

    # Labels and styling
    ax.set_ylabel(ylabel or "Value", fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)

    ax.legend(fontsize=fontsize)  # legend for UX
    ax.grid(True, alpha=0.3)      # grid improves readability


    return fig, axes  # type: ignore


def list_all_logs(inference: _NeuralInference) -> List:
    """Returns a list of all log dirs for an inference class."""
    method = inference.__class__.__name__
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
