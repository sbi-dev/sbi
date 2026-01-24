import shutil
from pathlib import Path

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import close
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.analysis.tensorboard_output import plot_summary


def test_plot_summary_grouped(tmp_path):
    log_dir = tmp_path / "logs"
    writer = SummaryWriter(log_dir)
    for i in range(10):
        writer.add_scalar("loss/train", np.exp(-i), i)
        writer.add_scalar("loss/val", np.exp(-i) + 0.1, i)
        writer.add_scalar("acc", i * 0.1, i)
    writer.close()

    # Test single tag
    fig, axes = plot_summary(log_dir, tags=["loss/train"])
    assert len(axes) == 1
    close()

    # Test grouped tags
    fig, axes = plot_summary(log_dir, tags=[["loss/train", "loss/val"], "acc"])
    assert len(axes) == 2
    assert axes[0].get_legend() is not None
    close()

    # Test grouped tags with ylabel
    fig, axes = plot_summary(
        log_dir,
        tags=[["loss/train", "loss/val"]],
        ylabel=["Custom Loss"]
    )
    assert axes[0].get_ylabel() == "Custom Loss"
    close()

    # Test titles
    fig, axes = plot_summary(
        log_dir,
        tags=["loss/train"],
        title="Figure Title",
        titles=["Subplot Title"]
    )
    assert fig._suptitle.get_text() == "Figure Title"
    assert axes[0].get_title() == "Subplot Title"
    close()
