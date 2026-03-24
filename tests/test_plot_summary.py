"""Tests for the enhanced plot_summary function."""

from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest

from sbi.analysis.tensorboard_output import plot_summary


@pytest.fixture
def mock_scalars():
    """Mock scalar data mimicking tensorboard event data."""
    return {
        "training_loss": {
            "step": list(range(100)),
            "value": [1.0 / (i + 1) for i in range(100)],
        },
        "validation_loss": {
            "step": list(range(100)),
            "value": [1.2 / (i + 1) for i in range(100)],
        },
    }


@pytest.fixture
def mock_inference(mock_scalars):
    """Patch event data loading so we don't need real tensorboard logs."""
    with patch(
        "sbi.analysis.tensorboard_output._get_event_data_from_log_dir"
    ) as mock_get:
        mock_get.return_value = {"scalars": mock_scalars}
        from pathlib import Path

        yield Path("/fake/log/dir")


class TestPlotSummaryBackwardCompat:
    """Existing behavior should not change."""

    def test_single_tag(self, mock_inference):
        fig, axes = plot_summary(
            mock_inference,
            tags=["training_loss"],
            disable_tensorboard_prompt=True,
        )
        assert axes.shape == (1,)
        plt.close(fig)

    def test_multiple_tags_separate_subplots(self, mock_inference):
        fig, axes = plot_summary(
            mock_inference,
            tags=["training_loss", "validation_loss"],
            disable_tensorboard_prompt=True,
        )
        assert axes.shape == (2,)
        plt.close(fig)


class TestPlotSummaryOverlay:
    """New overlay functionality."""

    def test_overlay_creates_single_axes(self, mock_inference):
        fig, axes = plot_summary(
            mock_inference,
            tags=["training_loss", "validation_loss"],
            overlay=True,
            disable_tensorboard_prompt=True,
        )
        assert axes.shape == (1,)
        # Should have 2 lines on the single axes
        assert len(axes[0].get_lines()) == 2
        plt.close(fig)

    def test_overlay_with_colors(self, mock_inference):
        fig, axes = plot_summary(
            mock_inference,
            tags=["training_loss", "validation_loss"],
            overlay=True,
            colors=["blue", "orange"],
            disable_tensorboard_prompt=True,
        )
        lines = axes[0].get_lines()
        assert lines[0].get_color() == "blue"
        assert lines[1].get_color() == "orange"
        plt.close(fig)

    def test_overlay_with_labels(self, mock_inference):
        fig, axes = plot_summary(
            mock_inference,
            tags=["training_loss", "validation_loss"],
            overlay=True,
            labels=["Train", "Val"],
            disable_tensorboard_prompt=True,
        )
        legend = axes[0].get_legend()
        assert legend is not None
        texts = [t.get_text() for t in legend.get_texts()]
        assert texts == ["Train", "Val"]
        plt.close(fig)

    def test_overlay_legend_disabled(self, mock_inference):
        fig, axes = plot_summary(
            mock_inference,
            tags=["training_loss", "validation_loss"],
            overlay=True,
            legend=False,
            disable_tensorboard_prompt=True,
        )
        assert axes[0].get_legend() is None
        plt.close(fig)


class TestPlotSummaryGrid:
    def test_grid_enabled(self, mock_inference):
        fig, axes = plot_summary(
            mock_inference,
            tags=["training_loss"],
            grid=True,
            disable_tensorboard_prompt=True,
        )
        assert axes[0].xaxis.get_gridlines()[0].get_visible()
        plt.close(fig)


class TestPlotSummaryTitle:
    def test_single_title(self, mock_inference):
        fig, axes = plot_summary(
            mock_inference,
            tags=["training_loss"],
            title="My Plot",
            disable_tensorboard_prompt=True,
        )
        assert axes[0].get_title() == "My Plot"
        plt.close(fig)
