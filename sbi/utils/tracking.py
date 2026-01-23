# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Any, Optional

from torch.utils.tensorboard.writer import SummaryWriter

from sbi.sbi_types import Tracker


class TensorBoardTracker:
    """Adapter for TensorBoard SummaryWriter."""

    def __init__(self, summary_writer: SummaryWriter) -> None:
        self._writer = summary_writer

    @property
    def log_dir(self) -> str:
        return self._writer.log_dir

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        self._writer.add_scalar(tag=name, scalar_value=value, global_step=step)

    def log_metrics(
        self, metrics: dict[str, float], step: Optional[int] = None
    ) -> None:
        for name, value in metrics.items():
            self.log_metric(name=name, value=value, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        for name, value in params.items():
            self._writer.add_text(tag=f"params/{name}", text_string=str(value))

    def add_figure(self, name: str, figure: Any, step: Optional[int] = None) -> None:
        self._writer.add_figure(tag=name, figure=figure, global_step=step)

    def flush(self) -> None:
        self._writer.flush()


__all__ = [
    "TensorBoardTracker",
    "Tracker",
]
