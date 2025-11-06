# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

# SPDX-FileCopyrightText: 2023-2025 sbi-dev
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Optional

from torch.utils.tensorboard.writer import SummaryWriter

from .base import BaseLogger


class TensorBoardLogger(BaseLogger):
    """Adapter that wraps a TensorBoard SummaryWriter to match BaseLogger."""

    def __init__(
        self, writer: Optional[SummaryWriter] = None, **summary_writer_kwargs: Any
    ) -> None:
        self._writer = (
            writer if writer is not None else SummaryWriter(**summary_writer_kwargs)
        )

    # Delegate common SummaryWriter APIs
    def add_scalar(
        self, tag: str, scalar_value: float | int, global_step: Optional[int] = None
    ) -> None:
        self._writer.add_scalar(
            tag=tag, scalar_value=scalar_value, global_step=global_step
        )

    def add_text(
        self, tag: str, text_string: str, global_step: Optional[int] = None
    ) -> None:
        self._writer.add_text(tag=tag, text_string=text_string, global_step=global_step)

    def add_figure(
        self, tag: str, figure: Any, global_step: Optional[int] = None
    ) -> None:
        # SummaryWriter.add_figure exists and accepts matplotlib figures
        self._writer.add_figure(tag=tag, figure=figure, global_step=global_step)

    def add_image(self, tag: str, img: Any, global_step: Optional[int] = None) -> None:
        self._writer.add_image(tag=tag, img_tensor=img, global_step=global_step)

    def add_histogram(
        self, tag: str, values: Any, global_step: Optional[int] = None, **kwargs: Any
    ) -> None:
        self._writer.add_histogram(
            tag=tag, values=values, global_step=global_step, **kwargs
        )

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()

    # Expose the underlying writer when needed (e.g., for advanced usage)
    @property
    def writer(self) -> SummaryWriter:
        return self._writer
