# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

# SPDX-FileCopyrightText: 2023-2025 sbi-dev
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional

from .base import BaseLogger


class WandBSummaryWriter(BaseLogger):
    """A lightweight WandB adapter that mimics TensorBoard's SummaryWriter.

    Notes:
    - Requires `wandb` to be installed (pip install sbi[wandb] or pip install wandb).
    - Only a subset of the TensorBoard SummaryWriter API is implemented; unsupported
      calls will be silently ignored to keep parity with BaseLogger.
    - Uses `wandb.log` with an explicit `step` so training loops remain deterministic.
    """

    def __init__(
        self,
        *,
        project: Optional[str] = None,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None,
        dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        reinit: bool = True,
        **wandb_init_kwargs: Any,
    ) -> None:
        try:
            import wandb  # type: ignore
        except Exception as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "wandb is not installed. Install with "
                "`pip install sbi[wandb]` or `pip install wandb`."
            ) from e

        # Delay init until first log? We init eagerly for clarity and to capture config.
        self._wandb: Any = wandb
        self._run: Any = self._wandb.init(
            project=project,
            name=name,
            entity=entity,
            group=group,
            job_type=job_type,
            dir=dir,
            config=config,
            tags=tags,
            reinit=reinit,
            **wandb_init_kwargs,
        )

    # --- SummaryWriter-like APIs ---
    def add_scalar(
        self, tag: str, scalar_value: float | int, global_step: Optional[int] = None
    ) -> None:
        self._wandb.log({tag: scalar_value}, step=global_step)

    def add_text(
        self, tag: str, text_string: str, global_step: Optional[int] = None
    ) -> None:
        self._wandb.log({tag: self._wandb.Html(text_string)}, step=global_step)

    def add_figure(
        self, tag: str, figure: Any, global_step: Optional[int] = None
    ) -> None:
        # Accepts matplotlib figures
        self._wandb.log({tag: self._wandb.Image(figure)}, step=global_step)

    def add_image(self, tag: str, img: Any, global_step: Optional[int] = None) -> None:
        self._wandb.log({tag: self._wandb.Image(img)}, step=global_step)

    def add_histogram(
        self, tag: str, values: Any, global_step: Optional[int] = None, **kwargs: Any
    ) -> None:
        self._wandb.log({tag: self._wandb.Histogram(values)}, step=global_step)

    # --- Extras ---
    def watch_model(
        self, model: Any, log: str = "gradients", log_freq: int | None = None
    ) -> None:
        # wandb.watch can log gradients and optionally parameters
        self._wandb.watch(model, log=log, log_freq=log_freq)

    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        art = self._wandb.Artifact(name or path.replace("/", "_"), type="dataset")
        art.add_file(path)
        self._run.log_artifact(art)

    def flush(self) -> None:  # noqa: D401
        # No explicit flush needed; keep for API parity.
        return None

    def close(self) -> None:
        # Finish the run cleanly
        with contextlib.suppress(Exception):
            self._wandb.finish()
