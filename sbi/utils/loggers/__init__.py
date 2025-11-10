# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

# SPDX-FileCopyrightText: 2023-2025 sbi-dev
# SPDX-License-Identifier: Apache-2.0
"""
Built-in logging adapters for popular experiment tracking backends.

This module provides small, dependency-light wrappers that duck-type the
TensorBoard SummaryWriter API so they can be plugged into existing `sbi`
training entry points via the `summary_writer` parameter.

Available adapters:
- TensorBoardLogger: wraps torch.utils.tensorboard.SummaryWriter
- WandBSummaryWriter: integrates with Weights & Biases (optional dependency)
"""

from .tensorboard import TensorBoardLogger
from .wandb import WandBSummaryWriter

# convenience exports / aliases
# Provide a user-friendly name for the WandB adapter without importing
# via an incorrect nested package path.
WandbLogger = WandBSummaryWriter

__all__ = [
    "TensorBoardLogger",
    "WandBSummaryWriter",
    "WandbLogger",
]
