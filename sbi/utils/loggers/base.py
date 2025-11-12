# SPDX-FileCopyrightText: 2023-2025 sbi-dev
# SPDX-License-Identifier: Apache-2.0

"""Base class for all loggers used in the SBI framework."""

from abc import ABC, abstractmethod
from typing import Any


class BaseLogger(ABC):
    """Abstract base class for all loggers (e.g., WandbLogger, TensorBoardLogger)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize base logger."""
        pass

    @abstractmethod
    def log(self, *args: Any, **kwargs: Any) -> None:
        """Log data to the target destination (must be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement the 'log' method.")
