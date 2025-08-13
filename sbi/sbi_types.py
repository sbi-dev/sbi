# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch
from pyro.distributions import TransformedDistribution  # type: ignore
from torch import Tensor
from torch.distributions import Distribution
from torch.distributions.transforms import Transform
from torch.utils.tensorboard.writer import SummaryWriter
from typing_extensions import TypeAlias

Array = Union[np.ndarray, torch.Tensor]
Shape = Union[torch.Size, Tuple[int, ...]]

T = TypeVar("T")
OneOrMore = Union[T, Sequence[T]]

ScalarFloat = Union[torch.Tensor, float]

transform_types = Optional[
    Union[
        torch.distributions.transforms.Transform,
        torch.distributions.transforms.ComposeTransform,
    ]
]

# Define alias types for better readability in type hints and checking.
# See PEP 613 for the reason why we need to use TypeAlias here.
TensorBoardSummaryWriter: TypeAlias = SummaryWriter
TorchDistribution: TypeAlias = Distribution
TorchTransform: TypeAlias = Transform
PyroTransformedDistribution: TypeAlias = TransformedDistribution
TorchTensor: TypeAlias = Tensor

__all__ = [
    "Array",
    "Shape",
    "OneOrMore",
    "ScalarFloat",
    "TensorBoardSummaryWriter",
    "TorchTransform",
    "transform_types",
    "TorchDistribution",
    "PyroTransformedDistribution",
    "TorchTensor",
]
