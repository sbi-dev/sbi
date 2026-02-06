# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional, Protocol, Sequence, Tuple, TypeVar, Union

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
VariationalDistribution: TypeAlias = TransformedDistribution
TorchTensor: TypeAlias = Tensor


class SampleProposal(Protocol):
    """Protocol for sample proposal callables used in rejection sampling.

    Any callable that takes a sample shape and optional keyword arguments
    and returns a Tensor of samples satisfies this protocol.
    """

    def __call__(self, sample_shape: torch.Size, **kwargs) -> Tensor: ...


class AcceptRejectFn(Protocol):
    """Protocol for accept/reject functions used in rejection sampling.

    Any callable that takes a batch of parameters (theta) and returns a boolean
    Tensor indicating which samples are accepted satisfies this protocol.
    """

    def __call__(self, theta: Tensor) -> Tensor: ...


__all__ = [
    "AcceptRejectFn",
    "Array",
    "Shape",
    "OneOrMore",
    "SampleProposal",
    "ScalarFloat",
    "TensorBoardSummaryWriter",
    "TorchTransform",
    "transform_types",
    "TorchDistribution",
    "VariationalDistribution",
    "TorchTensor",
]
