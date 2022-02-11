# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from typing import NewType, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch
from pyro.distributions import TransformedDistribution
from torch import Tensor
from torch.distributions import Distribution
from torch.distributions.transforms import Transform
from torch.nn import Module
from torch.utils.tensorboard.writer import SummaryWriter

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

# Define alias types because otherwise, the documentation by mkdocs became very long and
# made the website look ugly.
TensorboardSummaryWriter = NewType("Writer", SummaryWriter)
TorchTransform = NewType("torch Transform", Transform)
TorchModule = NewType("Module", Module)
TorchDistribution = NewType("torch Distribution", Distribution)
PyroTransformedDistribution = NewType(
    "pyro TransformedDistribution", TransformedDistribution
)
TorchTensor = NewType("Tensor", Tensor)

__all__ = [
    "Array",
    "Shape",
    "OneOrMore",
    "ScalarFloat",
    "TensorboardSummaryWriter",
    "TorchModule",
    "transform_types",
    "TorchDistribution",
    "PyroTransformedDistribution",
    "TorchTensor",
]
