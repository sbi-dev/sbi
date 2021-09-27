# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from typing import NewType, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

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
TorchModule = NewType("Module", Module)

__all__ = [
    "Array",
    "Shape",
    "OneOrMore",
    "ScalarFloat",
    "TensorboardSummaryWriter",
    "TorchModule",
    "transform_types",
]
