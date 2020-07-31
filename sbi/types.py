# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from typing import (
    Any,
    Callable,
    Dict,
    List,
    NewType,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

Array = Union[np.ndarray, torch.Tensor]
Shape = Union[torch.Size, Tuple[int, ...]]

T = TypeVar("T")
OneOrMore = Union[T, Sequence[T]]

ScalarFloat = Union[torch.Tensor, float]

# Define a `Writer` because otherwise, the documentation by mkdocs became very long and
# made the website look ugly.
TensorboardSummaryWriter = NewType("Writer", SummaryWriter)


__all__ = ["Array", "Shape", "OneOrMore", "ScalarFloat", "TensorboardSummaryWriter"]
