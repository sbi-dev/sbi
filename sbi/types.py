from typing import Sequence, TypeVar, Union

import numpy as np
import torch
from torch import Tensor

Array = Union[np.ndarray, torch.Tensor]

T = TypeVar("T")
OneOrMore = Union[T, Sequence[T]]

ScalarFloat = Union[Tensor, float]
