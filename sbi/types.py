from typing import Sequence, Union, Tuple, TypeVar
import numpy as np
import torch

Array = Union[np.ndarray, torch.Tensor]
Shape = Union[torch.Size, Tuple[int, ...]]

T = TypeVar("T")
OneOrMore = Union[T, Sequence[T]]

ScalarFloat = Union[torch.Tensor, float]


__all__ = ["Array", "Shape", "OneOrMore", "ScalarFloat"]
