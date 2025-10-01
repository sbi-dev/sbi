# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import collections
import logging
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import six
import torch

try:
    collectionsAbc = collections.abc  # type: ignore
except AttributeError:
    collectionsAbc = collections

# for circular import error
if TYPE_CHECKING:
    from sbi.analysis.plot import KwargsType


def hex2rgb(hex: str) -> List[int]:
    """Pass 16 to the integer function for change of base"""
    return [int(hex[i : i + 2], 16) for i in range(1, 6, 2)]


def rgb2hex(RGB: List[int]) -> str:
    """Components need to be integers for hex to make sense"""
    RGB = [int(x) for x in RGB]
    return "#" + "".join([
        "0{0:x}".format(v) if v < 16 else "{0:x}".format(v) for v in RGB
    ])


def to_list_string(
    x: Optional[Union[str, List[Optional[str]]]], len: int
) -> List[Optional[str]]:
    """If x is not a list, make it a list of strings of length `len`."""
    if not isinstance(x, list):
        return [x for _ in range(len)]
    return x


def to_list_kwargs(x: "KwargsType", len: int) -> List[Optional[Dict]]:
    """If x is not a list, make it a list of dicts of length `len`."""

    if not isinstance(x, list):
        x = [x for _ in range(len)]

    dict_list = []
    for value in x:
        if is_dataclass(value) and not isinstance(value, type):
            dict_list.append(asdict(value))
        elif value is None or isinstance(value, dict):
            dict_list.append(value)
        else:
            raise TypeError(
                f"Expected type of dataclass, dict or None, "
                f"but got type={type(value).__name__}"
            )

    return dict_list


def update(d: Dict, u: Optional[Dict]) -> Dict:
    """update dictionary with user input, see: https://stackoverflow.com/a/3233356"""
    if u is not None:
        for k, v in six.iteritems(u):
            dv = d.get(k, {})
            if not isinstance(dv, collectionsAbc.Mapping):  # type: ignore
                d[k] = v
            elif isinstance(v, collectionsAbc.Mapping):  # type: ignore
                d[k] = update(dv, v)
            else:
                d[k] = v
    return d


def ensure_numpy(t: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Returns np.ndarray if torch.Tensor was provided.

    Used because samples_nd() can only handle np.ndarray.
    """
    if isinstance(t, torch.Tensor):
        return t.numpy()
    elif not isinstance(t, np.ndarray):
        return np.array(t)
    return t


def handle_nan_infs(samples: List[np.ndarray]) -> List[np.ndarray]:
    """Check if there are NaNs or Infs in the samples."""
    for i in range(len(samples)):
        if np.isnan(samples[i]).any():
            logging.warning("NaNs found in samples, omitting datapoints.")
        if np.isinf(samples[i]).any():
            logging.warning("Infs found in samples, omitting datapoints.")
            # cast inf to nan, so they are omitted in the next step
            np.nan_to_num(
                samples[i], copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan
            )
        samples[i] = samples[i][~np.isnan(samples[i]).any(axis=1)]
    return samples


def convert_to_list_of_numpy(
    arr: Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor],
) -> List[np.ndarray]:
    """Converts a list of torch.Tensor to a list of np.ndarray."""
    if not isinstance(arr, list):
        arr = ensure_numpy(arr)
        return [arr]
    return [ensure_numpy(a) for a in arr]
