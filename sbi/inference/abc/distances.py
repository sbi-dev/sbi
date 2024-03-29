from functools import partial
from logging import warning
from typing import Callable, Dict, Optional, Union

import torch
from tqdm import tqdm

from sbi.utils.metrics import unbiased_mmd_squared, wasserstein_2_squared


class Distance:
    def __init__(
        self,
        distance: Union[str, Callable] = "l2",
        requires_iid_data: Optional[bool] = None,
        distance_kwargs: Optional[Dict] = None,
        batch_size=-1,
    ):
        """Distance class for ABC

        Args:
            distance:
            requires_iid_data:
            distance_kwargs:
        """
        self.batch_size = batch_size
        self.distance_kwargs = distance_kwargs or {}
        if isinstance(distance, Callable):
            if requires_iid_data is None:
                # By default, we assume that data should not come in batches
                warning(
                    "Please specify if your the custom distance requires "
                    "iid data or is evaluated between single datapoints. "
                    "By default, we assume that `requires_iid_data=False`"
                )
                requires_iid_data = False
            self.distance_fn = distance
            self._requires_iid_data = requires_iid_data
        else:
            implemented_pairwise_distances = ["l1", "l2", "mse"]
            implemented_statistical_distances = ["mmd", "wasserstein"]

            assert (
                distance
                in implemented_pairwise_distances + implemented_statistical_distances
            ), f"{distance} must be one of "
            f"{implemented_pairwise_distances + implemented_statistical_distances}."

            self._requires_iid_data = distance in implemented_statistical_distances

            distance_functions = {
                "mse": mse_distance,
                "l2": l2_distance,
                "l1": l1_distance,
                "mmd": partial(mmd, **self.distance_kwargs),
                "wasserstein": partial(wasserstein, **self.distance_kwargs),
            }
            try:
                self.distance_fn = distance_functions[distance]
            except KeyError as exc:
                raise KeyError(f"Distance {distance} not supported.") from exc

    def __call__(self, xo, x) -> torch.Tensor:
        """Distance evaluation between the reference data and the simulated data.

        Args:
            xo: Reference data
            x: Simulated data
        """
        if self.requires_iid_data:
            assert x.ndim >= 3, "simulated data needs batch dimension"
            assert xo.ndim + 1 == x.ndim
        else:
            assert x.ndim >= 2, "simulated data needs batch dimension"
        if self.batch_size == -1:
            return self.distance_fn(xo, x)
        else:
            return self._batched_distance(xo, x)

    def _batched_distance(self, xo, x):
        """Evaluate the distance is mini-batches.
        Especially for statistical distances, batching over two empirical
        datasets can lead to memory overflow. Batching can help to resolve
        the memory problems.

        Args:
            xo: Reference data
            x: Simulated data
        """
        num_batches = x.shape[0] // self.batch_size - 1
        remaining = x.shape[0] % self.batch_size
        if remaining == 0:
            remaining = self.batch_size

        distances = torch.empty(x.shape[0])
        for i in tqdm(range(num_batches)):
            distances[self.batch_size * i : (i + 1) * self.batch_size] = (
                self.distance_fn(xo, x[self.batch_size * i : (i + 1) * self.batch_size])
            )
        if remaining > 0:
            distances[-remaining:] = self.distance_fn(xo, x[-remaining:])

        return distances

    @property
    def requires_iid_data(self):
        return self._requires_iid_data


def mse_distance(xo, x):
    return torch.mean((xo - x) ** 2, dim=-1)


def l2_distance(xo, x):
    return torch.norm((xo - x), dim=-1)


def l1_distance(xo, x):
    return torch.mean(abs(xo - x), dim=-1)


def mmd(xo, x, scale=None):
    dist_fn = partial(unbiased_mmd_squared, scale=scale)
    return torch.vmap(dist_fn, in_dims=(None, 0))(xo, x)


def wasserstein(xo, x, epsilon=1e-3, max_iter=1000, tol=1e-9):
    batched_xo = xo.repeat((x.shape[0], *[1] * len(xo.shape)))
    return wasserstein_2_squared(
        batched_xo, x, epsilon=epsilon, max_iter=max_iter, tol=tol
    )
