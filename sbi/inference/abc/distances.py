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
            distance: A distance function comparing the simulations with 'x_o'.
            Implemented distances are the 'mse', 'l2', and 'l1' norm as pairwise
            distances, or the 'wasserstein' and 'mmd' as statistical distances.
            requires_iid_data: 'True' if the distance is a statistical distance.
            Only needs to be specified if 'distance' is a custom distance
            distance_kwargs: Arguments for the specific distance.
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

    def __call__(self, x_o, x) -> torch.Tensor:
        """Distance evaluation between the reference data and the simulated data.

        Args:
            x_o: Reference data
            x: Simulated data
        """
        if self.requires_iid_data:
            assert x.ndim >= 3, "simulated data needs batch dimension"
            assert x_o.ndim + 1 == x.ndim
        else:
            assert x.ndim >= 2, "simulated data needs batch dimension"
        if self.batch_size == -1:
            return self.distance_fn(x_o, x)
        else:
            return self._batched_distance(x_o, x)

    def _batched_distance(self, x_o, x):
        """Evaluate the distance is mini-batches.
        Especially for statistical distances, batching over two empirical
        datasets can lead to memory overflow. Batching can help to resolve
        the memory problems.

        Args:
            x_o: Reference data
            x: Simulated data
        """
        num_batches = x.shape[0] // self.batch_size - 1
        remaining = x.shape[0] % self.batch_size
        if remaining == 0:
            remaining = self.batch_size

        distances = torch.empty(x.shape[0])
        for i in tqdm(range(num_batches)):
            distances[self.batch_size * i : (i + 1) * self.batch_size] = (
                self.distance_fn(
                    x_o, x[self.batch_size * i : (i + 1) * self.batch_size]
                )
            )
        if remaining > 0:
            distances[-remaining:] = self.distance_fn(x_o, x[-remaining:])

        return distances

    @property
    def requires_iid_data(self):
        return self._requires_iid_data


def mse_distance(x_o, x):
    return torch.mean((x_o - x) ** 2, dim=-1)


def l2_distance(x_o, x):
    return torch.norm((x_o - x), dim=-1)


def l1_distance(x_o, x):
    return torch.mean(abs(x_o - x), dim=-1)


def mmd(x_o, x, scale=None):
    dist_fn = partial(unbiased_mmd_squared, scale=scale)
    return torch.vmap(dist_fn, in_dims=(None, 0))(x_o, x)


def wasserstein(x_o, x, epsilon=1e-3, max_iter=1000, tol=1e-9):
    batched_x_o = x_o.repeat((x.shape[0], *[1] * len(x_o.shape)))
    return wasserstein_2_squared(
        batched_x_o, x, epsilon=epsilon, max_iter=max_iter, tol=tol
    )
