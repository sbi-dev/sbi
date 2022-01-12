# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import torch
import torch.distributions.transforms as torch_tf
from torch import Tensor
from torch import multiprocessing as mp
from torch import nn

from sbi import utils as utils
from sbi.types import Array, Shape, TorchTransform
from sbi.utils.torchutils import (
    ScalarFloat,
    atleast_2d_float32_tensor,
    ensure_theta_batched,
    process_device,
)
from sbi.utils.user_input_checks import check_for_possibly_batched_x_shape, process_x


class NeuralPosterior(ABC):
    r"""Posterior $p(\theta|x)$ with `log_prob()` and `sample()` methods.<br/><br/>
    All inference methods in sbi train a neural network which is then used to obtain
    the posterior distribution. The `NeuralPosterior` class wraps the trained network
    such that one can directly evaluate the (unnormalized) log probability and draw
    samples from the posterior.
    """

    def __init__(
        self,
        potential_fn: Callable,
        theta_transform: Optional[TorchTransform] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            potential_fn: The potential function from which to draw samples.
            theta_transform: Transformation that will be applied during sampling.
                Allows to perform, e.g. MCMC in unconstrained space.
            device: Training device, e.g., "cpu", "cuda" or "cuda:0". If None,
                `potential_fn.device` is used.
        """
        if device is None:
            device = potential_fn.device

        # Ensure device string.
        self._device = process_device(device)

        self.potential_fn = potential_fn

        if theta_transform is None:
            self.theta_transform = torch_tf.IndependentTransform(
                torch_tf.identity_transform, reinterpreted_batch_ndims=1
            )
        else:
            self.theta_transform = theta_transform

        self._purpose = ""

        # If the sampler interface (#573) is used, the user might have passed `x_o`
        # already to the potential function builder. If so, this `x_o` will be used
        # as default x.
        self._x = self.potential_fn._x_o

    def potential(
        self, theta: Tensor, x: Optional[Tensor] = None, track_gradients: bool = False
    ) -> Tensor:
        r"""
        Evaluates $\theta$ under the potential that is used to sample the posterior.

        The potential is the unnormalized log-probability of $\theta$ under the
        posterior.

        Args:
            theta: Parameters $\theta$.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.
        """
        self.potential_fn.set_x(self._x_else_default_x(x))

        theta = ensure_theta_batched(torch.as_tensor(theta))
        return self.potential_fn(
            theta.to(self._device), track_gradients=track_gradients
        )

    def log_prob(
        self, theta: Tensor, x: Optional[Tensor] = None, track_gradients: bool = False
    ) -> Tensor:
        r"""
        Returns the log-probability of theta under the posterior.

        Args:
            theta: Parameters $\theta$.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.

        Returns:
            `len($\theta$)`-shaped log-probability.
        """
        warn(
            "`.log_prob()` is deprecated for methods that can only evaluate the log-probability up to a normalizing constant. Use `.potential()` instead."
        )
        warn("The log-probability is unnormalized!")

        self.potential_fn.set_x(self._x_else_default_x(x))

        theta = ensure_theta_batched(torch.as_tensor(theta))
        return self.potential_fn(
            theta.to(self._device), track_gradients=track_gradients
        )

    @abstractmethod
    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        show_progress_bars: bool = True,
        mcmc_method: Optional[str] = None,
        mcmc_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """See child classes for docstring."""
        pass

    @property
    def default_x(self) -> Optional[Tensor]:
        """Return default x used by `.sample(), .log_prob` as conditioning context."""
        return self._x

    @default_x.setter
    def default_x(self, x: Tensor) -> None:
        """See `set_default_x`."""
        self.set_default_x(x)

    def set_default_x(self, x: Tensor) -> "NeuralPosterior":
        """Set new default x for `.sample(), .log_prob` to use as conditioning context.
        This is a pure convenience to avoid having to repeatedly specify `x` in calls to
        `.sample()` and `.log_prob()` - only θ needs to be passed.
        This convenience is particularly useful when the posterior is focused, i.e.
        has been trained over multiple rounds to be accurate in the vicinity of a
        particular `x=x_o` (you can check if your posterior object is focused by
        printing it).
        NOTE: this method is chainable, i.e. will return the NeuralPosterior object so
        that calls like `posterior.set_default_x(my_x).sample(mytheta)` are possible.
        Args:
            x: The default observation to set for the posterior $p(theta|x)$.
        Returns:
            `NeuralPosterior` that will use a default `x` when not explicitly passed.
        """
        self._x = process_x(x, allow_iid_x=self.potential_fn.allow_iid_x).to(
            self._device
        )
        return self

    def _x_else_default_x(self, x: Optional[Array]) -> Tensor:
        if x is not None:
            return process_x(x, allow_iid_x=self.potential_fn.allow_iid_x)
        elif self.default_x is None:
            raise ValueError(
                "Context `x` needed when a default has not been set."
                "If you'd like to have a default, use the `.set_default_x()` method."
            )
        else:
            return self.default_x

    @abstractmethod
    def map(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 1_000,
        num_to_optimize: int = 100,
        learning_rate: float = 0.01,
        init_method: Union[str, Tensor] = "posterior",
        num_init_samples: int = 1_000,
        save_best_every: int = 10,
        show_progress_bars: bool = False,
    ) -> Tensor:
        raise NotImplementedError

    def __repr__(self):
        desc = f"""{self.__class__.__name__} sampler for potential_fn=<{self.potential_fn.__class__.__name__}>"""
        return desc

    def __str__(self):

        desc = (
            f"Posterior conditional density p(θ|x) of type {self.__class__.__name__}. "
            f"{self._purpose}"
        )

        return desc

    def __getstate__(self) -> Dict:
        """
        Returns the state of the object that is supposed to be pickled.

        Returns:
            Dictionary containing the state.
        """
        return self.__dict__

    def __setstate__(self, state_dict: Dict):
        """
        Sets the state when being loaded from pickle.

        For developers: for any new attribute added to `NeuralPosterior`, we have to
        add an entry here using `check_warn_and_setstate()`.

        Args:
            state_dict: State to be restored.
        """
        self.__dict__ = state_dict
