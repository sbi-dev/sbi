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
        device: str = "cpu",
    ):
        """
        Args:
            potential_fn: The potential function from which to draw samples.
            theta_transform: Transformation that will be applied during sampling.
                Allows to perform, e.g. MCMC in unconstrained space.
            device: Training device, e.g., "cpu", "cuda" or "cuda:0".
        """
        # Ensure device string.
        device = process_device(device)

        self.potential_fn = potential_fn

        if theta_transform is None:
            self.theta_transform = torch_tf.IndependentTransform(
                torch_tf.identity_transform, reinterpreted_batch_ndims=1
            )
        else:
            self.theta_transform = theta_transform

        self._device = device
        self._purpose = ""

    def potential(self, theta: Tensor, track_gradients: bool = False) -> Tensor:
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
        theta = ensure_theta_batched(torch.as_tensor(theta))
        return self.potential_fn(
            theta.to(self._device), track_gradients=track_gradients
        )

    def log_prob(self, theta: Tensor, track_gradients: bool = False) -> Tensor:
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

    def __repr__(self):
        desc = f"""{self.__class__.__name__} sampler for potential_fn=<{self.potential_fn.__name__}>"""
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
