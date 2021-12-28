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

from sbi.types import Array, Shape
from sbi.utils.sbiutils import (
    check_warn_and_setstate,
    mcmc_transform,
    optimize_potential_fn,
)
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
    samples from the posterior. The neural network itself can be accessed via the `.net`
    attribute.
    """

    def __init__(
        self,
        potential_fn: Callable,
        potential_tf: Optional[torch_tf.Transform] = None,
        device: str = "cpu",
    ):
        """
        Args:
            method_family: One of snpe, snl, snre_a or snre_b.
            neural_net: A classifier for SNRE, a density estimator for SNPE and SNL.
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            x_shape: Shape of the simulator data.
            sample_with: Method to use for sampling from the posterior. Must be one of
                [`mcmc` | `rejection`].
            mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`,
                `hmc`, `nuts`. Currently defaults to `slice_np` for a custom numpy
                implementation of slice sampling; select `hmc`, `nuts` or `slice` for
                Pyro-based sampling.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains,
                `init_strategy` for the initialisation strategy for chains; `prior`
                will draw init locations from prior, whereas `sir` will use Sequential-
                Importance-Resampling. Init strategies may have their own keywords
                which can also be set from `mcmc_parameters`.
            rejection_sampling_parameters: Dictionary overriding the default parameters
                for rejection sampling. The following parameters are supported:
                `proposal` as the proposal distribtution.
                `max_sampling_batch_size` as the batchsize of samples being drawn from
                the proposal at every iteration. `num_samples_to_find_max` as the
                number of samples that are used to find the maximum of the
                `potential_fn / proposal` ratio. `num_iter_to_find_max` as the number
                of gradient ascent iterations to find the maximum of that ratio. `m` as
                multiplier to that ratio.
            device: Training device, e.g., "cpu", "cuda" or "cuda:0".
        """
        # Ensure device string.
        device = process_device(device)

        self.potential_fn = potential_fn

        if potential_tf is None:
            self.potential_tf = torch_tf.IndependentTransform(
                torch_tf.identity_transform, reinterpreted_batch_ndims=1
            )
        else:
            self.potential_tf = potential_tf

        self._num_trained_rounds = 0
        self._num_iid_trials = None

        self._device = device

        self._purpose = ""

    def potential(self, theta: Tensor, track_gradients: bool = False) -> Tensor:
        theta = ensure_theta_batched(torch.as_tensor(theta))
        return self.potential_fn(
            theta.to(self._device), track_gradients=track_gradients
        )

    def log_prob(self, theta: Tensor, track_gradients: bool = False) -> Tensor:
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

    def __str__(self):
        msg = {0: "untrained", 1: "amortized"}

        focused_msg = "multi-round"

        desc = (
            f"Posterior conditional density p(θ|x) "
            f"({msg.get(self._num_trained_rounds, focused_msg)}).\n\n"
            f"This {self.__class__.__name__}-object"
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
