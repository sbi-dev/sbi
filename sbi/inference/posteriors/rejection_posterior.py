# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.
from math import ceil
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union
from warnings import warn
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC
import numpy as np
import torch
import torch.distributions.transforms as torch_tf
from torch import Tensor, nn

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.types import Shape
from sbi.utils import del_entries
from sbi.samplers.rejection.rejection import rejection_sample
from sbi.utils.torchutils import atleast_2d, ensure_theta_batched
from sbi import utils as utils


class RejectionPosterior(NeuralPosterior):
    r"""Posterior $p(\theta|x)$ with `log_prob()` and `sample()` methods, obtained with
    SNLE.<br/><br/>
    SNLE trains a neural network to approximate the likelihood $p(x|\theta)$. The
    `SNLE_Posterior` class wraps the trained network such that one can directly evaluate
    the unnormalized posterior log probability $p(\theta|x) \propto p(x|\theta) \cdot
    p(\theta)$ and draw samples from the posterior with MCMC.<br/><br/>
    The neural network itself can be accessed via the `.net` attribute.
    """

    def __init__(
        self,
        potential_fn: Callable,
        proposal: Any,
        potential_tf: torch_tf.Transform = torch_tf.identity_transform,
        max_sampling_batch_size: int = 10_000,
        num_samples_to_find_max: int = 10_000,
        num_iter_to_find_max: int = 100,
        m: float = 1.2,
        device: str = "cpu",
    ):
        """
        Args:
            potential_fn:
            proposal: The proposal distribtution.
            potential_tf: Only used for MAP.
            max_sampling_batch_size: the batchsize of samples being drawn from
                the proposal at every iteration.
            num_samples_to_find_max: the number of samples that are used to find the
                maximum of the `potential_fn / proposal` ratio.
            num_iter_to_find_max: the number of gradient ascent iterations to find the
                maximum of that ratio.
            m: multiplier to that ratio.
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
        """
        super().__init__(
            potential_fn,
            potential_tf=potential_tf,
            device=device,
        )

        self.proposal = proposal
        self.max_sampling_batch_size = max_sampling_batch_size
        self.num_samples_to_find_max = num_samples_to_find_max
        self.num_samples_to_find_max = num_samples_to_find_max
        self.num_iter_to_find_max = num_iter_to_find_max
        self.m = m

        self._purpose = (
            "It provides Rejection sampling to .sample() from the posterior and "
            "can evaluate the _unnormalized_ posterior density with .log_prob()."
        )

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        show_progress_bars: bool = True,
    ):
        num_samples = torch.Size(sample_shape).numel()

        potential = partial(self.potential_fn, track_gradients=True)

        samples, _ = rejection_sample(
            potential,
            proposal=self.proposal,
            num_samples=num_samples,
            show_progress_bars=show_progress_bars,
            warn_acceptance=0.01,
            max_sampling_batch_size=self.max_sampling_batch_size,
            num_samples_to_find_max=self.num_samples_to_find_max,
            num_iter_to_find_max=self.num_iter_to_find_max,
            m=self.m,
            device=self._device,
        )

        return samples.reshape((*sample_shape, -1))
