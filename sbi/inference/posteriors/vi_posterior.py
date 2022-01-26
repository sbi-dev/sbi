# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.
from typing import Any, Callable, Optional

import torch

from sbi import utils as utils
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.types import Shape, TorchTransform


class VIPosterior(NeuralPosterior):
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
        prior,
        theta_transform: Optional[TorchTransform] = None,
        device: str = "cpu",
    ):
        """
        Args:
            potential_fn:
            theta_transform:
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            proposal: the proposal distribtution (default is the prior).
            max_sampling_batch_size: the batchsize of samples being drawn from
                the proposal at every iteration.
            num_samples_to_find_max: the number of samples that are used to find the
                maximum of the `potential_fn / proposal` ratio.
            num_iter_to_find_max: the number of gradient ascent iterations to find the
                maximum of that ratio.
            m: multiplier to that ratio.
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
        """
        super().__init__(potential_fn, theta_transform, device)

        self._device = device

        self._purpose = (
            "It provides Variational inference to .sample() from the posterior and "
            "can evaluate the _normalized_ posterior density with .log_prob()."
        )

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        show_progress_bars: bool = True,
    ):
        raise NotImplementedError
