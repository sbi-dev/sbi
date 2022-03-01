# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.
from functools import partial
from typing import Any, Callable, Optional, Union
from warnings import warn

import torch
from torch import Tensor

from sbi import utils as utils
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.samplers.rejection.rejection import rejection_sample
from sbi.types import Shape, TorchTransform
from sbi.utils.torchutils import ensure_theta_batched


class RejectionPosterior(NeuralPosterior):
    r"""Provides rerjeciton sampling to sample from the posterior.<br/><br/>
    SNLE or SNRE train neural networks to approximate the likelihood(-ratios).
    `RejectionPosterior` allows to sample from the posterior with rejection sampling.
    """

    def __init__(
        self,
        potential_fn: Callable,
        proposal: Any,
        theta_transform: Optional[TorchTransform] = None,
        max_sampling_batch_size: int = 10_000,
        num_samples_to_find_max: int = 10_000,
        num_iter_to_find_max: int = 100,
        m: float = 1.2,
        device: Optional[str] = None,
        x_shape: Optional[torch.Size] = None,
    ):
        """
        Args:
            potential_fn: The potential function from which to draw samples.
            proposal: The proposal distribution.
            theta_transform: Transformation that is applied to parameters. Is not used
                during but only when calling `.map()`.
            max_sampling_batch_size: The batchsize of samples being drawn from
                the proposal at every iteration.
            num_samples_to_find_max: The number of samples that are used to find the
                maximum of the `potential_fn / proposal` ratio.
            num_iter_to_find_max: The number of gradient ascent iterations to find the
                maximum of the `potential_fn / proposal` ratio.
            m: Multiplier to the `potential_fn / proposal` ratio.
            device: Training device, e.g., "cpu", "cuda" or "cuda:0". If None,
                `potential_fn.device` is used.
            x_shape: Shape of a single simulator output. If passed, it is used to check
                the shape of the observed data and give a descriptive error.
        """
        super().__init__(
            potential_fn,
            theta_transform=theta_transform,
            device=device,
            x_shape=x_shape,
        )

        self.proposal = proposal
        self.max_sampling_batch_size = max_sampling_batch_size
        self.num_samples_to_find_max = num_samples_to_find_max
        self.num_samples_to_find_max = num_samples_to_find_max
        self.num_iter_to_find_max = num_iter_to_find_max
        self.m = m

        self._purpose = (
            "It provides rejection sampling to .sample() from the posterior and "
            "can evaluate the _unnormalized_ posterior density with .log_prob()."
        )

    def log_prob(
        self, theta: Tensor, x: Optional[Tensor] = None, track_gradients: bool = False
    ) -> Tensor:
        r"""Returns the log-probability of theta under the posterior.

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

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        max_sampling_batch_size: Optional[int] = None,
        num_samples_to_find_max: Optional[int] = None,
        num_iter_to_find_max: Optional[int] = None,
        m: Optional[float] = None,
        sample_with: Optional[str] = None,
        show_progress_bars: bool = True,
    ):
        r"""Return samples from posterior $p(\theta|x)$ via rejection sampling.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            sample_with: This argument only exists to keep backward-compatibility with
                `sbi` v0.17.2 or older. If it is set, we instantly raise an error.
            show_progress_bars: Whether to show sampling progress monitor.

        Returns:
            Samples from posterior.
        """
        num_samples = torch.Size(sample_shape).numel()
        self.potential_fn.set_x(self._x_else_default_x(x))

        potential = partial(self.potential_fn, track_gradients=True)

        if sample_with is not None:
            raise ValueError(
                f"You set `sample_with={sample_with}`. As of sbi v0.18.0, setting "
                f"`sample_with` is no longer supported. You have to rerun "
                f"`.build_posterior(sample_with={sample_with}).`"
            )
        # Replace arguments that were not passed with their default.
        max_sampling_batch_size = (
            self.max_sampling_batch_size
            if max_sampling_batch_size is None
            else max_sampling_batch_size
        )
        num_samples_to_find_max = (
            self.num_samples_to_find_max
            if num_samples_to_find_max is None
            else num_samples_to_find_max
        )
        num_iter_to_find_max = (
            self.num_iter_to_find_max
            if num_iter_to_find_max is None
            else num_iter_to_find_max
        )
        m = self.m if m is None else m

        samples, _ = rejection_sample(
            potential,
            proposal=self.proposal,
            num_samples=num_samples,
            show_progress_bars=show_progress_bars,
            warn_acceptance=0.01,
            max_sampling_batch_size=max_sampling_batch_size,
            num_samples_to_find_max=num_samples_to_find_max,
            num_iter_to_find_max=num_iter_to_find_max,
            m=m,
            device=self._device,
        )

        return samples.reshape((*sample_shape, -1))

    def map(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 1_000,
        num_to_optimize: int = 100,
        learning_rate: float = 0.01,
        init_method: Union[str, Tensor] = "proposal",
        num_init_samples: int = 1_000,
        save_best_every: int = 10,
        show_progress_bars: bool = False,
        force_update: bool = False,
    ) -> Tensor:
        r"""Returns the maximum-a-posteriori estimate (MAP).

        The method can be interrupted (Ctrl-C) when the user sees that the
        log-probability converges. The best estimate will be saved in `self._map` and
        can be accessed with `self.map()`. The MAP is obtained by running gradient
        ascent from a given number of starting positions (samples from the posterior
        with the highest log-probability). After the optimization is done, we select the
        parameter set that has the highest log-probability after the optimization.

        Warning: The default values used by this function are not well-tested. They
        might require hand-tuning for the problem at hand.

        For developers: if the prior is a `BoxUniform`, we carry out the optimization
        in unbounded space and transform the result back into bounded space.

        Args:
            x: Deprecated - use `.set_default_x()` prior to `.map()`.
            num_iter: Number of optimization steps that the algorithm takes
                to find the MAP.
            learning_rate: Learning rate of the optimizer.
            init_method: How to select the starting parameters for the optimization. If
                it is a string, it can be either [`posterior`, `prior`], which samples
                the respective distribution `num_init_samples` times. If it is a
                tensor, the tensor will be used as init locations.
            num_init_samples: Draw this number of samples from the posterior and
                evaluate the log-probability of all of them.
            num_to_optimize: From the drawn `num_init_samples`, use the
                `num_to_optimize` with highest log-probability as the initial points
                for the optimization.
            save_best_every: The best log-probability is computed, saved in the
                `map`-attribute, and printed every `save_best_every`-th iteration.
                Computing the best log-probability creates a significant overhead
                (thus, the default is `10`.)
            show_progress_bars: Whether or not to show a progressbar for sampling from
                the posterior.
            force_update: Whether to re-calculate the MAP when x is unchanged and
                have a cached value.
            log_prob_kwargs: Will be empty for SNLE and SNRE. Will contain
                {'norm_posterior': True} for SNPE.

        Returns:
            The MAP estimate.
        """
        return super().map(
            x=x,
            num_iter=num_iter,
            num_to_optimize=num_to_optimize,
            learning_rate=learning_rate,
            init_method=init_method,
            num_init_samples=num_init_samples,
            save_best_every=save_best_every,
            show_progress_bars=show_progress_bars,
            force_update=force_update,
        )
