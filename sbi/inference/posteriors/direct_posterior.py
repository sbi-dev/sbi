# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.
from functools import partial
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Union
from warnings import warn

import torch
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC
from torch import Tensor, log, nn

from sbi import utils as utils
from sbi.analysis import gradient_ascent
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.potentials.posterior_based_potential import posterior_potential
from sbi.samplers.rejection.rejection import rejection_sample_posterior_within_prior
from sbi.types import Shape
from sbi.utils import del_entries
from sbi.utils.sbiutils import match_theta_and_x_batch_shapes, within_support
from sbi.utils.torchutils import (
    atleast_2d,
    atleast_2d_float32_tensor,
    ensure_theta_batched,
)
from sbi.utils.user_input_checks import check_for_possibly_batched_x_shape


class DirectPosterior(NeuralPosterior):
    r"""Posterior $p(\theta|x_o)$ with `log_prob()` and `sample()` methods, only
    applicable to SNPE.<br/><br/>
    SNPE trains a neural network to directly approximate the posterior distribution.
    However, for bounded priors, the neural network can have leakage: it puts non-zero
    mass in regions where the prior is zero. The `DirectPosterior` class wraps the
    trained network to deal with these cases.<br/><br/>
    Specifically, this class offers the following functionality:<br/>
    - correct the calculation of the log probability such that it compensates for the
      leakage.<br/>
    - reject samples that lie outside of the prior bounds.<br/><br/>
    This class can not be used in combination with SNLE or SNRE.
    """

    def __init__(
        self,
        prior: Callable,
        posterior_model: nn.Module,
        x_o: Tensor,
        max_sampling_batch_size: int = 10_000,
        device: str = "cpu",
    ):
        """
        Args:
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            posterior_model: The trained neural posterior.
            x_o: Tensor at which to evaluate the `posterior_model`.
            max_sampling_batch_size: Batchsize of samples being drawn from
                the proposal at every iteration.
            device: Training device, e.g., "cpu", "cuda" or "cuda:0".
        """
        # Because `DirectPosterior` does not take the `potential_fn` as input, it
        # builds it itself. The `potential_fn` and `theta_transform` are used only for
        # obtaining the MAP.
        potential_fn, theta_transform = posterior_potential(posterior_model, prior, x_o)

        super().__init__(
            potential_fn=potential_fn, theta_transform=theta_transform, device=device
        )

        self.prior = prior
        self.posterior_model = posterior_model

        # once we move to pyroflows, we might get rid of this x_o and force that
        # posterior_nn has a default x_o
        self.x_o = atleast_2d_float32_tensor(x_o).to(device)
        check_for_possibly_batched_x_shape(self.x_o.shape)

        self.max_sampling_batch_size = max_sampling_batch_size
        self._leakage_density_correction_factor = None

        self._purpose = "It samples the posterior network but rejects samples that lie outside of the prior bounds."

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        show_progress_bars: bool = True,
    ):
        r"""
        Return samples from posterior distribution $p(\theta|x)$.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            show_progress_bars: Whether to show sampling progress monitor.
        """

        num_samples = torch.Size(sample_shape).numel()

        samples = rejection_sample_posterior_within_prior(
            posterior_nn=self.posterior_model,
            prior=self.prior,
            x=self.x_o,
            num_samples=num_samples,
            show_progress_bars=show_progress_bars,
            max_sampling_batch_size=self.max_sampling_batch_size,
        )[0]
        return samples

    def log_prob(
        self,
        theta: Tensor,
        norm_posterior: bool = True,
        track_gradients: bool = False,
        leakage_correction_params: Optional[dict] = None,
    ) -> Tensor:
        r"""
        Returns the log-probability of the posterior $p(\theta|x).$

        Args:
            theta: Parameters $\theta$.
            norm_posterior: Whether to enforce a normalized posterior density.
                Renormalization of the posterior is useful when some
                probability falls out or leaks out of the prescribed prior support.
                The normalizing factor is calculated via rejection sampling, so if you
                need speedier but unnormalized log posterior estimates set here
                `norm_posterior=False`. The returned log posterior is set to
                -∞ outside of the prior support regardless of this setting.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.
            leakage_correction_params: A `dict` of keyword arguments to override the
                default values of `leakage_correction()`. Possible options are:
                `num_rejection_samples`, `force_update`, `show_progress_bars`, and
                `rejection_sampling_batch_size`.
                These parameters only have an effect if `norm_posterior=True`.

        Returns:
            `(len(θ),)`-shaped log posterior probability $\log p(\theta|x)$ for θ in the
            support of the prior, -∞ (corresponding to 0 probability) outside.
        """

        # TODO Train exited here, entered after sampling?
        self.posterior_model.eval()

        theta = ensure_theta_batched(torch.as_tensor(theta))
        theta_repeated, x_repeated = match_theta_and_x_batch_shapes(theta, self.x_o)

        with torch.set_grad_enabled(track_gradients):

            # Evaluate on device, move back to cpu for comparison with prior.
            unnorm_log_prob = self.posterior_model.log_prob(
                theta_repeated, context=x_repeated
            )

            # Force probability to be zero outside prior support.
            in_prior_support = within_support(self.prior, theta_repeated)

            masked_log_prob = torch.where(
                in_prior_support,
                unnorm_log_prob,
                torch.tensor(float("-inf"), dtype=torch.float32, device=self._device),
            )

            if leakage_correction_params is None:
                leakage_correction_params = dict()  # use defaults
            log_factor = (
                log(self.leakage_correction(**leakage_correction_params))
                if norm_posterior
                else 0
            )

            return masked_log_prob - log_factor

    @torch.no_grad()
    def leakage_correction(
        self,
        num_rejection_samples: int = 10_000,
        show_progress_bars: bool = False,
        rejection_sampling_batch_size: int = 10_000,
    ) -> Tensor:
        r"""Return leakage correction factor for a leaky posterior density estimate.

        The factor is estimated from the acceptance probability during rejection
        sampling from the posterior.

        This is to avoid re-estimating the acceptance probability from scratch
        whenever `log_prob` is called and `norm_posterior=True`. Here, it
        is estimated only once for `self.default_x` and saved for later. We
        re-evaluate only whenever a new `x` is passed.

        Arguments:
            num_rejection_samples: Number of samples used to estimate correction factor.
            show_progress_bars: Whether to show a progress bar during sampling.
            rejection_sampling_batch_size: Batch size for rejection sampling.

        Returns:
            Saved or newly-estimated correction factor (as a scalar `Tensor`).
        """

        def acceptance_at(x: Tensor) -> Tensor:

            return rejection_sample_posterior_within_prior(
                posterior_nn=self.posterior_model,
                prior=self.prior,
                x=x.to(self._device),
                num_samples=num_rejection_samples,
                show_progress_bars=show_progress_bars,
                sample_for_correction_factor=True,
                max_sampling_batch_size=rejection_sampling_batch_size,
            )[1]

        if self._leakage_density_correction_factor is None:
            acceptance = acceptance_at(self.x_o)
            self._leakage_density_correction_factor = acceptance
            return acceptance
        else:
            return self._leakage_density_correction_factor

    def map(
        self,
        num_iter: int = 1_000,
        num_to_optimize: int = 100,
        learning_rate: float = 0.01,
        init_method: Union[str, Tensor] = "posterior",
        num_init_samples: int = 1_000,
        save_best_every: int = 10,
        show_progress_bars: bool = False,
    ) -> Tensor:
        r"""
        Returns the maximum-a-posteriori estimate (MAP).

        The method can be interrupted (Ctrl-C) when the user sees that the
        log-probability converges. The best estimate will be saved in `self.map_`.
        The MAP is obtained by running gradient ascent from a given number of starting
        positions (samples from the posterior with the highest log-probability). After
        the optimization is done, we select the parameter set that has the highest
        log-probability after the optimization.

        Warning: The default values used by this function are not well-tested. They
        might require hand-tuning for the problem at hand.

        For developers: if the prior is a `BoxUniform`, we carry out the optimization
        in unbounded space and transform the result back into bounded space.

        Args:
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
            log_prob_kwargs: Will be empty for SNLE and SNRE. Will contain
                {'norm_posterior': True} for SNPE.

        Returns:
            The MAP estimate.
        """

        if init_method == "posterior":
            inits = self.sample((num_init_samples,))
        elif init_method == "prior":
            inits = self.prior.sample((num_init_samples,))

        self.map_ = gradient_ascent(
            potential_fn=self.potential_fn,
            inits=inits,
            theta_transform=self.theta_transform,
            num_iter=num_iter,
            num_to_optimize=num_to_optimize,
            learning_rate=learning_rate,
            save_best_every=save_best_every,
            show_progress_bars=show_progress_bars,
        )[0]
        return self.map_
