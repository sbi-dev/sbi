# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.
from math import ceil
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union
from warnings import warn
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC
import torch
from torch import Tensor, nn, log

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.types import Shape
from sbi.utils import del_entries
from sbi.samplers.rejection.rejection import rejection_sample_posterior_within_prior
from sbi.utils.torchutils import (
    atleast_2d,
    atleast_2d_float32_tensor,
    ensure_theta_batched,
)
from sbi import utils as utils
from sbi.utils.sbiutils import (
    within_support,
    match_theta_and_x_batch_shapes,
)
from sbi.utils.user_input_checks import check_for_possibly_batched_x_shape
from sbi.inference.potentials.posterior_based_potential import posterior_potential


class DirectPosterior(NeuralPosterior):
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
        prior: Callable,
        posterior_model: nn.Module,
        xo: Tensor,
        max_sampling_batch_size: int = 10_000,
        device: str = "cpu",
    ):
        """
        Args:
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
        # Because `DirectPosterior` does not take the `potential_fn` as input, it
        # builds it itself. The `potential_fn` and `potential_tf` are used only for
        # obtaining the MAP.
        potential_fn, potential_tf = posterior_potential(posterior_model, prior, xo)

        super().__init__(
            potential_fn=potential_fn, potential_tf=potential_tf, device=device
        )

        self.prior = prior
        self.posterior_model = posterior_model

        # once we move to pyroflows, we might get rid of this xo and force that
        # posterior_nn has a default xo
        self.xo = atleast_2d_float32_tensor(xo).to(device)
        check_for_possibly_batched_x_shape(self.xo.shape)

        self.max_sampling_batch_size = max_sampling_batch_size
        self._leakage_density_correction_factor = None

        self._purpose = "It samples the posterior network but rejects samples that lie "
        "outside of the prior bounds."

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        show_progress_bars: bool = True,
    ):
        num_samples = torch.Size(sample_shape).numel()

        samples = rejection_sample_posterior_within_prior(
            posterior_nn=self.posterior_model,
            prior=self.prior,
            x=self.xo,
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
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x` passed to `set_default_x()`.
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
        theta_repeated, x_repeated = match_theta_and_x_batch_shapes(theta, self.xo)

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
                log(self.leakage_correction(x=self.xo, **leakage_correction_params))
                if norm_posterior
                else 0
            )

            return masked_log_prob - log_factor

    @torch.no_grad()
    def leakage_correction(
        self,
        x: Tensor,
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
            x: Conditioning context for posterior $p(\theta|x)$.
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
            acceptance = acceptance_at(x)
            self._leakage_density_correction_factor = acceptance
            return acceptance
        else:
            return self._leakage_density_correction_factor
