# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional, Union

import torch
from torch import Tensor, log
from torch.distributions import Distribution

from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.neural_nets.estimators.mixture_density_estimator import (
    MixtureDensityEstimator,
)
from sbi.neural_nets.estimators.mog import MoG
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
    reshape_to_sample_batch_event,
)
from sbi.samplers.rejection import rejection
from sbi.sbi_types import Shape
from sbi.utils.sbiutils import warn_if_outside_prior_support, within_support
from sbi.utils.torchutils import ensure_theta_batched


class NPE_A_Posterior(DirectPosterior):
    """Posterior for SNPE-A with analytical correction.

    This posterior extends DirectPosterior to apply the SNPE-A correction formula:
        p(θ|x) ∝ q(θ|x) × prior(θ) / proposal(θ)

    where q(θ|x) is the density estimator output, and proposal is the distribution
    used to generate training samples (typically the previous round's posterior).

    For first-round inference (proposal = prior), no correction is needed and this
    behaves like a standard DirectPosterior.

    For multi-round inference, the correction is applied analytically since all
    distributions are Mixtures of Gaussians (MoG).
    """

    def __init__(
        self,
        posterior_estimator: MixtureDensityEstimator,
        prior: Distribution,
        proposal_mog: Optional[MoG] = None,
        prior_mog: Optional[MoG] = None,
        max_sampling_batch_size: int = 10_000,
        device: Optional[Union[str, torch.device]] = None,
        enable_transform: bool = True,
    ):
        """Initialize NPE_A_Posterior.

        Args:
            posterior_estimator: The trained MixtureDensityEstimator.
            prior: Prior distribution (MultivariateNormal or BoxUniform).
            proposal_mog: MoG parameters from the proposal distribution (previous
                round's posterior). None for first round (no correction needed).
            prior_mog: MoG representation of the prior in z-scored space. None for
                uniform priors (which have zero precision).
            max_sampling_batch_size: Batch size for rejection sampling.
            device: Device for computation.
            enable_transform: Whether to enable transforms for MAP optimization.
        """
        super().__init__(
            posterior_estimator=posterior_estimator,
            prior=prior,
            max_sampling_batch_size=max_sampling_batch_size,
            device=device,
            enable_transform=enable_transform,
        )

        # Move MoG parameters to the correct device (handles cross-device multi-round)
        self._proposal_mog = (
            proposal_mog.to(self._device).detach() if proposal_mog is not None else None
        )
        self._prior_mog = (
            prior_mog.to(self._device).detach() if prior_mog is not None else None
        )
        self._apply_correction = proposal_mog is not None

    def get_mog_params(self, x: Tensor) -> MoG:
        """Get the (possibly corrected) MoG parameters for given observation.

        This method is needed for multi-round SNPE-A where this posterior
        becomes the proposal for the next round.

        Args:
            x: Observation tensor, shape (batch_dim, *condition_shape).

        Returns:
            MoG parameters (corrected if this is a multi-round posterior).
        """
        return self._get_corrected_mog(x)

    def _get_corrected_mog(self, x: Tensor) -> MoG:
        """Get corrected MoG for the given observation.

        Args:
            x: Observation tensor, shape (batch_dim, *condition_shape).

        Returns:
            Corrected MoG if correction is needed, otherwise raw MoG from estimator.
        """
        # Import here to avoid circular imports
        from sbi.inference.trainers.npe.npe_a import _correct_for_proposal

        density_mog = self.posterior_estimator.get_uncorrected_mog(x)

        if not self._apply_correction:
            return density_mog

        # When correction is applied, proposal_mog is guaranteed to be set
        assert self._proposal_mog is not None
        return _correct_for_proposal(density_mog, self._proposal_mog, self._prior_mog)

    def _corrected_sample(self, sample_shape: torch.Size, condition: Tensor) -> Tensor:
        """Sample from the corrected MoG distribution.

        Args:
            sample_shape: Shape of samples to draw.
            condition: Conditioning observation.

        Returns:
            Samples from corrected distribution, shape (*sample_shape, batch, dim).
        """
        corrected_mog = self._get_corrected_mog(condition)
        samples = corrected_mog.sample(sample_shape)

        # Undo z-score transform if applied
        if self.posterior_estimator.has_input_transform:
            samples = self.posterior_estimator._inverse_transform_input(samples)

        return samples

    def _corrected_log_prob(self, theta: Tensor, condition: Tensor) -> Tensor:
        """Compute log probability under the corrected MoG.

        Args:
            theta: Parameters to evaluate, shape (sample_dim, batch_dim, dim).
            condition: Conditioning observation.

        Returns:
            Log probabilities, shape (sample_dim, batch_dim).
        """
        corrected_mog = self._get_corrected_mog(condition)

        # Apply z-score transform if needed
        if self.posterior_estimator.has_input_transform:
            theta_transformed = self.posterior_estimator._transform_input(theta)
        else:
            theta_transformed = theta

        log_probs = corrected_mog.log_prob(theta_transformed)

        # Add log det jacobian for z-score transform
        log_probs = log_probs + self.posterior_estimator._log_det_jacobian_forward(
            theta
        )

        return log_probs

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        max_sampling_batch_size: int = 10_000,
        sample_with: Optional[str] = None,
        show_progress_bars: bool = True,
        reject_outside_prior: bool = True,
        max_sampling_time: Optional[float] = None,
    ) -> Tensor:
        r"""Draw samples from the posterior distribution $p(\theta|x)$.

        Args:
            sample_shape: Shape of samples to draw.
            x: Conditioning observation. Uses default_x if not provided.
            max_sampling_batch_size: Batch size for rejection sampling.
            sample_with: Deprecated, raises error if set.
            show_progress_bars: Whether to show progress during sampling.
            reject_outside_prior: If True, reject samples outside prior support.
            max_sampling_time: Maximum time for sampling in seconds.

        Returns:
            Samples of shape (*sample_shape, dim).
        """
        num_samples = torch.Size(sample_shape).numel()
        x = self._x_else_default_x(x)
        x = reshape_to_batch_event(
            x, event_shape=self.posterior_estimator.condition_shape
        )
        assert x is not None  # For type checker
        if x.shape[0] > 1:
            raise ValueError(
                ".sample() supports only `batchsize == 1`. If you intend "
                "to sample multiple observations, use `.sample_batched()`. "
            )

        max_sampling_batch_size = (
            self.max_sampling_batch_size
            if max_sampling_batch_size is None
            else max_sampling_batch_size
        )

        if sample_with is not None:
            raise ValueError(
                f"You set `sample_with={sample_with}`. As of sbi v0.18.0, setting "
                f"`sample_with` is no longer supported."
            )

        if reject_outside_prior:
            samples = rejection.accept_reject_sample(
                proposal=self._corrected_sample,
                accept_reject_fn=lambda theta: within_support(self.prior, theta),
                num_samples=num_samples,
                show_progress_bars=show_progress_bars,
                max_sampling_batch_size=max_sampling_batch_size,
                proposal_sampling_kwargs={"condition": x},
                alternative_method="build_posterior(..., sample_with='mcmc')",
                max_sampling_time=max_sampling_time,
            )[0]
        else:
            samples = self._corrected_sample(torch.Size([num_samples]), condition=x)
            warn_if_outside_prior_support(self.prior, samples[:, 0])

        return samples[:, 0]  # Remove batch dimension.

    def log_prob(
        self,
        theta: Tensor,
        x: Optional[Tensor] = None,
        norm_posterior: bool = True,
        track_gradients: bool = False,
        leakage_correction_params: Optional[dict] = None,
    ) -> Tensor:
        r"""Returns the log-probability of the posterior $p(\theta|x)$.

        Args:
            theta: Parameters to evaluate.
            x: Conditioning observation. Uses default_x if not provided.
            norm_posterior: Whether to normalize for leakage correction.
            track_gradients: Whether to track gradients.
            leakage_correction_params: Parameters for leakage correction.

        Returns:
            Log probabilities for each theta value.
        """
        x = self._x_else_default_x(x)

        theta = ensure_theta_batched(torch.as_tensor(theta))
        theta_density_estimator = reshape_to_sample_batch_event(
            theta, theta.shape[1:], leading_is_sample=True
        )
        x_density_estimator = reshape_to_batch_event(
            x, event_shape=self.posterior_estimator.condition_shape
        )
        if x_density_estimator.shape[0] > 1:
            raise ValueError(
                ".log_prob() supports only `batchsize == 1`. If you intend "
                "to evaluate given multiple observations, use `.log_prob_batched()`."
            )

        self.posterior_estimator.eval()

        with torch.set_grad_enabled(track_gradients):
            unnorm_log_prob = self._corrected_log_prob(
                theta_density_estimator, x_density_estimator
            )
            unnorm_log_prob = unnorm_log_prob.squeeze(dim=1)

            # Mask outside prior support
            in_prior_support = within_support(self.prior, theta)
            masked_log_prob = torch.where(
                in_prior_support,
                unnorm_log_prob,
                torch.tensor(float("-inf"), dtype=torch.float32, device=self._device),
            )

            if leakage_correction_params is None:
                leakage_correction_params = dict()
            log_factor = (
                log(self.leakage_correction(x=x, **leakage_correction_params))
                if norm_posterior
                else 0
            )

            return masked_log_prob - log_factor

    @torch.no_grad()
    def leakage_correction(
        self,
        x: Tensor,
        num_rejection_samples: int = 10_000,
        force_update: bool = False,
        show_progress_bars: bool = False,
        rejection_sampling_batch_size: int = 10_000,
    ) -> Tensor:
        """Return leakage correction factor for posterior density estimate.

        Overrides parent to use corrected sampling.
        """

        def acceptance_at(x: Tensor) -> Tensor:
            return rejection.accept_reject_sample(
                proposal=self._corrected_sample,
                accept_reject_fn=lambda theta: within_support(self.prior, theta),
                num_samples=num_rejection_samples,
                show_progress_bars=show_progress_bars,
                sample_for_correction_factor=True,
                max_sampling_batch_size=rejection_sampling_batch_size,
                proposal_sampling_kwargs={
                    "condition": reshape_to_batch_event(
                        x, event_shape=self.posterior_estimator.condition_shape
                    )
                },
            )[1]

        is_new_x = self.default_x is None or (
            x is not self.default_x and (x != self.default_x).any()
        )
        not_saved_at_default_x = self._leakage_density_correction_factor is None

        if is_new_x:
            return acceptance_at(x)
        elif not_saved_at_default_x or force_update:
            assert self.default_x is not None
            self._leakage_density_correction_factor = acceptance_at(self.default_x)

        return self._leakage_density_correction_factor
