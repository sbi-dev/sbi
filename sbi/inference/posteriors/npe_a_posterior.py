# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import math
from typing import Dict, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal

from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.neural_nets.estimators.mixture_density_estimator import (
    MixtureDensityEstimator,
    ProposalCorrectedMDN,
)
from sbi.neural_nets.estimators.mog import MoG


class NPE_A_Posterior(DirectPosterior):
    """Posterior for SNPE-A with analytical correction.

    This posterior extends DirectPosterior to apply the SNPE-A correction formula:
        p(θ|x) ∝ q(θ|x) × prior(θ) / proposal(θ)

    where q(θ|x) is the density estimator output, and proposal is the distribution
    used to generate training samples (typically the previous round's posterior).

    For first-round inference (proposal = prior), no correction is needed and this
    behaves like a standard DirectPosterior.

    For multi-round inference, the correction is applied analytically since all
    distributions are Mixtures of Gaussians (MoG). The density estimator is then
    wrapped in a `ProposalCorrectedMDN`, so all methods use the corrected density.

    Note:
        Z-scored space: When the density estimator uses z-scoring (input
        normalization), all MoG parameters (means, precisions) are in z-scored
        coordinates. The prior_mog must also be transformed to z-scored space
        for the correction to be valid. The NPE_A trainer handles this
        transformation automatically in ``build_posterior()``.
    """

    # NPE-A does not support `sample_with='mcmc'`.
    _alternative_sampling_method = "using fewer samples or increasing max_sampling_time"

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
        estimator = (
            posterior_estimator
            if proposal_mog is None
            else ProposalCorrectedMDN(posterior_estimator, proposal_mog, prior_mog)
        )
        super().__init__(
            posterior_estimator=estimator,
            prior=prior,
            max_sampling_batch_size=max_sampling_batch_size,
            device=device,
            enable_transform=enable_transform,
        )

    def __setstate__(self, state_dict: Dict):
        """Sets the state when being loaded from pickle.

        Posteriors pickled by sbi 0.27 keep the correction MoGs on the posterior and
        an unwrapped estimator. Without re-wrapping, they would silently skip the
        correction.

        Args:
            state_dict: State to be restored.
        """
        proposal_mog = state_dict.pop("_proposal_mog", None)
        prior_mog = state_dict.pop("_prior_mog", None)
        state_dict.pop("_apply_correction", None)
        if proposal_mog is not None:
            estimator = ProposalCorrectedMDN(
                state_dict["posterior_estimator"], proposal_mog, prior_mog
            )
            state_dict["posterior_estimator"] = estimator
            state_dict["potential_fn"].posterior_estimator = estimator
            # A cached MAP was computed from the uncorrected density.
            state_dict["_map"] = None
        super().__setstate__(state_dict)

    def get_mog_params(self, x: Tensor) -> MoG:
        """Get the (possibly corrected) MoG parameters for given observation.

        This method is needed for multi-round SNPE-A where this posterior
        becomes the proposal for the next round.

        Args:
            x: Observation tensor, shape (batch_dim, *condition_shape).

        Returns:
            MoG parameters (corrected if this is a multi-round posterior).
        """
        if isinstance(self.posterior_estimator, ProposalCorrectedMDN):
            return self.posterior_estimator.get_corrected_mog(x)
        return self.posterior_estimator.get_uncorrected_mog(x)

    def as_proposal(
        self, x: Optional[Tensor] = None, widening: float = 2.0
    ) -> "GaussianProposal":
        """Return this posterior as the proposal of the next NPE-A round.

        The proposal is the posterior's Gaussian with its covariance multiplied by
        `widening`, restricted to the support of the prior. The correction divides by
        the proposal in closed form and fails, or becomes unstable, when a component
        of the next density estimator is about as wide as the proposal (Papamakarios
        & Murray 2016, Appendix C). A wider proposal keeps a margin, and the
        correction stays exact because it divides by the proposal that was used.

        Args:
            x: Observation. If None, the default x is used.
            widening: Factor on the covariance, at least 1.

        Returns:
            The Gaussian proposal.

        Raises:
            ValueError: If `widening` is smaller than 1, if `x` has more than one
                observation, or if the posterior has more than one component, i.e.
                it comes from a round trained with `final_round=True`.
        """
        if widening < 1:
            raise ValueError(f"widening must be at least 1, got {widening}.")
        x = self._x_else_default_x(x)
        if x.shape[0] != 1:
            raise ValueError(f"as_proposal() needs one observation, got {x.shape[0]}.")
        with torch.no_grad():
            mog = self.get_mog_params(x)
            if mog.num_components != 1:
                raise ValueError(
                    f"This posterior has {mog.num_components} components, so it "
                    "comes from a round trained with `final_round=True` and cannot "
                    "be the proposal of another round."
                )
            estimator = self.posterior_estimator
            if isinstance(estimator, ProposalCorrectedMDN):
                estimator = estimator.net
            gaussian = estimator.inverse_transform_mog(mog)
            covariance = torch.linalg.inv(gaussian.precisions[0, 0])
        return GaussianProposal(
            gaussian.means[0, 0], widening * covariance, support=self.prior.support
        )


class GaussianProposal(Distribution):
    """Gaussian proposal for NPE-A, restricted to the support of the prior.

    `NPE_A_Posterior.as_proposal()` builds it. Samples outside the support are
    rejected. `log_prob()` is the Gaussian's log density and -inf outside the
    support, so it lacks the normalization of the restriction; NPE-A's correction
    only uses the mean and covariance.
    """

    arg_constraints: Dict = {}

    def __init__(self, mean: Tensor, covariance_matrix: Tensor, support) -> None:
        """Initialize the proposal.

        Args:
            mean: Mean, shape (dim,).
            covariance_matrix: Covariance, shape (dim, dim).
            support: Support of the prior; samples outside it are rejected.
        """
        self._gaussian = MultivariateNormal(mean, covariance_matrix)
        self._support = support
        super().__init__(event_shape=mean.shape, validate_args=False)

    @property
    def mean(self) -> Tensor:
        """Mean of the untruncated Gaussian."""
        return self._gaussian.mean

    @property
    def covariance_matrix(self) -> Tensor:
        """Covariance of the untruncated Gaussian."""
        return self._gaussian.covariance_matrix

    @property
    def support(self):
        """Support of the prior."""
        return self._support

    def _in_support(self, theta: Tensor) -> Tensor:
        within = self._support.check(theta)
        return within.all(-1) if within.dim() > theta.dim() - 1 else within

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draw samples, rejecting those outside the support.

        Raises:
            RuntimeError: If fewer than one in 10,000 samples lies in the support.
        """
        num_samples = math.prod(sample_shape)
        accepted, num_accepted, num_drawn = [], 0, 0
        while num_accepted < num_samples:
            batch = self._gaussian.sample((max(num_samples, 1000),))
            batch = batch[self._in_support(batch)]
            accepted.append(batch)
            num_accepted += batch.shape[0]
            num_drawn += max(num_samples, 1000)
            if num_drawn >= 100_000 and num_accepted < 1e-4 * num_drawn:
                raise RuntimeError(
                    "Almost no samples of the Gaussian proposal lie in the support "
                    "of the prior. Use a smaller `widening`."
                )
        samples = torch.cat(accepted)[:num_samples]
        return samples.reshape(*sample_shape, *self.event_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        """Gaussian log density, -inf outside the support (not renormalized)."""
        log_prob = self._gaussian.log_prob(value)
        return torch.where(
            self._in_support(value), log_prob, torch.full_like(log_prob, -math.inf)
        )
