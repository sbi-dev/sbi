# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

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
        transformation automatically via ``_compute_z_scored_prior_mog()``.
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
