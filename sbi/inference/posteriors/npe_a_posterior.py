# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from typing import Dict, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.neural_nets.estimators.mixture_density_estimator import (
    MixtureDensityEstimator,
    ProposalCorrectedMDN,
)
from sbi.neural_nets.estimators.mog import MoG
from sbi.neural_nets.estimators.shape_handling import reshape_to_batch_event

# A corrected posterior this much wider (in variance) than its proposal in some
# direction is a sign of an unstable correction.
_UNSTABLE_WIDTH_RATIO: float = 5.0


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

    def set_default_x(self, x: Tensor) -> "NPE_A_Posterior":
        """Set the default x, and warn if the correction looks unstable for it.

        With a Gaussian prior, a corrected posterior much wider than its proposal
        means that a component of the density estimator was barely narrower than
        the proposal, which makes the closed-form correction unstable (Papamakarios
        & Murray 2016, Appendix C).

        Args:
            x: The default observation.

        Returns:
            The posterior, with `x` as its default.
        """
        super().set_default_x(x)
        estimator = self.posterior_estimator
        if isinstance(estimator, ProposalCorrectedMDN):
            condition = reshape_to_batch_event(
                self.default_x, estimator.condition_shape
            )
            ratio = estimator.width_ratio_to_proposal(condition)
            if ratio is not None and ratio > _UNSTABLE_WIDTH_RATIO:
                warnings.warn(
                    f"The NPE-A posterior is {ratio:.1f} times wider (in variance) "
                    "than its proposal in some direction, which usually means the "
                    "closed-form correction is unstable (Papamakarios & Murray "
                    "2016, Appendix C). See the NPE-A section of the how-to guide "
                    "'How to run sequential methods', or use NPE_C.",
                    UserWarning,
                    stacklevel=2,
                )
        return self

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
