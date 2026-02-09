# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, Literal, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal

from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.trainers.npe.npe_base import (
    PosteriorEstimatorTrainer,
)
from sbi.neural_nets.estimators.base import (
    ConditionalDensityEstimator,
    ConditionalEstimatorBuilder,
)
from sbi.neural_nets.estimators.mixture_density_estimator import (
    MixtureDensityEstimator,
)
from sbi.neural_nets.estimators.mog import MoG
from sbi.sbi_types import TensorBoardSummaryWriter
from sbi.utils.sbiutils import del_entries
from sbi.utils.torchutils import BoxUniform

# =============================================================================
# SNPE-A Correction Functions
# =============================================================================

# Small constant for numerical stability in matrix operations
_CORRECTION_EPSILON: float = 1e-6


class NPE_A(PosteriorEstimatorTrainer):
    r"""Neural Posterior Estimation algorithm as in Papamakarios et al. (2016) [1].

    [1] *Fast epsilon-free Inference of Simulation Models with Bayesian
        Conditional Density Estimation*, Papamakarios et al., NeurIPS 2016.
        https://arxiv.org/abs/1605.06376

    Like all NPE methods, this method trains a deep neural density estimator to
    directly approximate the posterior. Also like all other NPE methods, in the
    first round, this density estimator is trained with a maximum-likelihood loss.

    This class implements NPE-A. NPE-A trains across multiple rounds with a
    maximum-likelihood loss. This will make training converge to the proposal
    posterior instead of the true posterior. To correct for this, SNPE-A applies a
    post-hoc correction after training. This correction is performed analytically
    and requires Mixture of Gaussians (MoG) density estimators.

    Note:
        In multi-round SNPE-A, the number of MoG components grows multiplicatively
        with each round: if the proposal has L components and the density estimator
        has K components, the corrected posterior has L×K components. For many
        rounds, consider using SNPE-C (APT) instead, which handles multi-round
        inference more efficiently."""

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[
            Literal["mdn_snpe_a"],
            ConditionalEstimatorBuilder[ConditionalDensityEstimator],
        ] = "mdn_snpe_a",
        num_components: int = 10,
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[TensorBoardSummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        r"""Initialize NPE-A [1].

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. Any
                object with `.log_prob()`and `.sample()` (for example, a PyTorch
                distribution) can be used.
            density_estimator: If it is a string (only "mdn_snpe_a" is valid), use a
                pre-configured mixture of densities network. Alternatively, a function
                that builds a custom neural network, which adheres to
                `ConditionalEstimatorBuilder` protocol can be provided. The function
                will be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. The
                density estimator needs to provide the methods `.log_prob` and
                `.sample()` and must return a `ConditionalDensityEstimator`.
            num_components: Number of components of the mixture of Gaussians.
                Note: In multi-round SNPE-A, the number of components grows
                multiplicatively with each round due to the analytical correction
                (L components in proposal × K components in density = L*K posterior
                components).
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: A tensorboard `SummaryWriter` to control, among others, log
                file location (default is `<current working directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during training.
        """

        # Catch invalid inputs.
        if not ((density_estimator == "mdn_snpe_a") or callable(density_estimator)):
            raise TypeError(
                "The `density_estimator` passed to SNPE_A needs to be a "
                "callable or the string 'mdn_snpe_a'!"
            )

        self._num_components = num_components

        # WARNING: sneaky trick ahead. We proxy the parent's `train` here,
        # requiring the signature to have `num_components`, save it for use below, and
        # continue. It's sneaky because we are using the object (self) as a namespace
        # to pass arguments between functions, and that's implicit state management.
        kwargs = del_entries(
            locals(),
            entries=("self", "__class__", "num_components"),
        )
        super().__init__(**kwargs)

    def train(
        self,
        training_batch_size: int = 200,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        resume_training: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
    ) -> ConditionalDensityEstimator:
        r"""Return density estimator that approximates the proposal posterior.

        [1] _Fast epsilon-free Inference of Simulation Models with Bayesian Conditional
            Density Estimation_, Papamakarios et al., NeurIPS 2016,
            https://arxiv.org/abs/1605.06376.

        Training is performed with maximum likelihood on samples from the latest round,
        which leads the algorithm to converge to the proposal posterior.

        Args:
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. Otherwise,
                we train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            calibration_kernel: A function to calibrate the loss with respect to the
                simulations `x`. See Lueckmann, Gonçalves et al., NeurIPS 2017.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round. Not supported for
                SNPE-A.
            show_train_summary: Whether to print the number of epochs and validation
                loss and leakage after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)

        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """

        assert not retrain_from_scratch, """Retraining from scratch is not supported in
            SNPE-A yet. The reason for this is that, if we reininitialized the density
            estimator, the z-scoring would change, which would break the posthoc
            correction. This is a pure implementation issue."""

        kwargs = del_entries(
            locals(),
            entries=(
                "self",
                "__class__",
            ),
        )

        # SNPE-A always discards the prior samples.
        kwargs["discard_prior_samples"] = True
        kwargs["force_first_round_loss"] = True

        self._round = max(self._data_round_index)

        # Always use the specified number of components
        self._build_neural_net = partial(
            self._build_neural_net, num_components=self._num_components
        )

        return super().train(**kwargs)

    def correct_for_proposal(
        self,
        density_estimator: Optional[ConditionalDensityEstimator] = None,
    ) -> ConditionalDensityEstimator:
        r"""Build mixture of Gaussians that approximates the posterior.

        Applies the SNPE-A post-hoc correction to the density estimator and
        returns a wrapped estimator. For round 0 (when proposal is prior), no
        correction is needed and the original estimator is returned.
        For later rounds, the correction compensates for training on samples from
        the proposal distribution rather than the prior.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.

        Returns:
            ConditionalDensityEstimator with correction applied (if needed).
            For first round, returns the original MixtureDensityEstimator.
            For later rounds, returns a _CorrectedMDN wrapper.

        Raises:
            TypeError: If density_estimator is not a MixtureDensityEstimator or
                if the prior/proposal types are not supported.
            ValueError: If the proposal posterior doesn't have default_x set or
                if default_x has batch size != 1.
        """
        if density_estimator is None:
            density_estimator = deepcopy(self._neural_net)

        # Validate density estimator type
        if not isinstance(density_estimator, MixtureDensityEstimator):
            raise TypeError(
                "NPE_A requires MixtureDensityEstimator, "
                f"got {type(density_estimator).__name__}. "
                "Use density_estimator='mdn_snpe_a' when initializing NPE_A."
            )

        # Determine the proposal for this round
        if (
            self._proposal_roundwise[-1] is self._prior
            or self._proposal_roundwise[-1] is None
        ):
            proposal = self._prior
            if not isinstance(proposal, (MultivariateNormal, BoxUniform)):
                raise TypeError(
                    "Prior must be `torch.distributions.MultivariateNormal` or "
                    f"`sbi.utils.BoxUniform`, got {type(proposal).__name__}"
                )
        else:
            if not isinstance(self._proposal_roundwise[-1], DirectPosterior):
                proposal_type = type(self._proposal_roundwise[-1]).__name__
                raise TypeError(
                    "The proposal you passed to `append_simulations` is neither the "
                    "prior nor a `DirectPosterior`. SNPE-A currently only supports "
                    f"these scenarios. Got {proposal_type}"
                )
            proposal = self._proposal_roundwise[-1]

        # First round: proposal is prior, no correction needed
        if isinstance(proposal, (MultivariateNormal, BoxUniform)):
            return density_estimator

        # Later rounds: proposal is DirectPosterior from previous round
        # Get the default_x from the proposal (needed to evaluate the proposal MoG)
        default_x = proposal.default_x
        if default_x is None:
            raise ValueError(
                "Proposal posterior must have a default_x set for SNPE-A correction."
            )

        # Validate batch size - SNPE-A only supports single observation
        if default_x.shape[0] != 1:
            raise ValueError(
                f"SNPE-A requires default_x batch size of 1, got {default_x.shape[0]}. "
                "SNPE-A only supports single observations for correction."
            )

        # Get the proposal MoG from the previous round's posterior
        proposal_estimator = proposal.posterior_estimator
        if isinstance(proposal_estimator, MixtureDensityEstimator):
            # Round 2: proposal is raw MDN from round 1
            proposal_mog = proposal_estimator.get_uncorrected_mog(default_x)
        elif isinstance(proposal_estimator, _CorrectedMDN):
            # Round 3+: proposal is corrected MDN from previous round
            proposal_mog = proposal_estimator.get_corrected_mog(default_x)
        else:
            raise TypeError(
                f"Proposal posterior estimator must be MixtureDensityEstimator, "
                f"got {type(proposal_estimator).__name__}. Multi-round SNPE-A "
                "requires consistent use of MixtureDensityEstimator across rounds."
            )

        # Compute the z-scored prior as a MoG (or None for uniform priors)
        prior_mog = self._compute_z_scored_prior_mog(density_estimator)

        # Return wrapped estimator with correction
        return _CorrectedMDN(
            estimator=density_estimator,
            proposal_mog=proposal_mog,
            prior_mog=prior_mog,
        )

    def _compute_z_scored_prior_mog(
        self, density_estimator: MixtureDensityEstimator
    ) -> Optional[MoG]:
        """Compute the prior as a MoG in z-scored space (if applicable).

        For SNPE-A correction, the prior needs to be in the same coordinate system
        as the density estimator's output. When z-scoring is applied to inputs,
        the density estimator outputs MoG parameters in z-scored space, so the
        prior must also be transformed to z-scored space.

        For uniform priors (BoxUniform), returns None since uniform priors have
        zero precision (infinite covariance) and are handled specially in the
        correction formula.

        Mathematical background:
            For z-score transform: z = (theta - shift) / scale
            If theta ~ N(mu, Sigma), then:
            z ~ N((mu - shift) / scale, Sigma / (scale ⊗ scale))
            where (scale ⊗ scale)_ij = scale_i * scale_j

        Args:
            density_estimator: The MixtureDensityEstimator (to get z-score parameters).

        Returns:
            MoG representation of the z-scored prior for Gaussian priors,
            or None for uniform priors.
        """
        # Uniform priors have zero precision, return None
        if isinstance(self._prior, BoxUniform):
            return None

        if not isinstance(self._prior, MultivariateNormal):
            raise TypeError(
                f"Prior must be MultivariateNormal or BoxUniform, "
                f"got {type(self._prior).__name__}"
            )

        # Get prior parameters
        prior_mean = self._prior.mean
        prior_cov = self._prior.covariance_matrix

        # Apply z-score transform if enabled
        if density_estimator.has_input_transform:
            shift = density_estimator._transform_shift
            scale = density_estimator._transform_scale

            # Validate z-score parameters
            if not torch.all(torch.isfinite(shift)):
                raise ValueError(
                    "Z-score shift contains non-finite values. "
                    "Check training data for NaN/Inf values."
                )
            if not torch.all(torch.isfinite(scale)):
                raise ValueError(
                    "Z-score scale contains non-finite values. "
                    "Check training data for NaN/Inf values."
                )
            if torch.any(scale.abs() < 1e-10):
                raise ValueError(
                    "Z-score scale contains near-zero values, which would cause "
                    "numerical instability. This may indicate constant or "
                    "near-constant features in the training data."
                )

            # Z-scored mean: (mu - shift) / scale
            z_mean = (prior_mean - shift) / scale

            # Z-scored covariance: Sigma_z[i,j] = Sigma[i,j] / (scale_i * scale_j)
            scale_outer = scale.unsqueeze(-1) * scale.unsqueeze(-2)
            z_cov = prior_cov / scale_outer
        else:
            z_mean = prior_mean
            z_cov = prior_cov

        # Validate covariance is positive definite
        try:
            torch.linalg.cholesky(z_cov)
        except RuntimeError as e:
            raise ValueError(
                "Z-scored prior covariance is not positive definite. "
                "This may indicate numerical issues with the z-score transform. "
                f"Original error: {e}"
            ) from e

        # Convert to MoG
        return MoG.from_gaussian(z_mean, z_cov)

    def build_posterior(
        self,
        density_estimator: Optional[torch.nn.Module] = None,
        prior: Optional[Distribution] = None,
        **kwargs,
    ) -> "DirectPosterior":
        r"""Build posterior from the neural density estimator.

        This method first corrects the estimated density with `correct_for_proposal`
        and then returns a `DirectPosterior`.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
            prior: Prior distribution.

        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods.
        """
        if prior is None:
            assert (
                self._prior is not None
            ), """You did not pass a prior. You have to pass the prior either at
                initialization `inference = SNPE_A(prior)` or to `.build_posterior
                (prior=prior)`."""
            prior = self._prior

        wrapped_density_estimator = self.correct_for_proposal(
            density_estimator=density_estimator  # type: ignore[arg-type]
        )
        self._posterior = super().build_posterior(
            density_estimator=wrapped_density_estimator,
            prior=prior,
            **kwargs,
        )
        return deepcopy(self._posterior)  # type: ignore

    def _log_prob_proposal_posterior(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
    ) -> Tensor:
        """Return the log-probability of the proposal posterior.

        For SNPE-A this is the same as `self._neural_net.log_prob(theta, x)` in
        `_loss()` to be found in `snpe_base.py`.

        Args:
            theta: Batch of parameters θ.
            x: Batch of data.
            masks: Mask that is True for prior samples in the batch in order to train
                them with prior loss.
            proposal: Proposal distribution.

        Returns: Log-probability of the proposal posterior.
        """
        return self._neural_net.log_prob(theta, x)


def _correct_for_proposal(
    density_mog: MoG,
    proposal_mog: MoG,
    prior_mog: Optional[MoG] = None,
) -> MoG:
    """Compute SNPE-A corrected posterior from density estimator output.

    Implements Appendix C of Papamakarios et al. 2016 (SNPE-A paper).

    The true posterior is computed as:
        posterior = density_estimator * prior / proposal

    Since all distributions are MoGs, this can be done analytically.
    If the proposal has L components and the density has K components,
    the posterior has L*K components.

    For uniform priors, pass `prior_mog=None`. The prior term is then omitted
    from the correction (uniform has zero precision).

    Warning:
        Component count grows multiplicatively across rounds. In multi-round
        SNPE-A, if round N has L components, round N+1 will have L*K components
        (where K is the density estimator's component count). This can lead to
        memory issues for many rounds. However, the original SNPE-A algorithm
        uses a single Gaussian (K=1) in intermediate rounds, expanding to
        multiple components only in the final round, which avoids this issue.

    Args:
        density_mog: MoG from the density estimator for current observation.
        proposal_mog: MoG from previous round's proposal distribution.
        prior_mog: MoG representation of the prior. Use `MoG.from_gaussian()`
            for Gaussian priors. Pass None for uniform priors.

    Returns:
        Corrected MoG representing the true posterior.

    Raises:
        ValueError: If posterior precision is not positive definite.
    """
    num_comps_proposal = proposal_mog.num_components
    num_comps_density = density_mog.num_components
    dim = density_mog.dim

    # Compute posterior precisions (Eq. 23)
    # prec_post = prec_density - prec_proposal + prec_prior
    # For uniform priors, prec_prior = 0
    prec_proposal_rep = proposal_mog.precisions.repeat_interleave(
        num_comps_density, dim=1
    )
    prec_density_rep = density_mog.precisions.repeat(1, num_comps_proposal, 1, 1)

    prec_post = prec_density_rep - prec_proposal_rep

    # Add prior precision term only for Gaussian priors
    if prior_mog is not None:
        prec_prior_rep = prior_mog.precisions.repeat(
            1, num_comps_proposal * num_comps_density, 1, 1
        )
        prec_post = prec_post + prec_prior_rep

    # Add small epsilon to diagonal for numerical stability
    eye = torch.eye(dim, device=prec_post.device, dtype=prec_post.dtype)
    prec_post_stabilized = prec_post + _CORRECTION_EPSILON * eye

    # Compute precision factors via Cholesky (also validates positive definiteness)
    try:
        precf_post = torch.linalg.cholesky(prec_post_stabilized, upper=True)
    except torch.linalg.LinAlgError as e:
        raise ValueError(
            "Posterior precision matrix is not positive definite. "
            "This is a known issue with SNPE-A when the proposal and density "
            "estimator don't align well. Try different hyperparameters. "
            f"Original error: {e}"
        ) from e

    # Compute posterior covariances using solve for numerical stability
    batch_shape = prec_post_stabilized.shape[:-2]
    eye_expanded = eye.expand(*batch_shape, dim, dim)
    cov_post = torch.linalg.solve(prec_post_stabilized, eye_expanded)

    # Compute posterior means (Eq. 24)
    # mean_post = cov_post @ (prec_density @ mean_density
    #                         - prec_proposal @ mean_proposal
    #                         + prec_prior @ mean_prior)
    prec_mean_proposal = _batched_mv(proposal_mog.precisions, proposal_mog.means)
    prec_mean_density = _batched_mv(density_mog.precisions, density_mog.means)

    prec_mean_proposal_rep = prec_mean_proposal.repeat_interleave(
        num_comps_density, dim=1
    )
    prec_mean_density_rep = prec_mean_density.repeat(1, num_comps_proposal, 1)

    summed_prec_mean = prec_mean_density_rep - prec_mean_proposal_rep

    # Add prior mean term only for Gaussian priors
    if prior_mog is not None:
        prec_mean_prior = _batched_mv(prior_mog.precisions, prior_mog.means)
        prec_mean_prior_rep = prec_mean_prior.repeat(
            1, num_comps_proposal * num_comps_density, 1
        )
        summed_prec_mean = summed_prec_mean + prec_mean_prior_rep

    mean_post = _batched_mv(cov_post, summed_prec_mean)

    # Compute posterior logits (Eqs. 25-26)
    logits_post = _compute_posterior_logits(
        mean_post,
        prec_post,
        cov_post,
        proposal_mog.logits,
        proposal_mog.means,
        proposal_mog.precisions,
        density_mog.logits,
        density_mog.means,
        density_mog.precisions,
        num_comps_proposal,
        num_comps_density,
    )

    return MoG(
        logits=logits_post,
        means=mean_post,
        precisions=prec_post_stabilized,
        precision_factors=precf_post,
    )


def _compute_posterior_logits(
    mean_post: Tensor,
    prec_post: Tensor,
    cov_post: Tensor,
    logits_proposal: Tensor,
    mean_proposal: Tensor,
    prec_proposal: Tensor,
    logits_density: Tensor,
    mean_density: Tensor,
    prec_density: Tensor,
    num_comps_proposal: int,
    num_comps_density: int,
) -> Tensor:
    """Compute posterior logits using Eqs. 25-26 from SNPE-A paper.

    Computes unnormalized log weights for the posterior MoG components.
    The formula combines logit differences, log-determinant ratios, and
    quadratic form differences from the proposal, density, and posterior.

    Args:
        mean_post: Posterior means, shape (batch, L*K, dim).
        prec_post: Posterior precisions, shape (batch, L*K, dim, dim).
        cov_post: Posterior covariances, shape (batch, L*K, dim, dim).
        logits_proposal: Proposal logits, shape (batch, L).
        mean_proposal: Proposal means, shape (batch, L, dim).
        prec_proposal: Proposal precisions, shape (batch, L, dim, dim).
        logits_density: Density logits, shape (batch, K).
        mean_density: Density means, shape (batch, K, dim).
        prec_density: Density precisions, shape (batch, K, dim, dim).
        num_comps_proposal: Number of proposal components (L).
        num_comps_density: Number of density components (K).

    Returns:
        Posterior logits of shape (batch, L*K).
    """
    # Compute logit factors
    logits_proposal_rep = logits_proposal.repeat_interleave(num_comps_density, dim=1)
    logits_density_rep = logits_density.repeat(1, num_comps_proposal)
    logit_factors = logits_density_rep - logits_proposal_rep

    # Compute log-determinant terms using slogdet for numerical stability
    _, logdet_cov_post = torch.linalg.slogdet(cov_post)
    _, logdet_prec_proposal = torch.linalg.slogdet(prec_proposal)
    _, logdet_prec_density = torch.linalg.slogdet(prec_density)
    logdet_cov_proposal = -logdet_prec_proposal
    logdet_cov_density = -logdet_prec_density

    logdet_cov_proposal_rep = logdet_cov_proposal.repeat_interleave(
        num_comps_density, dim=1
    )
    logdet_cov_density_rep = logdet_cov_density.repeat(1, num_comps_proposal)

    log_sqrt_det_ratio = 0.5 * (
        logdet_cov_post + logdet_cov_proposal_rep - logdet_cov_density_rep
    )

    # Compute quadratic form terms (m^T P m)
    exponent_proposal = _batched_vmv(prec_proposal, mean_proposal)
    exponent_density = _batched_vmv(prec_density, mean_density)
    exponent_post = _batched_vmv(prec_post, mean_post)

    exponent_proposal_rep = exponent_proposal.repeat_interleave(
        num_comps_density, dim=1
    )
    exponent_density_rep = exponent_density.repeat(1, num_comps_proposal)

    exponent = -0.5 * (exponent_density_rep - exponent_proposal_rep - exponent_post)

    return logit_factors + log_sqrt_det_ratio + exponent


def _batched_mv(matrix: Tensor, vector: Tensor) -> Tensor:
    """Batched matrix-vector product with component dimension.

    Args:
        matrix: Shape (batch, num_components, dim, dim).
        vector: Shape (batch, num_components, dim).

    Returns:
        Product of shape (batch, num_components, dim).
    """
    return torch.einsum("bcij,bcj->bci", matrix, vector)


def _batched_vmv(matrix: Tensor, vector: Tensor) -> Tensor:
    """Batched vector-matrix-vector product (quadratic form).

    Args:
        matrix: Shape (batch, num_components, dim, dim).
        vector: Shape (batch, num_components, dim).

    Returns:
        Quadratic form v^T M v of shape (batch, num_components).
    """
    mv = torch.einsum("bcij,bcj->bci", matrix, vector)
    return torch.einsum("bci,bci->bc", vector, mv)


# =============================================================================
# Corrected MDN Wrapper
# =============================================================================


class _CorrectedMDN(ConditionalDensityEstimator):
    """Wrapper that applies SNPE-A correction to a MixtureDensityEstimator.

    This class wraps a trained MixtureDensityEstimator and applies the SNPE-A
    post-hoc correction on every call to log_prob() and sample(). The correction
    compensates for training on samples from a proposal distribution rather than
    the prior.

    This class is internal to NPE_A and should not be used directly.

    Note:
        This class intentionally accesses private methods of the wrapped
        MixtureDensityEstimator (e.g., _transform_input, _inverse_transform_input)
        to apply the same z-score transforms. This tight coupling is by design:
        _CorrectedMDN is specifically built to wrap MixtureDensityEstimator and
        needs intimate knowledge of its internals to apply corrections correctly.
    """

    def __init__(
        self,
        estimator: MixtureDensityEstimator,
        proposal_mog: MoG,
        prior_mog: Optional[MoG],
    ) -> None:
        """Initialize the corrected MDN wrapper.

        Args:
            estimator: The trained MixtureDensityEstimator to wrap.
            proposal_mog: MoG from previous round's posterior (the proposal).
            prior_mog: MoG representation of the (z-scored) prior, or None for
                uniform priors.
        """
        # Initialize base class with the wrapped estimator's properties
        super().__init__(
            net=estimator.net,
            input_shape=estimator.input_shape,
            condition_shape=estimator.condition_shape,
        )
        self._estimator = estimator
        self._proposal_mog = proposal_mog.detach()
        self._prior_mog = prior_mog.detach() if prior_mog is not None else None

    def _get_corrected_mog(self, condition: Tensor) -> MoG:
        """Get corrected MoG for the given condition."""
        density_mog = self._estimator.get_uncorrected_mog(condition)
        return _correct_for_proposal(density_mog, self._proposal_mog, self._prior_mog)

    def get_corrected_mog(self, condition: Tensor) -> MoG:
        """Get the corrected MoG for the given condition.

        This method is needed for multi-round SNPE-A where a corrected estimator
        becomes the proposal for subsequent rounds.

        Args:
            condition: Conditioning input, shape (batch_dim, *condition_shape).

        Returns:
            Corrected MoG for the given condition.
        """
        return self._get_corrected_mog(condition)

    def log_prob(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        """Compute corrected log probability.

        Args:
            input: Inputs to evaluate, shape (sample_dim, batch_dim, *input_shape)
                or (batch_dim, *input_shape).
            condition: Conditions, shape (batch_dim, *condition_shape).

        Returns:
            Log probabilities with same shape convention as input.
        """
        self._check_condition_shape(condition)
        self._check_input_shape(input)

        # Handle input with or without sample dimension
        has_sample_dim = input.dim() > len(self.input_shape) + 1
        if not has_sample_dim:
            input = input.unsqueeze(0)

        # Apply z-score transform if the estimator has one
        if self._estimator.has_input_transform:
            transformed_input = self._estimator._transform_input(input)
        else:
            transformed_input = input

        # Get corrected MoG and compute log prob
        corrected_mog = self._get_corrected_mog(condition)
        log_probs = corrected_mog.log_prob(transformed_input)

        # Add log det jacobian for z-score transform
        log_probs = log_probs + self._estimator._log_det_jacobian_forward(input)

        if not has_sample_dim:
            log_probs = log_probs.squeeze(0)

        return log_probs

    def loss(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        """Training loss is not supported for corrected estimators.

        Raises:
            NotImplementedError: Always raised. Corrected estimators are meant
                for inference only, not for training.
        """
        raise NotImplementedError(
            "_CorrectedMDN is a post-hoc corrected estimator for inference only. "
            "It cannot be used for training. Use the underlying "
            "MixtureDensityEstimator for training instead."
        )

    def sample(self, sample_shape: torch.Size, condition: Tensor, **kwargs) -> Tensor:
        """Sample from the corrected distribution.

        Args:
            sample_shape: Shape prefix for samples.
            condition: Conditions, shape (batch_dim, *condition_shape).

        Returns:
            Samples, shape (*sample_shape, batch_dim, *input_shape).
        """
        self._check_condition_shape(condition)

        # Get corrected MoG and sample
        corrected_mog = self._get_corrected_mog(condition)
        samples = corrected_mog.sample(sample_shape)

        # Apply inverse z-score transform if needed
        if self._estimator.has_input_transform:
            samples = self._estimator._inverse_transform_input(samples)

        return samples

    @property
    def embedding_net(self) -> torch.nn.Module:
        """Return the embedding network from the wrapped estimator."""
        return self._estimator.embedding_net
