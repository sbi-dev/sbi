# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, Literal, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.inference.posteriors.npe_a_posterior import NPE_A_Posterior
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
from sbi.sbi_types import Tracker
from sbi.utils.sbiutils import del_entries
from sbi.utils.torchutils import BoxUniform

# Constant for numerical stability in matrix operations.
_CORRECTION_EPSILON: float = 1e-4


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
        summary_writer: Optional[SummaryWriter] = None,
        tracker: Optional[Tracker] = None,
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
            summary_writer: Deprecated alias for the TensorBoard summary writer.
                Use ``tracker`` instead.
            tracker: Tracking adapter used to log training metrics. If None, a
                TensorBoard tracker is used with a default log directory.
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

        if len(self._data_round_index) == 0:
            raise RuntimeError(
                "No simulations found. You must call .append_simulations() "
                "before calling .train()."
            )

        self._round = max(self._data_round_index)

        # Always use the specified number of components
        self._build_neural_net = partial(
            self._build_neural_net, num_components=self._num_components
        )

        density_estimator = super().train(**kwargs)

        return density_estimator

    def _get_proposal_mog(
        self,
        proposal: Union["NPE_A_Posterior", MultivariateNormal, MoG, Any],
    ) -> MoG:
        """Extract MoG parameters from a proposal distribution.

        Supports multiple proposal types:
        - NPE_A_Posterior: extracts corrected MoG via get_mog_params()
        - MultivariateNormal: converts to single-component MoG
        - MoG: uses directly
        - Any object with get_mog_params(x) method

        Args:
            proposal: The proposal distribution from the previous round.

        Returns:
            MoG parameters from the proposal.

        Raises:
            ValueError: If NPE_A_Posterior proposal doesn't have default_x set.
            TypeError: If proposal type is not supported.
        """
        if isinstance(proposal, NPE_A_Posterior):
            default_x = proposal.default_x
            if default_x is None:
                raise ValueError(
                    "Proposal posterior must have a default_x set for SNPE-A "
                    "correction. Call posterior.set_default_x(x_o) before using "
                    "as proposal."
                )
            if default_x.shape[0] != 1:
                raise ValueError(
                    f"SNPE-A requires default_x batch size of 1, got "
                    f"{default_x.shape[0]}. SNPE-A only supports single "
                    "observations for correction."
                )
            return proposal.get_mog_params(default_x)

        if isinstance(proposal, MultivariateNormal):
            mean: Tensor = proposal.mean.to(self._device)  # type: ignore[assignment]
            cov: Tensor = proposal.covariance_matrix.to(  # type: ignore[assignment]
                self._device
            )
            return MoG.from_gaussian(mean.unsqueeze(0), cov.unsqueeze(0))

        if isinstance(proposal, MoG):
            return proposal.to(self._device)

        # Case 4: Any object with get_mog_params method
        if hasattr(proposal, "get_mog_params"):
            # Try to get default_x if available
            default_x = getattr(proposal, "default_x", None)
            if default_x is None:
                raise ValueError(
                    "Proposal has get_mog_params() but no default_x set. "
                    "Call proposal.set_default_x(x_o) before using as proposal."
                )
            if default_x.shape[0] != 1:
                raise ValueError(
                    f"SNPE-A requires default_x batch size of 1, got "
                    f"{default_x.shape[0]}."
                )
            mog = proposal.get_mog_params(default_x)
            if not isinstance(mog, MoG):
                raise TypeError(
                    f"Proposal's get_mog_params() must return MoG, "
                    f"got {type(mog).__name__}."
                )
            return mog.to(self._device)

        # Unsupported type
        raise TypeError(
            f"For multi-round SNPE-A, proposal must be one of: NPE_A_Posterior, "
            f"MultivariateNormal, MoG, or an object with get_mog_params() method. "
            f"Got {type(proposal).__name__}. For custom proposals, construct "
            f"NPE_A_Posterior directly with your proposal_mog parameter."
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
        density_estimator: Optional[ConditionalDensityEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: Literal["direct"] = "direct",
        **kwargs,
    ) -> NPE_A_Posterior:
        r"""Build posterior from the neural density estimator.

        Returns an NPE_A_Posterior that applies the SNPE-A correction formula:
            p(θ|x) ∝ q(θ|x) × prior(θ) / proposal(θ)

        Note:
            NPE_A only supports `sample_with="direct"`. The corrected posterior is a
            Mixture of Gaussians (MoG) which can be sampled directly and efficiently.
            MCMC, VI, rejection, and importance sampling methods do not provide
            benefits over direct MoG sampling and are therefore not supported.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
            prior: Prior distribution.
            sample_with: Must be "direct". Other sampling methods are not supported.
            **kwargs: Additional arguments passed to NPE_A_Posterior.

        Returns:
            NPE_A_Posterior with the SNPE-A correction applied.

        Raises:
            ValueError: If sample_with is not "direct".
        """
        if sample_with != "direct":
            raise ValueError(
                f"NPE_A only supports sample_with='direct', got '{sample_with}'. "
                "The corrected posterior is a Mixture of Gaussians which can be "
                "sampled directly and efficiently. MCMC, VI, rejection, and "
                "importance sampling do not provide benefits over direct MoG sampling."
            )

        if prior is None:
            assert self._prior is not None, (
                "You did not pass a prior. You have to pass the prior either at "
                "initialization `inference = NPE_A(prior)` or to "
                "`.build_posterior(prior=prior)`."
            )
            prior = self._prior

        # Resolve and validate density estimator
        if density_estimator is None:
            density_estimator = deepcopy(self._neural_net)

        if not isinstance(density_estimator, MixtureDensityEstimator):
            raise TypeError(
                "NPE_A requires MixtureDensityEstimator, "
                f"got {type(density_estimator).__name__}. "
                "Use density_estimator='mdn_snpe_a' when initializing NPE_A."
            )

        # Compute correction parameters
        proposal = self._proposal_roundwise[-1]
        is_first_round = proposal is self._prior or proposal is None

        if is_first_round:
            proposal_mog = None
            prior_mog = None
        else:
            proposal_mog = self._get_proposal_mog(proposal)
            prior_mog = self._compute_z_scored_prior_mog(density_estimator)

        # Build the posterior
        self._posterior = NPE_A_Posterior(
            posterior_estimator=density_estimator,
            prior=prior,
            proposal_mog=proposal_mog,
            prior_mog=prior_mog,
            device=self._device,
            **kwargs,
        )

        return self._posterior

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
