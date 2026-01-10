# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
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
    MultivariateGaussianMDN,
)
from sbi.sbi_types import TensorBoardSummaryWriter
from sbi.utils.sbiutils import del_entries
from sbi.utils.torchutils import BoxUniform


class NPE_A(PosteriorEstimatorTrainer):
    r"""Neural Posterior Estimation algorithm as in Papamakarios et al. (2016) [1].

    [1] *Fast epsilon-free Inference of Simulation Models with Bayesian
        Conditional Density Estimation*, Papamakarios et al., NeurIPS 2016.
        https://arxiv.org/abs/1605.06376

    Like all NPE methods, this method trains a deep neural density estimator to
    directly approximate the posterior. Also like all other NPE methods, in the
    first round, this density estimator is trained with a maximum-likelihood loss.

    This class implements NPE-A. NPE-A trains across multiple rounds with a
    maximum-likelihood-loss. This will make training converge to the proposal
    posterior instead of the true posterior. To correct for this, SNPE-A applies a
    post-hoc correction after training. This correction has to be performed
    analytically. Thus, NPE-A is limited to Gaussian distributions for all but the
    last round. In the last round, NPE-A can use a Mixture of Gaussians."""

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
                Note that until the last round only a single (multivariate) Gaussian
                component is used for training (seeAlgorithm 1 in [1]). In the last
                round, this component is replicated `num_components` times,
                its parameters are perturbed with a very small noise, and then the last
                training round is done with the expanded Gaussian mixture as estimator
                for the proposal posterior.
            num_components: Number of components of the mixture of Gaussians in the
                last round. This overrides the `num_components` value passed to
                `posterior_nn()`.
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

        # `num_components` will be used to replicate the Gaussian in the last round.
        self._num_components = num_components
        self._ran_final_round = False

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
        final_round: bool = False,
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
        component_perturbation: float = 5e-3,
    ) -> ConditionalDensityEstimator:
        r"""Return density estimator that approximates the proposal posterior.

        [1] _Fast epsilon-free Inference of Simulation Models with Bayesian Conditional
            Density Estimation_, Papamakarios et al., NeurIPS 2016,
            https://arxiv.org/abs/1605.06376.

        Training is performed with maximum likelihood on samples from the latest round,
        which leads the algorithm to converge to the proposal posterior.

        Args:
            final_round: Whether we are in the last round of training or not. For all
                but the last round, Algorithm 1 from [1] is executed. In last the
                round, Algorithm 2 from [1] is executed once.
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
            component_perturbation: The standard deviation applied to all weights and
                biases when, in the last round, the Mixture of Gaussians is build from
                a single Gaussian. This value can be problem-specific and also depends
                on the number of mixture components.

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
                "final_round",
                "component_perturbation",
            ),
        )

        # SNPE-A always discards the prior samples.
        kwargs["discard_prior_samples"] = True
        kwargs["force_first_round_loss"] = True

        self._round = max(self._data_round_index)

        if final_round:
            # If there is (will be) only one round, train with Algorithm 2 from [1].
            if self._round == 0:
                self._build_neural_net = partial(
                    self._build_neural_net, num_components=self._num_components
                )
            # Run Algorithm 2 from [1].
            elif not self._ran_final_round:
                # Now switch to the specified number of components. This method will
                # only be used if `retrain_from_scratch=True`. Otherwise,
                # the MDN will be built from replicating the single-component net for
                # `num_component` times (via `_expand_mog()`).
                self._build_neural_net = partial(
                    self._build_neural_net, num_components=self._num_components
                )

                # Extend the MDN to the originally desired number of components.
                self._expand_mog(eps=component_perturbation)
            else:
                warnings.warn(
                    "You have already run SNPE-A with `final_round=True`. Running it"
                    "again with this setting will not allow computing the posthoc"
                    "correction applied in SNPE-A. Thus, you will get an error when "
                    "calling `.build_posterior()` after training.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            # Run Algorithm 1 from [1].
            # Wrap the function that builds the MDN such that we can make
            # sure that there is only one component when running.
            self._build_neural_net = partial(self._build_neural_net, num_components=1)

        if final_round:
            self._ran_final_round = True

        return super().train(**kwargs)

    def correct_for_proposal(
        self,
        density_estimator: Optional[MixtureDensityEstimator] = None,
    ) -> MixtureDensityEstimator:
        r"""Build mixture of Gaussians that approximates the posterior.

        Applies the SNPE-A post-hoc correction to the density estimator and
        returns it. For round 0 (when proposal is prior), no correction is needed.
        For later rounds, the correction compensates for training on samples from
        the proposal distribution rather than the prior.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.

        Returns:
            MixtureDensityEstimator with correction applied (if needed).

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
        if not isinstance(proposal_estimator, MixtureDensityEstimator):
            raise TypeError(
                f"Proposal posterior estimator must be MixtureDensityEstimator, "
                f"got {type(proposal_estimator).__name__}. Multi-round SNPE-A "
                "requires consistent use of MixtureDensityEstimator across rounds."
            )

        proposal_mog = proposal_estimator.get_mog(default_x)

        # Compute the z-scored prior if z-scoring is enabled
        z_scored_prior = self._compute_z_scored_prior(density_estimator)

        # Validate z-scored prior
        self._validate_z_scored_prior(z_scored_prior)

        # Apply the correction
        density_estimator.apply_correction(proposal_mog.detach(), z_scored_prior)

        return density_estimator

    def _compute_z_scored_prior(
        self, density_estimator: MixtureDensityEstimator
    ) -> Union[MultivariateNormal, BoxUniform]:
        """Compute the prior transformed to z-scored space if z-scoring is enabled.

        For SNPE-A correction, the prior needs to be in the same coordinate system
        as the density estimator's output. When z-scoring is applied to inputs,
        the density estimator outputs MoG parameters in z-scored space, so the
        prior must also be transformed to z-scored space.

        Mathematical background:
            For z-score transform: z = (theta - shift) / scale
            If theta ~ N(mu, Sigma), then:
            z ~ N((mu - shift) / scale, Sigma / (scale ⊗ scale))
            where (scale ⊗ scale)_ij = scale_i * scale_j

        Args:
            density_estimator: The MixtureDensityEstimator (to get z-score parameters).

        Returns:
            The (potentially z-scored) prior distribution.
        """
        if not density_estimator.has_input_transform:
            # No z-scoring, return original prior
            return self._prior

        # Get the z-score transform parameters
        # In our convention: z = (theta - shift) / scale
        # where shift = estimated mean, scale = estimated std
        shift = density_estimator._transform_shift
        scale = density_estimator._transform_scale

        if isinstance(self._prior, MultivariateNormal):
            # Transform mean and covariance to z-space
            prior_mean = self._prior.mean
            prior_cov = self._prior.covariance_matrix

            # Z-scored mean: (mu - shift) / scale
            z_mean = (prior_mean - shift) / scale

            # Z-scored covariance: Sigma_z[i,j] = Sigma[i,j] / (scale_i * scale_j)
            scale_outer = scale.unsqueeze(-1) * scale.unsqueeze(-2)
            z_cov = prior_cov / scale_outer

            return MultivariateNormal(z_mean, z_cov)

        elif isinstance(self._prior, BoxUniform):
            # Transform bounds to z-space
            z_low = (self._prior.low - shift) / scale
            z_high = (self._prior.high - shift) / scale
            return BoxUniform(z_low, z_high)

        else:
            raise TypeError(
                f"Prior must be MultivariateNormal or BoxUniform, "
                f"got {type(self._prior).__name__}"
            )

    def _validate_z_scored_prior(
        self, z_scored_prior: Union[MultivariateNormal, BoxUniform]
    ) -> None:
        """Validate that the z-scored prior has valid parameters.

        Args:
            z_scored_prior: The z-scored prior to validate.

        Raises:
            ValueError: If the z-scored prior has invalid parameters.
        """
        if isinstance(z_scored_prior, MultivariateNormal):
            # Check that covariance is positive definite
            try:
                torch.linalg.cholesky(z_scored_prior.covariance_matrix)
            except RuntimeError as e:
                raise ValueError(
                    "Z-scored prior covariance is not positive definite. "
                    "This may indicate numerical issues with the z-score transform. "
                    f"Original error: {e}"
                ) from e
        elif isinstance(z_scored_prior, BoxUniform) and not torch.all(
            z_scored_prior.low < z_scored_prior.high
        ):
            raise ValueError(
                "Z-scored prior bounds are invalid (low >= high). "
                "This may indicate issues with the z-score transform parameters."
            )

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
            density_estimator=density_estimator
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

    def _expand_mog(self, eps: float = 1e-5):
        """Replicate a single Gaussian to multiple components for final round.

        In SNPE-A, Algorithm 1 trains a single-component Gaussian. Before the
        final round (Algorithm 2), this method expands it to `num_components`
        mixture components by replicating the output layer parameters and adding
        small perturbations to break symmetry.

        Args:
            eps: Standard deviation for the random perturbation added to break
                symmetry between replicated components.

        Raises:
            TypeError: If the neural network is not a MixtureDensityEstimator
                or doesn't contain a MultivariateGaussianMDN.
        """
        if not isinstance(self._neural_net, MixtureDensityEstimator):
            raise TypeError(
                "Expected MixtureDensityEstimator, "
                f"got {type(self._neural_net).__name__}. "
                "Use density_estimator='mdn_snpe_a' when initializing NPE_A."
            )

        mdn_net = self._neural_net.net
        if not isinstance(mdn_net, MultivariateGaussianMDN):
            raise TypeError(
                f"Expected MultivariateGaussianMDN, got {type(mdn_net).__name__}"
            )

        # Update the number of components in the MDN
        mdn_net._num_components = self._num_components

        # Define the exact parameter paths that need expansion
        # These are the output layers of the MDN that produce per-component outputs
        expansion_layer_names = {
            "net._logits_layer",
            "net._means_layer",
            "net._unconstrained_diagonal_layer",
        }
        # Upper layer only exists for dim > 1
        if mdn_net._upper_layer is not None:
            expansion_layer_names.add("net._upper_layer")

        # Track which parameters were expanded for validation
        expanded_params = set()

        for name, param in self._neural_net.named_parameters():
            # Check if this parameter belongs to one of the expansion layers
            layer_prefix = ".".join(name.rsplit(".", 1)[:-1])  # Remove .weight/.bias
            if layer_prefix in expansion_layer_names:
                if name.endswith(".bias"):
                    param.data = param.data.repeat(self._num_components)
                    param.data.add_(torch.randn_like(param.data) * eps)
                    param.grad = None  # Let autograd construct a new gradient
                    expanded_params.add(name)
                elif name.endswith(".weight"):
                    param.data = param.data.repeat(self._num_components, 1)
                    param.data.add_(torch.randn_like(param.data) * eps)
                    param.grad = None
                    expanded_params.add(name)

        # Validate that we expanded the expected parameters
        expected_count = len(expansion_layer_names) * 2  # weight + bias per layer
        if len(expanded_params) != expected_count:
            warnings.warn(
                f"Expected to expand {expected_count} parameters but expanded "
                f"{len(expanded_params)}. Expanded: {expanded_params}",
                UserWarning,
                stacklevel=2,
            )
