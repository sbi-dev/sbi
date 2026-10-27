# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from copy import deepcopy
from dataclasses import replace
from functools import partial
from typing import Any, Callable, Dict, Literal, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.inference.posteriors.npe_a_posterior import NPE_A_Posterior
from sbi.inference.trainers.npe.npe_base import (
    PosteriorEstimatorTrainer,
    _z_scored_prior_mog,
)
from sbi.neural_nets.estimators.base import (
    ConditionalDensityEstimator,
    ConditionalEstimatorBuildFn,
)
from sbi.neural_nets.estimators.mixture_density_estimator import (
    MixtureDensityEstimator,
    MultivariateGaussianMDN,
    ProposalCorrectedMDN,
)
from sbi.neural_nets.estimators.mog import MoG, _check_single_gaussian_proposal
from sbi.neural_nets.net_builders.estimator_configs import MDNConfig
from sbi.sbi_types import Tracker
from sbi.utils.sbiutils import del_entries

_MDN_DEFAULT_COMPONENTS = MDNConfig().num_components


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
        The correction divides by the proposal in closed form, which needs a
        proposal with a single Gaussian (Appendix C of [1]). As in [1], NPE-A
        therefore trains a single Gaussian in every round, and a mixture of
        `num_components` Gaussians only in the round trained with
        `train(final_round=True)`. The posterior of that round cannot be the
        proposal of another round.

    Example:
    --------

    ::

        import torch
        from sbi.inference import NPE_A
        from sbi.utils import BoxUniform

        # 1. Setup simulator, prior, and observation
        prior = BoxUniform(low=torch.zeros(3), high=torch.ones(3))

        def simulator(theta):
            return theta + torch.randn_like(theta) * 0.1

        x_o = simulator(prior.sample((1,)))  # Observed data

        # 2. Multi-round inference
        inference = NPE_A(prior=prior, num_components=5)
        proposal = prior
        num_rounds = 5

        for r in range(num_rounds):
            theta = proposal.sample((100,))
            x = simulator(theta)
            inference.append_simulations(theta, x, proposal=proposal).train(
                final_round=(r == num_rounds - 1)
            )
            posterior = inference.build_posterior()
            proposal = posterior.set_default_x(x_o)

        # 3. Sample from final posterior
        samples = posterior.sample((1000,), x=x_o)
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[
            Literal["mdn_snpe_a"],
            MDNConfig,
            ConditionalEstimatorBuildFn[ConditionalDensityEstimator],
            None,
        ] = None,
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
            density_estimator: If `None` (default), uses `MDNConfig()`. NPE-A
                needs a mixture of Gaussians, so `MDNConfig` is the only config it
                accepts; its `num_components` is set by this class's
                `num_components` argument. If it is a string (deprecated), only
                "mdn_snpe_a" is valid. Alternatively, a function that builds a
                custom neural network, which adheres to
                `ConditionalEstimatorBuildFn` protocol can be provided. The function
                will be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. The
                density estimator needs to provide the methods `.log_prob` and
                `.sample()` and must return a `MixtureDensityEstimator`. It is also
                called with the keyword `num_components`, and has to build that many
                mixture components.
            num_components: Number of components of the mixture of Gaussians in the
                round trained with `train(final_round=True)`. All other rounds train
                a single Gaussian.
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: Deprecated alias for the TensorBoard summary writer.
                Use ``tracker`` instead.
            tracker: Tracking adapter used to log training metrics. If None, a
                TensorBoard tracker is used with a default log directory.
            show_progress_bars: Whether to show a progressbar during training.
        """

        if density_estimator is None:
            density_estimator = MDNConfig(num_components=num_components)
        elif isinstance(density_estimator, MDNConfig):
            if (
                density_estimator.num_components != _MDN_DEFAULT_COMPONENTS
                and density_estimator.num_components != num_components
            ):
                raise ValueError(
                    "`num_components` was set both on the config "
                    f"({density_estimator.num_components}) and on NPE_A "
                    f"({num_components}). For NPE-A it belongs on the trainer."
                )
            density_estimator = replace(
                density_estimator, num_components=num_components
            )
        elif isinstance(density_estimator, str):
            if density_estimator != "mdn_snpe_a":
                raise TypeError(
                    "The `density_estimator` passed to NPE_A needs to be a "
                    "MDNConfig, a callable, or the string 'mdn_snpe_a'!"
                )
            warnings.warn(
                "Passing a string for `density_estimator` is deprecated. "
                "Use MDNConfig() instead, e.g. "
                "`from sbi.neural_nets import MDNConfig`.",
                FutureWarning,
                stacklevel=2,
            )
            density_estimator = MDNConfig(num_components=num_components)
        elif not callable(density_estimator):
            raise TypeError(
                "The `density_estimator` passed to NPE_A needs to be a "
                "MDNConfig, a callable, or the string 'mdn_snpe_a'!"
            )

        kwargs = del_entries(
            locals(),
            entries=("self", "__class__", "num_components"),
        )
        super().__init__(**kwargs)

        self._num_components = num_components
        # `train()` binds the number of components when the network is built.
        builder: Callable[..., ConditionalDensityEstimator]
        if isinstance(density_estimator, MDNConfig):
            config, wrap_builder = density_estimator, self._wrap_builder

            def build_from_config(batch_theta, batch_x, num_components):
                config_k = replace(config, num_components=num_components)
                return wrap_builder(config_k)(batch_theta, batch_x)

            builder = build_from_config
        else:
            builder = density_estimator
        self._build_neural_net = partial(_build_mdn, builder=builder)

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
        final_round: bool = False,
        component_perturbation: float = 5e-3,
    ) -> ConditionalDensityEstimator:
        r"""Return density estimator that approximates the proposal posterior.

        [1] _Fast epsilon-free Inference of Simulation Models with Bayesian Conditional
            Density Estimation_, Papamakarios et al., NeurIPS 2016,
            https://arxiv.org/abs/1605.06376.

        Training is performed with maximum likelihood on samples from the latest round,
        which leads the algorithm to converge to the proposal posterior.

        As in [1], the density estimator is a single Gaussian in every round but the
        final one, so that its posterior can be the proposal of the next round
        (Algorithm 1). A round trained with `final_round=True` copies that Gaussian
        `num_components` times with small noise and trains the mixture (Algorithm 2).
        Its posterior cannot be a proposal anymore.

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
            final_round: Whether this is the last round. Set it in the last iteration
                of the loop over rounds, or call `train(final_round=True)` once more
                after the loop, which retrains on the data of the last round.
            component_perturbation: Standard deviation of the noise added to the
                copies of the Gaussian in the final round. It breaks the symmetry
                between the copies.

        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.

        Raises:
            ValueError: If the proposal is a mixture, e.g. the posterior of a final
                round, or if `final_round=True` would expand the network while
                `resume_training=True`.
        """

        assert not retrain_from_scratch, """Retraining from scratch is not supported in
            NPE-A yet. The reason for this is that, if we reininitialized the density
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

        if len(self._data_round_index) == 0:
            raise RuntimeError(
                "No simulations found. You must call .append_simulations() "
                "before calling .train()."
            )

        self._round = max(self._data_round_index)

        # Fail before training if the correction would fail after it. The count does
        # not need the network, which is not built yet if the first round already
        # came from a proposal.
        proposal = self._proposal_roundwise[-1]
        if isinstance(proposal, NPE_A_Posterior) and proposal.default_x is not None:
            with torch.no_grad():
                _check_single_gaussian_proposal(
                    proposal.get_mog_params(proposal.default_x)
                )
        elif isinstance(proposal, MoG):
            _check_single_gaussian_proposal(proposal)

        num_components = self._num_components if final_round else 1
        if self._neural_net is None:
            self._build_neural_net = partial(
                self._build_neural_net, num_components=num_components
            )
        elif num_components > 1:
            mdn = self._neural_net.net
            assert isinstance(mdn, MultivariateGaussianMDN)
            if mdn.num_components == 1:
                if resume_training:
                    raise ValueError(
                        "`final_round=True` replaces the output layers of the "
                        "network, so the optimizer cannot be resumed. Call "
                        "`train(final_round=True)` without `resume_training`."
                    )
                mdn.expand_components(num_components, component_perturbation)

        density_estimator = super().train(**kwargs)

        return density_estimator

    def _get_proposal_mog(
        self,
        proposal: Union["NPE_A_Posterior", MultivariateNormal, MoG, Any],
        density_estimator: MixtureDensityEstimator,
    ) -> MoG:
        """Extract MoG parameters from a proposal distribution.

        Supports multiple proposal types:
        - NPE_A_Posterior: its corrected MoG, from get_mog_params()
        - MultivariateNormal: converts to single-component MoG over parameters
        - MoG: a MoG over parameters

        Args:
            proposal: The proposal distribution from the previous round.
            density_estimator: The estimator whose z-scored space the proposal is
                mapped into.

        Returns:
            MoG parameters from the proposal, in the z-scored space.

        Raises:
            ValueError: If NPE_A_Posterior proposal doesn't have default_x set.
            TypeError: If proposal type is not supported.
        """
        if isinstance(proposal, NPE_A_Posterior):
            default_x = proposal.default_x
            if default_x is None:
                raise ValueError(
                    "Proposal posterior must have a default_x set for NPE-A "
                    "correction. Call posterior.set_default_x(x_o) before using "
                    "as proposal."
                )
            if default_x.shape[0] != 1:
                raise ValueError(
                    f"NPE-A requires default_x batch size of 1, got "
                    f"{default_x.shape[0]}. NPE-A only supports single "
                    "observations for correction."
                )
            # The proposal's MoG is in the z-scored space of its own estimator.
            proposal_estimator = proposal.posterior_estimator
            if isinstance(proposal_estimator, ProposalCorrectedMDN):
                proposal_estimator = proposal_estimator.net
            assert isinstance(proposal_estimator, MixtureDensityEstimator)
            return density_estimator.transform_mog_from(
                proposal_estimator, proposal.get_mog_params(default_x)
            )

        if isinstance(proposal, MultivariateNormal):
            return density_estimator.transform_mog(
                MoG.from_gaussian(proposal.mean, proposal.covariance_matrix)
            )

        if isinstance(proposal, MoG):
            return density_estimator.transform_mog(proposal)

        raise TypeError(
            f"For multi-round NPE-A, proposal must be one of: NPE_A_Posterior, "
            f"MultivariateNormal or MoG. Got {type(proposal).__name__}. For custom "
            f"proposals, construct NPE_A_Posterior directly with your proposal_mog "
            f"parameter, in the estimator's z-scored space."
        )

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
                "Use density_estimator=MDNConfig() when initializing NPE_A."
            )

        # Compute correction parameters
        proposal = self._proposal_roundwise[-1]
        is_first_round = proposal is self._prior or proposal is None

        if is_first_round:
            proposal_mog = None
            prior_mog = None
        else:
            proposal_mog = self._get_proposal_mog(proposal, density_estimator)
            prior_mog = _z_scored_prior_mog(self._prior, density_estimator)

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

    def _multiround_loss_is_per_row(self, proposal: Optional[Any]) -> bool:
        """Return True: NPE-A trains on the plain log-prob for any proposal.

        The proposal correction is applied analytically after training, not in the loss.

        Args:
            proposal: Unused, the answer does not depend on it.

        Returns:
            True.
        """
        return True

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


def _build_mdn(
    batch_theta: Tensor, batch_x: Tensor, builder: Callable, num_components: int
) -> MixtureDensityEstimator:
    """Build the NPE-A network and check that it has `num_components` components.

    Args:
        batch_theta: Parameters, used for shape inference and z-scoring.
        batch_x: Simulation outputs, used for shape inference and z-scoring.
        builder: Function that builds the network from `batch_theta`, `batch_x` and
            the keyword `num_components`.
        num_components: Number of mixture components to build.

    Returns:
        The mixture density estimator.

    Raises:
        TypeError: If the builder does not return a `MixtureDensityEstimator`.
        ValueError: If the builder ignores `num_components`.
    """
    estimator = builder(batch_theta, batch_x, num_components=num_components)
    if not isinstance(estimator, MixtureDensityEstimator):
        raise TypeError(
            "NPE_A requires a MixtureDensityEstimator, the density estimator "
            f"builder returned {type(estimator).__name__}. Use MDNConfig."
        )
    if estimator.net.num_components != num_components:
        raise ValueError(
            f"NPE-A asked the density estimator builder for {num_components} "
            f"mixture components and got {estimator.net.num_components}. A builder "
            "passed to NPE_A has to build its MDN with the `num_components` keyword "
            "it is called with: NPE-A trains one Gaussian in every round but the "
            "final one."
        )
    return estimator
