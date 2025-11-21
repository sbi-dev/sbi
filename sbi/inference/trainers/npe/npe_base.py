# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, Union

from torch import Tensor, ones
from torch.distributions import Distribution
from torch.utils.tensorboard.writer import SummaryWriter
from typing_extensions import Self

from sbi.inference.posteriors import DirectPosterior
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.posteriors.posterior_parameters import (
    DirectPosteriorParameters,
    ImportanceSamplingPosteriorParameters,
    MCMCPosteriorParameters,
    RejectionPosteriorParameters,
    VIPosteriorParameters,
)
from sbi.inference.potentials import posterior_estimator_based_potential
from sbi.inference.potentials.posterior_based_potential import PosteriorBasedPotential
from sbi.inference.trainers._contracts import (
    LossArgsNPE,
    StartIndexContext,
    TrainConfig,
)
from sbi.inference.trainers.base import (
    LossArgs,
    NeuralInference,
    check_if_proposal_has_default_x,
)
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.estimators import ConditionalDensityEstimator
from sbi.neural_nets.estimators.base import ConditionalEstimatorBuilder
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
    reshape_to_sample_batch_event,
)
from sbi.sbi_types import TorchTransform
from sbi.utils import (
    RestrictedPrior,
    check_estimator_arg,
    handle_invalid_x,
    nle_nre_apt_msg_on_invalid_x,
    npe_msg_on_invalid_x,
    test_posterior_net_for_multi_d_x,
    validate_theta_and_x,
    warn_if_zscoring_changes_data,
)
from sbi.utils.sbiutils import (
    ImproperEmpirical,
    mask_sims_from_prior,
)
from sbi.utils.torchutils import assert_all_finite


class PosteriorEstimatorTrainer(NeuralInference[ConditionalDensityEstimator], ABC):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[
            Literal["nsf", "maf", "mdn", "made"],
            ConditionalEstimatorBuilder[ConditionalDensityEstimator],
        ] = "maf",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        """Base class for Sequential Neural Posterior Estimation methods.

        We note that the different subclasses (NPE_A, NPE_B, NPE_C) have the same
        loss function in the first round and only differ when they are being run in
        a sequential scheme.

        Args:
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network, which adheres to
                `ConditionalEstimatorBuilder` protocol can be provided. The function
                will be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. The
                density estimator needs to provide the methods `.log_prob` and
                `.sample()` and must return a `ConditionalDensityEstimator`.

        See docstring of `NeuralInference` class for all other arguments.
        """

        super().__init__(
            prior=prior,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
        )

        # As detailed in the docstring, `density_estimator` is either a string or
        # a callable. The function creating the neural network is attached to
        # `_build_neural_net`. It will be called in the first round and receive
        # thetas and xs as inputs, so that they can be used for shape inference and
        # potentially for z-scoring.
        check_estimator_arg(density_estimator)
        if isinstance(density_estimator, str):
            self._build_neural_net = posterior_nn(model=density_estimator)
        else:
            self._build_neural_net = density_estimator

        self._proposal_roundwise = []
        self.use_non_atomic_loss = False

    @abstractmethod
    def _log_prob_proposal_posterior(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
    ) -> Tensor: ...

    def append_simulations(
        self,
        theta: Tensor,
        x: Tensor,
        proposal: Optional[DirectPosterior] = None,
        exclude_invalid_x: Optional[bool] = None,
        data_device: Optional[str] = None,
    ) -> Self:
        r"""Store parameters and simulation outputs to use them for later training.

        Data are stored as entries in lists for each type of variable (parameter/data).

        Stores $\theta$, $x$, prior_masks (indicating if simulations are coming from the
        prior or not) and an index indicating which round the batch of simulations came
        from.

        Args:
            theta: Parameter sets.
            x: Simulation outputs.
            proposal: The distribution that the parameters $\theta$ were sampled from.
                Pass `None` if the parameters were sampled from the prior. If not
                `None`, it will trigger a different loss-function.
            exclude_invalid_x: Whether invalid simulations are discarded during
                training. For single-round SNPE, it is fine to discard invalid
                simulations, but for multi-round SNPE (atomic), discarding invalid
                simulations gives systematically wrong results. If `None`, it will
                be `True` in the first round and `False` in later rounds.
            data_device: Where to store the data, default is on the same device where
                the training is happening. If training a large dataset on a GPU with not
                much VRAM can set to 'cpu' to store data on system memory instead.

        Returns:
            NeuralInference object (returned so that this function is chainable).
        """
        if (
            proposal is None
            or proposal is self._prior
            or (
                isinstance(proposal, RestrictedPrior) and proposal._prior is self._prior
            )
        ):
            # The `_data_round_index` will later be used to infer if one should train
            # with MLE loss or with atomic loss (see, in `train()`:
            # self._round = max(self._data_round_index))
            current_round = 0
        else:
            if not self._data_round_index:
                # This catches a pretty specific case: if, in the first round, one
                # passes data that does not come from the prior.
                current_round = 1
            else:
                current_round = max(self._data_round_index) + 1

        if exclude_invalid_x is None:
            exclude_invalid_x = current_round == 0

        if data_device is None:
            data_device = self._device

        theta, x = validate_theta_and_x(
            theta, x, data_device=data_device, training_device=self._device
        )

        is_valid_x, num_nans, num_infs = handle_invalid_x(
            x, exclude_invalid_x=exclude_invalid_x
        )

        x = x[is_valid_x]
        theta = theta[is_valid_x]

        # Check for problematic z-scoring
        warn_if_zscoring_changes_data(x)
        if (
            type(self).__name__ == "SNPE_C"
            and current_round > 0
            and not self.use_non_atomic_loss
        ):
            nle_nre_apt_msg_on_invalid_x(
                num_nans,
                num_infs,
                exclude_invalid_x,
                "Multiround SNPE-C (atomic)",
            )
        else:
            npe_msg_on_invalid_x(
                num_nans, num_infs, exclude_invalid_x, "Single-round NPE"
            )

        self._check_proposal(proposal)

        self._data_round_index.append(current_round)
        prior_masks = mask_sims_from_prior(int(current_round > 0), theta.size(0))

        self._theta_roundwise.append(theta)
        self._x_roundwise.append(x)
        self._prior_masks.append(prior_masks)

        self._proposal_roundwise.append(proposal)

        if self._prior is None or isinstance(self._prior, ImproperEmpirical):
            if proposal is not None:
                raise ValueError(
                    "You did not passed a prior at initialization, but now you "
                    "passed a proposal. If you want to run multi-round SNPE, you have "
                    "to specify a prior (set the `.prior` argument or re-initialize "
                    "the object with a prior distribution). If the samples you passed "
                    "to `append_simulations()` were sampled from the prior, you can "
                    "run single-round inference with "
                    "`append_simulations(..., proposal=None)`."
                )
            theta_prior = self.get_simulations()[0].to(self._device)
            self._prior = ImproperEmpirical(
                theta_prior, ones(theta_prior.shape[0], device=self._device)
            )

        return self

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
        force_first_round_loss: bool = False,
        discard_prior_samples: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[dict] = None,
    ) -> ConditionalDensityEstimator:
        r"""Return density estimator that approximates the distribution $p(\theta|x)$.

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
            calibration_kernel: A function to calibrate the loss with respect
                to the simulations `x` (optional). See Lueckmann, Gonçalves et al.,
                NeurIPS 2017. If `None`, no calibration is used.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)

        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """

        train_config = TrainConfig(
            max_num_epochs=max_num_epochs,
            stop_after_epochs=stop_after_epochs,
            learning_rate=learning_rate,
            resume_training=resume_training,
            show_train_summary=show_train_summary,
            training_batch_size=training_batch_size,
            retrain_from_scratch=retrain_from_scratch,
            validation_fraction=validation_fraction,
            clip_max_norm=clip_max_norm,
        )

        # Calibration kernels proposed in Lueckmann, Gonçalves et al., 2017.
        if calibration_kernel is None:

            def default_calibration_kernel(x):
                return ones([len(x)], device=self._device)

            calibration_kernel = default_calibration_kernel

        start_idx = self._get_start_index(
            context=StartIndexContext(
                discard_prior_samples=discard_prior_samples,
                force_first_round_loss=force_first_round_loss,
                resume_training=train_config.resume_training,
            )
        )

        # Set the proposal to the last proposal that was passed by the user. For
        # atomic SNPE, it does not matter what the proposal is. For non-atomic
        # SNPE, we only use the latest data that was passed, i.e. the one from the
        # last proposal.
        proposal = self._proposal_roundwise[-1]

        train_loader, val_loader = self.get_dataloaders(
            start_idx,
            train_config.training_batch_size,
            train_config.validation_fraction,
            train_config.resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )

        self._initialize_neural_network(
            retrain_from_scratch=train_config.retrain_from_scratch,
            start_idx=start_idx,
        )

        loss_args = LossArgsNPE(
            proposal=proposal,
            calibration_kernel=calibration_kernel,
            force_first_round_loss=force_first_round_loss,
        )

        return self._run_training_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            train_config=train_config,
            loss_args=loss_args,
        )

    def build_posterior(
        self,
        density_estimator: Optional[ConditionalDensityEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: Literal[
            "mcmc", "rejection", "vi", "importance", "direct"
        ] = "direct",
        mcmc_method: Literal[
            "slice_np",
            "slice_np_vectorized",
            "hmc_pyro",
            "nuts_pyro",
            "slice_pymc",
            "hmc_pymc",
            "nuts_pymc",
        ] = "slice_np_vectorized",
        vi_method: Literal["rKL", "fKL", "IW", "alpha"] = "rKL",
        direct_sampling_parameters: Optional[Dict[str, Any]] = None,
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        vi_parameters: Optional[Dict[str, Any]] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
        importance_sampling_parameters: Optional[Dict[str, Any]] = None,
        posterior_parameters: Optional[
            Union[
                DirectPosteriorParameters,
                MCMCPosteriorParameters,
                VIPosteriorParameters,
                RejectionPosteriorParameters,
                ImportanceSamplingPosteriorParameters,
            ]
        ] = None,
    ) -> NeuralPosterior:
        r"""Build posterior from the neural density estimator.

        For SNPE, the posterior distribution that is returned here implements the
        following functionality over the raw neural density estimator:
        - correct the calculation of the log probability such that it compensates for
            the leakage.
        - reject samples that lie outside of the prior bounds.
        - alternatively, if leakage is very high (which can happen for multi-round
            SNPE), sample from the posterior with MCMC.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
            prior: Prior distribution.
            sample_with: Method to use for sampling from the posterior. Must be one of
                [`direct` | `mcmc` | `rejection` | `vi` | `importance`].
            mcmc_method: Method used for MCMC sampling, one of `slice_np`,
                `slice_np_vectorized`, `hmc_pyro`, `nuts_pyro`, `slice_pymc`,
                `hmc_pymc`, `nuts_pymc`. `slice_np` is a custom
                numpy implementation of slice sampling. `slice_np_vectorized` is
                identical to `slice_np`, but if `num_chains>1`, the chains are
                vectorized for `slice_np_vectorized` whereas they are run sequentially
                for `slice_np`. The samplers ending on `_pyro` are using Pyro, and
                likewise the samplers ending on `_pymc` are using PyMC.
            vi_method: Method used for VI, one of [`rKL`, `fKL`, `IW`, `alpha`]. Note
                some of the methods admit a `mode seeking` property (e.g. rKL) whereas
                some admit a `mass covering` one (e.g fKL).
            direct_sampling_parameters: Additional kwargs passed to `DirectPosterior`.
            mcmc_parameters: Additional kwargs passed to `MCMCPosterior`.
            vi_parameters: Additional kwargs passed to `VIPosterior`.
            rejection_sampling_parameters: Additional kwargs passed to
                `RejectionPosterior`.
            importance_sampling_parameters: Additional kwargs passed to
                `ImportanceSamplingPosterior`.
            posterior_parameters: Configuration passed to the init method for the
                posterior. Must be one of the following
                - `VIPosteriorParameters`
                - `ImportanceSamplingPosteriorParameters`
                - `MCMCPosteriorParameters`
                - `DirectPosteriorParameters`
                - `RejectionPosteriorParameters`

        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods
            (the returned log-probability is unnormalized).
        """

        self._check_prior_for_rejection_sampling(
            prior, sample_with, posterior_parameters
        )

        return super().build_posterior(
            density_estimator,
            prior,
            sample_with,
            posterior_parameters,
            mcmc_method=mcmc_method,
            vi_method=vi_method,
            mcmc_parameters=mcmc_parameters,
            vi_parameters=vi_parameters,
            rejection_sampling_parameters=rejection_sampling_parameters,
            importance_sampling_parameters=importance_sampling_parameters,
            direct_sampling_parameters=direct_sampling_parameters,
        )

    def _get_potential_function(
        self,
        prior: Distribution,
        estimator: ConditionalDensityEstimator,
    ) -> Tuple[PosteriorBasedPotential, TorchTransform]:
        r"""Gets the potential for posterior-based methods.

        It also returns a transformation that can be used to transform the potential
        into unconstrained space.

        The potential is the same as the log-probability of the `posterior_estimator`,
        but it is set to $-\inf$ outside of the prior bounds.

        Args:
            prior: The prior distribution.
            estimator: The neural network modelling the posterior.

        Returns:
            The potential function and a transformation that maps
            to unconstrained space.
        """

        potential_fn, theta_transform = posterior_estimator_based_potential(
            posterior_estimator=estimator,
            prior=prior,
            x_o=None,
        )
        return potential_fn, theta_transform

    def _check_prior_for_rejection_sampling(
        self,
        prior: Optional[Distribution],
        sample_with: Literal["mcmc", "rejection", "vi", "importance", "direct"],
        posterior_parameters: Optional[
            Union[
                DirectPosteriorParameters,
                MCMCPosteriorParameters,
                VIPosteriorParameters,
                RejectionPosteriorParameters,
                ImportanceSamplingPosteriorParameters,
            ]
        ],
    ) -> None:
        """
        Validates that when using rejection sampling, a prior distribution
        is explicitly provided.

        Args:
            prior: Prior distribution.
            sample_with: The sampling method used. Must be one of
                "mcmc", "rejection", "vi", "importance", or "direct".
            posterior_parameters: Configuration for building the posterior.
        """

        if (
            sample_with == "rejection"
            or isinstance(posterior_parameters, RejectionPosteriorParameters)
        ) and prior is None:
            raise ValueError(
                "You indicated sampling via rejection sampling but "
                "haven't passed a prior. As of sbi v0.23.0, you either have"
                " to pass a prior to perform rejection sampling using the prior"
                " as proposal, or to use the posterior as proposal, you have to"
                " use a DirectPosterior via `sample_with='direct' or"
                " `posterior_parameters=DirectPosteriorParameters`."
            )

    def _loss(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
        calibration_kernel: Callable,
        force_first_round_loss: bool = False,
    ) -> Tensor:
        """Return loss with proposal correction (`round_>0`) or without it (`round_=0`).

        The loss is the negative log prob. Irrespective of the round or SNPE method
        (A, B, or C), it can be weighted with a calibration kernel.

        Returns:
            Calibration kernel-weighted negative log prob.
            force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
        """
        if self._round == 0 or force_first_round_loss:
            theta = reshape_to_batch_event(
                theta, event_shape=self._neural_net.input_shape
            )
            x = reshape_to_batch_event(x, event_shape=self._neural_net.condition_shape)
            # Use posterior log prob (without proposal correction) for first round.
            loss = self._neural_net.loss(theta, x)
        else:
            # Currently only works for `DensityEstimator` objects.
            # Must be extended ones other Estimators are implemented. See #966,
            loss = -self._log_prob_proposal_posterior(theta, x, masks, proposal)

        assert_all_finite(loss, "NPE loss")
        return calibration_kernel(x) * loss

    def _check_proposal(self, proposal):
        """
        Check for validity of the provided proposal distribution.

        If the proposal is a `NeuralPosterior`, we check if the default_x is set.
        If the proposal is **not** a `NeuralPosterior`, we warn since it is likely that
        the user simply passed the prior, but this would still trigger atomic loss.
        """
        if proposal is not None:
            check_if_proposal_has_default_x(proposal)

            if isinstance(proposal, RestrictedPrior):
                if proposal._prior is not self._prior:
                    warnings.warn(
                        "The proposal you passed is a `RestrictedPrior`, but the "
                        "proposal distribution it uses is not the prior (it can be "
                        "accessed via `RestrictedPrior._prior`). We do not "
                        "recommend to mix the `RestrictedPrior` with multi-round "
                        "SNPE.",
                        stacklevel=2,
                    )
            elif (
                not isinstance(proposal, NeuralPosterior)
                and proposal is not self._prior
            ):
                warnings.warn(
                    "The proposal you passed is neither the prior nor a "
                    "`NeuralPosterior` object. If you are an expert user and did so "
                    "for research purposes, this is fine. If not, you might be doing "
                    "something wrong: feel free to create an issue on Github.",
                    stacklevel=2,
                )
        elif self._round > 0:
            raise ValueError(
                "A proposal was passed but no prior was passed at initialisation. When "
                "running multi-round inference, a prior needs to be specified upon "
                "initialisation. Potential fix: setting the `._prior` attribute or "
                "re-initialisation. If the samples passed to `append_simulations()` "
                "were sampled from the prior, single-round inference can be performed "
                "with `append_simulations(..., proprosal=None)`."
            )

    def _get_start_index(self, context: StartIndexContext) -> int:
        """
        Determine the starting index for training based on previous rounds.

        Args:
            context: StartIndexContext dataclass values used to determine the starting
                index of the training set.
        Returns:
            The method will return 1 to skip samples from round 0; otherwise,
            it returns 0.
        """

        # Load data from most recent round.
        self._round = max(self._data_round_index)

        if (
            self._round == 0
            and self._neural_net is not None
            and not (context.force_first_round_loss or context.resume_training)
        ):
            raise ValueError(
                "This neural network has already been trained. "
                "If you want to continue training without adding new simulations, "
                "set resume_training=True. "
                "If you appended new simulations, you must either provide a proposal "
                "in append_simulations(), or set force_first_round_loss=True for "
                "simulations drawn from the prior. "
                "Warning: Setting force_first_round_loss=True with simulations not "
                "drawn from the prior will produce the proposal posterior instead of "
                "the true posterior, which is typically more narrow."
            )

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(context.discard_prior_samples and self._round > 0)

        # For non-atomic loss, we can not reuse samples from previous rounds as of now.
        # SNPE-A can, by construction of the algorithm, only use samples from the last
        # round. SNPE-A is the only algorithm that has an attribute `_ran_final_round`,
        # so this is how we check for whether or not we are using SNPE-A.
        if self.use_non_atomic_loss or hasattr(self, "_ran_final_round"):
            start_idx = self._round

        return start_idx

    def _initialize_neural_network(
        self,
        retrain_from_scratch: bool,
        start_idx: int,
    ) -> None:
        """
        Initialize the neural network if it is None or retraining from scratch.

        Args:
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            start_idx: The index of the first round to retrieve simulation data from.
        """

        # First round or if retraining from scratch:
        # Call the `self._build_neural_net` with the rounds' thetas and xs as
        # arguments, which will build the neural network.
        # This is passed into NeuralPosterior, to create a neural posterior which
        # can `sample()` and `log_prob()`. The network is accessible via `.net`.
        if self._neural_net is None or retrain_from_scratch:
            # Get theta,x to initialize NN
            theta, x, _ = self.get_simulations(starting_round=start_idx)
            # Use only training data for building the neural net (z-scoring transforms)

            self._neural_net = self._build_neural_net(
                theta[self.train_indices].to("cpu"),
                x[self.train_indices].to("cpu"),
            )

            theta = reshape_to_sample_batch_event(
                theta.to("cpu"), self._neural_net.input_shape
            )
            x = reshape_to_batch_event(x.to("cpu"), self._neural_net.condition_shape)
            test_posterior_net_for_multi_d_x(self._neural_net, theta, x)

            del theta, x

    def _get_losses(self, batch: Sequence[Tensor], loss_args: LossArgs) -> Tensor:
        """
        Compute losses for a batch of data.

        Args:
            batch: A batch of data.
            loss_args: Additional arguments passed to self._loss fn.

        Returns:
            A tensor containing the computed losses for each sample in the batch.
        """

        theta_batch, x_batch, masks_batch = (
            batch[0].to(self._device),
            batch[1].to(self._device),
            batch[2].to(self._device),
        )

        if not isinstance(loss_args, LossArgsNPE):
            raise TypeError(
                "Expected type of loss_args to be LossArgsNPE,"
                f" but got {type(loss_args)}"
            )

        # Take negative loss here to get validation log_prob.
        losses = self._loss(theta_batch, x_batch, masks_batch, **asdict(loss_args))

        return losses
