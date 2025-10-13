# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, ones
from torch.distributions import Distribution
from torch.utils import data
from torch.utils.tensorboard.writer import SummaryWriter
from typing_extensions import Self

from sbi import utils as utils
from sbi.inference import NeuralInference
from sbi.inference.posteriors import DirectPosterior
from sbi.inference.potentials.vector_field_potential import (
    VectorFieldBasedPotential,
    vector_field_estimator_based_potential,
)
from sbi.inference.trainers._contracts import LossArgsVF, StartIndexContext, TrainConfig
from sbi.inference.trainers.base import LossArgs
from sbi.neural_nets.estimators import ConditionalVectorFieldEstimator
from sbi.neural_nets.estimators.base import ConditionalEstimatorBuilder
from sbi.sbi_types import TorchTransform
from sbi.utils import (
    check_estimator_arg,
    handle_invalid_x,
    npe_msg_on_invalid_x,
    test_posterior_net_for_multi_d_x,
    validate_theta_and_x,
    warn_if_zscoring_changes_data,
)
from sbi.utils.sbiutils import ImproperEmpirical, mask_sims_from_prior
from sbi.utils.torchutils import assert_all_finite


class VectorFieldTrainer(NeuralInference[ConditionalVectorFieldEstimator], ABC):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        vector_field_estimator_builder: Union[
            Literal["mlp", "ada_mlp", "transformer", "transformer_cross_attn"],
            ConditionalEstimatorBuilder[ConditionalVectorFieldEstimator],
        ] = "mlp",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        **kwargs,
    ):
        """Base class for vector field inference methods. It is used
        both for NPSE and FMPE.

        NOTE: Vector field inference does not support multi-round inference
        with flexible proposals yet. You can try to run multi-round with
        truncated proposals, but note that this is not tested yet.

        Args:
            prior: Prior distribution.
            vector_field_estimator_builder: Neural network architecture for the
                vector field estimator. Can be a string (e.g. 'mlp' or 'ada_mlp') or a
                callable that implements the `ConditionalEstimatorBuilder` protocol
                with `__call__` that receives `theta` and `x` and returns a
                `ConditionalVectorFieldEstimator`.
            device: Device to run the training on.
            logging_level: Logging level for the training. Can be an integer or a
                string.
            summary_writer: Tensorboard summary writer.
            show_progress_bars: Whether to show progress bars during training.
            kwargs: Additional keyword arguments passed to the default builder if
                `vector_field_estimator_builder` is a string.
        """

        super().__init__(
            prior=prior,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
        )

        # As detailed in the docstring, `vector_field_estimator` is either a string or
        # a callable. The function creating the neural network is attached to
        # `_build_neural_net`. It will be called in the first round and receive
        # thetas and xs as inputs, so that they can be used for shape inference and
        # potentially for z-scoring.
        check_estimator_arg(vector_field_estimator_builder)
        if isinstance(vector_field_estimator_builder, str):
            self._build_neural_net = self._build_default_nn_fn(
                model=vector_field_estimator_builder, **kwargs
            )
        else:
            self._build_neural_net = vector_field_estimator_builder

        self._proposal_roundwise = []

    @abstractmethod
    def _build_default_nn_fn(
        self,
        model: Literal["mlp", "ada_mlp", "transformer", "transformer_cross_attn"],
        **kwargs,
    ) -> ConditionalEstimatorBuilder[ConditionalVectorFieldEstimator]: ...

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
                Pass `None` if the parameters were sampled from the prior. Multi-round
                training is not yet implemented, so anything other than `None` will
                raise an error.
            exclude_invalid_x: Whether invalid simulations are discarded during
                training. For single-round training, it is fine to discard invalid
                simulations, but for multi-round sequential (atomic) training,
                discarding invalid simulations gives systematically wrong results. If
                `None`, it will be `True` in the first round and `False` in later
                rounds. Note that multi-round training is not yet implemented.
            data_device: Where to store the data, default is on the same device where
                the training is happening. If training a large dataset on a GPU with not
                much VRAM can set to 'cpu' to store data on system memory instead.

        Returns:
            VectorFieldTrainer object (returned so that this function is chainable).
        """
        inference_name = self.__class__.__name__
        assert proposal is None, (
            f"Multi-round {inference_name} is not yet implemented. "
            f"Please use single-round {inference_name}."
        )
        current_round = 0

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

        npe_msg_on_invalid_x(
            num_nans,
            num_infs,
            exclude_invalid_x,
            algorithm=f"Single-round {inference_name}",
        )

        self._data_round_index.append(current_round)
        prior_masks = mask_sims_from_prior(int(current_round > 0), theta.size(0))

        self._theta_roundwise.append(theta)
        self._x_roundwise.append(x)
        self._prior_masks.append(prior_masks)

        self._proposal_roundwise.append(proposal)

        if self._prior is None or isinstance(self._prior, ImproperEmpirical):
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
        ema_loss_decay: float = 0.1,
        validation_times: Union[Tensor, int] = 10,
        validation_times_nugget: float = 0.05,
        resume_training: bool = False,
        force_first_round_loss: bool = False,
        discard_prior_samples: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[dict] = None,
    ) -> ConditionalVectorFieldEstimator:
        r"""Returns a vector field estimator that approximates the posterior
        $p(\theta|x)$ through a continuous transformation from the base distribution
        to the target posterior.

        NOTE: This method is common for both score-based methods (NPSE) and flow
        matching methods (FMPE).

        The denoising score matching loss has a high
        variance, which makes it more difficult to detect converegence. To reduce this
        variance, we evaluate the validation loss at a fixed set of times. We also use
        the exponential moving average of the training and validation losses, as opposed
        to the other `trainer` classes, which track the loss directly.

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
            ema_loss_decay: Loss decay strength for exponential moving average of
                training and validation losses.
            validation_times: Diffusion times at which to evaluate the validation loss
                to reduce variance of validation loss.
            validation_times_nugget: As both diffusion and flow matching losses often
                have high variance losses at the end, we add a small nugget to compute
                the validation loss. Default is 0.05 i.e. t_min + 0.05 or t_max - 0.5.
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
            Vector field estimator that approximates the posterior.
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

        if isinstance(validation_times, int):
            validation_times = torch.linspace(
                self._neural_net.t_min + validation_times_nugget,
                self._neural_net.t_max - validation_times_nugget,
                validation_times,
            )

        loss_args = LossArgsVF(
            proposal=proposal,
            calibration_kernel=calibration_kernel,
            force_first_round_loss=force_first_round_loss,
            times=validation_times,
        )

        summarization_kwargs = dict(ema_loss_decay=ema_loss_decay)

        return self._run_training_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            train_config=train_config,
            loss_args=loss_args,
            summarization_kwargs=summarization_kwargs,
        )

    def _converged(self, epoch: int, stop_after_epochs: int) -> bool:
        """Return whether the training converged yet and save best model state so far.

        Diffusion or flow matching objectives are inherently more stochastic than MLE
        for e.g. NPE because they additionally add "noise" by construction. We hence
        use a statistical approach to detect convergence by tracking standard deviation
        of validation losses. Training is considered converged when the current loss is
        significantly worse than the best loss for a sustained period (more than 2 std
        deviations above best).

        NOTE: The standard deviation of the `validation_loss `is computed in a running
            fashion over the most recent 2 × stop_after_epochs loss values.

        Args:
            epoch: Current epoch in training.
            stop_after_epochs: How many fruitless epochs to let pass before stopping.

        Returns:
            Whether the training has stopped improving, i.e. has converged.
        """
        converged = False

        assert self._neural_net is not None
        neural_net = self._neural_net

        # Initialize tracking variables if not exists
        if not hasattr(self, '_best_val_loss'):
            self._best_val_loss = float('inf')
            self._epochs_since_last_improvement = 0
            self._best_model_state_dict = None

        # Check if we have a new best loss
        if self._val_loss < self._best_val_loss:
            self._best_val_loss = self._val_loss
            self._epochs_since_last_improvement = 0
            self._best_model_state_dict = deepcopy(neural_net.state_dict())
        else:
            # Only start statistical analysis after we have enough data
            if len(self._summary["validation_loss"]) >= stop_after_epochs:
                # Calculate running statistics of recent losses
                recent_losses = torch.tensor(
                    self._summary["validation_loss"][-stop_after_epochs * 2 :]
                )
                loss_std = recent_losses.std().item()

                # Calculate how many standard deviations the current loss is from the
                # best
                diff_to_best_normalized = (
                    self._val_loss - self._best_val_loss
                ) / loss_std
                # Consider it "no improvement" if current loss is significantly
                # worse than the best loss (more than 2 std deviations above best)
                # This accounts for natural fluctuations while being sensitive to
                # real degradation
                if diff_to_best_normalized > 2.0:
                    self._epochs_since_last_improvement += 1
                else:
                    # Reset counter if loss is within acceptable range
                    self._epochs_since_last_improvement = 0
            else:
                return False

        # If no validation improvement over many epochs, stop training.
        if self._epochs_since_last_improvement > stop_after_epochs - 1:
            if self._best_model_state_dict is not None:
                neural_net.load_state_dict(self._best_model_state_dict)
            converged = True

        return converged

    def _get_potential_function(
        self, prior: Distribution, estimator: ConditionalVectorFieldEstimator
    ) -> Tuple[VectorFieldBasedPotential, TorchTransform]:
        r"""Gets the potential function gradient for vector field estimators.

        Args:
            prior: The prior distribution.
            estimator: The neural network modelling the vector field.
        Returns:
            The potential function and a transformation that maps
            to unconstrained space.
        """
        potential_fn, theta_transform = vector_field_estimator_based_potential(
            estimator,
            prior,
            x_o=None,
        )
        return potential_fn, theta_transform

    def _loss_proposal_posterior(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
    ) -> Tensor:
        cls_name = self.__class__.__name__
        raise NotImplementedError(f"Multi-round {cls_name} is not yet implemented.")

    def _loss(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
        calibration_kernel: Callable,
        times: Optional[Tensor] = None,
        force_first_round_loss: bool = False,
    ) -> Tensor:
        r"""Return loss from vector field estimator. Currently only single-round
        training is implemented, i.e., no proposal correction is applied for later
        rounds.

        The loss can be weighted with a calibration kernel.

        Args:
            theta: Parameter sets :math:`\theta`.
            x: Simulation outputs :math:`x`.
            masks: Prior masks. Ignored for now.
            proposal: Proposal distribution. Ignored for now.
            calibration_kernel: Calibration kernel.
            times: Times :math:`t`.
            force_first_round_loss: If `True`, ignore the correction for using a
                proposal distribution different from the prior. Since the
                correction is not implemented yet, `False` will raise an error
                on any round other than the first one.

        Returns:
            Calibration kernel-weighted loss implemented by the vector field estimator.
        """

        if times is not None:
            times = times.to(self._device)

        cls_name = self.__class__.__name__
        if self._round == 0 or force_first_round_loss:
            # First round loss.
            loss = self._neural_net.loss(theta, x, times=times)
        else:
            raise NotImplementedError(
                f"Multi-round {cls_name} with arbitrary proposals is not implemented"
            )

        assert_all_finite(loss, f"{cls_name} loss")
        return calibration_kernel(x) * loss

    def _get_losses(self, batch: Sequence[Tensor], loss_args: LossArgs) -> Tensor:
        """
        Compute losses for a batch of data.

        Args:
            batch: A batch of data.
            loss_args: Additional keyword arguments passed to self._loss fn.

        Returns:
            A tensor containing the computed losses for each sample in the batch.
        """

        # Get batches on current device.
        theta_batch, x_batch, masks_batch = (
            batch[0].to(self._device),
            batch[1].to(self._device),
            batch[2].to(self._device),
        )

        if not isinstance(loss_args, LossArgsVF):
            raise TypeError(
                "Expected type of loss_args to be LossArgsVF,"
                f" but got {type(loss_args)}"
            )

        validation_times = loss_args.times
        if validation_times is not None:
            # For validation loss, we evaluate at a fixed set of times to reduce
            # the variance in the validation loss, for improved convergence
            # checks. We evaluate the entire validation batch at all times, so
            # we repeat the batches here to match.
            val_batch_size = theta_batch.shape[0]
            times_batch = validation_times.shape[0]
            theta_batch = theta_batch.repeat(
                times_batch, *([1] * (theta_batch.ndim - 1))
            )
            x_batch = x_batch.repeat(times_batch, *([1] * (x_batch.ndim - 1)))
            masks_batch = masks_batch.repeat(
                times_batch, *([1] * (masks_batch.ndim - 1))
            )

            validation_times_rep = validation_times.repeat_interleave(
                val_batch_size, dim=0
            )

            loss_args = replace(loss_args, times=validation_times_rep)

        losses = self._loss(
            theta=theta_batch,
            x=x_batch,
            masks=masks_batch,
            **asdict(loss_args),
        )

        return losses

    def _train_epoch(
        self,
        train_loader: data.DataLoader,
        clip_max_norm: Optional[float],
        loss_args: LossArgs | None,
    ) -> float:
        """
        Override the parent method for performing a single training epoch over the
        provided training data to set `times` to None as it is only used
        when calculating the validation loss.

        Args:
            train_loader: Dataloader for training.
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            loss_args: Additional arguments passed to self._loss fn.

        Returns:
            The average training loss over all samples in the epoch.
        """

        if not isinstance(loss_args, LossArgsVF):
            raise TypeError(
                "Expected type of loss_args to be LossArgsVF,"
                f" but got {type(loss_args)}"
            )

        loss_args = replace(loss_args, **dict(times=None))

        return super()._train_epoch(
            train_loader=train_loader,
            clip_max_norm=clip_max_norm,
            loss_args=loss_args,
        )

    def _summarize_epoch(
        self,
        train_loss: float,
        val_loss: float,
        epoch_start_time: float,
        summarization_kwargs: Dict[str, Any],
    ) -> None:
        """
        Override base class method to pass additional arguments through
        summarization_kwargs and log exponential moving average for the validation
        and training losses.

        Args:
            train_loss: The average training loss for the epoch.
            val_loss: The average validation loss for the epoch.
            epoch_start_time: Timestamp when the epoch started, used to compute
                duration.
            summarization_kwargs: Additional keyword arguments for customizing
                the summarization.
        """

        ema_loss_decay = summarization_kwargs.get("ema_loss_decay")
        assert ema_loss_decay is not None and isinstance(ema_loss_decay, float)

        # NOTE: Due to the inherently noisy nature we do instead log a exponential
        # moving average of the training loss.
        if len(self._summary["training_loss"]) == 0:
            train_loss_ema = train_loss
        else:
            previous_loss = self._summary["training_loss"][-1]
            train_loss_ema = (
                1.0 - ema_loss_decay
            ) * previous_loss + ema_loss_decay * train_loss

        if len(self._summary["validation_loss"]) == 0:
            val_loss_ema = val_loss
        else:
            previous_loss = self._summary["validation_loss"][-1]
            val_loss_ema = (
                1 - ema_loss_decay
            ) * previous_loss + ema_loss_decay * val_loss

        super()._summarize_epoch(
            train_loss=train_loss_ema,
            val_loss=val_loss_ema,
            epoch_start_time=epoch_start_time,
            summarization_kwargs=summarization_kwargs,
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

        if self._round == 0 and self._neural_net is not None:
            assert context.force_first_round_loss or context.resume_training, (
                "You have already trained this neural network. After you had trained "
                "the network, you again appended simulations with `append_simulations"
                "(theta, x)`, but you did not provide a proposal. If the new "
                "simulations are sampled from the prior, you can set "
                "`.train(..., force_first_round_loss=True`). However, if the new "
                "simulations were not sampled from the prior, you should pass the "
                "proposal, i.e. `append_simulations(theta, x, proposal)`. If "
                "your samples are not sampled from the prior and you do not pass a "
                "proposal and you set `force_first_round_loss=True`, the result of "
                "NPSE will not be the true posterior. Instead, it will be the proposal "
                "posterior, which (usually) is more narrow than the true posterior."
            )

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(context.discard_prior_samples and self._round > 0)

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
        if self._neural_net is None or retrain_from_scratch:
            # Get theta,x to initialize NN
            theta, x, _ = self.get_simulations(starting_round=start_idx)
            # Use only training data for building the neural net (z-scoring transforms)

            self._neural_net = self._build_neural_net(
                theta[self.train_indices].to("cpu"),
                x[self.train_indices].to("cpu"),
            )

            test_posterior_net_for_multi_d_x(
                self._neural_net,
                theta.to("cpu"),
                x.to("cpu"),
            )

            del theta, x
