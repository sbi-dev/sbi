# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Optional, Protocol, Union

import torch
from torch import Tensor, ones
from torch.distributions import Distribution
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.adam import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from sbi import utils as utils
from sbi.inference import NeuralInference
from sbi.inference.posteriors import (
    DirectPosterior,
)
from sbi.inference.posteriors.vector_field_posterior import VectorFieldPosterior
from sbi.neural_nets.estimators import ConditionalVectorFieldEstimator
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


class VectorFieldEstimatorBuilder(Protocol):
    """Protocol for building a vector field estimator from data."""

    def __call__(self, theta: Tensor, x: Tensor) -> ConditionalVectorFieldEstimator:
        """Build a vector field estimator from theta and x, which mainly
        inform the shape of the input and the condition to the neural network.
        Generally, it can also be used to z-score the data, but not in the case
        of vector field estimators.

        Args:
            theta: Parameter sets.
            x: Simulation outputs.

        Returns:
            Vector field estimator.
        """
        ...


class VectorFieldInference(NeuralInference, ABC):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        vector_field_estimator_builder: Union[str, VectorFieldEstimatorBuilder] = "mlp",
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
                callable that implements the `VectorFieldEstimatorBuilder` protocol
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
                vector_field_estimator_builder=vector_field_estimator_builder, **kwargs
            )
        else:
            self._build_neural_net = vector_field_estimator_builder

        self._proposal_roundwise = []

    @abstractmethod
    def _build_default_nn_fn(self, **kwargs) -> VectorFieldEstimatorBuilder:
        pass

    def append_simulations(
        self,
        theta: Tensor,
        x: Tensor,
        proposal: Optional[DirectPosterior] = None,
        exclude_invalid_x: Optional[bool] = None,
        data_device: Optional[str] = None,
    ) -> "VectorFieldInference":
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
            VectorFieldInference object (returned so that this function is chainable).
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
        stop_after_epochs: int = 50,
        max_num_epochs: int = 500,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        ema_loss_decay: float = 0.1,
        validation_times: Union[Tensor, int] = 20,
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
        # Load data from most recent round.
        self._round = max(self._data_round_index)

        if self._round == 0 and self._neural_net is not None:
            assert force_first_round_loss or resume_training, (
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

        # Calibration kernels proposed in Lueckmann, Gonçalves et al., 2017.
        if calibration_kernel is None:

            def default_calibration_kernel(x):
                return ones([len(x)], device=self._device)

            calibration_kernel = default_calibration_kernel

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(discard_prior_samples and self._round > 0)

        # Set the proposal to the last proposal that was passed by the user. For
        # atomic SNPE, it does not matter what the proposal is. For non-atomic
        # SNPE, we only use the latest data that was passed, i.e. the one from the
        # last proposal.
        proposal = self._proposal_roundwise[-1]

        train_loader, val_loader = self.get_dataloaders(
            start_idx,
            training_batch_size,
            validation_fraction,
            resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )
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

        # Move entire net to device for training.
        self._neural_net.to(self._device)

        if isinstance(validation_times, int):
            validation_times = torch.linspace(
                self._neural_net.t_min, self._neural_net.t_max, validation_times
            )
        assert isinstance(
            validation_times, Tensor
        )  # let pyright know validation_times is a Tensor.

        if not resume_training:
            self.optimizer = Adam(list(self._neural_net.parameters()), lr=learning_rate)

            self.epoch, self._val_loss = 0, float("Inf")

        while self.epoch <= max_num_epochs and not self._converged(
            self.epoch, stop_after_epochs
        ):
            # Train for a single epoch.
            self._neural_net.train()
            train_loss_sum = 0
            epoch_start_time = time.time()
            for batch in train_loader:
                self.optimizer.zero_grad()
                # Get batches on current device.
                theta_batch, x_batch, masks_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                    batch[2].to(self._device),
                )

                train_losses = self._loss(
                    theta=theta_batch,
                    x=x_batch,
                    masks=masks_batch,
                    proposal=proposal,
                    calibration_kernel=calibration_kernel,
                    force_first_round_loss=force_first_round_loss,
                )

                train_loss = torch.mean(train_losses)

                train_loss_sum += train_losses.sum().item()

                train_loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._neural_net.parameters(), max_norm=clip_max_norm
                    )
                self.optimizer.step()

            self.epoch += 1

            train_loss_average = train_loss_sum / (
                len(train_loader) * train_loader.batch_size  # type: ignore
            )

            # NOTE: Due to the inherently noisy nature we do instead log a exponential
            # moving average of the training loss.
            if len(self._summary["training_loss"]) == 0:
                self._summary["training_loss"].append(train_loss_average)
            else:
                previous_loss = self._summary["training_loss"][-1]
                self._summary["training_loss"].append(
                    (1.0 - ema_loss_decay) * previous_loss
                    + ema_loss_decay * train_loss_average
                )

            # Calculate validation performance.
            self._neural_net.eval()
            val_loss_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch, masks_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                        batch[2].to(self._device),
                    )

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

                    # Take negative loss here to get validation log_prob.
                    val_losses = self._loss(
                        theta=theta_batch,
                        x=x_batch,
                        masks=masks_batch,
                        proposal=proposal,
                        calibration_kernel=calibration_kernel,
                        times=validation_times_rep,
                        force_first_round_loss=force_first_round_loss,
                    )

                    val_loss_sum += val_losses.sum().item()

            # Take mean over all validation samples.
            val_loss = val_loss_sum / (
                len(val_loader) * val_loader.batch_size * times_batch  # type: ignore
            )

            if len(self._summary["validation_loss"]) == 0:
                val_loss_ema = val_loss
            else:
                previous_loss = self._summary["validation_loss"][-1]
                val_loss_ema = (
                    1 - ema_loss_decay
                ) * previous_loss + ema_loss_decay * val_loss

            self._val_loss = val_loss_ema
            self._summary["validation_loss"].append(self._val_loss)
            self._summary["epoch_durations_sec"].append(time.time() - epoch_start_time)

            self._maybe_show_progress(self._show_progress_bars, self.epoch)

        self._report_convergence_at_end(self.epoch, stop_after_epochs, max_num_epochs)

        # Update summary.
        self._summary["epochs_trained"].append(self.epoch)
        self._summary["best_validation_loss"].append(self._val_loss)

        # Update tensorboard and summary dict.
        self._summarize(round_=self._round)

        # Update description for progress bar.
        if show_train_summary:
            print(self._describe_round(self._round, self._summary))

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self._neural_net.zero_grad(set_to_none=True)

        return deepcopy(self._neural_net)

    def _build_posterior(
        self,
        vector_field_estimator: Optional[ConditionalVectorFieldEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: str = "sde",
        **kwargs,
    ) -> VectorFieldPosterior:
        r"""Build posterior from the vector field estimator.

        For NPSE, the posterior distribution that is returned here implements the
        following functionality over the raw neural density estimator:
        - correct the calculation of the log probability such that it compensates for
            the leakage.
        - reject samples that lie outside of the prior bounds.

        Args:
            vector_field_estimator: The vector field estimator that the posterior
                is based on. If `None`, use the latest vector field estimator that was
                trained.
            prior: Prior distribution.
            sample_with: Method to use for sampling from the posterior. Can be one of
                'sde' (default) or 'ode'. The 'sde' method uses the vector field to
                do a Langevin diffusion step, while the 'ode' solves a probabilistic ODE
                with a numerical ODE solver.
            **kwargs: Additional keyword arguments passed to
                `VectorFieldBasedPotential`.

        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods.
        """
        if prior is None:
            cls_name = self.__class__.__name__
            assert self._prior is not None, (
                "You did not pass a prior. You have to pass the prior either at "
                f"initialization `inference = {cls_name}(prior)` or to "
                "`.build_posterior(prior=prior)`."
            )
            prior = self._prior
        else:
            utils.check_prior(prior)

        if vector_field_estimator is None:
            vector_field_estimator = self._neural_net
            # If internal net is used device is defined.
            device = self._device
        # Otherwise, infer it from the device of the net parameters.
        else:
            device = str(next(vector_field_estimator.parameters()).device)

        posterior = VectorFieldPosterior(
            vector_field_estimator,
            prior,
            device=device,
            sample_with=sample_with,
            **kwargs,
        )

        self._posterior = posterior
        # Store models at end of each round.
        self._model_bank.append(deepcopy(self._posterior))

        return deepcopy(self._posterior)

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
