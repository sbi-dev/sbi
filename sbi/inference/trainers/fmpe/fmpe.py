# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License v2.0, see <https://www.apache.org/licenses/LICENSE-2.0>.


import time
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch.distributions import Distribution
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.adam import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from sbi import utils as utils
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.trainers.base import NeuralInference
from sbi.neural_nets import flowmatching_nn
from sbi.neural_nets.estimators import ConditionalDensityEstimator
from sbi.utils import (
    RestrictedPrior,
    handle_invalid_x,
    npe_msg_on_invalid_x,
    validate_theta_and_x,
    warn_if_zscoring_changes_data,
)
from sbi.utils.sbiutils import mask_sims_from_prior


class FMPE(NeuralInference):
    """Implements the Flow Matching Posterior Estimator (FMPE) for
    simulation-based inference.
    """

    def __init__(
        self,
        prior: Optional[Distribution],
        density_estimator: Union[str, Callable] = "mlp",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
    ) -> None:
        """Initialization method for the FMPE class.

        Args:
            prior: Prior distribution.
            density_estimator: Neural network architecture used to learn the vector
                field for flow matching. Can be a string, e.g., 'mlp' or 'resnet', or a
                `Callable` that builds a corresponding neural network.
            device: Device to use for training.
            logging_level: Logging level.
            summary_writer: Summary writer for tensorboard.
            show_progress_bars: Whether to show progress bars.
        """
        # obtain the shape of the prior samples
        if isinstance(density_estimator, str):
            self._build_neural_net = flowmatching_nn(model=density_estimator)
        else:
            self._build_neural_net = density_estimator

        super().__init__(
            prior=prior,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
        )

    def append_simulations(
        self,
        theta: torch.Tensor,
        x: torch.Tensor,
        proposal: Optional[DirectPosterior] = None,
        exclude_invalid_x: Optional[bool] = None,
        data_device: Optional[str] = None,
    ) -> NeuralInference:
        if (
            proposal is None
            or proposal is self._prior
            or (
                isinstance(proposal, RestrictedPrior) and proposal._prior is self._prior
            )
        ):
            current_round = 0
        else:
            raise NotImplementedError(
                "Sequential FMPE with proposal different from prior is not implemented."
            )

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
        # Check whether there are NaNs or Infs in the data and remove accordingly.
        npe_msg_on_invalid_x(
            num_nans=num_nans,
            num_infs=num_infs,
            exclude_invalid_x=exclude_invalid_x,
            algorithm="Single-round FMPE",
        )

        self._data_round_index.append(current_round)
        prior_masks = mask_sims_from_prior(int(current_round > 0), theta.size(0))

        self._theta_roundwise.append(theta)
        self._x_roundwise.append(x)
        self._prior_masks.append(prior_masks)

        return self

    def train(
        self,
        training_batch_size: int = 200,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        resume_training: bool = False,
        force_first_round_loss: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[dict] = None,
    ) -> ConditionalDensityEstimator:
        """Train the flow matching estimator.

        Args:
            training_batch_size: Batch size for training. Defaults to 50.
            learning_rate: Learning rate for training. Defaults to 5e-4.
            validation_fraction: Fraction of the data to use for validation.
            stop_after_epochs: Number of epochs to train for. Defaults to 20.
            max_num_epochs: Maximum number of epochs to train for.
            clip_max_norm: Maximum norm for gradient clipping. Defaults to 5.0.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            force_first_round_loss: Whether to allow training with
                simulations that have not been sampled from the prior, e.g., in a
                sequential inference setting. Note that can lead to biased inference
                results.
            show_train_summary: Whether to show the training summary. Defaults to False.
            dataloader_kwargs: Additional keyword arguments for the dataloader.

        Returns:
            DensityEstimator: Trained flow matching estimator.
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
                "FMPE will not be the true posterior. Instead, it will be the proposal "
                "posterior, which (usually) is more narrow than the true posterior."
            )

        start_idx = 0  # as there is no multi-round FMPE yet

        train_loader, val_loader = self.get_dataloaders(
            start_idx,
            training_batch_size,
            validation_fraction,
            resume_training=resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )

        if self._neural_net is None:
            # Get theta, x to initialize NN
            theta, x, _ = self.get_simulations(starting_round=start_idx)

            # Use only training data for building the neural net (z-scoring transforms)
            self._neural_net = self._build_neural_net(
                theta[self.train_indices].to("cpu"),
                x[self.train_indices].to("cpu"),
            )

            del theta, x

        # Move entire net to device for training.
        self._neural_net.to(self._device)

        # initialize optimizer and training parameters
        if not resume_training:
            self.optimizer = Adam(
                list(self._neural_net.net.parameters()), lr=learning_rate
            )
            self.epoch = 0
            # NOTE: in the FMPE context we use MSE loss, not log probs.
            self._val_loss = float("Inf")

        while self.epoch <= max_num_epochs and not self._converged(
            self.epoch, stop_after_epochs
        ):
            self._neural_net.net.train()
            train_loss_sum = 0
            epoch_start_time = time.time()
            for batch in train_loader:
                self.optimizer.zero_grad()
                # get batches on current device.
                theta_batch, x_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                )

                train_loss = self._neural_net.loss(theta_batch, x_batch).mean()
                train_loss_sum += train_loss.item()

                train_loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._neural_net.net.parameters(), max_norm=clip_max_norm
                    )
                self.optimizer.step()

            self.epoch += 1

            train_loss_average = train_loss_sum / len(train_loader)  # type: ignore
            self._summary["training_loss"].append(train_loss_average)

            # Calculate validation performance.
            self._neural_net.eval()
            val_loss_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                    )
                    # Aggregate the validation losses.
                    val_losses = self._neural_net.loss(theta_batch, x_batch)
                    val_loss_sum += val_losses.sum().item()

            # Take mean over all validation samples.
            self._val_loss = val_loss_sum / (
                len(val_loader) * val_loader.batch_size  # type: ignore
            )
            # Log validation loss for every epoch.
            self._summary["validation_loss"].append(self._val_loss)
            self._summary["epoch_durations_sec"].append(time.time() - epoch_start_time)

            self._maybe_show_progress(self._show_progress_bars, self.epoch)

        self._report_convergence_at_end(self.epoch, stop_after_epochs, max_num_epochs)

        # Update summary.
        self._summary["epochs_trained"].append(self.epoch)
        self._summary["best_validation_loss"].append(self._best_val_loss)

        # Update tensorboard and summary dict.
        self._summarize(round_=self._round)

        # Update description for progress bar.
        if show_train_summary:
            print(self._describe_round(self._round, self._summary))

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self._neural_net.zero_grad(set_to_none=True)

        return deepcopy(self._neural_net)

    def build_posterior(
        self,
        density_estimator: Optional[ConditionalDensityEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: str = "direct",
        direct_sampling_parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> DirectPosterior:
        """Build the posterior distribution.

        Args:
            density_estimator: Density estimator for the posterior.
            prior: Prior distribution.
            sample_with: Sampling method, currently only "direct" is supported.
            direct_sampling_parameters: kwargs for DirectPosterior.

        Returns:
            DirectPosterior: Posterior distribution.
        """
        if sample_with != "direct":
            raise NotImplementedError(
                "Currently, only direct sampling is supported for FMPE."
            )

        if prior is None:
            assert self._prior is not None, (
                "You did not pass a prior. You have to pass the prior either at "
                "initialization `inference = SNPE(prior)` or to "
                "`.build_posterior(prior=prior)`."
            )
            prior = self._prior
        else:
            utils.check_prior(prior)

        if density_estimator is None:
            posterior_estimator = self._neural_net
            # If internal net is used device is defined.
            device = self._device
        else:
            posterior_estimator = density_estimator
            # Otherwise, infer it from the device of the net parameters.
            device = next(density_estimator.parameters()).device.type

        self._posterior = DirectPosterior(
            posterior_estimator=posterior_estimator,  # type: ignore
            prior=prior,
            device=device,
            **direct_sampling_parameters or {},
        )

        return deepcopy(self._posterior)
