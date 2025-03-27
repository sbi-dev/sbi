# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.adam import Adam
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.neural_nets.estimators import UnconditionalDensityEstimator
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
)
from sbi.neural_nets.factory import marginal_nn
from sbi.utils import check_estimator_arg, get_log_root
from sbi.utils.torchutils import assert_all_finite, process_device


class MarginalTrainer:
    def __init__(
        self,
        density_estimator: Union[
            Literal["bpf", "maf", "naf", "ncsf", "nsf", "sospf", "unaf"],
            Callable[[Tensor], UnconditionalDensityEstimator],
        ] = "nsf",
        device: str = "cpu",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        """Utility class for training a marginal estimator method.

        Args:
            density_estimator: Density estimator to use. Can be a string or a callable.
                If a string, it must be one of the following:
                - "bpf": Bijector Polynomial Flow
                - "maf": Masked Autoregressive Flow
                - "naf": Neural Autoregressive Flow
                - "ncsf": Neural Conditional Spline Flow
                - "nsf": Neural Spline Flow
                - "sospf": Sum-of-Squares Polynomial Flow
                - "unaf": Unconditional Neural Autoregressive Flow
                If a callable, it must be a function that returns a neural network
                that inherits from `UnconditionalDensityEstimator`.
            device: Device to use for training. Can be "cpu" or "cuda".
            summary_writer: Summary writer for logging training progress. If None,
                a new writer is created.
            show_progress_bars: Whether to show progress bars during training.
        """

        self._device = process_device(device)
        self._neural_net = None

        self._show_progress_bars = show_progress_bars
        self._val_loss = float("Inf")

        self._summary_writer = (
            self._default_summary_writer() if summary_writer is None else summary_writer
        )

        # Logging during training (by SummaryWriter).
        self._summary = dict(
            epochs_trained=[],
            best_validation_loss=[],
            validation_loss=[],
            training_loss=[],
            epoch_durations_sec=[],
        )

        check_estimator_arg(density_estimator)
        if isinstance(density_estimator, str):
            self._build_neural_net = marginal_nn(model=density_estimator)
        else:
            self._build_neural_net = density_estimator

    def get_dataloaders(
        self,
        training_batch_size: int = 200,
        validation_fraction: float = 0.1,
        dataloader_kwargs: Optional[dict] = None,
    ) -> Tuple[data.DataLoader, data.DataLoader]:
        """Return training and validation dataloaders."""

        x = self.get_samples()
        dataset = data.TensorDataset(x)

        # Get total number of training examples.
        num_examples = x.size(0)
        # Select random train and validation splits from (theta, x) pairs.
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples

        # Separate indices for training and validation
        permuted_indices = torch.randperm(num_examples)
        self.train_indices, self.val_indices = (
            permuted_indices[:num_training_examples],
            permuted_indices[num_training_examples:],
        )

        train_loader_kwargs = {
            "batch_size": min(training_batch_size, num_training_examples),
            "drop_last": True,
            "sampler": SubsetRandomSampler(self.train_indices.tolist()),
        }
        val_loader_kwargs = {
            "batch_size": min(training_batch_size, num_validation_examples),
            "shuffle": False,
            "drop_last": True,
            "sampler": SubsetRandomSampler(self.val_indices.tolist()),
        }
        if dataloader_kwargs is not None:
            train_loader_kwargs = dict(train_loader_kwargs, **dataloader_kwargs)
            val_loader_kwargs = dict(val_loader_kwargs, **dataloader_kwargs)

        train_loader = data.DataLoader(dataset, **train_loader_kwargs)
        val_loader = data.DataLoader(dataset, **val_loader_kwargs)

        return train_loader, val_loader

    def append_samples(self, x) -> "MarginalTrainer":
        self._x = x
        return self

    def get_samples(self) -> Tensor:
        return self._x

    def loss(self, x: Tensor) -> Tensor:
        """Return loss.

        The loss is the negative log prob

        Returns:
            Negative log prob.
        """
        if self._neural_net is None:
            raise ValueError(
                "Neural network has not been initialized. Please call `train` first."
            )
        else:
            x = reshape_to_batch_event(x, event_shape=self._neural_net.input_shape)
        loss = self._neural_net.loss(x)
        assert_all_finite(loss, "loss")
        return loss

    def train(
        self,
        training_batch_size: int = 200,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        dataloader_kwargs: Optional[dict] = None,
    ) -> UnconditionalDensityEstimator:
        r"""Return density estimator that approximates the distribution $p(x)$.

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
            show_train_summary: Whether to print the number of epochs and validation
                loss after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)

        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """

        # fake round setting just for compatibility with NeuralInference
        self._round = 0

        train_loader, val_loader = self.get_dataloaders(
            training_batch_size,
            validation_fraction,
            dataloader_kwargs=dataloader_kwargs,
        )

        if self._neural_net is None:
            # Get x to initialize NN
            x = self.get_samples()
            # Use only training data for building the neural net (z-scoring transforms)

            self._neural_net = self._build_neural_net(
                x[self.train_indices].to("cpu"),
            )

        self.optimizer = Adam(list(self._neural_net.parameters()), lr=learning_rate)
        self.epoch, self._val_loss = 0, float("Inf")

        self._neural_net.to(self._device)
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
                x_batch = batch[0].to(self._device)

                train_losses = self.loss(x_batch)
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
            self._summary["training_loss"].append(train_loss_average)

            # Calculate validation performance.
            self._neural_net.eval()
            val_loss_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    x_batch = batch[0].to(self._device)
                    # Take negative loss here to get validation log_prob.
                    val_losses = self.loss(x_batch)
                    val_loss_sum += val_losses.sum().item()

            # Take mean over all validation samples.
            self._val_loss = val_loss_sum / (
                len(val_loader) * val_loader.batch_size  # type: ignore
            )
            # Log validation loss for every epoch.
            self._summary["validation_loss"].append(self._val_loss)
            self._summary["epoch_durations_sec"].append(time.time() - epoch_start_time)

            self._maybe_show_progress(self._show_progress_bars, self.epoch)

        # Update summary.
        self._summary["epochs_trained"].append(self.epoch)
        self._summary["best_validation_loss"].append(self._best_val_loss)

        # Update tensorboard and summary dict.
        self._summarize(round_=self._round)

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self._neural_net.zero_grad(set_to_none=True)

        return deepcopy(self._neural_net)

    def _default_summary_writer(self) -> SummaryWriter:
        """Return summary writer logging to method- and simulator-specific directory."""

        method = self.__class__.__name__
        logdir = Path(
            get_log_root(), method, datetime.now().isoformat().replace(":", "_")
        )
        return SummaryWriter(logdir)

    def _converged(self, epoch: int, stop_after_epochs: int) -> bool:
        """Return whether the training converged yet and save best model state so far.

        Checks for improvement in validation performance over previous epochs.

        Args:
            epoch: Current epoch in training.
            stop_after_epochs: How many fruitless epochs to let pass before stopping.

        Returns:
            Whether the training has stopped improving, i.e. has converged.
        """
        converged = False

        assert self._neural_net is not None
        neural_net = self._neural_net

        # (Re)-start the epoch count with the first epoch or any improvement.
        if epoch == 0 or self._val_loss < self._best_val_loss:
            self._best_val_loss = self._val_loss
            self._epochs_since_last_improvement = 0
            self._best_model_state_dict = deepcopy(neural_net.state_dict())
        else:
            self._epochs_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        if self._epochs_since_last_improvement > stop_after_epochs - 1:
            neural_net.load_state_dict(self._best_model_state_dict)
            converged = True

        return converged

    @staticmethod
    def _maybe_show_progress(show: bool, epoch: int) -> None:
        if show:
            # end="\r" deletes the print statement when a new one appears.
            # https://stackoverflow.com/questions/3419984/. `\r` in the beginning due
            # to #330.
            print("\r", f"Training neural network. Epochs trained: {epoch}", end="")

    def _summarize(
        self,
        round_: int,
    ) -> None:
        """Update the summary_writer with statistics for a given round.

        During training several performance statistics are added to the summary, e.g.,
        using `self._summary['key'].append(value)`. This function writes these values
        into summary writer object.

        Args:
            round: index of round

        Scalar tags:
            - epochs_trained:
                number of epochs trained
            - best_validation_loss:
                best validation loss (for each round).
            - validation_loss:
                validation loss for every epoch (for each round).
            - training_loss
                training loss for every epoch (for each round).
            - epoch_durations_sec
                epoch duration for every epoch (for each round)

        """

        # Add most recent training stats to summary writer.
        self._summary_writer.add_scalar(
            tag="epochs_trained",
            scalar_value=self._summary["epochs_trained"][-1],
            global_step=round_ + 1,
        )

        self._summary_writer.add_scalar(
            tag="best_validation_loss",
            scalar_value=self._summary["best_validation_loss"][-1],
            global_step=round_ + 1,
        )

        # Add validation loss for every epoch.
        # Offset with all previous epochs.
        offset = (
            torch.tensor(self._summary["epochs_trained"][:-1], dtype=torch.int)
            .sum()
            .item()
        )
        for i, vlp in enumerate(self._summary["validation_loss"][offset:]):
            self._summary_writer.add_scalar(
                tag="validation_loss",
                scalar_value=vlp,
                global_step=offset + i,
            )

        for i, tlp in enumerate(self._summary["training_loss"][offset:]):
            self._summary_writer.add_scalar(
                tag="training_loss",
                scalar_value=tlp,
                global_step=offset + i,
            )

        for i, eds in enumerate(self._summary["epoch_durations_sec"][offset:]):
            self._summary_writer.add_scalar(
                tag="epoch_durations_sec",
                scalar_value=eds,
                global_step=offset + i,
            )

        self._summary_writer.flush()
