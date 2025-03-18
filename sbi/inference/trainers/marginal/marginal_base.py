# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import time
from abc import ABC
from copy import deepcopy
from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.adam import Adam
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.inference.trainers.base import NeuralInference
from sbi.neural_nets.estimators import UnconditionalDensityEstimator
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
)
from sbi.neural_nets.net_builders.flow import build_zuko_unconditional_flow
from sbi.sbi_types import Shape
from sbi.utils import check_estimator_arg
from sbi.utils.torchutils import assert_all_finite


def marginal_nn(
    model: str,
    z_score_x: Optional[str] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    num_components: int = 10,
    **kwargs: Any,
) -> Callable:
    r"""
    Returns a function that builds a density estimator for learning the marginal.

    Args:
        model: The type of density estimator that will be created. One of [`mdn`,
            `made`, `maf`, `maf_rqs`, `nsf`].
        z_score_x: Whether to z-score samples $x$ before passing them into
            the network, can take one of the following:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms when a flow is used. Only relevant if
            density estimator is a normalizing flow (i.e. currently either a `maf` or a
            `nsf`). Ignored if density estimator is a `mdn` or `made`.
        num_bins: Number of bins used for the splines in `nsf`. Ignored if density
            estimator not `nsf`.
        num_components: Number of mixture components for a mixture of Gaussians.
            Ignored if density estimator is not an mdn.
        kwargs: additional custom arguments passed to downstream build functions.
    """

    kwargs = dict(
        zip(
            (
                "z_score_x",
                "hidden_features",
                "num_transforms",
                "num_bins",
                "num_components",
            ),
            (
                z_score_x,
                hidden_features,
                num_transforms,
                num_bins,
                num_components,
            ),
            strict=False,
        ),
        **kwargs,
    )

    def build_fn(batch_x):
        return build_zuko_unconditional_flow(which_nf=model, batch_x=batch_x)

    return build_fn


class MarginalEstimator(NeuralInference, ABC):
    def __init__(
        self,
        density_estimator: Union[str, Callable] = "MAF",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        """Base class for Marginal estimation method."""

        super().__init__(
            prior=None,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
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
        x = self.get_simulations()
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

    def append_simulations(self, x) -> "MarginalEstimator":
        self._x = x
        return self

    def get_simulations(self) -> Tensor:
        return self._x

    def train(
        self,
        training_batch_size: int = 200,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        resume_training: bool = False,
        show_train_summary: bool = False,
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
            x = self.get_simulations()
            # Use only training data for building the neural net (z-scoring transforms)

            self._neural_net = self._build_neural_net(
                x[self.train_indices].to("cpu"),
            )

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
                x_batch = batch[0].to(self._device)

                train_losses = self._loss(x_batch)
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
                    val_losses = self._loss(x_batch)
                    val_loss_sum += val_losses.sum().item()

            # Take mean over all validation samples.
            self._val_loss = val_loss_sum / (
                len(val_loader) * val_loader.batch_size  # type: ignore
            )
            # Log validation loss for every epoch.
            self._summary["validation_loss"].append(self._val_loss)
            self._summary["epoch_durations_sec"].append(time.time() - epoch_start_time)

            self._maybe_show_progress(self._show_progress_bars, self.epoch)

        # self._report_convergence_at_end(self.epoch, stop_after_epochs, max_num_epochs)

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

    def _loss(
        self,
        x: Tensor,
    ) -> Tensor:
        """Return loss.

        The loss is the negative log prob

        Returns:
            Negative log prob.
        """
        x = reshape_to_batch_event(x, event_shape=self._neural_net.input_shape)
        loss = self._neural_net.loss(x)
        assert_all_finite(loss, "loss")
        return loss

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
    ):
        return self._neural_net.sample(sample_shape)
