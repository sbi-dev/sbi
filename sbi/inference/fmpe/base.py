# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License v2.0, see <https://www.apache.org/licenses/LICENSE-2.0>.


import time
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Union, Tuple

import torch
from torch import optim
from torch.distributions import Distribution
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter

from sbi import utils as utils
from sbi.inference.base import NeuralInference
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.neural_nets.density_estimators.zuko_flow_estimator import (
    ZukoFlowMatchingEstimator,
)
from sbi.neural_nets.density_estimators import ConditionalDensityEstimator
from sbi.utils.sbiutils import mask_sims_from_prior

from sbi.utils import (
    RestrictedPrior,
    check_estimator_arg,
    handle_invalid_x,
    nle_nre_apt_msg_on_invalid_x,
    npe_msg_on_invalid_x,
    test_posterior_net_for_multi_d_x,
    validate_theta_and_x,
    warn_if_zscoring_changes_data,
    x_shape_from_simulation,
)


class FMPE(NeuralInference):
    """Implements the Flow Matching Posterior Estimator (FMPE) for 
    simulation-based inference. 
    """
    def __init__(
        self,
        prior: Optional[Distribution],
        density_estimator: Optional[ZukoFlowMatchingEstimator] = None,
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
    ) -> None:
        """Initialization method for the FMPE class.
        
        Args:
            prior: Prior distribution.
            density_estimator: Density estimator for the FMPE. Defaults to None.
            device: Device to use for training. Defaults to "cpu".
            logging_level: Logging level. Defaults to "WARNING".
            summary_writer: Summary writer for tensorboard. Defaults to None.
            show_progress_bars: Whether to show progress bars. Defaults to True.
        """
        # obtain the shape of the prior samples
        if density_estimator is None:
            self._build_neural_net = utils.posterior_nn(model="zuko_fm")

        super().__init__(
            prior=prior,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
        )

    # todo: this is not correct, the method should return a vector field
    # estimator and not a density est.
    def train(
        self,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        discard_prior_samples: bool = False,
        resume_training: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[dict] = None,
    ) -> ConditionalDensityEstimator:
        """Train the density estimator.
        
        Args:
            training_batch_size: Batch size for training. Defaults to 50.
            learning_rate: Learning rate for training. Defaults to 5e-4.
            validation_fraction: Fraction of the data to use for validation. Defaults to 0.1.
            stop_after_epochs: Number of epochs to train for. Defaults to 20.
            max_num_epochs: Maximum number of epochs to train for. Defaults to 2**31 - 1.
            clip_max_norm: Maximum norm for gradient clipping. Defaults to 5.0.
            discard_prior_samples: Whether to discard prior samples. Defaults to False.
            resume_training: Whether to resume training. Defaults to False.
            show_train_summary: Whether to show the training summary. Defaults to False.
            dataloader_kwargs: Additional keyword arguments for the dataloader. Defaults to None.
        
        Returns:
            DensityEstimator: Trained density estimator.
        """
        if resume_training:
            raise NotImplementedError("Resume training is not yet implemented.")

        start_idx = 0  # as there is no multi-round FMPE yet
        current_round = 1  # as there is no multi-round FMPE yet
        self._data_round_index.append(current_round)

        train_loader, val_loader = self.get_dataloaders(
            start_idx,
            training_batch_size,
            validation_fraction,
            resume_training=False,
            dataloader_kwargs=dataloader_kwargs,
        )

        if self._neural_net is None:
            # Get theta,x to initialize NN
            theta, x, _ = self.get_simulations(starting_round=start_idx)
            # Use only training data for building the neural net (z-scoring transforms)

            self._neural_net = self._build_neural_net(
                theta[self.train_indices].to("cpu"),
                x[self.train_indices].to("cpu"),
            )
            self._x_shape = x_shape_from_simulation(x.to("cpu"))

            # test_posterior_net_for_multi_d_x(
            #     self._neural_net,
            #     theta.to("cpu"),
            #     x.to("cpu"),
            # )

            del theta, x

        # Move entire net to device for training.
        self._neural_net.to(self._device)
        
        # initialize optimizer and training parameters
        if not resume_training:
            self.optimizer = optim.Adam(
                list(self._neural_net.net.parameters()), lr=learning_rate
            )
            self.epoch, self._val_log_prob = 0, float("-Inf")

        # while self.epoch <= max_num_epochs and not self._converged(
        #     self.epoch, stop_after_epochs
        # ):
        while self.epoch <= max_num_epochs:
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
            # self._neural_net.eval()
            val_log_prob_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                    )
                    # Take negative loss here to get validation log_prob.
                    val_losses = self._neural_net.loss(theta_batch, x_batch)
                    val_log_prob_sum -= val_losses.sum().item()

            # Take mean over all validation samples.
            val_log_prob = val_log_prob_sum / (
                len(val_loader) * val_loader.batch_size  # type: ignore
            )
            # Log validation log prob for every epoch.
            self._summary["validation_loss"].append(val_log_prob)
            self._summary["epoch_durations_sec"].append(time.time() - epoch_start_time)

            self._maybe_show_progress(self._show_progress_bars, self.epoch)

        # self._report_convergence_at_end(self.epoch, stop_after_epochs, max_num_epochs)

        # Update summary.
        # self._summary["epochs_trained"].append(self.epoch)
        # self._summary["best_validation_log_prob"].append(self._best_val_log_prob)

        # Update tensorboard and summary dict.
        # self._summarize(round_=self._round)

        # Update description for progress bar.
        # if show_train_summary:
        #     print(self._describe_round(self._round, self._summary))

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        # self._neural_net.zero_grad(set_to_none=True)

        return deepcopy(self._neural_net)

    def build_posterior(
        self,
        density_estimator: Optional[ConditionalDensityEstimator] = None,
        prior: Optional[Distribution] = None,
        direct_sampling_parameters: Optional[Dict[str, Any]] = None,
    ) -> DirectPosterior:
        """Build the posterior distribution.
        
        Args:
            density_estimator: Density estimator for the posterior. Defaults to None.
            prior: Prior distribution. Defaults to None.
            direct_sampling_parameters: Direct sampling parameters. Defaults to None.

        Returns:
            DirectPosterior: Posterior distribution.
        """
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
            # todo: what we call density estimator is often saved as _neural_net in
            # other classes
            posterior_estimator = self._neural_net
            # If internal net is used device is defined.
            device = self._device
        else:
            posterior_estimator = density_estimator
            # Otherwise, infer it from the device of the net parameters.
            device = next(density_estimator.parameters()).device.type

        # todo: what's the `posterior_estimator_based_potential` ???
        # in the super SNPE class, there's the following block:
        # potential_fn, theta_transform = posterior_estimator_based_potential(
        #     posterior_estimator=posterior_estimator,
        #     prior=prior,
        #     x_o=None,
        # )

        self._posterior = DirectPosterior(
            posterior_estimator=posterior_estimator,  # type: ignore
            prior=prior,
            # todo: what's the x_shape used for??
            # x_shape=self.x_dim,
            device=device,
            **direct_sampling_parameters or {},
        )

        return deepcopy(self._posterior)

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
            # The `_data_round_index` will later be used to infer if one should train
            # with MLE loss or with atomic loss (see, in `train()`:
            # self._round = max(self._data_round_index))
            current_round = 0
        else:
            raise NotImplementedError("Mutli-round FMPE is currently not supported.")

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
        # todo: check what this does exactly
        npe_msg_on_invalid_x(num_nans, num_infs, exclude_invalid_x, "Single-round FMPE")

        prior_masks = mask_sims_from_prior(int(current_round > 0), theta.size(0))

        self._theta_roundwise.append(theta)
        self._x_roundwise.append(x)
        self._prior_masks.append(prior_masks)

        # self._proposal_roundwise.append(proposal)

        return self

    # def get_dataloaders(
    #     self,
    #     training_batch_size: int,
    #     validation_fraction: float,
    #     dataloader_kwargs: Optional[Dict[str, Any]] = None,
    # ) -> Tuple[DataLoader, DataLoader]:
    #     # todo: utilize method from super
    #     dataset = TensorDataset(self._theta, self._x)

    #     num_examples = self._theta.size(0)

    #     num_training_examples = int((1 - validation_fraction) * num_examples)
    #     num_validation_examples = num_examples - num_training_examples

    #     train_loader_kwargs = {
    #         "batch_size": min(training_batch_size, num_training_examples),
    #         "drop_last": True,
    #         # "sampler": SubsetRandomSampler(self.train_indices.tolist()),
    #     }
    #     val_loader_kwargs = {
    #         "batch_size": min(training_batch_size, num_validation_examples),
    #         "shuffle": False,
    #         "drop_last": True,
    #         # "sampler": SubsetRandomSampler(self.val_indices.tolist()),
    #     }

    #     if dataloader_kwargs is not None:
    #         train_loader_kwargs = dict(train_loader_kwargs, **dataloader_kwargs)
    #         val_loader_kwargs = dict(val_loader_kwargs, **dataloader_kwargs)

    #     train_loader = DataLoader(dataset, **train_loader_kwargs)
    #     val_loader = DataLoader(dataset, **val_loader_kwargs)

    #     return train_loader, val_loader