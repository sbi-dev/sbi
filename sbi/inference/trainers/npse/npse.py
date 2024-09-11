# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
import time
from copy import deepcopy
from typing import Any, Callable, Optional, Union

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
from sbi.inference.posteriors.score_posterior import ScorePosterior
from sbi.neural_nets.estimators.score_estimator import ConditionalScoreEstimator
from sbi.neural_nets.factory import posterior_score_nn
from sbi.utils import (
    check_estimator_arg,
    handle_invalid_x,
    npe_msg_on_invalid_x,
    test_posterior_net_for_multi_d_x,
    validate_theta_and_x,
    warn_if_zscoring_changes_data,
)
from sbi.utils.sbiutils import ImproperEmpirical, mask_sims_from_prior


class NPSE(NeuralInference):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        score_estimator: Union[str, Callable] = "mlp",
        sde_type: str = "ve",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        **kwargs,
    ):
        """Base class for Neural Posterior Score Estimation methods.

        Instead of performing conditonal *density* estimation, NPSE methods perform
        conditional *score* estimation i.e. they estimate the gradient of the log
        density using denoising score matching loss.

        NOTE: NPSE does not support multi-round inference with flexible proposals yet.
        You can try to run multi-round with truncated proposals, but note that this is
        not tested yet.

        Args:
            prior: Prior distribution.
            score_estimator: Neural network architecture for the score estimator. Can be
                a string (e.g. 'mlp' or 'ada_mlp') or a callable that returns a neural
                network.
            sde_type: Type of SDE to use. Must be one of ['vp', 've', 'subvp'].
            device: Device to run the training on.
            logging_level: Logging level for the training. Can be an integer or a
                string.
            summary_writer: Tensorboard summary writer.
            show_progress_bars: Whether to show progress bars during training.
            kwargs: Additional keyword arguments.

        References:
            - Geffner, Tomas, George Papamakarios, and Andriy Mnih. "Score modeling for
              simulation-based inference." ICML 2023.
            - Sharrock, Louis, et al. "Sequential neural score estimation: Likelihood-
              free inference with conditional score based diffusion models." ICML 2024.
        """

        super().__init__(
            prior=prior,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
        )

        # As detailed in the docstring, `score_estimator` is either a string or
        # a callable. The function creating the neural network is attached to
        # `_build_neural_net`. It will be called in the first round and receive
        # thetas and xs as inputs, so that they can be used for shape inference and
        # potentially for z-scoring.
        check_estimator_arg(score_estimator)
        if isinstance(score_estimator, str):
            self._build_neural_net = posterior_score_nn(
                sde_type=sde_type, score_net_type=score_estimator, **kwargs
            )
        else:
            self._build_neural_net = score_estimator

        self._proposal_roundwise = []

    def append_simulations(
        self,
        theta: Tensor,
        x: Tensor,
        proposal: Optional[DirectPosterior] = None,
        exclude_invalid_x: Optional[bool] = None,
        data_device: Optional[str] = None,
    ) -> "NPSE":
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
        assert (
            proposal is None
        ), "Multi-round NPSE is not yet implemented. Please use single-round NPSE."
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

        npe_msg_on_invalid_x(num_nans, num_infs, exclude_invalid_x, "Single-round NPE")

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
        stop_after_epochs: int = 200,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        ema_loss_decay: float = 0.1,
        resume_training: bool = False,
        force_first_round_loss: bool = False,
        discard_prior_samples: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[dict] = None,
    ) -> ConditionalScoreEstimator:
        r"""Returns a score estimator that approximates the score
        $\nabla_\theta \log p(\theta|x)$.

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
            Score estimator that approximates the posterior score.
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
                    theta_batch,
                    x_batch,
                    masks_batch,
                    proposal,
                    calibration_kernel,
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
                    # Take negative loss here to get validation log_prob.
                    val_losses = self._loss(
                        theta_batch,
                        x_batch,
                        masks_batch,
                        proposal,
                        calibration_kernel,
                        force_first_round_loss=force_first_round_loss,
                    )
                    val_loss_sum += val_losses.sum().item()

            # Take mean over all validation samples.
            val_loss = val_loss_sum / (
                len(val_loader) * val_loader.batch_size  # type: ignore
            )

            # NOTE: Due to the inherently noisy nature we do instead log a exponential
            # moving average of the validation loss.
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

    def build_posterior(
        self,
        score_estimator: Optional[ConditionalScoreEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: str = "sde",
    ) -> ScorePosterior:
        r"""Build posterior from the score estimator.

        For NPSE, the posterior distribution that is returned here implements the
        following functionality over the raw neural density estimator:
        - correct the calculation of the log probability such that it compensates for
            the leakage.
        - reject samples that lie outside of the prior bounds.

        Args:
            score_estimator: The score estimator that the posterior is based on.
                If `None`, use the latest neural score estimator that was trained.
            prior: Prior distribution.
            sample_with: Method to use for sampling from the posterior. Can be one of
                'sde' (default) or 'ode'. The 'sde' method uses the score to
                do a Langevin diffusion step, while the 'ode' method uses the score to
                define a probabilistic ODE and solves it with a numerical ODE solver.

        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods.
        """
        if prior is None:
            assert self._prior is not None, (
                "You did not pass a prior. You have to pass the prior either at "
                "initialization `inference = NPSE(prior)` or to "
                "`.build_posterior(prior=prior)`."
            )
            prior = self._prior
        else:
            utils.check_prior(prior)

        if score_estimator is None:
            score_estimator = self._neural_net
            # If internal net is used device is defined.
            device = self._device
        # Otherwise, infer it from the device of the net parameters.
        else:
            # TODO: Add protocol for checking if the score estimator has forward and
            # loss methods with the correct signature.
            device = str(next(score_estimator.parameters()).device)

        posterior = ScorePosterior(
            score_estimator,  # type: ignore
            prior,
            device=device,
            sample_with=sample_with,
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
        raise NotImplementedError("Multi-round NPSE is not yet implemented.")

    def _loss(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
        calibration_kernel: Callable,
        force_first_round_loss: bool = False,
    ) -> Tensor:
        """Return loss from score estimator. Currently only single-round NPSE
         is implemented, i.e., no proposal correction is applied for later rounds.

        The loss is the negative log prob. Irrespective of the round or SNPE method
        (A, B, or C), it can be weighted with a calibration kernel.

        Returns:
            Calibration kernel-weighted negative log prob.
            force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
        """
        if self._round == 0 or force_first_round_loss:
            # First round loss.
            loss = self._neural_net.loss(theta, x)
        else:
            raise NotImplementedError(
                "Multi-round NPSE with arbitrary proposals is not implemented"
            )

        return calibration_kernel(x) * loss

    def _converged(self, epoch: int, stop_after_epochs: int) -> bool:
        """Check if training has converged.

        Unlike the `._converged` method in base.py, this method does not reset to the
        best model. We noticed that this improves performance. Deleting this method
        will make C2ST tests fail. This is because the loss is very stochastic, so
        resetting might reset to an underfitted model. Ideally, we would write a
        custom `._converged()` method which checks whether the loss is still going
        down **for all t**.

        Args:
            epoch: Current epoch.
            stop_after_epochs: Number of epochs to wait for improvement on the
                validation set before terminating training.

        Returns:
            Whether training has converged.
        """
        converged = False

        # No checkpointing, just check if the validation loss has improved.

        # (Re)-start the epoch count with the first epoch or any improvement.
        if epoch == 0 or self._val_loss < self._best_val_loss:
            self._best_val_loss = self._val_loss
            self._epochs_since_last_improvement = 0
        else:
            self._epochs_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        if self._epochs_since_last_improvement > stop_after_epochs - 1:
            converged = True

        return converged
