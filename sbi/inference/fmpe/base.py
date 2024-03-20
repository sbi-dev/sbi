# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


import time
from copy import deepcopy
from typing import Optional, Union

import torch
from torch import optim
from torch.distributions import Distribution
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.inference.base import NeuralInference
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.neural_nets.density_estimators.zuko_flow_estimator import ZukoFlowMatchingEstimator

from sbi.types import Shape


class FMPE(NeuralInference):
    def __init__(
        self,
        x_shape: torch.Size,
        prior: Optional[Distribution] = None,
        max_sampling_batch_size: int = 10_000,
        enable_transform: bool = True,
        density_estimator: ZukoFlowMatchingEstimator = None,
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
    ) -> None:
        # obtain the shape of the prior samples
        self.theta_dim = prior.sample((1,)).shape[-1]
        if not density_estimator:
            # init default Flow Matching Estimator using a MLP as regressor network
            density_estimator = ZukoFlowMatchingEstimator(
                theta_shape=self.theta_dim,
                condition_shape=x_shape,
                device=device,
            )
        else:
            density_estimator = density_estimator

        super().__init__(
            prior=prior,
            density_estimator=density_estimator,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
        )

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
    ) -> VectorFieldEstimator:

        if resume_training:
            raise NotImplementedError("Resume training is not yet implemented.")

        # todo: clarify this part
        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(discard_prior_samples and self._round > 0)

        train_loader, val_loader = self.get_dataloaders(
            start_idx,
            training_batch_size,
            validation_fraction,
            resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )

        # initialize optimizer and training parameters
        if not resume_training:
            self.optimizer = optim.Adam(
                list(self.vf_estimator.net.parameters()), lr=learning_rate
            )
            self.epoch, self._val_log_prob = 0, float("-Inf")

        while self.epoch <= max_num_epochs and not self._converged(
            self.epoch, stop_after_epochs
        ):
            self.vf_estimator.net.train()
            train_log_probs_sum = 0
            epoch_start_time = time.time()
            for batch in train_loader:
                self.optimizer.zero_grad()
                # get batches on current device.
                # todo: clarify what masks_batch does here ... currently skipped
                theta_batch, x_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                )

                train_losses = self.vf_estimator.loss(theta_batch, x_batch)
                train_loss = torch.mean(train_losses)
                train_log_probs_sum -= train_losses.sum().item()

                train_loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._neural_net.parameters(), max_norm=clip_max_norm
                    )
                self.optimizer.step()

            self.epoch += 1

            train_log_prob_average = train_log_probs_sum / (
                len(train_loader) * train_loader.batch_size  # type: ignore
            )
            self._summary["training_log_probs"].append(train_log_prob_average)

            # Calculate validation performance.
            self._neural_net.eval()
            val_log_prob_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    # todo: clarify what masks_batch does here ... currently skipped
                    theta_batch, x_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                    )
                    # Take negative loss here to get validation log_prob.
                    val_losses = self.vf_estimator.loss(theta_batch, x_batch)
                    val_log_prob_sum -= val_losses.sum().item()

            # Take mean over all validation samples.
            self._val_log_prob = val_log_prob_sum / (
                len(val_loader) * val_loader.batch_size  # type: ignore
            )
            # Log validation log prob for every epoch.
            self._summary["validation_log_probs"].append(self._val_log_prob)
            self._summary["epoch_durations_sec"].append(time.time() - epoch_start_time)

            self._maybe_show_progress(self._show_progress_bars, self.epoch)

        self._report_convergence_at_end(self.epoch, stop_after_epochs, max_num_epochs)

        # Update summary.
        self._summary["epochs_trained"].append(self.epoch)
        self._summary["best_validation_log_prob"].append(self._best_val_log_prob)

        # Update tensorboard and summary dict.
        self._summarize(round_=self._round)

        # Update description for progress bar.
        if show_train_summary:
            print(self._describe_round(self._round, self._summary))

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self._neural_net.zero_grad(set_to_none=True)

        return deepcopy(self.vf_estimator.net)

    def build_posterior(self) -> DirectPosterior:
        raise NotImplementedError
