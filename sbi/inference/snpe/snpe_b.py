from __future__ import annotations
from typing import Callable, Optional
import logging

import torch
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter

from sbi.inference.snpe.snpe_base import SnpeBase


class SnpeB(SnpeBase):
    def __init__(
        self,
        simulator: Callable,
        prior,
        x_o: Tensor,
        density_estimator: Optional[nn.Module] = None,
        calibration_kernel: Optional[Callable] = None,
        use_combined_loss: bool = False,
        z_score_x: bool = True,
        z_score_min_std: float = 1e-7,
        simulation_batch_size: Optional[int] = 1,
        retrain_from_scratch_each_round: bool = False,
        discard_prior_samples: bool = False,
        summary_writer: Optional[SummaryWriter] = None,
        num_workers: int = 1,
        worker_batch_size: int = 20,
        device: Optional[torch.device] = None,
        skip_input_checks: bool = False,
        show_progressbar: bool = True,
        show_round_summary: bool = False,
        logging_level: int = logging.WARNING,
    ):
        r"""SNPE-B [1]

        [1] _Flexible statistical inference for mechanistic models of neural dynamics_,
            Lueckmann et al., NeurIPS 2017, https://arxiv.org/abs/1711.01861.

        Args:
            use_combined_loss: Whether to train jointly on prior samples using maximum
                likelihood and on all samples using importance-weighted loss.

        See docstring of `SnpeBase` class for all other arguments.
        """

        self._use_combined_loss = use_combined_loss

        super().__init__(
            simulator=simulator,
            prior=prior,
            x_o=x_o,
            density_estimator=density_estimator,
            calibration_kernel=calibration_kernel,
            z_score_x=z_score_x,
            z_score_min_std=z_score_min_std,
            simulation_batch_size=simulation_batch_size,
            retrain_from_scratch_each_round=retrain_from_scratch_each_round,
            discard_prior_samples=discard_prior_samples,
            num_workers=num_workers,
            worker_batch_size=worker_batch_size,
            device=device,
            skip_input_checks=skip_input_checks,
            show_progressbar=show_progressbar,
            show_round_summary=show_round_summary,
            logging_level=logging_level,
        )

    def _get_log_prob_proposal_posterior(
        self, theta: Tensor, x: Tensor, masks: Tensor
    ) -> Tensor:
        """
        Return importance-weighted log probability (Lueckmann, Goncalves et al., 2017).

        Args:
            theta: Batch of parameters θ.
            x: Batch of data.
            masks: Whether to retrain with prior loss (for each prior sample).

        Returns:
            Log probability of proposal posterior.
        """

        batch_size = theta.shape[0]

        # Evaluate posterior.
        log_prob_posterior = self._neural_posterior.neural_net.log_prob(theta, x)
        log_prob_posterior = log_prob_posterior.reshape(batch_size)
        self._assert_all_finite(log_prob_posterior, "posterior eval")

        # Evaluate prior.
        log_prob_prior = self._prior.log_prob(theta).reshape(batch_size)
        self._assert_all_finite(log_prob_prior, "prior eval.")

        # Evaluate proposal.
        log_prob_proposal = self._model_bank[-1].neural_net.log_prob(theta, x)
        self._assert_all_finite(log_prob_proposal, "proposal posterior eval")

        # Compute log prob with importance weights.
        log_prob = (
            self.calibration_kernel(x)
            * torch.exp(log_prob_prior - log_prob_proposal)
            * log_prob_posterior
        )

        # XXX This evaluates the posterior on _all_ prior samples.
        if self._use_combined_loss:
            log_prob_posterior_non_atomic = self._neural_posterior.neural_net.log_prob(
                theta, x
            )
            masks = masks.reshape(-1)
            log_prob = masks * log_prob_posterior_non_atomic + log_prob

        return log_prob
