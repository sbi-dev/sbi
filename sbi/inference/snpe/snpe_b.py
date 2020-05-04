from __future__ import annotations

import os

import torch
from torch import distributions
from torch.utils.tensorboard import SummaryWriter

import sbi.utils as utils
from sbi.inference.snpe.snpe_base import SnpeBase


class SnpeB(SnpeBase):
    def __init__(
        self,
        simulator,
        prior,
        x_o,
        num_pilot_sims=100,
        density_estimator="maf",
        calibration_kernel=None,
        use_combined_loss=False,
        z_score_x=True,
        simulation_batch_size: int = 1,
        retrain_from_scratch_each_round=False,
        discard_prior_samples=False,
        summary_writer=None,
        device=None,
        z_score_min_std: float = 1e-7,
        skip_input_checks: bool = False,
    ):
        r"""

        Implementation of __Flexible statistical inference for mechanistic models of
         neural dynamics__ by Lueckmann et al.,
         NeurIPS 2017, https://arxiv.org/abs/1711.01861
        
        Args:
            num_pilot_sims: number of simulations that are run when
                instantiating an object. Used to z-score the data $x$.
            density_estimator: neural density estimator
            calibration_kernel: a function to calibrate the data $x$
            z_score_x: whether to z-score the data features $x$
            use_combined_loss: whether to jointly neural_net prior samples 
                using maximum likelihood. Useful to prevent density leaking when using
                 box uniform priors.
            retrain_from_scratch_each_round: whether to retrain the conditional
                density estimator for the posterior from scratch each round.
            discard_prior_samples: whether to discard prior samples from round
                two onwards.
            z_score_min_std: Minimum value of the standard deviation to use when
                standardizing inputs. This is typically needed when some simulator outputs are deterministic or nearly so.
            skip_simulator_checks: Flag to turn off input checks,
                e.g., for saving simulation budget as the input checks run the
                simulator a couple of times.
        """

        super(SnpeB, self).__init__(
            simulator=simulator,
            prior=prior,
            x_o=x_o,
            num_pilot_sims=num_pilot_sims,
            density_estimator=density_estimator,
            calibration_kernel=calibration_kernel,
            use_combined_loss=use_combined_loss,
            z_score_x=z_score_x,
            simulation_batch_size=simulation_batch_size,
            retrain_from_scratch_each_round=retrain_from_scratch_each_round,
            discard_prior_samples=discard_prior_samples,
            device=device,
            z_score_min_std=z_score_min_std,
            skip_input_checks=skip_input_checks,
        )

    def _get_log_prob_proposal_posterior(
        self, theta: torch.Tensor, x: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Return importance weighted log probability as proposed in
         Lueckmann, Goncalves et al 2017.

        Args:
            theta: batch of parameters $\theta$.
            x: batch of data $x$.
            masks: binary, whether or not to retrain with prior loss on specific prior
                 sample

        Returns: log_prob_proposal_posterior

        """

        batch_size = theta.shape[0]

        # Evaluate posterior
        log_prob_posterior = self._neural_posterior.neural_net.log_prob(theta, x)
        assert torch.isfinite(
            log_prob_posterior
        ).all(), "NaN/inf detected in posterior eval."
        log_prob_posterior = log_prob_posterior.reshape(batch_size)

        # Evaluate prior
        log_prob_prior = self._prior.log_prob(theta)
        log_prob_prior = log_prob_prior.reshape(batch_size)
        assert torch.isfinite(log_prob_prior).all(), "NaN/inf detected in prior eval."

        # evaluate proposal
        log_prob_proposal = self._model_bank[-1].neural_net.log_prob(theta, x)
        assert torch.isfinite(
            log_prob_proposal
        ).all(), "NaN/inf detected in proposal posterior eval."

        # Compute log prob with importance weights
        log_prob = (
            self.calibration_kernel(x)
            * torch.exp(log_prob_prior - log_prob_proposal)
            * log_prob_posterior
        )

        # todo: this implementation is not perfect: it evaluates the posterior
        # todo:     at all prior samples
        if self._use_combined_loss:
            log_prob_posterior_non_atomic = self._neural_posterior.neural_net.log_prob(
                theta, x
            )
            masks = masks.reshape(-1)
            log_prob = masks * log_prob_posterior_non_atomic + log_prob
        return log_prob
