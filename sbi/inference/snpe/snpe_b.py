import os

import torch
from torch import distributions
from torch.utils.tensorboard import SummaryWriter

import sbi.utils as utils
from sbi.inference.snpe.snpe_base import SnpeBase


class SnpeB(SnpeBase):
    """
    Implementation of
    'Flexible statistical inference for mechanistic
        models of neural dynamics'
    Lueckmann et al.
    NeurIPS 2017
    https://arxiv.org/abs/1711.01861
    """

    def __init__(
        self,
        simulator,
        prior,
        true_observation,
        num_pilot_samples=100,
        density_estimator="maf",
        calibration_kernel=None,
        use_combined_loss=False,
        z_score_obs=True,
        simulation_batch_size: int = 1,
        retrain_from_scratch_each_round=False,
        discard_prior_samples=False,
        summary_writer=None,
        device=None,
    ):
        """
        See snpe_base.SnpeBase for docstring.
        """

        super(SnpeB, self).__init__(
            simulator=simulator,
            prior=prior,
            true_observation=true_observation,
            num_pilot_samples=num_pilot_samples,
            density_estimator=density_estimator,
            calibration_kernel=calibration_kernel,
            use_combined_loss=use_combined_loss,
            z_score_obs=z_score_obs,
            simulation_batch_size=simulation_batch_size,
            retrain_from_scratch_each_round=retrain_from_scratch_each_round,
            discard_prior_samples=discard_prior_samples,
            device=device,
        )

    def _get_log_prob_proposal_posterior(
        self, inputs: torch.Tensor, context: torch.Tensor, masks: torch.Tensor
    ):
        """
        XXX: Improve docstring here, it is not clear what log_prob refers to. isnt this the snpeB "loss"?
        Return log prob under proposal posterior.
        
        We have two main options when evaluating the proposal posterior.
        (1) Generate atoms from the proposal prior.
        (2) Generate atoms from a more targeted distribution,
        such as the most recent posterior.
        If we choose the latter, it is likely beneficial not to do this in the first
        round, since we would be sampling from a randomly initialized neural density
        estimator.

        Args:
            inputs: torch.Tensor Batch of parameters.
            context: torch.Tensor Batch of observations.
            masks: torch.Tensor
                binary, whether or not to retrain with prior loss on specific prior sample

        Returns: torch.Tensor [1] log_prob_proposal_posterior

        """

        batch_size = inputs.shape[0]

        # Evaluate posterior
        log_prob_posterior = self._neural_posterior.log_prob(
            inputs, context, normalize_snpe_density=False
        )
        assert torch.isfinite(
            log_prob_posterior
        ).all(), "NaN/inf detected in posterior eval."
        log_prob_posterior = log_prob_posterior.reshape(batch_size)

        # Evaluate prior
        log_prob_prior = self._prior.log_prob(inputs)
        log_prob_prior = log_prob_prior.reshape(batch_size)
        assert torch.isfinite(log_prob_prior).all(), "NaN/inf detected in prior eval."

        # evaluate proposal
        log_prob_proposal = self._model_bank[-1].log_prob(
            inputs, context, normalize_snpe_density=False
        )
        assert torch.isfinite(
            log_prob_proposal
        ).all(), "NaN/inf detected in proposal posterior eval."

        # Compute log prob with importance weights
        log_prob = (
            self.calibration_kernel(context)
            * torch.exp(log_prob_prior - log_prob_proposal)
            * log_prob_posterior
        )

        # todo: this implementation is not perfect: it evaluates the posterior
        # todo:     at all prior samples
        if self._use_combined_loss:
            log_prob_posterior_non_atomic = self._neural_posterior.log_prob(
                inputs, context, normalize_snpe_density=False
            )
            masks = masks.reshape(-1)
            log_prob = masks * log_prob_posterior_non_atomic + log_prob
        return log_prob
