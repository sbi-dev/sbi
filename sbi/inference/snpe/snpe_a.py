from __future__ import annotations

import os

import torch
from torch import distributions
from torch.utils.tensorboard import SummaryWriter

import sbi.utils as utils
from sbi.inference.snpe.snpe_base import SnpeBase


class SnpeA(SnpeBase):
    def __init__(
        self,
        simulator,
        prior,
        x_o,
        num_pilot_samples=100,
        density_estimator="maf",
        use_combined_loss=False,
        z_score_x=True,
        simulation_batch_size: int = 1,
        retrain_from_scratch_each_round=False,
        discard_prior_samples=False,
        summary_writer=None,
        device=None,
    ):
        """SNPE-A

        Implementation of _Fast epsilon-free Inference of Simulation Models with Bayesian Conditional Density Estimation_ by Papamakarios et al., NeurIPS 2016, 
        https://arxiv.org/abs/1605.06376
        
        Args:
            num_pilot_samples: number of simulations that are run when
                instantiating an object. Used to z-score the data x.
            density_estimator: neural density estimator
            calibration_kernel: a function to calibrate the data x
            z_score_x: whether to z-score the data features x
            use_combined_loss: whether to jointly neural_net prior samples 
                using maximum likelihood. Useful to prevent density leaking when using box uniform priors.
            retrain_from_scratch_each_round: whether to retrain the conditional
                density estimator for the posterior from scratch each round.
            discard_prior_samples: whether to discard prior samples from round
                two onwards.
        """

        raise NotImplementedError

        super(SnpeA, self).__init__(
            simulator=simulator,
            prior=prior,
            x_o=x_o,
            num_pilot_samples=num_pilot_samples,
            density_estimator=density_estimator,
            use_combined_loss=use_combined_loss,
            z_score_x=z_score_x,
            simulation_batch_size=simulation_batch_size,
            retrain_from_scratch_each_round=retrain_from_scratch_each_round,
            discard_prior_samples=discard_prior_samples,
            device=device,
        )

    def _get_log_prob_proposal_posterior(self, theta, x, masks):
        """
        We have two main options when evaluating the proposal posterior.
        (1) Generate atoms from the proposal prior.
        (2) Generate atoms from a more targeted distribution,
        such as the most recent posterior.
        If we choose the latter, it is likely beneficial not to do this in the first
        round, since we would be sampling from a randomly initialized neural density
        estimator.

        Args:
            theta: torch.Tensor Batch of parameters theta.
            x: torch.Tensor Batch of data x.
            masks: torch.Tensor
                binary, whether or not to retrain with prior loss on specific prior sample

        Returns: torch.Tensor [1] log_prob_proposal_posterior

        """

        log_prob_posterior_non_atomic = self._neural_posterior.log_prob(theta, x)

        batch_size = theta.shape[0]

        num_atoms = self._num_atoms if self._num_atoms > 0 else batch_size

        # Each set of theta atoms is evaluated using the same x,
        # so we repeat rows of the x.
        # e.g. [1, 2] -> [1, 1, 2, 2]
        repeated_x = utils.repeat_rows(x, num_atoms)

        # To generate the full set of atoms for a given item in the batch,
        # we sample without replacement num_atoms - 1 times from the rest
        # of the thetas in the batch.
        assert 0 < num_atoms - 1 < batch_size
        probs = (
            (1 / (batch_size - 1))
            * torch.ones(batch_size, batch_size)
            * (1 - torch.eye(batch_size))
        )
        choices = torch.multinomial(probs, num_samples=num_atoms - 1, replacement=False)
        contrasting_theta = theta[choices]

        # We can now create our sets of atoms from the contrasting thetas
        # we have generated.
        atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(
            batch_size * num_atoms, -1
        )

        # Evaluate large batch giving (batch_size * num_atoms) log prob posterior evals.
        log_prob_posterior = self._neural_posterior.log_prob(atomic_theta, repeated_x)
        assert torch.isfinite(
            log_prob_posterior
        ).all(), "NaN/inf detected in posterior eval."
        log_prob_posterior = log_prob_posterior.reshape(batch_size, num_atoms)

        # Get (batch_size * num_atoms) log prob prior evals.
        log_prob_prior = self._prior.log_prob(atomic_theta)
        log_prob_prior = log_prob_prior.reshape(batch_size, num_atoms)
        assert torch.isfinite(log_prob_prior).all(), "NaN/inf detected in prior eval."

        # Compute unnormalized proposal posterior.
        unnormalized_log_prob_proposal_posterior = log_prob_posterior - log_prob_prior

        # Normalize proposal posterior across discrete set of atoms.
        log_prob_proposal_posterior = unnormalized_log_prob_proposal_posterior[
            :, 0
        ] - torch.logsumexp(unnormalized_log_prob_proposal_posterior, dim=-1)
        assert torch.isfinite(
            log_prob_proposal_posterior
        ).all(), "NaN/inf detected in proposal posterior eval."

        if self._use_combined_loss:
            masks = masks.reshape(-1)

            log_prob_proposal_posterior = (
                masks * log_prob_posterior_non_atomic + log_prob_proposal_posterior
            )
