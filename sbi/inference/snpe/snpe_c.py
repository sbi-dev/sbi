from __future__ import annotations

import os

import torch

import sbi.utils as utils
from sbi.inference.snpe.snpe_base import SnpeBase
from torch import Tensor, isfinite, ones, eye


class SnpeC(SnpeBase):
    def __init__(
        self,
        simulator,
        prior,
        num_atoms: Optional[int] = None,
        z_score_min_std: float = 1e-7,
        skip_input_checks: bool = False,
    ):
        r"""SNPE-C / APT

        Implementation of _Automatic Posterior Transformation for Likelihood-free
        Inference_ by Greenberg et al., ICML 2019, https://arxiv.org/abs/1905.07488

        Args:
            num_pilot_sims: number of simulations that are run when
                instantiating an object. Used to z-score the observations.   
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
            num_atoms: int
                Number of atoms to use for classification.
                If -1, use all other thetas in minibatch.
            z_score_min_std: Minimum value of the standard deviation to use when
                standardizing inputs. This is typically needed when some simulator outputs are deterministic or nearly so.
            num_atoms: Number of atoms to use for classification. If None, use all
                other parameters $\theta$ in minibatch.
        """

        self._num_atoms = num_atoms if num_atoms is not None else 0
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
            sample_with_mcmc=sample_with_mcmc,
            mcmc_method=mcmc_method,
            z_score_min_std=z_score_min_std,
            skip_input_checks=skip_input_checks,
        )

        assert isinstance(num_atoms, int), "Number of atoms must be an integer."
        self._num_atoms = num_atoms

    def _get_log_prob_proposal_posterior(
        self, theta: Tensor, x: Tensor, masks: Tensor
    ) -> Tensor:
        r"""
        We have two main options when evaluating the proposal posterior.
        (1) Generate atoms from the proposal prior.
        (2) Generate atoms from a more targeted distribution,
        such as the most recent posterior.
        If we choose the latter, it is likely beneficial not to do this in the first
        round, since we would be sampling from a randomly initialized neural density
        estimator.

        Args:
            theta: batch of parameters $\theta$.
            x: batch of data $x$.
            masks: binary, whether or not to retrain with prior loss on specific prior
             sample

        Returns: log_prob_proposal_posterior

        """

        batch_size = theta.shape[0]

        num_atoms = self._num_atoms if self._num_atoms > 0 else batch_size

        # Each set of parameter atoms is evaluated using the same x,
        # so we repeat rows of the data x.
        # e.g. [1, 2] -> [1, 1, 2, 2]
        repeated_x = utils.repeat_rows(x, num_atoms)

        # To generate the full set of atoms for a given item in the batch,
        # we sample without replacement num_atoms - 1 times from the rest
        # of the theta in the batch.
        assert 0 < num_atoms - 1 < batch_size
        probs = (
            (1 / (batch_size - 1))
            * ones(batch_size, batch_size)
            * (1 - eye(batch_size))
        )
        choices = torch.multinomial(probs, num_samples=num_atoms - 1, replacement=False)
        contrasting_theta = theta[choices]

        # We can now create our sets of atoms from the contrasting parameter sets
        # we have generated.
        atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(
            batch_size * num_atoms, -1
        )

        # Evaluate large batch giving (batch_size * num_atoms) log prob posterior evals.
        log_prob_posterior = self._neural_posterior.neural_net.log_prob(
            atomic_theta, repeated_x
        )
        assert isfinite(log_prob_posterior).all(), "NaN/inf detected in posterior eval."
        log_prob_posterior = log_prob_posterior.reshape(batch_size, num_atoms)

        # Get (batch_size * num_atoms) log prob prior evals.
        log_prob_prior = self._prior.log_prob(atomic_theta)
        log_prob_prior = log_prob_prior.reshape(batch_size, num_atoms)
        assert isfinite(log_prob_prior).all(), "NaN/inf detected in prior eval."

        # Compute unnormalized proposal posterior.
        unnormalized_log_prob_proposal_posterior = log_prob_posterior - log_prob_prior

        # Normalize proposal posterior across discrete set of atoms.
        log_prob_proposal_posterior = self.calibration_kernel(
            x
        ) * unnormalized_log_prob_proposal_posterior[:, 0] - torch.logsumexp(
            unnormalized_log_prob_proposal_posterior, dim=-1
        )
        assert isfinite(
            log_prob_proposal_posterior
        ).all(), "NaN/inf detected in proposal posterior eval."

        # todo: this implementation is not perfect: it evaluates the posterior
        # todo: at all prior samples
        if self._use_combined_loss:
            log_prob_posterior_non_atomic = self._neural_posterior.neural_net.log_prob(
                theta, x
            )
            masks = masks.reshape(-1)
            log_prob_proposal_posterior = (
                masks * log_prob_posterior_non_atomic + log_prob_proposal_posterior
            )

        return log_prob_proposal_posterior

    def _get_log_prob_proposal_MoG(self, theta, x, masks):

        raise NotImplementedError
