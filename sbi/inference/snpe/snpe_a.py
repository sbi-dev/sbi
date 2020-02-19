import sbi.utils as utils
from sbi.inference.snpe.base_snpe import base_snpe
import torch
import os
from torch import distributions
from torch.utils.tensorboard import SummaryWriter


class SNPE_A(base_snpe):
    """
    Implementation of
    'Fast epsilon-free Inference of Simulation Models
        with Bayesian Conditional Density Estimation'
    Papamakarios et al.
    NeurIPS 2016
    https://arxiv.org/abs/1605.06376
    """

    def __init__(
        self,
        simulator,
        prior,
        true_observation,
        num_pilot_samples=100,
        density_estimator='maf',
        use_combined_loss=False,
        z_score_obs=True,
        retrain_from_scratch_each_round=False,
        discard_prior_samples=False,
        summary_writer=None,
        device=None,
    ):
        """
        See base_snpe for docstring.

        Args:
            num_atoms: int
                Number of atoms to use for classification.
                If -1, use all other parameters in minibatch.
        """

        super(SNPE_A, self).__init__(simulator=simulator,
                                      prior=prior,
                                      true_observation=true_observation,
                                      num_pilot_samples=num_pilot_samples,
                                      density_estimator=density_estimator,
                                      use_combined_loss=use_combined_loss,
                                      z_score_obs=z_score_obs,
                                      retrain_from_scratch_each_round=retrain_from_scratch_each_round,
                                      discard_prior_samples=discard_prior_samples,
                                      device=device,
                                      )

        # Each APT run has an associated log directory for TensorBoard output.
        if summary_writer is None:
            log_dir = os.path.join(
                utils.get_log_root(), "snpe-a", simulator.name, utils.get_timestamp()
            )
            self._summary_writer = SummaryWriter(log_dir)
        else:
            self._summary_writer = summary_writer

        raise NameError('Not implemented yet')



    def _get_log_prob_proposal_posterior(self, inputs, context, masks):
        """
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

        log_prob_posterior_non_atomic = self._neural_posterior.log_prob(
            inputs, context
        )

        batch_size = inputs.shape[0]

        num_atoms = self._num_atoms if self._num_atoms > 0 else batch_size

        # Each set of parameter atoms is evaluated using the same observation,
        # so we repeat rows of the context.
        # e.g. [1, 2] -> [1, 1, 2, 2]
        repeated_context = utils.repeat_rows(context, num_atoms)

        # To generate the full set of atoms for a given item in the batch,
        # we sample without replacement num_atoms - 1 times from the rest
        # of the parameters in the batch.
        assert 0 < num_atoms - 1 < batch_size
        probs = (
                (1 / (batch_size - 1))
                * torch.ones(batch_size, batch_size)
                * (1 - torch.eye(batch_size))
        )
        choices = torch.multinomial(
            probs, num_samples=num_atoms - 1, replacement=False
        )
        contrasting_inputs = inputs[choices]

        # We can now create our sets of atoms from the contrasting parameter sets
        # we have generated.
        atomic_inputs = torch.cat(
            (inputs[:, None, :], contrasting_inputs), dim=1
        ).reshape(batch_size * num_atoms, -1)

        # Evaluate large batch giving (batch_size * num_atoms) log prob posterior evals.
        log_prob_posterior = self._neural_posterior.log_prob(
            atomic_inputs, repeated_context
        )
        assert utils.notinfnotnan(
            log_prob_posterior
        ), "NaN/inf detected in posterior eval."
        log_prob_posterior = log_prob_posterior.reshape(batch_size, num_atoms)

        # Get (batch_size * num_atoms) log prob prior evals.
        if isinstance(self._prior, distributions.Uniform):
            log_prob_prior = self._prior.log_prob(atomic_inputs).sum(-1)
            # log_prob_prior = torch.zeros(log_prob_prior.shape)
        else:
            log_prob_prior = self._prior.log_prob(atomic_inputs)
        log_prob_prior = log_prob_prior.reshape(batch_size, num_atoms)
        assert utils.notinfnotnan(log_prob_prior), "NaN/inf detected in prior eval."

        # Compute unnormalized proposal posterior.
        unnormalized_log_prob_proposal_posterior = (
                log_prob_posterior - log_prob_prior
        )

        # Normalize proposal posterior across discrete set of atoms.
        log_prob_proposal_posterior = unnormalized_log_prob_proposal_posterior[
                                      :, 0
                                      ] - torch.logsumexp(unnormalized_log_prob_proposal_posterior, dim=-1)
        assert utils.notinfnotnan(
            log_prob_proposal_posterior
        ), "NaN/inf detected in proposal posterior eval."

        if self._use_combined_loss:
            masks = masks.reshape(-1)

            log_prob_proposal_posterior = (
                    masks * log_prob_posterior_non_atomic + log_prob_proposal_posterior
            )

        return log_prob_proposal_posterior

