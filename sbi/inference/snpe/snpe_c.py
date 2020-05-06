from __future__ import annotations
from typing import Callable, Optional

import torch
from torch import Tensor, eye, nn, ones
from torch.utils.tensorboard import SummaryWriter

from sbi.inference.snpe.snpe_base import SnpeBase
import sbi.utils as utils


class SnpeC(SnpeBase):
    def __init__(
        self,
        simulator: Callable,
        prior,
        x_o: Tensor,
        num_atoms: Optional[int] = None,
        density_estimator: Optional[nn.Module] = None,
        calibration_kernel: Optional[Callable] = None,
        use_combined_loss: bool = False,
        z_score_x: bool = True,
        z_score_min_std: float = 1e-7,
        simulation_batch_size: Optional[int] = 1,
        retrain_from_scratch_each_round: bool = False,
        discard_prior_samples: bool = False,
        summary_writer: Optional[SummaryWriter] = None,
        device: Optional[torch.device] = None,
        sample_with_mcmc: bool = False,
        mcmc_method: str = "slice-np",
        skip_input_checks: bool = False,
    ):
        r"""SNPE-C / APT [1]

        [1] _Automatic Posterior Transformation for Likelihood-free Inference_,
            Greenberg et al., ICML 2019, https://arxiv.org/abs/1905.07488.

        Args:
            density_estimator: Neural density estimator.
            calibration_kernel: A function to calibrate the data $x$.
            z_score_x: Whether to z-score the data features $x$.
            z_score_min_std: Minimum value of the standard deviation to use when
                standardizing inputs. This is typically needed when some simulator outputs are deterministic or nearly so.
            use_combined_loss: Whether to train the neural_net jointly on prior samples 
                using maximum likelihood and on all samples using atomic loss. Useful to prevent density leaking when using bounded priors.
            retrain_from_scratch_each_round: Whether to retrain the conditional
                density estimator for the posterior from scratch each round.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less specific samples.
            num_atoms: Number of atoms to use for classification. If None, use all
                other parameters $\theta$ in minibatch.
            skip_input_checks: Whether to turn off input checks. This saves
                simulation time because the input checks test-run the simulator to ensure it's correct.
        """

        self._num_atoms = num_atoms if num_atoms is not None else 0
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
            device=device,
            sample_with_mcmc=sample_with_mcmc,
            mcmc_method=mcmc_method,
            skip_input_checks=skip_input_checks,
        )

    def _get_log_prob_proposal_posterior(
        self, theta: Tensor, x: Tensor, masks: Tensor
    ) -> Tensor:
        """
        Return log probability of the proposal posterior.

        We have two main options when evaluating the proposal posterior.
            (1) Generate atoms from the proposal prior.
            (2) Generate atoms from a more targeted distribution, such as the most
                recent posterior.
        If we choose the latter, it is likely beneficial not to do this in the first
        round, since we would be sampling from a randomly-initialized neural density
        estimator.

        Args:
            theta: Batch of parameters θ.
            x: Batch of data.
            masks: Whetherto retrain with prior loss (for each prior sample).

        Returns:
            Log-probability of the proposal posterior. 
        """

        batch_size = theta.shape[0]

        num_atoms = self._num_atoms if self._num_atoms > 0 else batch_size

        # Each set of parameter atoms is evaluated using the same x,
        # so we repeat rows of the data x, e.g. [1, 2] -> [1, 1, 2, 2]
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
        self._assert_all_finite(log_prob_posterior, "posterior eval")
        log_prob_posterior = log_prob_posterior.reshape(batch_size, num_atoms)

        # Get (batch_size * num_atoms) log prob prior evals.
        log_prob_prior = self._prior.log_prob(atomic_theta)
        log_prob_prior = log_prob_prior.reshape(batch_size, num_atoms)
        self._assert_all_finite(log_prob_prior, "prior eval")

        # Compute unnormalized proposal posterior.
        unnormalized_log_prob_proposal_posterior = log_prob_posterior - log_prob_prior

        # Normalize proposal posterior across discrete set of atoms.
        log_prob_proposal_posterior = self.calibration_kernel(
            x
        ) * unnormalized_log_prob_proposal_posterior[:, 0] - torch.logsumexp(
            unnormalized_log_prob_proposal_posterior, dim=-1
        )
        self._assert_all_finite(log_prob_proposal_posterior, "proposal posterior eval")

        # XXX This evaluates the posterior on _all_ prior samples
        if self._use_combined_loss:
            log_prob_posterior_non_atomic = self._neural_posterior.neural_net.log_prob(
                theta, x
            )
            masks = masks.reshape(-1)
            log_prob_proposal_posterior = (
                masks * log_prob_posterior_non_atomic + log_prob_proposal_posterior
            )

        return log_prob_proposal_posterior

    def _get_log_prob_proposal_MoG(
        self, theta: Tensor, x: Tensor, masks: Tensor
    ) -> Tensor:

        raise NotImplementedError
