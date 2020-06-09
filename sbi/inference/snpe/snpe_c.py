from __future__ import annotations
from typing import Callable, Optional, Union

import torch
from torch import Tensor, eye, nn, ones
from torch.utils.tensorboard import SummaryWriter

from sbi.inference.posterior import NeuralPosterior
from sbi.inference.snpe.snpe_base import PosteriorEstimator
from sbi.types import OneOrMore
from sbi.utils import clamp_and_warn, del_entries, repeat_rows
from sbi.utils.torchutils import get_default_device


class SNPE_C(PosteriorEstimator):
    def __init__(
        self,
        prior,
        simulator: Callable,
        x_shape: Optional[torch.Size] = None,
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        density_estimator: Union[str, nn.Module] = "maf",
        z_score_x: bool = True,
        z_score_min_std: float = 1e-7,
        sample_with_mcmc: bool = False,
        mcmc_method: str = "slice_np",
        calibration_kernel: Optional[Callable] = None,
        use_combined_loss: bool = False,
        retrain_from_scratch_each_round: bool = False,
        discard_prior_samples: bool = False,
        skip_input_checks: bool = False,
        exclude_invalid_x: bool = True,
        device: Union[torch.device, str] = get_default_device(),
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progressbar: bool = True,
        show_round_summary: bool = False,
    ):
        r"""SNPE-C / APT [1].

        [1] _Automatic Posterior Transformation for Likelihood-free Inference_,
            Greenberg et al., ICML 2019, https://arxiv.org/abs/1905.07488.

        Args:
            use_combined_loss: Whether to train the neural net also on prior samples
                using maximum likelihood in addition to training it on all samples using
                atomic loss. The extra MLE loss helps prevent density leaking with
                bounded priors.

        See docstring of `PosteriorEstimator` class for all other arguments.
        """

        self._use_combined_loss = use_combined_loss

        super().__init__(
            prior=prior,
            simulator=simulator,
            x_shape=x_shape,
            num_workers=num_workers,
            simulation_batch_size=simulation_batch_size,
            density_estimator=density_estimator,
            z_score_x=z_score_x,
            z_score_min_std=z_score_min_std,
            sample_with_mcmc=sample_with_mcmc,
            mcmc_method=mcmc_method,
            calibration_kernel=calibration_kernel,
            retrain_from_scratch_each_round=retrain_from_scratch_each_round,
            discard_prior_samples=discard_prior_samples,
            skip_input_checks=skip_input_checks,
            exclude_invalid_x=exclude_invalid_x,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progressbar=show_progressbar,
            show_round_summary=show_round_summary,
        )

    def __call__(
        self,
        num_rounds: int,
        num_simulations_per_round: OneOrMore[int],
        x_o: Optional[Tensor] = None,
        num_atoms: int = 10,
        batch_size: int = 100,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
    ) -> NeuralPosterior:
        """
        Args:
            num_atoms: Number of atoms to use for classification.
        Returns:
        """

        # WARNING: sneaky trick ahead. We proxy the parent's `__call__` here,
        # requiring the signature to have `num_atoms`, save it for use below, and
        # continue. It's sneaky because we are using the object (self) as a namespace
        # to pass arguments between functions, and that's implicit state management.

        self._num_atoms = num_atoms
        kwargs = del_entries(locals(), entries=("self", "__class__", "num_atoms"))

        return super().__call__(**kwargs)

    def _log_prob_proposal_posterior(
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
            masks: Mask that is True for prior samples in the batch in order to train
                them with prior loss.

        Returns:
            Log-probability of the proposal posterior.
        """

        batch_size = theta.shape[0]

        num_atoms = clamp_and_warn(
            "num_atoms", self._num_atoms, min_val=2, max_val=batch_size
        )

        # Each set of parameter atoms is evaluated using the same x,
        # so we repeat rows of the data x, e.g. [1, 2] -> [1, 1, 2, 2]
        repeated_x = repeat_rows(x, num_atoms)

        # To generate the full set of atoms for a given item in the batch,
        # we sample without replacement num_atoms - 1 times from the rest
        # of the theta in the batch.
        probs = ones(batch_size, batch_size) * (1 - eye(batch_size)) / (batch_size - 1)

        choices = torch.multinomial(probs, num_samples=num_atoms - 1, replacement=False)
        contrasting_theta = theta[choices]

        # We can now create our sets of atoms from the contrasting parameter sets
        # we have generated.
        atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(
            batch_size * num_atoms, -1
        )

        # Evaluate large batch giving (batch_size * num_atoms) log prob posterior evals.
        log_prob_posterior = self._posterior.net.log_prob(atomic_theta, repeated_x)
        self._assert_all_finite(log_prob_posterior, "posterior eval")
        log_prob_posterior = log_prob_posterior.reshape(batch_size, num_atoms)

        # Get (batch_size * num_atoms) log prob prior evals.
        log_prob_prior = self._prior.log_prob(atomic_theta)
        log_prob_prior = log_prob_prior.reshape(batch_size, num_atoms)
        self._assert_all_finite(log_prob_prior, "prior eval")

        # Compute unnormalized proposal posterior.
        unnormalized_log_prob = log_prob_posterior - log_prob_prior

        # Normalize proposal posterior across discrete set of atoms.
        log_prob_proposal_posterior = unnormalized_log_prob[:, 0] - torch.logsumexp(
            unnormalized_log_prob, dim=-1
        )
        self._assert_all_finite(log_prob_proposal_posterior, "proposal posterior eval")

        # XXX This evaluates the posterior on _all_ prior samples
        if self._use_combined_loss:
            log_prob_posterior_non_atomic = self._posterior.net.log_prob(theta, x)
            masks = masks.reshape(-1)
            log_prob_proposal_posterior = (
                masks * log_prob_posterior_non_atomic + log_prob_proposal_posterior
            )

        return log_prob_proposal_posterior

    def _log_prob_proposal_MoG(self, theta: Tensor, x: Tensor, masks: Tensor) -> Tensor:
        raise NotImplementedError
