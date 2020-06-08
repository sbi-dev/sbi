from __future__ import annotations
from abc import ABC
from typing import Callable, Dict, Optional, Union
from copy import deepcopy

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from sbi.inference.posterior import NeuralPosterior
from sbi.inference import NeuralInference
from sbi.types import ScalarFloat, OneOrMore
import sbi.utils as utils
from sbi.utils import handle_invalid_x, warn_on_invalid_x


class LikelihoodEstimator(NeuralInference, ABC):
    def __init__(
        self,
        simulator: Callable,
        prior,
        x_shape: Optional[torch.Size] = None,
        density_estimator: Optional[nn.Module] = None,
        simulation_batch_size: Optional[int] = 1,
        retrain_from_scratch_each_round: bool = False,
        summary_writer: Optional[SummaryWriter] = None,
        device: Optional[torch.device] = None,
        num_workers: int = 1,
        mcmc_method: str = "slice_np",
        skip_input_checks: bool = False,
        show_progressbar: bool = True,
        show_round_summary: bool = False,
        logging_level: Union[int, str] = "warning",
        exclude_invalid_x: bool = False,
    ):
        r"""Sequential Neural Likelihood.

        Sequential Neural Likelihood: Fast Likelihood-free Inference with
        Autoregressive Flows_ by Papamakarios et al., AISTATS 2019,
        https://arxiv.org/abs/1805.07226

        Args:
            density_estimator: Conditional density estimator $q(x|\theta)$, a nn.Module
                with `.log_prob()` and `.sample()`

        See docstring of `NeuralInference` class for all other arguments.
        """

        super().__init__(
            simulator=simulator,
            prior=prior,
            x_shape=x_shape,
            simulation_batch_size=simulation_batch_size,
            retrain_from_scratch_each_round=retrain_from_scratch_each_round,
            device=device,
            summary_writer=summary_writer,
            num_workers=num_workers,
            skip_input_checks=skip_input_checks,
            show_progressbar=show_progressbar,
            show_round_summary=show_round_summary,
            logging_level=logging_level,
            exclude_invalid_x=exclude_invalid_x,
        )

        if density_estimator is None:
            density_estimator = utils.likelihood_nn(
                model="maf",
                theta_shape=self._prior.sample().shape,
                x_o_shape=self._x_shape,
            )

        # Create neural posterior which can sample().
        # TODO Notice use of `snle_a` used, so long as it is the sole descendant.´
        self._posterior = NeuralPosterior(
            algorithm_family="snle_a",
            neural_net=density_estimator,
            prior=self._prior,
            mcmc_method=mcmc_method,
            get_potential_function=PotentialFunctionProvider(),
            x_shape=self._x_shape,
        )

        self._posterior.net.train(True)

        # SNLE-specific summary_writer fields.
        self._summary.update({"mcmc_times": []})  # type: ignore

    def __call__(
        self,
        num_rounds: int,
        num_simulations_per_round: OneOrMore[int],
        x_o: Optional[Tensor] = None,
        batch_size: int = 100,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
    ) -> NeuralPosterior:
        r"""Run SNLE.

        This runs SNLE for `num_rounds` rounds, using num_simulations_per_round calls to
        the simulator.

        Args:
            num_rounds: Number of rounds to run.
            num_simulations_per_round: Number of simulator calls per round.
            x_o: When SNLE is used in a multi-round setting, training samples for every
                round after the first round are drawn from the most recent posterior
                estimate given $x_o$, i.e. $p(\theta|x_o)$.
            batch_size: Size of batch to use for training.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If `max_num_epochs`
                is reached, we stop training even if the validation loss is still
                decreasing. If None, we train until validation loss increases (see
                argument `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.

        Returns:
            Posterior $p(\theta|x_o)$ that can be sampled and evaluated.
        """

        self._handle_x_o_wrt_amortization(x_o, num_rounds)

        # If we're retraining from scratch each round,
        # keep a copy of the original untrained model for reinitialization.
        if self._retrain_from_scratch_each_round:
            self._untrained_posterior = deepcopy(self._posterior)

        max_num_epochs = 2 ** 31 - 1 if max_num_epochs is None else max_num_epochs

        #
        num_sims_per_round = self._ensure_list(num_simulations_per_round, num_rounds)

        for round_, num_sims in enumerate(num_sims_per_round):

            # Generate parameters theta from prior in first round, and from most recent
            # posterior estimate in subsequent rounds.
            if round_ == 0:
                theta = self._prior.sample((num_sims,))
            else:
                theta = self._posterior.sample(
                    num_sims, show_progressbar=self._show_progressbar
                )

            x = self._batched_simulator(theta)

            # Check for NaNs in simulations.
            is_valid_x, num_nans, num_infs = handle_invalid_x(x, self.exclude_invalid_x)
            warn_on_invalid_x(num_nans, num_infs, self.exclude_invalid_x)

            # Store (theta, x) pairs.
            self._theta_bank.append(theta[is_valid_x])
            self._x_bank.append(x[is_valid_x])

            # Fit neural likelihood to newly aggregated dataset.
            self._train(
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_fraction=validation_fraction,
                stop_after_epochs=stop_after_epochs,
                max_num_epochs=max_num_epochs,
                clip_max_norm=clip_max_norm,
            )

            # Update description for progress bar.
            if self._show_round_summary:
                print(self._describe_round(round_, self._summary))

            # Update TensorBoard and summary dict.
            self._summarize(
                round_=round_,
                x_o=self._posterior.x_o,
                theta_bank=self._theta_bank,
                x_bank=self._x_bank,
            )

        self._posterior._num_trained_rounds = num_rounds
        return self._posterior

    def _train(
        self,
        batch_size: int,
        learning_rate: float,
        validation_fraction: float,
        stop_after_epochs: int,
        max_num_epochs: int,
        clip_max_norm: Optional[float],
    ) -> None:
        r"""
        Train the conditional density estimator for the likelihood.

        Update the conditional density estimator weights to maximize the
        likelihood on the most recently aggregated bank of $(\theta, x)$ pairs.

        Uses performance on a held-out validation set as a terminating condition (early
        stopping).
        """

        # Get total number of training examples.
        num_examples = sum(len(theta) for theta in self._theta_bank)

        # Select random train and validation splits from (theta, x) pairs.
        permuted_indices = torch.randperm(num_examples)
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples
        train_indices, val_indices = (
            permuted_indices[:num_training_examples],
            permuted_indices[num_training_examples:],
        )

        # Dataset is shared for training and validation loaders.
        dataset = data.TensorDataset(
            torch.cat(self._x_bank), torch.cat(self._theta_bank)
        )

        # Create neural net and validation loaders using a subset sampler.
        train_loader = data.DataLoader(
            dataset,
            batch_size=min(batch_size, num_training_examples),
            drop_last=True,
            sampler=SubsetRandomSampler(train_indices),
        )
        val_loader = data.DataLoader(
            dataset,
            batch_size=min(batch_size, num_validation_examples),
            shuffle=False,
            drop_last=False,
            sampler=SubsetRandomSampler(val_indices),
        )

        optimizer = optim.Adam(self._posterior.net.parameters(), lr=learning_rate)

        # If we're retraining from scratch each round, reset the neural posterior
        # to the untrained copy we made at the start.
        if self._retrain_from_scratch_each_round:
            self._posterior = deepcopy(self._untrained_posterior)
            optimizer = optim.Adam(self._posterior.net.parameters(), lr=learning_rate)

        epoch, self._val_log_prob = 0, float("-Inf")
        while epoch <= max_num_epochs and not self._converged(epoch, stop_after_epochs):

            # Train for a single epoch.
            self._posterior.net.train()
            for batch in train_loader:
                optimizer.zero_grad()
                theta_batch, x_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                )
                log_prob = self._posterior.net.log_prob(theta_batch, context=x_batch)
                loss = -torch.mean(log_prob)
                loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._posterior.net.parameters(), max_norm=clip_max_norm,
                    )
                optimizer.step()

            epoch += 1

            # Calculate validation performance.
            self._posterior.net.eval()
            log_prob_sum = 0
            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                    )
                    log_prob = self._posterior.net.log_prob(
                        theta_batch, context=x_batch
                    )
                    log_prob_sum += log_prob.sum().item()
            self._val_log_prob = log_prob_sum / num_validation_examples

            self._maybe_show_progress(self._show_progressbar, epoch)

        self._report_convergence_at_end(epoch, stop_after_epochs, max_num_epochs)

        # Update summary.
        self._summary["epochs"].append(epoch)
        self._summary["best_validation_log_probs"].append(self._best_val_log_prob)


class PotentialFunctionProvider:
    """
    This class is initialized without arguments during the initialization of the
     Posterior class. When called, it specializes to the potential function appropriate
     to the requested mcmc_method.

    NOTE: Why use a class?
    ----------------------
    During inference, we use deepcopy to save untrained posteriors in memory. deepcopy
    uses pickle which can't serialize nested functions
    (https://stackoverflow.com/a/12022055).

    It is important to NOT initialize attributes upon instantiation, because we need the
    most current trained posterior neural net.

    Returns:
        Potential function for use by either numpy or pyro sampler.
    """

    def __call__(
        self, prior, likelihood_nn: nn.Module, x: Tensor, mcmc_method: str,
    ) -> Callable:
        r"""Return potential function for posterior $p(\theta|x)$.

        Switch on numpy or pyro potential function based on mcmc_method.

        Args:
            prior: Prior distribution that can be evaluated.
            likelihood_nn: Neural likelihood estimator that can be evaluated.
            x: Conditioning variable for posterior $p(\theta|x)$.
            mcmc_method: One of `slice_np`, `slice`, `hmc` or `nuts`.

        Returns:
            Potential function for sampler.
        """
        self.likelihood_nn = likelihood_nn
        self.prior = prior
        self.x = x

        if mcmc_method in ("slice", "hmc", "nuts"):
            return self.pyro_potential
        else:
            return self.np_potential

    def np_potential(self, theta: np.array) -> ScalarFloat:
        r"""Return posterior log prob. of theta $p(\theta|x)$"

        Args:
            theta: Parameters $\theta$, batch dimension 1.

        Returns:
            Posterior log probability of the theta, $-\infty$ if impossible under prior.
        """
        theta = torch.as_tensor(theta, dtype=torch.float32)
        log_likelihood = self.likelihood_nn.log_prob(
            inputs=self.x.reshape(1, -1), context=theta.reshape(1, -1)
        )

        # Notice opposite sign to pyro potential.
        return log_likelihood + self.prior.log_prob(theta)

    def pyro_potential(self, theta: Dict[str, Tensor]) -> Tensor:
        r"""Return posterior log probability of parameters $p(\theta|x)$.

         Args:
            theta: Parameters $\theta$. The tensor's shape will be
                (1, shape_of_single_theta) if running a single chain or just
                (shape_of_single_theta) for multiple chains.

        Returns:
            The potential $-[\log r(x_o, \theta) + \log p(\theta)]$.
        """

        theta = next(iter(theta.values()))

        log_likelihood = self.likelihood_nn.log_prob(
            inputs=self.x.reshape(1, -1), context=theta.reshape(1, -1)
        )

        return -(log_likelihood + self.prior.log_prob(theta))
