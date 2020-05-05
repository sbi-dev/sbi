from __future__ import annotations
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sbi.inference.base import NeuralInference
from sbi.inference.posteriors.sbi_posterior import Posterior
import sbi.utils as utils


class SNL(NeuralInference):
    def __init__(
        self,
        simulator: Callable,
        prior,
        x_o: Tensor,
        density_estimator: Optional[nn.Module] = None,
        simulation_batch_size: int = 1,
        summary_writer: SummaryWriter = None,
        device: torch.device = None,
        mcmc_method: str = "slice-np",
        skip_input_checks: bool = False,
    ):
        r"""Sequential Neural Likelihood
        
        Implementation of Sequential Neural Likelihood: Fast Likelihood-free Inference
         with Autoregressive Flows_ by Papamakarios et al., AISTATS 2019,
         https://arxiv.org/abs/1805.07226

        Args:
            density_estimator: Conditional density estimator $q(x|\theta)$, a nn.Module
                with `.log_prob()` and `.sample()`
            skip_simulator_checks: Flag to turn off input checks,
                e.g., for saving simulation budget as the input checks run the
                simulator a couple of times.
        """

        super().__init__(
            simulator,
            prior,
            x_o,
            simulation_batch_size,
            device,
            summary_writer,
            skip_input_checks=skip_input_checks,
        )

        if density_estimator is None:
            density_estimator = utils.likelihood_nn(
                model="maf", prior=self._prior, x_o=self._x_o,
            )

        # create neural posterior which can sample()
        self._neural_posterior = Posterior(
            algorithm_family="snl",
            neural_net=density_estimator,
            prior=prior,
            x_o=x_o,
            mcmc_method=mcmc_method,
            get_potential_function=PotentialFunctionProvider(),
        )

        # XXX why not density_estimator.train(True)???
        self._neural_posterior.neural_net.train(True)

        # SNL-specific summary_writer fields
        self._summary.update({"mcmc_times": []})

    def __call__(
        self,
        num_rounds: int,
        num_simulations_per_round: Union[List[int], int],
        batch_size: int = 100,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
    ) -> Posterior:
        r"""Run SNL

        This runs SNL for num_rounds rounds, using num_simulations_per_round calls to
        the simulator
        
        Args:
            num_rounds: Number of rounds to run
            num_simulations_per_round: Number of simulator calls per round
            batch_size: Size of batch to use for training.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.

        Returns:
            Posterior $p(\theta|x_o)$ that can be sampled and evaluated
        """
        num_sims_per_round = self._ensure_list(num_simulations_per_round, num_rounds)

        tbar = tqdm(enumerate(num_sims_per_round))
        for round_, num_sims in tbar:

            # Generate parameters theta from prior in first round, and from most recent
            # posterior estimate in subsequent rounds.
            if round_ == 0:
                theta = self._prior.sample((num_sims,))
            else:
                theta = self._neural_posterior.sample(num_sims)

            x = self._batched_simulator(theta)
            # Store (theta, x) pairs.
            self._theta_bank.append(theta)
            self._x_bank.append(x)

            # Fit neural likelihood to newly aggregated dataset.
            self._train(
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_fraction=validation_fraction,
                stop_after_epochs=stop_after_epochs,
            )

            # Update description for progress bar.
            tbar.set_description(self._describe_round(round_, self._summary))

            # Update TensorBoard and summary dict.
            self._summary_writer, self._summary = utils.summarize(
                summary_writer=self._summary_writer,
                summary=self._summary,
                round_=round_,
                x_o=self._x_o,
                theta_bank=self._theta_bank,
                x_bank=self._x_bank,
                simulator=self._simulator,
            )

        self._neural_posterior._num_trained_rounds = num_rounds
        return self._neural_posterior

    def _train(self, batch_size, learning_rate, validation_fraction, stop_after_epochs):
        r"""
        Trains the conditional density estimator for the likelihood by maximum
         likelihood on the most recently aggregated bank of $(\theta, x)$ pairs.
         Uses early stopping on a held-out validation set as a terminating condition.
        """

        # Get total number of training examples.
        num_examples = torch.cat(self._theta_bank).shape[0]

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

        # Create neural_net and validation loaders using a subset sampler.
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

        optimizer = optim.Adam(
            self._neural_posterior.neural_net.parameters(), lr=learning_rate
        )
        # Keep track of best_validation log_prob seen so far.
        best_validation_log_prob = -1e100
        # Keep track of number of epochs since last improvement.
        epochs_since_last_improvement = 0
        # Keep track of model with best validation performance.
        best_model_state_dict = None

        epochs = 0
        while True:

            # Train for a single epoch.
            self._neural_posterior.neural_net.train()
            for batch in train_loader:
                optimizer.zero_grad()
                theta_batch, x_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                )
                log_prob = self._neural_posterior.neural_net.log_prob(
                    theta_batch, context=x_batch
                )
                loss = -torch.mean(log_prob)
                loss.backward()
                clip_grad_norm_(
                    self._neural_posterior.neural_net.parameters(), max_norm=5.0
                )
                optimizer.step()

            epochs += 1

            # Calculate validation performance.
            self._neural_posterior.neural_net.eval()
            log_prob_sum = 0
            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                    )
                    log_prob = self._neural_posterior.neural_net.log_prob(
                        theta_batch, context=x_batch
                    )
                    log_prob_sum += log_prob.sum().item()
            validation_log_prob = log_prob_sum / num_validation_examples

            # Check for improvement in validation performance over previous epochs.
            if validation_log_prob > best_validation_log_prob:
                best_validation_log_prob = validation_log_prob
                epochs_since_last_improvement = 0
                best_model_state_dict = deepcopy(
                    self._neural_posterior.neural_net.state_dict()
                )
            else:
                epochs_since_last_improvement += 1

            # If no validation improvement over many epochs, stop training.
            if epochs_since_last_improvement > stop_after_epochs - 1:
                self._neural_posterior.neural_net.load_state_dict(best_model_state_dict)
                break

        # Update summary.
        self._summary["epochs"].append(epochs)
        self._summary["best_validation_log_probs"].append(best_validation_log_prob)

    @property
    def summary(self):
        return self._summary


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
    most current trained Posterior.neural_net.
    
    Returns:
        [Callable]: potential function for use by either numpy or pyro sampler
    """

    def __call__(
        self, prior, likelihood_nn: nn.Module, x: Tensor, mcmc_method: str,
    ) -> Callable:
        r"""Return potential function for posterior $p(\theta|x)$.
        
        Switch on numpy or pyro potential function based on mcmc_method.
        
        Args:
            prior: prior distribution that can be evaluated
            likelihood_nn: neural likelihood estimator that can be evaluated
            x: conditioning variable for posterior $p(\theta|x)$.
            mcmc_method (str): one of slice-np, slice, hmc or nuts
        
        Returns:
            Callable: potential function for sampler.
        """ """        
        
        Args: 
        
        """
        self.likelihood_nn = likelihood_nn
        self.prior = prior
        self.x = x

        if mcmc_method in ("slice", "hmc", "nuts"):
            return self.pyro_potential
        else:
            return self.np_potential

    def np_potential(self, theta: np.array) -> Union[Tensor, float]:
        r"""Return posterior log prob. of theta $p(\theta|x)$"
        
        Args:
            theta: parameters $\theta$, batch dimension 1
        
        Returns:
            Posterior log probability of the theta, -Inf if impossible under prior.
        """
        theta = torch.as_tensor(theta, dtype=torch.float32)
        log_likelihood = self.likelihood_nn.log_prob(
            inputs=self.x.reshape(1, -1), context=theta.reshape(1, -1)
        )

        # notice opposite sign to pyro potential
        return log_likelihood + self.prior.log_prob(theta)

    def pyro_potential(self, theta: Dict[str, Tensor]) -> Tensor:
        r"""Return posterior log prob. of parameters $p(\theta|x)$.
        
         Args:
            theta: parameters $\theta$. The tensor's shape will be
            (1, shape_of_single_theta) if running a single chain or just
             (shape_of_single_theta) for multiple  chains.
        
        Returns:
            potential: $-[\log r(x_o, \theta) + \log p(\theta)]$
        """

        theta = next(iter(theta.values()))

        log_likelihood = self.likelihood_nn.log_prob(
            inputs=self.x.reshape(1, -1), context=theta.reshape(1, -1)
        )

        return -(log_likelihood + self.prior.log_prob(theta))
