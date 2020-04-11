from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC
from torch import Tensor, nn, optim
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sbi.simulators as simulators
import sbi.utils as utils
from sbi.inference.base import NeuralInference
from sbi.inference.posteriors.sbi_posterior import Posterior
from sbi.utils.torchutils import ensure_observation_batched, ensure_parameter_batched


class SRE(NeuralInference):
    def __init__(
        self,
        simulator: Callable,
        prior,
        true_observation: Tensor,
        classifier: nn.Module,
        num_atoms: int = -1,
        simulation_batch_size: int = 1,
        mcmc_method: str = "slice-np",
        summary_net: Optional[nn.Module] = None,
        classifier_loss: str = "sre",
        retrain_from_scratch_each_round: bool = False,
        summary_writer: Optional[SummaryWriter] = None,
        device: Optional[torch.device] = None,
    ):
        """Sequential Ratio Estimation

        As presented in _Likelihood-free MCMC with Amortized Approximate Likelihood Ratios_ by Hermans et al., Pre-print 2019, https://arxiv.org/abs/1903.04057

        See NeuralInference docstring for all other arguments.

        Args:
            classifier: Binary classifier
            num_atoms: Number of atoms to use for classification.
                If -1, use all other parameters in minibatch
            retrain_from_scratch_each_round: whether to retrain from scratch
                each round
            summary_net: Optional network which may be used to produce feature
                vectors f(x) for high-dimensional observations
            classifier_loss: `sre` implements the algorithm suggested in Durkan et al. 
                2019, whereas `aalr` implements the algorithm suggested in Hermans et al. 2019. `sre` can use more than two atoms, potentially boosting performance, but does not allow for exact posterior density evaluation (only up to a normalizing constant), even when training only one round. `aalr` is limited to `num_atoms=2`, but allows for density evaluation when training for one round.
        """

        super().__init__(
            simulator,
            prior,
            true_observation,
            simulation_batch_size,
            device,
            summary_writer,
        )

        self._classifier_loss = classifier_loss

        assert isinstance(num_atoms, int), "Number of atoms must be an integer."
        self._num_atoms = num_atoms

        if classifier is None:
            classifier = utils.classifier_nn(
                model="resnet", prior=self._prior, context=self._true_observation,
            )

        # create posterior object which can sample()
        self._neural_posterior = Posterior(
            algorithm_family=self._classifier_loss,
            neural_net=classifier,
            prior=prior,
            context=true_observation,
            mcmc_method=mcmc_method,
            get_potential_function=PotentialFunctionProvider(),
        )

        # XXX why not classifier.train(True)???
        self._neural_posterior.neural_net.train(True)

        # We may want to summarize high-dimensional observations.
        # This may be either a fixed or learned transformation.
        if summary_net is None:
            self._summary_net = nn.Identity()
        else:
            self._summary_net = summary_net

        # If we're retraining from scratch each round,
        # keep a copy of the original untrained model for reinitialization.
        self._retrain_from_scratch_each_round = retrain_from_scratch_each_round
        if self._retrain_from_scratch_each_round:
            self._untrained_classifier = deepcopy(classifier)
        else:
            self._untrained_classifier = None

        # SRE-specific summary_writer fields
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
        """Run SRE

        This runs SRE for num_rounds rounds, using num_simulations_per_round calls to
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
            Posterior that can be sampled and evaluated.
        """
        round_description = ""
        tbar = tqdm(range(num_rounds))
        for round_ in tbar:

            tbar.set_description(round_description)

            # Generate parameters from prior in first round, and from most recent posterior
            # estimate in subsequent rounds.
            if round_ == 0:
                parameters, observations = simulators.simulate_in_batches(
                    simulator=self._simulator,
                    parameter_sample_fn=lambda num_samples: self._prior.sample(
                        (num_samples,)
                    ),
                    num_samples=num_simulations_per_round,
                    simulation_batch_size=self._simulation_batch_size,
                )
            else:
                parameters, observations = simulators.simulate_in_batches(
                    simulator=self._simulator,
                    parameter_sample_fn=lambda num_samples: self._neural_posterior.sample(
                        num_samples
                    ),
                    num_samples=num_simulations_per_round,
                    simulation_batch_size=self._simulation_batch_size,
                )

            # Store (parameter, observation) pairs.
            self._parameter_bank.append(torch.Tensor(parameters))
            self._observation_bank.append(torch.Tensor(observations))

            # Fit posterior using newly aggregated data set.
            self._train(
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_fraction=validation_fraction,
                stop_after_epochs=stop_after_epochs,
            )

            # Update description for progress bar.
            round_description = (
                f"-------------------------\n"
                f"||||| ROUND {round_ + 1} STATS |||||:\n"
                f"-------------------------\n"
                f"Epochs trained: {self._summary['epochs'][-1]}\n"
                f"Best validation performance: {self._summary['best_validation_log_probs'][-1]:.4f}\n\n"
            )

            # Update tensorboard and summary dict.
            self._summary_writer, self._summary = utils.summarize(
                summary_writer=self._summary_writer,
                summary=self._summary,
                round_=round_,
                true_observation=self._true_observation,
                parameter_bank=self._parameter_bank,
                observation_bank=self._observation_bank,
                simulator=self._simulator,
            )

        self._neural_posterior._num_trained_rounds = num_rounds
        return self._neural_posterior

    def _train(
        self, batch_size, learning_rate, validation_fraction, stop_after_epochs,
    ):
        """
        Trains the classifier by maximizing a Bernoulli likelihood which distinguishes
        between jointly distributed (parameter, observation) pairs and randomly chosen
        (parameter, observation) pairs.
        Uses early stopping on a held-out validation set as a terminating condition.
        """

        # Get total number of training examples.
        num_examples = torch.cat(self._parameter_bank).shape[0]

        # Select random train and validation splits from (parameter, observation) pairs.
        permuted_indices = torch.randperm(num_examples)
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples
        train_indices, val_indices = (
            permuted_indices[:num_training_examples],
            permuted_indices[num_training_examples:],
        )

        # Dataset is shared for training and validation loaders.
        dataset = data.TensorDataset(
            torch.cat(self._parameter_bank), torch.cat(self._observation_bank)
        )

        # Create neural_net and validation loaders using a subset sampler.

        # NOTE: The batch_size is clipped to num_validation samples
        clipped_batch_size = min(batch_size, num_validation_examples)
        train_loader = data.DataLoader(
            dataset,
            batch_size=clipped_batch_size,
            drop_last=True,
            sampler=SubsetRandomSampler(train_indices),
        )
        val_loader = data.DataLoader(
            dataset,
            batch_size=clipped_batch_size,
            shuffle=False,
            drop_last=False,
            sampler=SubsetRandomSampler(val_indices),
        )

        optimizer = optim.Adam(
            list(self._neural_posterior.neural_net.parameters())
            + list(self._summary_net.parameters()),
            lr=learning_rate,
        )

        # only used if classifier_loss == "aalr"
        criterion = nn.BCELoss()

        # Keep track of best_validation log_prob seen so far.
        best_validation_log_prob = -1e100
        # Keep track of number of epochs since last improvement.
        epochs_since_last_improvement = 0
        # Keep track of model with best validation performance.
        best_model_state_dict = None

        # If we're retraining from scratch each round, reset the neural posterior
        # to the untrained copy we made at the start.
        if self._retrain_from_scratch_each_round:
            self._neural_posterior = deepcopy(self._neural_posterior)

        def _get_loss(parameters, observations):

            # num_atoms = parameters.shape[0]
            num_atoms = self._num_atoms if self._num_atoms > 0 else clipped_batch_size

            if self._classifier_loss == "aalr":
                assert num_atoms == 2, "aalr allows only two atoms, i.e. num_atoms=2."

            repeated_observations = utils.repeat_rows(observations, num_atoms)

            # Choose between 1 and num_atoms - 1 parameters from the rest
            # of the batch for each observation.
            assert 0 < num_atoms - 1 < clipped_batch_size
            probs = (
                (1 / (clipped_batch_size - 1))
                * torch.ones(clipped_batch_size, clipped_batch_size)
                * (1 - torch.eye(clipped_batch_size))
            )
            choices = torch.multinomial(
                probs, num_samples=num_atoms - 1, replacement=False
            )
            contrasting_parameters = parameters[choices]

            atomic_parameters = torch.cat(
                (parameters[:, None, :], contrasting_parameters), dim=1
            ).reshape(clipped_batch_size * num_atoms, -1)

            inputs = torch.cat((atomic_parameters, repeated_observations), dim=1)

            if self._classifier_loss == "aalr":
                network_outputs = self._neural_posterior.neural_net(inputs)
                likelihood = torch.squeeze(torch.sigmoid(network_outputs))

                # the first clipped_batch_size elements are the ones where theta and x
                # are sampled from the joint p(theta, x) and are labelled 1s.
                # The second clipped_batch_size elements are the ones where theta and x
                # are sampled from the marginals p(theta)p(x) and are labelled 0s.
                labels = torch.cat(
                    (torch.ones(clipped_batch_size), torch.zeros(clipped_batch_size))
                )
                # binary cross entropy to learn the likelihood
                loss = criterion(likelihood, labels)
            else:
                logits = self._neural_posterior.neural_net(inputs).reshape(
                    clipped_batch_size, num_atoms
                )
                # index 0 is the parameter set sampled from the joint
                log_prob = logits[:, 0] - torch.logsumexp(logits, dim=-1)
                loss = -torch.mean(log_prob)

            return loss

        epochs = 0
        while True:

            # Train for a single epoch.
            self._neural_posterior.neural_net.train()
            for parameters, observations in train_loader:
                optimizer.zero_grad()
                loss = _get_loss(parameters, observations)
                loss.backward()
                optimizer.step()

            epochs += 1

            # calculate validation performance
            self._neural_posterior.neural_net.eval()
            log_prob_sum = 0
            with torch.no_grad():
                for parameters, observations in val_loader:
                    log_prob = _get_loss(parameters, observations)
                    log_prob_sum += log_prob.sum().item()
                validation_log_prob = log_prob_sum / num_validation_examples

            # check for improvement
            if validation_log_prob > best_validation_log_prob:
                best_model_state_dict = deepcopy(
                    self._neural_posterior.neural_net.state_dict()
                )
                best_validation_log_prob = validation_log_prob
                epochs_since_last_improvement = 0
            else:
                epochs_since_last_improvement += 1

            # if no validation improvement over many epochs, stop training
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
    This class is initialized without arguments during the initialization of the Posterior class. When called, it specializes to the potential function appropriate to the requested mcmc_method.
    
   
    NOTE: Why use a class?
    ----------------------
    During inference, we use deepcopy to save untrained posteriors in memory. deepcopy uses pickle which can't serialize nested functions (https://stackoverflow.com/a/12022055).
    
    It is important to NOT initialize attributes upon instantiation, because we need the most current trained Posterior.neural_net.
    
    Returns:
        [Callable]: potential function for use by either numpy or pyro sampler
    """

    def __call__(
        self, prior, classifier: nn.Module, observation: Tensor, mcmc_method: str,
    ) -> Callable:
        """Return potential function. 
        
        Switch on numpy or pyro potential function based on mcmc_method.
        
        Args:
            prior: prior distribution that can be evaluated.
            classifier: binary classifier approximating the likelihood up to a constant.
       
            observation: actually observed conditioning context, x_o
            mcmc_method (str): one of slice-np, slice, hmc or nuts.
        
        Returns:
            Callable: potential function for sampler.
        """ """        
        
        Args: 
        
        """
        self.classifier = classifier
        self.prior = prior
        self.observation = observation

        if mcmc_method in ("slice", "hmc", "nuts"):
            return self.pyro_potential
        else:
            return self.np_potential

    def np_potential(self, parameters: np.array) -> Union[Tensor, float]:
        """Return potential for Numpy slice sampler."
        
        Args:
            parameters ([np.array]): parameter vector, batch dimension 1
        
        Returns:
            [tensor or -inf]: posterior log probability of the parameters.
        """
        parameter = torch.as_tensor(parameters, dtype=torch.float32)

        # parameter and observation should have shape (1, dim)
        parameter = ensure_parameter_batched(parameter)
        observation = ensure_observation_batched(self.observation)

        log_ratio = self.classifier(
            torch.cat((parameter, observation), dim=1).reshape(1, -1)
        )

        # notice opposite sign to pyro potential
        return log_ratio + self.prior.log_prob(parameter)

    def pyro_potential(self, parameters: Dict[str, Tensor]) -> Tensor:
        """Return potential for Pyro sampler.
        
        Args:
            parameters: {name: tensor, ...} dictionary (from pyro sampler). The tensor's shape will be (1, x) if running a single chain or just (x) for multiple chains.
        
        Returns:
            potential: -[log r(x0, theta) + log p(theta)]
        """

        parameter = next(iter(parameters.values()))

        # parameter and observation should have shape (1, dim)
        parameter = ensure_parameter_batched(parameter)
        observation = ensure_observation_batched(self.observation)

        log_ratio = self.classifier(
            torch.cat((parameter, observation), dim=1).reshape(1, -1)
        )

        return -(log_ratio + self.prior.log_prob(parameter))
