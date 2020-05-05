from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn, optim, ones
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sbi.utils as utils
from sbi.inference.base import NeuralInference
from sbi.inference.posteriors.sbi_posterior import Posterior
from sbi.utils.torchutils import ensure_x_batched, ensure_theta_batched


class SRE(NeuralInference):
    def __init__(
        self,
        simulator: Callable,
        prior,
        x_o: Tensor,
        classifier: Optional[nn.Module] = None,
        num_atoms: Optional[int] = None,
        simulation_batch_size: int = 1,
        mcmc_method: str = "slice-np",
        summary_net: Optional[nn.Module] = None,
        classifier_loss: str = "sre",
        retrain_from_scratch_each_round: bool = False,
        summary_writer: Optional[SummaryWriter] = None,
        device: Optional[torch.device] = None,
        skip_input_checks: bool = False,
    ):
        r"""Sequential Ratio Estimation

        As presented in _Likelihood-free MCMC with Amortized Approximate Likelihood
         Ratios_ by Hermans et al., Pre-print 2019, https://arxiv.org/abs/1903.04057

        See NeuralInference docstring for all other arguments.

        Args:
            classifier: Binary classifier
            num_atoms: Number of atoms to use for classification. If None, use all
                other parameters $\theta$ in minibatch.
            retrain_from_scratch_each_round: whether to retrain from scratch
                each round
            summary_net: Optional network which may be used to produce feature
                vectors f(x) for high-dimensional simulation outputs $x$.
            classifier_loss: `sre` implements the algorithm suggested in Durkan et al. 
                2019, whereas `aalr` implements the algorithm suggested in
                Hermans et al. 2019. `sre` can use more than two atoms, potentially
                boosting performance, but does not allow for exact posterior density
                evaluation (only up to a normalizing constant), even when training
                only one round. `aalr` is limited to `num_atoms=2`, but allows for
                density evaluation when training for one round.
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

        self._classifier_loss = classifier_loss
        self._num_atoms = num_atoms if num_atoms is not None else 0

        if classifier is None:
            classifier = utils.classifier_nn(
                model="resnet",
                theta_shape=self._prior.sample().shape,
                x_o_shape=self._x_o.shape,
            )

        # create posterior object which can sample()
        self._neural_posterior = Posterior(
            algorithm_family=self._classifier_loss,
            neural_net=classifier,
            prior=prior,
            x_o=x_o,
            mcmc_method=mcmc_method,
            get_potential_function=PotentialFunctionProvider(),
        )

        # XXX why not classifier.train(True)???
        self._neural_posterior.neural_net.train(True)

        # We may want to summarize high-dimensional x.
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
        num_sims_per_round = self._ensure_list(num_simulations_per_round, num_rounds)

        tbar = tqdm(enumerate(num_sims_per_round))
        for round_, num_sims in tbar:

            # Generate theta from prior in first round, and from most recent posterior
            # estimate in subsequent rounds.
            if round_ == 0:
                theta = self._prior.sample((num_sims,))
            else:
                theta = self._neural_posterior.sample(num_sims)

            x = self._batched_simulator(theta)

            # Store (theta, x) pairs.
            self._theta_bank.append(theta)
            self._x_bank.append(x)

            # Fit posterior using newly aggregated data set.
            self._train(
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_fraction=validation_fraction,
                stop_after_epochs=stop_after_epochs,
            )

            # Update description for progress bar.
            tbar.set_description(self._describe_round(round_, self._summary))

            # Update tensorboard and summary dict.
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

    def _train(
        self, batch_size, learning_rate, validation_fraction, stop_after_epochs,
    ):
        r"""
        Trains the classifier by maximizing a Bernoulli likelihood which distinguishes
        between jointly distributed $(\theta, x)$ pairs and randomly chosen
        $(\theta, x)$ pairs.

        Uses early stopping on a held-out validation set as a terminating condition.
        """

        # Get total number of training examples.
        # todo: We're really concatenating here just to get a shape, that's crazy!
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
            torch.cat(self._theta_bank), torch.cat(self._x_bank)
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

        def _get_loss(theta, x):

            # num_atoms = theta.shape[0]
            num_atoms = self._num_atoms if self._num_atoms > 0 else clipped_batch_size

            if self._classifier_loss == "aalr":
                assert num_atoms == 2, "aalr allows only two atoms, i.e. num_atoms=2."

            repeated_x = utils.repeat_rows(x, num_atoms)

            # Choose between 1 and num_atoms - 1 thetas from the rest
            # of the batch for each x.
            assert 0 < num_atoms - 1 < clipped_batch_size
            probs = (
                (1 / (clipped_batch_size - 1))
                * ones(clipped_batch_size, clipped_batch_size)
                * (1 - torch.eye(clipped_batch_size))
            )
            choices = torch.multinomial(
                probs, num_samples=num_atoms - 1, replacement=False
            )
            contrasting_theta = theta[choices]

            atomic_theta = torch.cat(
                (theta[:, None, :], contrasting_theta), dim=1
            ).reshape(clipped_batch_size * num_atoms, -1)

            theta_and_x = torch.cat((atomic_theta, repeated_x), dim=1)

            if self._classifier_loss == "aalr":
                network_outputs = self._neural_posterior.neural_net(theta_and_x)
                likelihood = torch.squeeze(torch.sigmoid(network_outputs))

                # Alternating pairs where there is one sampled from the joint and one
                # sampled from the marginals. The first element is sampled from the
                # joint p(theta, x) and is labelled 1. The second element is sampled
                # from the marginals p(theta)p(x) and is labelled 0. And so on.
                labels = ones(2 * clipped_batch_size)  # two atoms
                labels[1::2] = 0.0
                # binary cross entropy to learn the likelihood
                loss = criterion(likelihood, labels)
            else:
                logits = self._neural_posterior.neural_net(theta_and_x).reshape(
                    clipped_batch_size, num_atoms
                )
                # index 0 is the theta sampled from the joint
                log_prob = logits[:, 0] - torch.logsumexp(logits, dim=-1)
                loss = -torch.mean(log_prob)

            return loss

        epochs = 0
        while True:

            # Train for a single epoch.
            self._neural_posterior.neural_net.train()
            for theta_batch, x_batch in train_loader:
                optimizer.zero_grad()
                loss = _get_loss(theta_batch, x_batch)
                loss.backward()
                optimizer.step()

            epochs += 1

            # calculate validation performance
            self._neural_posterior.neural_net.eval()
            log_prob_sum = 0
            with torch.no_grad():
                for theta_batch, x_batch in val_loader:
                    log_prob = _get_loss(theta_batch, x_batch)
                    log_prob_sum -= log_prob.sum().item()
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
        self, prior, classifier: nn.Module, x: Tensor, mcmc_method: str,
    ) -> Callable:
        r"""Return potential function for posterior $p(\theta|x)$. 
        
        Switch on numpy or pyro potential function based on mcmc_method.
        
        Args:
            prior: prior distribution that can be evaluated.
            classifier: binary classifier approximating the likelihood up to a constant.
       
            x: conditioning variable for posterior $p(\theta|x)$.
            mcmc_method (str): one of slice-np, slice, hmc or nuts.
        
        Returns:
            Callable: potential function for sampler.
        """ """        
        
        Args: 
        
        """
        self.classifier = classifier
        self.prior = prior
        self.x = x

        if mcmc_method in ("slice", "hmc", "nuts"):
            return self.pyro_potential
        else:
            return self.np_potential

    def np_potential(self, theta: np.array) -> Union[Tensor, float]:
        """Return potential for Numpy slice sampler."
        
        Args:
            theta: parameters $\theta$, batch dimension 1
        
        Returns:
            [tensor or -inf]: posterior log probability of theta.
        """
        theta = torch.as_tensor(theta, dtype=torch.float32)

        # theta and x should have shape (1, dim)
        theta = ensure_theta_batched(theta)
        x = ensure_x_batched(self.x)

        log_ratio = self.classifier(torch.cat((theta, x), dim=1).reshape(1, -1))

        # notice opposite sign to pyro potential
        return log_ratio + self.prior.log_prob(theta)

    def pyro_potential(self, theta: Dict[str, Tensor]) -> Tensor:
        """Return potential for Pyro sampler.
        
        Args:
            theta: parameters $\theta$. The tensor's shape will be
             (1, shape_of_single_theta) if running a single chain or just
             (shape_of_single_theta) for multiple chains.
        
        Returns:
            potential: $-[\log r(x_o, \theta) + \log p(\theta)]$
        """

        theta = next(iter(theta.values()))

        # theta and x should have shape (1, dim)
        theta = ensure_theta_batched(theta)
        x = ensure_x_batched(self.x)

        log_ratio = self.classifier(torch.cat((theta, x), dim=1).reshape(1, -1))

        return -(log_ratio + self.prior.log_prob(theta))
