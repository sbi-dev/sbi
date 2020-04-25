from abc import ABC
from copy import deepcopy
from typing import Callable, List, Optional, Union
import warnings

import numpy as np
from pyknos.mdn.mdn import MultivariateGaussianMDN
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


class SnpeBase(NeuralInference, ABC):
    def __init__(
        self,
        simulator: Callable,
        prior,
        x_o: Tensor,
        num_pilot_sims: int = 100,
        density_estimator=None,
        calibration_kernel: Optional[Callable] = None,
        z_score_x: bool = True,
        simulation_batch_size: int = 1,
        use_combined_loss: bool = False,
        retrain_from_scratch_each_round: bool = False,
        discard_prior_samples: bool = False,
        device: Optional[torch.device] = None,
        sample_with_mcmc: bool = False,
        mcmc_method: str = "slice-np",
        summary_writer: Optional[SummaryWriter] = None,
    ):
        """
        See NeuralInference docstring for all other arguments.

        Args:
            num_pilot_sims: number of simulations that are run when
                instantiating an object. Used to z-score the data x.
            density_estimator: neural density estimator
            calibration_kernel: a function to calibrate the data x
            z_score_x: whether to z-score the data features x
            use_combined_loss: whether to jointly neural_net prior samples 
                using maximum likelihood. Useful to prevent density leaking when using
                box uniform priors.
            retrain_from_scratch_each_round: whether to retrain the conditional
                density estimator for the posterior from scratch each round.
            discard_prior_samples: whether to discard prior samples from round
                two onwards.
        """

        super().__init__(
            simulator, prior, x_o, simulation_batch_size, device, summary_writer,
        )

        self._z_score_x = z_score_x

        self._num_pilot_sims = num_pilot_sims
        self._use_combined_loss = use_combined_loss
        self._discard_prior_samples = discard_prior_samples

        self._prior_masks = []
        self._model_bank = []

        self._retrain_from_scratch_each_round = retrain_from_scratch_each_round

        self.pilot_theta = self._prior.sample((num_pilot_sims,))
        self.pilot_x = self._batched_simulator(self.pilot_theta)

        # create the deep neural density estimator
        if density_estimator is None:
            density_estimator = utils.posterior_nn(
                model="maf", prior=self._prior, x_o=self._x_o,
            )
        # else: check density estimator for valid prior etc.
        # XXX: here, the user could sneak in an invalid prior and x_o by providing a
        # density estimator with invalid .prior and .x_o, thus bypassing
        # the input checks.

        # create the neural posterior which can sample(), log_prob()
        self._neural_posterior = Posterior(
            algorithm_family="snpe",
            neural_net=density_estimator,
            prior=self._prior,
            x_o=self._x_o,
            sample_with_mcmc=sample_with_mcmc,
            mcmc_method=mcmc_method,
            get_potential_function=PotentialFunctionProvider(),
        )

        # obtain z-score for data x and define embedding net
        if self._z_score_x:
            self.x_mean = torch.mean(self.pilot_x, dim=0)
            self.x_std = torch.std(self.pilot_x, dim=0)
        else:
            self.x_mean = torch.zeros(self._x_o.shape)
            self.x_std = torch.ones(self._x_o.shape)

        # new embedding_net contains z-scoring
        if not isinstance(self._neural_posterior.neural_net, MultivariateGaussianMDN):
            embedding = nn.Sequential(
                utils.Standardize(self.x_mean, self.x_std),
                self._neural_posterior.neural_net._embedding_net,
            )
            self._neural_posterior.set_embedding_net(embedding)
        elif z_score_x:
            warnings.warn("z-scoring of data x not implemented for MDNs")

        # calibration kernels proposed in Lueckmann, Goncalves et al 2017
        if calibration_kernel is None:
            self.calibration_kernel = lambda x: torch.ones([len(x)])
        else:
            self.calibration_kernel = calibration_kernel

        # If we're retraining from scratch each round,
        # keep a copy of the original untrained model for reinitialization.
        self._untrained_neural_posterior = deepcopy(self._neural_posterior)

        # extra SNPE-specific fields summary_writer
        self._summary.update({"rejection_sampling_acceptance_rates": []})

    def __call__(
        self,
        num_rounds: int,
        num_simulations_per_round: Union[List[int], int],
        batch_size: int = 100,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        clip_grad_norm: bool = True,
    ) -> Posterior:
        r"""Run SNPE

        Return posterior $p(\theta|x_o)$ after inference over several rounds.

        Args:
            num_rounds: Number of rounds to run
            num_simulations_per_round: Number of simulator calls per round
            batch_size: Size of batch to use for training.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            clip_grad_norm: Whether to clip norm of gradients or not.
            
        Returns:
            Posterior that can be sampled and evaluated.
        """
        try:
            assert (
                len(num_simulations_per_round) == num_rounds
            ), "Please provide list with number of simulations per round for each round, or a single integer to be used for all rounds."
        except TypeError:
            num_simulations_per_round = [num_simulations_per_round] * num_rounds

        round_description = ""
        tbar = tqdm(range(num_rounds))
        for round_ in tbar:

            tbar.set_description(round_description)

            # run simulations for the round
            self._run_simulations(round_, num_simulations_per_round[round_])

            # Fit posterior using newly aggregated data set.
            self._train(
                round_=round_,
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_fraction=validation_fraction,
                stop_after_epochs=stop_after_epochs,
                clip_grad_norm=clip_grad_norm,
            )

            # Store models at end of each round.
            self._model_bank.append(deepcopy(self._neural_posterior))
            self._model_bank[-1].neural_net.eval()

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
                x_o=self._x_o,
                theta_bank=self._theta_bank,
                x_bank=self._x_bank,
                simulator=self._simulator,
                posterior_samples_acceptance_rate=self._neural_posterior.get_leakage_correction(
                    x=self._x_o
                ),
            )

        self._neural_posterior._num_trained_rounds = num_rounds
        return self._neural_posterior

    def _get_log_prob_proposal_posterior(self, theta: Tensor, x: Tensor, masks: Tensor):
        r"""
        Evaluate the log-probability used for the loss. Depending on
        the algorithm, this evaluates a different term.

        Args:
            theta: parameters $\theta$
            x: simulation outputs $x$
            masks: binary, indicates whether to
                use prior samples

        Returns: log-probability
        """
        raise NotImplementedError

    def _run_simulations(
        self, round_: int, num_sims: int,
    ):
        r"""
        Run the simulations for a given round.

        Args:
            round_: Round
            num_sims: Number of simulations in the round.

        Returns:
            self._theta_bank: $\theta$ used for training
            self._x_bank: $x$ used for training
            self._prior_masks: boolean masks of for each prior sample,
                indicating whether prior sample will be used in current round
        """
        # Generate theta from prior in first round, and from most recent posterior
        # estimate in subsequent rounds.
        if round_ == 0:
            # One in mask -> prior simulations not used in rounds>=1.
            prior_mask = torch.ones((num_sims, 1))

            num_missing_sims = max(num_sims - self._num_pilot_sims, 0)

            # We take at most the total requested number from the pilot run.
            theta = self.pilot_theta[:num_sims]
            x = self.pilot_x[:num_sims]
            if num_missing_sims > 0:
                # If missing, we produce extra simulations as needed.
                missing_theta = self._prior.sample((num_missing_sims,))
                missing_x = self._batched_simulator(missing_theta)
                theta = torch.cat((theta, missing_theta), dim=0)
                x = torch.cat((x, missing_x), dim=0)

        else:
            # Zero in mask -> prior simulations not used in rounds>=1.
            prior_mask = torch.zeros((num_sims, 1))
            # XXX Align signature of posterior.sample() with that of prior.sample().
            theta = self._neural_posterior.sample(num_sims, x=self._x_o)
            x = self._batched_simulator(theta)

        # XXX Return here round-wise outputs, accumulate elsewhere
        # XXX Rename bank -> rounds/roundwise.
        self._theta_bank.append(theta)
        self._x_bank.append(x)
        self._prior_masks.append(prior_mask)

    def _train(
        self,
        round_,
        batch_size,
        learning_rate,
        validation_fraction,
        stop_after_epochs,
        clip_grad_norm,
    ):
        r"""Train

        Trains the conditional density estimator for the posterior $p(\theta|x)$ by
         maximizing the proposal posterior using the most recently aggregated bank of
         $(\theta, x)$ pairs.
        
        Uses early stopping on a held-out validation set as a terminating condition.
        """

        # get the start index for what training set to use. Either 0 or 1
        ix = int(self._discard_prior_samples and (round_ > 0))

        # Get total number of training examples.
        num_examples = torch.cat(self._theta_bank[ix:]).shape[0]

        # Select random neural_net and validation splits from (theta, x) pairs.
        permuted_indices = torch.randperm(num_examples)
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples
        train_indices, val_indices = (
            permuted_indices[:num_training_examples],
            permuted_indices[num_training_examples:],
        )

        # Dataset is shared for training and validation loaders.
        dataset = data.TensorDataset(
            torch.cat(self._theta_bank[ix:]),
            torch.cat(self._x_bank[ix:]),
            torch.cat(self._prior_masks[ix:]),
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
            drop_last=True,
            sampler=SubsetRandomSampler(val_indices),
        )

        optimizer = optim.Adam(
            list(self._neural_posterior.neural_net.parameters()), lr=learning_rate,
        )
        # Keep track of best_validation log_prob seen so far.
        best_validation_log_prob = -1e100
        # Keep track of number of epochs since last improvement.
        epochs_since_last_improvement = 0
        # Keep track of model with best validation performance.
        best_model_state_dict = None

        # If retraining from scratch each round, reset the neural posterior
        # to the untrained copy.
        if self._retrain_from_scratch_each_round and round_ > 0:
            self._neural_posterior = deepcopy(self._untrained_neural_posterior)

        epochs = 0
        converged = False
        while not converged:

            # Train for a single epoch.
            self._neural_posterior.neural_net.train()
            for batch in train_loader:
                optimizer.zero_grad()
                theta_batch, x_batch, masks = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                    batch[2].to(self._device),
                )

                # just do maximum likelihood in the first round
                if round_ == 0:
                    log_prob = self._neural_posterior.neural_net.log_prob(
                        theta_batch, x_batch
                    )
                else:  # or call the APT loss
                    log_prob = self._get_log_prob_proposal_posterior(
                        theta_batch, x_batch, masks
                    )
                loss = -torch.mean(log_prob)
                loss.backward()
                if clip_grad_norm:
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
                    theta_batch, x_batch, masks = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                        batch[2].to(self._device),
                    )
                    # just do maximum likelihood in the first round
                    if round_ == 0:
                        log_prob = self._neural_posterior.neural_net.log_prob(
                            theta_batch, x_batch
                        )
                    else:
                        log_prob = self._get_log_prob_proposal_posterior(
                            theta_batch, x_batch, masks
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
                converged = True

        # Update summary.
        self._summary["epochs"].append(epochs)
        self._summary["best_validation_log_probs"].append(best_validation_log_prob)


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
        self, prior, posterior_nn: nn.Module, x: Tensor, mcmc_method: str,
    ) -> Callable:
        """Return potential function. 
        
        Switch on numpy or pyro potential function based on mcmc_method.
        
        """
        self.posterior_nn = posterior_nn
        self.prior = prior
        self.x = x

        if mcmc_method in ("slice", "hmc", "nuts"):
            return self.pyro_potential
        else:
            return self.np_potential

    def np_potential(self, theta: np.ndarray) -> Union[Tensor, float]:
        r"""Return posterior log prob. of theta $p(\theta|x)$, -inf if outside prior."
        
        Args:
            theta: parameters $\theta$, batch dimension 1
        
        Returns:
            posterior log probability $\log(p(\theta|x))$
        """
        theta = torch.as_tensor(theta, dtype=torch.float32)

        is_within_prior = torch.isfinite(self.prior.log_prob(theta))
        if is_within_prior:
            target_log_prob = self.posterior_nn.log_prob(
                inputs=theta.reshape(1, -1), context=self.x.reshape(1, -1),
            )
        else:
            target_log_prob = -float("Inf")

        return target_log_prob

    def pyro_potential(self, theta: dict) -> Tensor:
        r"""Return posterior log prob. of theta $p(\theta|x)$, -inf where outside prior.
        
        Args:
            theta: parameters $\theta$ (from pyro sampler)
        
        Returns:
            Posterior log probability $p(\theta|x)$, masked outside of prior
        """

        theta = next(iter(theta.values()))
        # XXX: notice sign, check convention pyro vs. numpy
        log_prob_posterior = -self.posterior_nn.log_prob(inputs=theta, context=self.x,)
        log_prob_prior = self.prior.log_prob(theta)

        within_prior = torch.isfinite(log_prob_prior)

        return torch.where(within_prior, log_prob_posterior, log_prob_prior)
