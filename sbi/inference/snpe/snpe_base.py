import warnings
from copy import deepcopy
from typing import Callable, Union

import numpy as np
import torch
from pyknos.mdn.mdn import MultivariateGaussianMDN
from torch import float32, nn, optim
from torch.distributions import Distribution
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

import sbi.utils as utils
from sbi.inference.posteriors.sbi_posterior import Posterior
from sbi.utils.torchutils import get_default_device
from sbi.simulators.simutils import (
    set_simulator_attributes,
    check_prior_and_data_dimensions,
    simulate_in_batches,
)


class SnpeBase:
    def __init__(
        self,
        simulator,
        prior,
        true_observation,
        num_pilot_samples=100,
        density_estimator=None,
        calibration_kernel=None,
        z_score_obs=True,
        simulation_batch_size: int = 1,
        use_combined_loss=False,
        retrain_from_scratch_each_round=False,
        discard_prior_samples=False,
        device=None,
        train_with_mcmc=False,
        mcmc_method="slice-np",
    ):
        """
        Args:
            simulator: Python object with 'simulate' method which takes a torch.Tensor
                of parameter values, and returns a simulation result for each parameter
                as a torch.Tensor.
            prior: torch.distribution
                Distribution object with 'log_prob' and 'sample' methods.
            true_observation: torch.Tensor [observation_dim] or [1, observation_dim]
                True observation x0 for which to perform inference on the posterior p(theta | x0).
            num_pilot_samples: int
                Number of simulations that are run when instantiating an object.
                Used to z-score the observations.
            density_estimator: Neural density estimator
                Density estimator to use.
            z_score_obs: bool
                Whether to z-score (=normalize) the data features x
            simulation_batch_size: the number of parameter sets the simulator takes and converts to data x at
                the same time. If simulation_batch_size==-1, we simulate all parameter sets at the same time.
                If simulation_batch_size==1, the simulator has to process data of shape (1, num_dim).
                If simulation_batch_size>1, the simulator has to process data of shape (simulation_batch_size, num_dim).
            use_combined_loss: bool
                Whether to jointly neural_net prior samples using maximum likelihood.
                Useful to prevent density leaking when using box uniform priors.
            retrain_from_scratch_each_round: bool
                Whether to retrain the conditional density estimator for the posterior
                from scratch each round.
            discard_prior_samples: bool
                Whether to discard prior samples from round two onwards.
            device: torch.device
                Optionally pass device
                If None, will infer it
        """

        true_observation = utils.torchutils.atleast_2d(true_observation)
        check_prior_and_data_dimensions(prior, true_observation)
        self._simulator = set_simulator_attributes(simulator, prior, true_observation)
        self._prior = prior
        self._true_observation = true_observation
        self._device = get_default_device() if device is None else device
        self.z_score_obs = z_score_obs
        self._simulation_batch_size = simulation_batch_size
        self.num_pilot_samples = num_pilot_samples
        self._use_combined_loss = use_combined_loss
        self._discard_prior_samples = discard_prior_samples
        # Need somewhere to store (parameter, observation) pairs from each round.
        self._parameter_bank, self._observation_bank, self._prior_masks = [], [], []
        self._model_bank = []
        self._retrain_from_scratch_each_round = retrain_from_scratch_each_round

        # run prior samples
        (self.pilot_parameters, self.pilot_observations,) = simulate_in_batches(
            simulator=self._simulator,
            parameter_sample_fn=lambda num_samples: self._prior.sample((num_samples,)),
            num_samples=num_pilot_samples,
            simulation_batch_size=self._simulation_batch_size,
            x_dim=self._true_observation.shape[1:],  # do not pass batch_dim
        )

        # create the deep neural density estimator
        if density_estimator is None:
            density_estimator = utils.posterior_nn(
                model="maf", prior=self._prior, context=self._true_observation,
            )

        # create the neural posterior which can sample(), log_prob()
        self._neural_posterior = Posterior(
            algorithm_family="snpe",
            neural_net=density_estimator,
            prior=prior,
            context=true_observation,
            train_with_mcmc=train_with_mcmc,
            mcmc_method=mcmc_method,
            get_potential_function=PotentialFunctionProvider(),
        )

        # obtain z-score for observations and define embedding net
        if self.z_score_obs:
            self.obs_mean = torch.mean(self.pilot_observations, dim=0)
            self.obs_std = torch.std(self.pilot_observations, dim=0)
        else:
            self.obs_mean = torch.zeros(self._true_observation.shape)
            self.obs_std = torch.ones(self._true_observation.shape)

        # new embedding_net contains z-scoring
        if not isinstance(self._neural_posterior.neural_net, MultivariateGaussianMDN):
            embedding = nn.Sequential(
                utils.Normalize(self.obs_mean, self.obs_std),
                self._neural_posterior.neural_net._embedding_net,
            )
            self._neural_posterior.set_embedding_net(embedding)
        elif z_score_obs:
            warnings.warn("z-scoring of observation not implemented for MDNs")

        # calibration kernels proposed in Lueckmann, Goncalves et al 2017
        if calibration_kernel is None:
            self.calibration_kernel = lambda context_input: torch.ones(
                [len(context_input)]
            )
        else:
            self.calibration_kernel = calibration_kernel

        # If we're retraining from scratch each round,
        # keep a copy of the original untrained model for reinitialization.
        self._untrained_neural_posterior = deepcopy(self._neural_posterior)

        # Each run also has a dictionary of summary statistics which are populated
        # over the course of training.
        self._summary = {
            "mmds": [],
            "median-observation-distances": [],
            "negative-log-probs-true-parameters": [],
            "neural-net-fit-times": [],
            "epochs": [],
            "best-validation-log-probs": [],
            "rejection-sampling-acceptance-rates": [],
        }

    def __call__(self, num_rounds, num_simulations_per_round, **kwargs):
        """
        Return posterior density after inference over several rounds.

        Args:
            num_rounds: int
                Number of rounds to run.
            num_simulations_per_round: list or int
                list or int: Number of simulator calls per round.

        Returns: Posterior that can be sampled and evaluated.

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
            self._run_sims(round_, num_simulations_per_round[round_])

            # Fit posterior using newly aggregated data set.
            self._train(round_=round_, **kwargs)

            # Store models at end of each round.
            self._model_bank.append(deepcopy(self._neural_posterior))
            self._model_bank[-1].neural_net.eval()

            # Update description for progress bar.
            round_description = (
                f"-------------------------\n"
                f"||||| ROUND {round_ + 1} STATS |||||:\n"
                f"-------------------------\n"
                f"Epochs trained: {self._summary['epochs'][-1]}\n"
                f"Best validation performance: {self._summary['best-validation-log-probs'][-1]:.4f}\n\n"
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
                posterior_samples_acceptance_rate=self._neural_posterior.get_leakage_correction(
                    context=self._true_observation
                ),
            )

        self._neural_posterior._num_trained_rounds = num_rounds
        return self._neural_posterior

    def _get_log_prob_proposal_posterior(self, inputs, context, masks):
        """
        Evaluate the log-probability used for the loss. Depending on
        the algorithm, this evaluates a different term.

        Args:
            inputs: torch.tensor(), parameters theta
            context: torch.tensor(), data x
            masks: torch.tensor(), binary, indicates whether to
                use prior samples

        Returns: log-probability
        """
        raise NotImplementedError

    def _run_sims(
        self, round_, num_simulations_per_round,
    ):
        """
        Runs the simulations at the beginning of each round.

        Args:
            round_: int. Round
            num_simulations_per_round: int. Number of simulations in current round

        Returns:
            self._parameter_bank: torch.tensor. theta used for training
            self._observation_bank: torch.tensor. x used for training
            self._prior_masks: torch.tensor. Masks of 0/1 for each prior sample,
                indicating whether prior sample will be used in current round
        """
        # Generate parameters from prior in first round, and from most recent posterior
        # estimate in subsequent rounds.
        if round_ == 0:
            parameters, observations = simulate_in_batches(
                simulator=self._simulator,
                parameter_sample_fn=lambda num_samples: self._prior.sample(
                    (num_samples,)
                ),
                num_samples=np.maximum(
                    0, num_simulations_per_round - self.num_pilot_samples
                ),
                simulation_batch_size=self._simulation_batch_size,
                x_dim=self._true_observation.shape[1:],  # do not pass batch_dim
            )
            parameters = torch.cat(
                (parameters, self.pilot_parameters[:num_simulations_per_round]), dim=0
            )
            observations = torch.cat(
                (observations, self.pilot_observations[:num_simulations_per_round]),
                dim=0,
            )
        else:
            parameters, observations = simulate_in_batches(
                simulator=self._simulator,
                parameter_sample_fn=lambda num_samples: self._neural_posterior.sample(
                    num_samples, context=self._true_observation,
                ),
                num_samples=num_simulations_per_round,
                simulation_batch_size=self._simulation_batch_size,
                x_dim=self._true_observation.shape[1:],  # do not pass batch_dim
            )

        # Store (parameter, observation) pairs.
        self._parameter_bank.append(parameters)
        self._observation_bank.append(observations)
        self._prior_masks.append(
            torch.ones(num_simulations_per_round, 1)
            if round_ == 0
            else torch.zeros(num_simulations_per_round, 1)
        )

    def _train(
        self,
        round_,
        batch_size=100,
        learning_rate=5e-4,
        validation_fraction=0.1,
        stop_after_epochs=20,
        clip_grad_norm=True,
    ):
        """
        Trains the conditional density estimator for the posterior by maximizing the
        proposal posterior using the most recently aggregated bank of (parameter, observation)
        pairs.
        Uses early stopping on a held-out validation set as a terminating condition.

        Args:
            round_: int
                Which round we're currently in. Needed when sampling procedure is
                not simply sampling from (proposal) marginal.
            batch_size: int
                Size of batch to use for training.
            learning_rate: float
                Learning rate for Adam optimizer.
            validation_fraction: float in [0, 1]
                The fraction of data to use for validation.
            stop_after_epochs: int
                The number of epochs to wait for improvement on the
                validation set before terminating training.
            clip_grad_norm: bool
                Whether to clip norm of gradients or not.

        Returns: None
        """

        # get the start index for what training set to use. Either 0 or 1
        ix = int(self._discard_prior_samples and (round_ > 0))

        # Get total number of training examples.
        num_examples = torch.cat(self._parameter_bank[ix:]).shape[0]

        if round_ > 0:
            assert (
                validation_fraction * num_examples >= batch_size
            ), "There are fewer samples in the validation set than the batchsize."

        # Select random neural_net and validation splits from (parameter, observation) pairs.
        permuted_indices = torch.randperm(num_examples)
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples
        train_indices, val_indices = (
            permuted_indices[:num_training_examples],
            permuted_indices[num_training_examples:],
        )

        # Dataset is shared for training and validation loaders.
        dataset = data.TensorDataset(
            torch.cat(self._parameter_bank[ix:]),
            torch.cat(self._observation_bank[ix:]),
            torch.cat(self._prior_masks[ix:]),
        )

        # Create neural_net and validation loaders using a subset sampler.
        train_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
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

        # If we're retraining from scratch each round, reset the neural posterior
        # to the untrained copy we made at the start.
        if self._retrain_from_scratch_each_round and round_ > 0:
            self._neural_posterior = deepcopy(self._untrained_neural_posterior)
            # self._neural_posterior = deepcopy(self._model_bank[0])

        epochs = 0
        converged = False
        while not converged:

            # Train for a single epoch.
            self._neural_posterior.neural_net.train()
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, context, masks = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                    batch[2].to(self._device),
                )

                # just do maximum likelihood in the first round
                if round_ == 0:
                    log_prob = self._neural_posterior.neural_net.log_prob(
                        inputs, context
                    )
                else:  # or call the APT loss
                    log_prob = self._get_log_prob_proposal_posterior(
                        inputs, context, masks
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
                    inputs, context, masks = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                        batch[2].to(self._device),
                    )
                    # just do maximum likelihood in the first round
                    if round_ == 0:
                        log_prob = self._neural_posterior.neural_net.log_prob(
                            inputs, context
                        )
                    else:
                        log_prob = self._get_log_prob_proposal_posterior(
                            inputs, context, masks
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
        self._summary["best-validation-log-probs"].append(best_validation_log_prob)


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
        self,
        prior: Distribution,
        posterior_nn: torch.nn.Module,
        observation: torch.Tensor,
        mcmc_method: str,
    ) -> Callable:
        """Return potential function. 
        
        Switch on numpy or pyro potential function based on mcmc_method.
        
        """
        self.posterior_nn = posterior_nn
        self.prior = prior
        self.observation = observation

        if mcmc_method in ("slice", "hmc", "nuts"):
            return self.pyro_potential
        else:
            return self.np_potential

    def np_potential(self, parameters: np.array) -> Union[torch.Tensor, float]:
        """Return posterior log prob. of parameters, -inf if outside prior."
        
        Args:
            parameters ([np.array]): parameter vector, batch dimension 1
        
        Returns:
            [tensor or -inf]: posterior log probability of the parameters.
        """
        parameters = torch.tensor(parameters, dtype=float32)

        is_within_prior = torch.isfinite(self.prior.log_prob(parameters))
        if is_within_prior:
            target_log_prob = self.posterior_nn.log_prob(
                inputs=parameters.reshape(1, -1),
                context=self.observation.reshape(1, -1),
            )
        else:
            target_log_prob = -float("Inf")

        return target_log_prob

    def pyro_potential(self, parameters: dict) -> torch.Tensor:
        """Return posterior log prob. of parameters, -inf where outside prior.
        
        Args:
            parameters (dict): parameters (from pyro sampler)
        
        Returns:
            torch.Tensor: posterior log probability, masked outside of prior
        """

        parameter = next(iter(parameters.values()))
        # XXX: notice sign, check convention pyro vs. numpy
        log_prob_posterior = -self.posterior_nn.log_prob(
            inputs=parameter, context=self.observation,
        )
        log_prob_prior = self.prior.log_prob(parameter)

        within_prior = torch.isfinite(log_prob_prior)

        return torch.where(within_prior, log_prob_posterior, log_prob_prior)
